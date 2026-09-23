// SysNav observation-only extraction. Original algorithm provenance: SOURCE_NOTICE.md.
#include <cmath>
#include <stdexcept>
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tare_planner/msg/object_node_list.hpp>
#include <tare_planner/msg/viewpoint_rep.hpp>
#include "pointcloud_manager/pointcloud_manager.h"
#include "representation/representation.h"
#include "rolling_occupancy_grid/rolling_occupancy_grid.h"
#include "utils/pointcloud_utils.h"

/** Publish SysNav observation representatives using the original occupancy model.
 *
 * 中文核心模型：令 C 为本次射线遍历体素，P 为上次代表点覆盖体素，I=C∩P。
 * 添加候选的条件为 |I| < floor(rho * (2*pi*R^3/3) / h^3) 且 |I|/|C| < rho，
 * 或当前可见对象分数超过 4（新对象权重 1，已有可见 viewpoint 的对象权重 0.2）。
 * Representation 原模型再执行 2 米三维距离去重，实际新增时发布 ID 和里程计时间。
 * 节点只拥有观测输出；任务目标与运动授权由 HuaweiNav 现有链路管理。
 */
class ViewpointProvider : public rclcpp::Node {
 public:
  ViewpointProvider() : Node("sysnav_viewpoint_provider") {
    declare_parameter("world_frame", "map");
    declare_parameter("odom_topic", "/aft_mapped_to_init");
    declare_parameter("cloud_topic", "/cloud_registered");
    declare_parameter("object_topic", "/object_nodes_list");
    declare_parameter("viewpoint_topic", "/viewpoint_rep_header");
    declare_parameter("max_odom_offset_s", 0.25);
    // Defaults follow SensorCoveragePlanner3D::ReadParameters.
    declare_parameter("kKeyposeCloudDwzFilterLeafSize", 0.2);
    declare_parameter("kSurfaceCloudDwzLeafSize", 0.2);
    declare_parameter("kPointCloudRowNum", 20);
    declare_parameter("kPointCloudColNum", 20);
    declare_parameter("kPointCloudLevelNum", 10);
    declare_parameter("kMaxCellPointNum", 100000);
    declare_parameter("kPointCloudCellSize", 24.0);
    declare_parameter("kPointCloudCellHeight", 3.0);
    declare_parameter("kPointCloudManagerNeighborCellNum", 5);
    declare_parameter("rolling_occupancy_grid/resolution_x", 0.3);
    declare_parameter("rolling_occupancy_grid/resolution_y", 0.3);
    declare_parameter("rolling_occupancy_grid/resolution_z", 0.3);
    declare_parameter("kViewPointCollisionMarginZPlus", 0.5);
    declare_parameter("kViewPointCollisionMarginZMinus", 0.5);
    declare_parameter("kRepSensorRange", 5.0);
    declare_parameter("rep_threshold_", 0.1);
    declare_parameter("camera_height_offset_m", 0.265);
  }

  /** Construct original models after shared ownership of the ROS node exists. */
  void initialize() {
    frame_ = get_parameter("world_frame").as_string();
    leaf_ = get_parameter("kKeyposeCloudDwzFilterLeafSize").as_double();
    range_ = get_parameter("kRepSensorRange").as_double();
    ratio_threshold_ = get_parameter("rep_threshold_").as_double();
    double resolution = get_parameter("rolling_occupancy_grid/resolution_x").as_double();
    voxel_threshold_ = int(ratio_threshold_ * (2.0 / 3.0 * M_PI * std::pow(range_, 3)) /
                           std::pow(resolution, 3));
    storage_ = std::make_shared<pointcloud_manager_ns::PointCloudManager>(
        get_parameter("kPointCloudRowNum").as_int(), get_parameter("kPointCloudColNum").as_int(),
        get_parameter("kPointCloudLevelNum").as_int(), get_parameter("kMaxCellPointNum").as_int(),
        get_parameter("kPointCloudCellSize").as_double(), get_parameter("kPointCloudCellHeight").as_double(),
        get_parameter("kPointCloudManagerNeighborCellNum").as_int());
    storage_->SetCloudDwzFilterLeafSize() = get_parameter("kSurfaceCloudDwzLeafSize").as_double();
    grid_ = std::make_shared<rolling_occupancy_grid_ns::RollingOccupancyGrid>(shared_from_this());
    reps_ = std::make_shared<representation_ns::Representation>(shared_from_this(), frame_);
    stacked_.reset(new pcl::PointCloud<pcl::PointXYZ>());
    publisher_ = create_publisher<tare_planner::msg::ViewpointRep>(get_parameter("viewpoint_topic").as_string(), 5);
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(get_parameter("odom_topic").as_string(), 10,
        [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) {
          require_frame(msg->header.frame_id);
          odom_ = msg;
        });
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(get_parameter("cloud_topic").as_string(), 5,
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { on_cloud(*msg); });
    object_sub_ = create_subscription<tare_planner::msg::ObjectNodeList>(get_parameter("object_topic").as_string(), 10,
        [this](tare_planner::msg::ObjectNodeList::ConstSharedPtr msg) { on_objects(*msg); });
  }

 private:
  void require_frame(const std::string& actual) const {
    if (actual != frame_) {
      throw std::runtime_error("viewpoint input frame=" + actual + "; expected world_frame=" + frame_);
    }
  }

  // Original PlanningEnv::UpdateRobotPosition transfers occupancy across rolling cells.
  void update_grid_position(const geometry_msgs::msg::Point& position) {
    bool storage_rolled = storage_->UpdateRobotPosition(position);
    grid_->InitializeOrigin(storage_->GetNeighborCellsOrigin());
    bool grid_rolled = grid_->UpdateRobotPosition(Eigen::Vector3d(position.x, position.y, position.z));
    if (storage_rolled) {
      auto incoming = storage_->GetRolledInOccupancyCloud();
      storage_->ClearNeighborCellOccupancyCloud();
      grid_->UpdateOccupancyStatus(incoming);
    }
    if (grid_rolled) {
      auto outgoing = grid_->GetRolledOutOccupancyCloud();
      storage_->StoreOccupancyCloud(outgoing);
    }
  }

  void on_objects(const tare_planner::msg::ObjectNodeList& msg) {
    for (const auto& object : msg.nodes) {
      if (!object.status) {
        for (int id : object.object_id) {
          reps_->GetObjectNodeRepMapMutable().erase(id);
          reps_->GetLatestObjectNodeIndicesMutable().erase(id);
          for (auto& viewpoint : reps_->GetViewPointRepsMutable()) {
            viewpoint.DeleteObjectIndex(id);
            viewpoint.DeleteDirectObjectIndex(id);
          }
        }
      } else if (!object.cloud.data.empty()) {
        if (object.object_id.empty()) {
          throw std::runtime_error("viewpoint object update requires object_id");
        }
        // mapping 与 provider 共享本次运行的 viewpoint 编号空间。
        if (object.viewpoint_id >= 0) reps_->GetViewPointRepNode(object.viewpoint_id);
        reps_->UpdateObjectNode(std::make_shared<tare_planner::msg::ObjectNode>(object));
        reps_->GetLatestObjectNodeIndicesMutable().insert(object.object_id.front());
      }
    }
  }

  // The original visibility predicate accepts an object when any in-range object voxel is visible.
  bool visible(const geometry_msgs::msg::Point& from, const representation_ns::ObjectNodeRep& object) {
    Eigen::Vector3d origin(from.x, from.y, from.z + get_parameter("camera_height_offset_m").as_double());
    const auto& p = object.GetPosition();
    if ((origin - Eigen::Vector3d(p.x, p.y, p.z)).norm() > range_) return false;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(object.GetCloud(), cloud);
    auto start = grid_->Pos2Sub(origin);
    for (const auto& point : cloud) {
      auto end = grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, point.z));
      if (grid_->InRange(end) && grid_->CheckLineOfSight(start, end)) return true;
    }
    return false;
  }

  double update_visibility_and_score(const geometry_msgs::msg::Point& position) {
    std::vector<int> associated;
    for (int id : reps_->GetLatestObjectNodeIndices()) {
      auto& object = reps_->GetObjectNodeRep(id);
      for (auto& viewpoint : reps_->GetViewPointRepsMutable()) {
        if (viewpoint.HasObjectIndex(id)) {
          object.AddVisibleViewpoint(viewpoint.GetId());
        } else if (object.GetVisibleViewpointIndices().count(viewpoint.GetId())) {
          viewpoint.AddObjectIndex(id);
          viewpoint.AddDirectObjectIndex(id);
          associated.push_back(id);
        } else if (visible(viewpoint.GetPosition(), object)) {
          viewpoint.AddObjectIndex(id);
          object.AddVisibleViewpoint(viewpoint.GetId());
          associated.push_back(id);
        }
      }
    }
    double score = 0.0;
    for (int id : reps_->GetLatestObjectNodeIndices()) {
      const auto& object = reps_->GetObjectNodeRep(id);
      if (visible(position, object)) score += object.GetVisibleViewpointIndices().empty() ? 1.0 : 0.2;
    }
    for (int id : associated) reps_->GetLatestObjectNodeIndicesMutable().erase(id);
    return score;
  }

  void on_cloud(const sensor_msgs::msg::PointCloud2& msg) {
    require_frame(msg.header.frame_id);
    if (!odom_) return;
    if (std::abs((rclcpp::Time(msg.header.stamp) - rclcpp::Time(odom_->header.stamp)).seconds()) >
        get_parameter("max_odom_offset_s").as_double()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "viewpoint waits for timestamp-aligned odometry");
      return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(msg, *cloud);
    if (cloud->empty()) return;
    *stacked_ += *cloud;
    downsizer_.Downsize(cloud, leaf_, leaf_, leaf_);
    const auto& position = odom_->pose.pose.position;
    update_grid_position(position);
    scan_count_ = (scan_count_ + 1) % 5;
    grid_->UpdateOccupancy<pcl::PointXYZ>(cloud);
    grid_->RayTrace(Eigen::Vector3d(position.x, position.y, position.z), scan_count_ == 0);
    if (scan_count_ != 0) return;
    downsizer_.Downsize(stacked_, leaf_, leaf_, leaf_);
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr keypose(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::copyPointCloud(*stacked_, *keypose);
    stacked_->clear();
    double score = update_visibility_and_score(position);
    std::vector<int> current, intersection;
    grid_->GetUpdatedVoxelInds(current);
    misc_utils_ns::SetIntersection(current, previous_, intersection);
    // 空观测没有覆盖率定义；对象可见性仍可独立触发原始 score 分支。
    bool changed = !current.empty() && int(intersection.size()) < voxel_threshold_ &&
        double(intersection.size()) / current.size() < ratio_threshold_;
    if (!changed && score <= 4.0) return;
    auto covered = grid_->GetUpdatedCloudInRange();
    auto before = reps_->GetViewPointReps().size();
    int id = reps_->AddViewPointRep(position, keypose, covered, odom_->header.stamp);
    if (reps_->GetViewPointReps().size() != before) {
      tare_planner::msg::ViewpointRep event;
      event.header = odom_->header;
      event.viewpoint_id = id;
      publisher_->publish(event);
      const auto& point = reps_->GetViewPointRepNodePos(id);
      grid_->UpdateCoveredVoxels(Eigen::Vector3d(point.x, point.y, point.z));
      grid_->GetCurrentObsVoxelInds(previous_);
      reps_->GetLatestObjectNodeIndicesMutable().clear();
    }
    auto& viewpoint = reps_->GetViewPointRepNode(id);
    for (auto& entry : reps_->GetObjectNodeRepMapMutable()) {
      if (!viewpoint.HasObjectIndex(entry.first) && visible(viewpoint.GetPosition(), entry.second)) {
        viewpoint.AddObjectIndex(entry.first);
        entry.second.AddVisibleViewpoint(id);
      }
    }
  }

  std::string frame_;
  double leaf_, range_, ratio_threshold_;
  int voxel_threshold_, scan_count_ = 0;
  std::vector<int> previous_;
  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr stacked_;
  pointcloud_utils_ns::PointCloudDownsizer<pcl::PointXYZ> downsizer_;
  std::shared_ptr<pointcloud_manager_ns::PointCloudManager> storage_;
  std::shared_ptr<rolling_occupancy_grid_ns::RollingOccupancyGrid> grid_;
  std::shared_ptr<representation_ns::Representation> reps_;
  rclcpp::Publisher<tare_planner::msg::ViewpointRep>::SharedPtr publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<tare_planner::msg::ObjectNodeList>::SharedPtr object_sub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ViewpointProvider>();
  node->initialize();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
