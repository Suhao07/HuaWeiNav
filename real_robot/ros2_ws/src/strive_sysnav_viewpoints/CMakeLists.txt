cmake_minimum_required(VERSION 3.8)
project(strive_sysnav_viewpoints)

set(CMAKE_CXX_STANDARD 17)
# 占据栅格遍历属于实时处理路径；显式 Debug 构建仍保留调试配置。
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tare_planner REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters kdtree segmentation)
find_package(OpenCV REQUIRED)

# 复用原始观测模型：这些模块独立于 TSP 求解器和运动目标发布。
add_library(sysnav_viewpoint_models STATIC
  src/rolling_grid/rolling_grid.cpp
  src/rolling_occupancy_grid/rolling_occupancy_grid.cpp
  src/pointcloud_manager/pointcloud_manager.cpp
  src/representation/representation.cpp
  src/utils/misc_utils.cpp
  src/utils/pointcloud_utils.cpp
)
target_include_directories(sysnav_viewpoint_models PUBLIC include ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(sysnav_viewpoint_models
  rclcpp nav_msgs sensor_msgs geometry_msgs visualization_msgs tf2_ros
  pcl_conversions cv_bridge message_filters tare_planner)
target_link_libraries(sysnav_viewpoint_models ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(sysnav_viewpoint_provider src/viewpoint_provider.cpp)
target_link_libraries(sysnav_viewpoint_provider sysnav_viewpoint_models)
ament_target_dependencies(sysnav_viewpoint_provider rclcpp tare_planner)
install(TARGETS sysnav_viewpoint_provider DESTINATION lib/${PROJECT_NAME})
install(FILES LICENSE SOURCE_NOTICE.md SOURCE_SHA256SUMS DESTINATION share/${PROJECT_NAME})

ament_package()
