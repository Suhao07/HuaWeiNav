# STRIVE 实物部署接口设计

本文档整理 STRIVE 从 HM3D/Habitat 仿真迁移到真实机器人时的接口设计、模块边界、输入输出数据流和上下层规划闭环。目标不是把仿真代码直接搬到机器人上，而是建立一个可插拔的 real-robot runtime，让 STRIVE 的高层语义导航能力复用真实机器人传感器、SLAM 和底层运动控制。

## 1. 设计目标

实物模式需要满足四个边界：

1. 保持 benchmark 模式不变。
2. 将真实传感器数据适配成 STRIVE 内部统一观测格式。
3. STRIVE 只负责语义目标理解、目标确认、关系验证和高层子目标选择。
4. 底层局部避障、路径跟踪、速度控制和安全停止交给 ROS/机器人规划控制栈。

推荐总体形态：

```text
Real sensors / ROS topics
  -> real_robot adapters
  -> STRIVE semantic mapper and instruction planner
  -> NavigationIntent
  -> ROS navigation bridge
  -> local planner / path follower / robot base
```

## 2. 硬件与传感器

STRIVE 论文中的真实平台配置：

```text
Base:
  Mecanum wheel platform

RGB / RGB-D sensor:
  Primary: Ricoh Theta Z1 360-degree panoramic camera
  Optional: Intel RealSense RGB-D camera

Spatial sensor:
  Livox Mid-360 LiDAR

Compatibility:
  LiDAR point clouds can be converted to depth maps when STRIVE needs
  simulation-like RGB-D inputs.
```

默认配置和 SysNav 的 wheeled robot 分支高度一致。RealSense 也可以接入，
但应作为另一个 `CameraAdapter` 实现，而不是替换接口中的相机抽象。
SysNav 使用 ROS2，将真实机器人分解为多个节点：

```text
livox_ros_driver2
  -> Mid-360 point cloud

arise_slam_mid360
  -> state estimation / odometry

semantic_mapping
  -> object detection, SAM2 segmentation, object graph

vlm_node
  -> instruction decomposition, room selection, object/anchor verification

tare_planner / local_planner / pathFollower
  -> exploration, local path, cmd_vel, serial control
```

## 3. 当前 STRIVE 仿真接口

当前 HM3D runtime 的主入口仍是 Habitat 风格：

```text
objnav_benchmark_with_process_obs.py
  -> Habitat env
  -> HM3D_Objnav_Agent
  -> Instruct_Mapper
```

单步输入：

```text
obs["rgb"]   : H x W x 3 uint8
obs["depth"] : H x W x 1 float32
pose         : Habitat sensor state
```

单步输出：

```text
Habitat discrete action:
  move_forward / turn_left / turn_right / stop
```

这套接口在真实机器人上不能直接使用。真实机器人没有 Habitat dense depth，也不应该由高层直接输出离散 action 或 `/cmd_vel`。

## 4. 推荐真实机器人分层

### 4.1 Sensor Adapter

职责：订阅真实传感器和 SLAM 输出，生成 STRIVE 可消费的统一观测。

输入建议：

```text
/camera/image
  sensor_msgs/Image
  Ricoh Theta Z1 panoramic RGB, or RealSense RGB

/registered_scan
  sensor_msgs/PointCloud2
  Livox Mid-360 registered point cloud

/camera/aligned_depth_to_color/image_raw
  sensor_msgs/Image
  Optional RealSense aligned depth

/state_estimation
  nav_msgs/Odometry
  SLAM pose in map frame
```

输出 contract：

```python
RealObservation:
    rgb: np.ndarray
    pointcloud: np.ndarray
    pose: SE3
    timestamp: float
    camera_model: str
    intrinsics: dict
    extrinsics: SE3
    fov: dict
    rgb_pano: np.ndarray | None
    depth_pano: np.ndarray | None
    depth: np.ndarray | None
    depth_valid_mask: np.ndarray | None
    frame_ids: dict[str, str]
```

关键原则：

```text
RGB 是语义主输入。
LiDAR point cloud 是几何主输入。
RealSense aligned depth 是局部 pinhole RGB-D 的直接几何输入。
projected depth 只是兼容层，不能假设和 Habitat dense depth 等价。
```

相机模型建议显式标注：

```python
CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray | None
    camera_model: Literal["panorama", "pinhole"]
    intrinsics: dict
    extrinsics: SE3
    fov: dict
    timestamp: float
```

Theta Z1 对应 `camera_model="panorama"`；RealSense 对应
`camera_model="pinhole"`。上层 planner 不应直接判断相机品牌，而应只看
camera model、FOV、depth availability 和当前 evidence quality。

### 4.2 Depth / Cloud Fusion Adapter

职责：将 LiDAR 点云与相机图像对齐，并统一处理全景和 pinhole 相机。

输入：

```text
rgb or rgb_pano
registered point cloud
camera intrinsics / panorama projection model / pinhole projection model
camera-to-lidar extrinsic
lidar-to-map pose
optional RealSense aligned depth
```

输出：

```text
projected_depth
depth_valid_mask
colored point cloud
camera-frame object points
```

核心算法：

```text
1. 按 timestamp 对齐 RGB、LiDAR、odom。
2. 将 LiDAR 点从 map/sensor frame 变换到 camera frame。
3. 根据 camera_model 选择投影模型：
   - panorama: 使用 Theta 全景投影得到 pixel coordinate。
   - pinhole: 使用 RealSense intrinsics 投影到局部 RGB frame。
4. 对每个 pixel 保留最近 depth。
5. 输出 sparse depth 和 valid mask。
```

注意事项：

```text
不要用 sparse depth 直接替代 Habitat depth 做所有三维重建。
小物体附近的 depth 缺失必须保留为 unknown，而不是插值成虚假表面。
RealSense FOV 较窄，不具备 Theta 一次观测 360 度上下文的能力。
```

RealSense 接入策略：

```text
RealSense RGB + aligned depth
  -> RealSenseCameraAdapter
  -> pinhole CameraFrame
  -> depth 直接反投影为局部点云
  -> 若需要全景上下文：
       机器人原地旋转采集多帧
       或只在局部视角内执行 detection / verification
```

经验判断：

```text
Theta 更适合房间级语义判断、快速全局观察和远距离上下文。
RealSense 更适合近距离目标确认、小物体检测和稳定 RGB-D 几何。
```

### 4.3 Detector Adapter

职责：统一不同检测器的输出格式。

当前 STRIVE benchmark 使用：

```text
MMDINOSAM_Perceiver / GroundingDINO + SAM
```

SysNav 真实机器人使用：

```text
YOLO World / YOLOE tracking
SAM2 segmentation
```

推荐统一输出：

```python
DetectionFrame:
    image: np.ndarray
    boxes_xyxy: np.ndarray
    labels: list[str]
    confidences: list[float]
    masks: list[np.ndarray] | None
    track_ids: list[int] | None
    timestamp: float
```

可插拔实现：

```text
SimulationDetectorAdapter
  -> calls current STRIVE perceiver

ROSDetectionResultAdapter
  -> subscribes /detection_result

DirectRealDetectorAdapter
  -> runs detector inside STRIVE real runtime
```

建议优先复用 SysNav 的 `/detection_result`，降低实物模式初期风险。

### 4.4 Semantic Map Adapter

职责：把真实检测、分割、点云、pose 融合成对象图和导航图。

STRIVE 当前内部状态：

```text
mapper.objects
mapper.nodes
mapper.room_nodes
mapper.grid_map
mapper.frontiers
mapper.navigable_pcd
mapper.obstacle_pcd
```

推荐导出统一快照：

```python
SemanticMapSnapshot:
    timestamp: float
    robot_pose: SE3
    objects: list[ObjectNode]
    nav_nodes: list[NavNode]
    rooms: list[RoomNode]
    frontiers: list[Frontier]
```

对象节点：

```python
ObjectNode:
    uid: str | int
    label: str
    confidence: float
    position: np.ndarray
    bbox2d: list[float] | None
    bbox3d_center: np.ndarray | None
    bbox3d_extent: np.ndarray | None
    image_ref: str | None
    pointcloud_ref: str | None
    room_id: int | None
    visible_viewpoints: list[int]
    verified_state: str
```

这层应保持独立于 ROS message。ROS bridge 可以负责消息转换，STRIVE 内部只消费 Python contract。

## 5. 高层导航器接口

真实机器人模式下，STRIVE 高层不输出 discrete action，而输出语义导航意图。

推荐 contract：

```python
NavigationIntent:
    mode: str
    goal_pose: Pose2D | None
    target_object_uid: str | None
    anchor_object_uid: str | None
    relation_edge_id: str | None
    stop_allowed: bool
    reason: str
```

典型 mode：

```text
explore_room
go_to_frontier
go_to_object
go_to_anchor
improve_view
stop
wait
```

高层模块职责：

```text
Instruction parser:
  原始自然语言 -> InstructionPlan

Concept grounding:
  target / anchor / support region 概念归一

Runtime concept matcher:
  observed object -> target/anchor concept

Constraint evaluator:
  room / attribute / count / sequence / relation

Dynamic relation verifier:
  object-object relation edge, e.g. on / near / inside

Final instruction verifier:
  原始 prompt 是否已满足

View controller:
  语义满足但视角不足时，围绕 pinned target/relation 改善视角
```

这部分可以直接复用当前 instruction adapter 的核心设计。

## 6. 下层规划与控制接口

SysNav 下层已经给出了很好的真实机器人闭环：

```text
high-level planner
  -> /way_point

localPlanner
  subscribes:
    /state_estimation
    /registered_scan
    /terrain_map
    /way_point
    /navigation_boundary
    /added_obstacles
    /check_obstacle
  publishes:
    /path
    /slow_down
    /free_paths

pathFollower
  subscribes:
    /state_estimation
    /path
    /joy
    /speed
    /stop
  publishes:
    /cmd_vel
  optional:
    serial /dev/ttyACM0
```

STRIVE 实物模式建议只发布 waypoint：

```text
NavigationIntent.goal_pose
  -> geometry_msgs/PointStamped
  -> /way_point
```

不要让 STRIVE 高层直接发 `/cmd_vel`。原因：

```text
cmd_vel 需要实时安全控制。
局部避障、急停、速度限制和手柄接管都应该在下层闭环中完成。
语义层调用 VLM/LLM，延迟不可控，不适合直接控制底盘。
```

### 6.1 STRIVE 仿真 action API 与实物 motion API 的差异

当前 STRIVE/Habitat runtime 使用同步离散控制接口。高层 planner 先生成
连续空间中的 waypoint 或 better-view viewpoint，然后通过 Habitat
shortest-path follower 转成离散动作：

```text
STRIVE selected waypoint/viewpoint
  -> habitat_waypoint()
  -> Habitat shortest-path follower
  -> discrete action: STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT
  -> env.step(action)
  -> synchronous RGB-D observation
```

这套接口成立依赖 Habitat 提供的几个仿真假设：

```text
global navmesh is available
shortest-path query is reliable
agent motion is discretized and deterministic enough
env.step(action) immediately returns synchronized RGB / depth / pose
collision and kinematic details are absorbed by the simulator
```

真实机器人不具备这个同步 `env.step` 抽象。SysNav 的底层是连续控制链路：

```text
semantic / exploration planner
  -> geometry_msgs/PointStamped on /way_point
  -> localPlanner selects a collision-aware local path
  -> nav_msgs/Path on /path
  -> pathFollower tracks path and publishes /cmd_vel
  -> robot moves asynchronously
  -> sensors publish RGB / LiDAR / odom with latency
```

因此实物模式不能复刻 Habitat discrete action。STRIVE 应保留
“生成可解释语义子目标和视角目标”的能力，但必须把“执行目标”的责任交给
一个异步 motion layer。

### 6.2 MotionController contract

建议在 real-robot runtime 中新增 `MotionController` 抽象，统一仿真和实物
两种执行模型：

```python
class MotionController:
    def send_goal(self, goal: MotionGoal) -> str:
        """Submit a navigation or viewpoint goal and return a goal id."""

    def poll_status(self, goal_id: str) -> NavigationStatus:
        """Return reached / running / blocked / timeout / preempted."""

    def cancel(self, goal_id: str) -> None:
        """Cancel the active lower-level motion goal."""

    def hold(self) -> None:
        """Ask the lower layer to stop safely without taking over velocity control."""
```

仿真实现可以包装当前 Habitat action loop：

```text
HabitatDiscreteController
  MotionGoal -> self.waypoint
  poll_status -> repeated planner.get_next_action() and env.step(action)
```

实物实现应包装 SysNav ROS 接口：

```text
RosWaypointController
  MotionGoal -> /way_point
  poll_status -> /state_estimation + /path + timeout + progress monitor
  hold -> /stop or controller-specific safe hold
```

上层 planner 只看到 `MotionGoal` 和 `NavigationStatus`，不关心底层是离散
action、ROS waypoint，还是某个实物平台的自定义导航接口。

### 6.3 ViewpointGoal 与异步证据采集

STRIVE 的 better-view 逻辑会生成多个候选 viewpoint。仿真中这些 viewpoint
可以直接通过 Habitat pathfinder 可达性检查和离散动作执行；实物中 viewpoint
必须被建模为一个异步目标：

```python
ViewpointGoal:
    pose: Pose2D | Pose3D
    look_at: Point3D | None
    target_uid: str | None
    anchor_uid: str | None
    relation_edge_id: str | None
    purpose: explore | verify_target | verify_relation | improve_view
    tolerance: dict
```

执行结果也必须显式返回：

```python
ViewpointResult:
    status: reached | blocked | timeout | preempted
    final_pose: Pose
    evidence: ViewEvidence | None
    path_length: float | None
    reason: str
```

真实机器人最终确认流程应是：

```text
send ViewpointGoal
  -> wait for NavigationStatus
  -> if reached or best available: acquire RGB / LiDAR / pose snapshot
  -> project target / anchor evidence
  -> final verifier checks semantic, relation, and view quality
  -> accept / try next viewpoint / abandon this instance
```

这里的关键边界是：VLM 负责判断语义、关系和视觉证据质量；motion layer 负责
判断是否到达、是否可达、是否被障碍阻断、是否超时。VLM 不应直接声明物理
可达性，motion layer 也不应直接判断自然语言任务是否满足。

### 6.4 与 SysNav 的对接方式

SysNav 已经把真实机器人底层拆成可复用链路：

```text
/way_point -> localPlanner -> /path -> pathFollower -> /cmd_vel / serial
```

STRIVE 实物模式应先对接这个最小公共接口，而不是复制 SysNav 的整套 planner。
推荐桥接：

```text
NavigationIntent(mode="go_to_object" / "improve_view")
  -> MotionGoal / ViewpointGoal
  -> RosWaypointController publishes /way_point
  -> SysNav localPlanner/pathFollower executes continuous motion
  -> bridge monitors odom/path progress
  -> STRIVE acquires evidence and updates verifier state
```

如果后续接入不同实物平台，只需要替换 `MotionController` 和 sensor adapters：

```text
Mecanum + SysNav localPlanner:
  RosWaypointController

Nav2-based platform:
  Nav2ActionController

Quadruped / humanoid:
  PlatformMotionController

Offline bag replay:
  ReplayMotionController
```

这样 STRIVE 的 instruction parser、concept grounding、relation verifier、
final verifier 和 view-control state 都可以保持平台无关。

## 7. 上下层闭环

推荐真实机器人闭环：

```text
1. SensorAdapter 读取 RGB / LiDAR / odom。
2. DetectorAdapter 生成 detection frame。
3. SemanticMapBuilder 更新对象图、房间、frontier 和导航节点。
4. Instruction planner 读取 SemanticMapSnapshot。
5. Planner 输出 NavigationIntent。
6. ROSNavigationBridge 发布 /way_point。
7. localPlanner 基于点云和地形生成 /path。
8. pathFollower 生成 /cmd_vel 或串口控制。
9. 机器人移动产生新 RGB / LiDAR / odom。
10. 高层根据新证据更新 verifier / ledger / relation edge。
```

停止条件应由三部分共同决定：

```text
instruction_satisfied == true
view_sufficient_for_stop == true
robot_stable_or_goal_reached == true
```

其中：

```text
instruction_satisfied:
  FinalInstructionVerifier 对原始自然语言确认。

view_sufficient_for_stop:
  目标/anchor/relation 在当前视角中有足够证据。

robot_stable_or_goal_reached:
  底层报告 waypoint 到达，或已无法继续改善视角且证据充分。
```

## 8. 与 SysNav 的对照

| 层级 | SysNav | STRIVE 当前 | STRIVE 实物建议 |
| --- | --- | --- | --- |
| 传感器 | ROS2 topics | Habitat observation | RealObservationAdapter |
| RGB / RGB-D | `/camera/image`, optional RealSense aligned depth | `obs["rgb"]`, `obs["depth"]` | CameraFrame |
| 几何 | `/registered_scan` | `obs["depth"]` | pointcloud + optional projected/aligned depth |
| 位姿 | `/state_estimation` | Habitat sensor state | SE3 pose |
| 检测 | YOLOE / YOLO World | MMDINO/SAM | DetectorAdapter |
| 分割 | SAM2 | SAM | 可插拔 |
| 语义地图 | `/object_nodes_list` | mapper.objects | SemanticMapSnapshot |
| 任务理解 | VLM node | instruction_adapter | 复用 instruction_adapter |
| 房间选择 | VLM room navigation | room_policy / LLM | 可接 room snapshot |
| 高层输出 | `/way_point` | discrete action | NavigationIntent |
| 局部规划 | localPlanner | Habitat SPF | ROSNavigationBridge |
| 底盘控制 | pathFollower / serial | env.step | pathFollower / cmd_vel |

## 9. 推荐代码结构

建议在 `robotic` 分支新增：

```text
real_robot/
  __init__.py
  contracts.py
  detector_vocabulary.py
  sysnav_ros_adapters.py
  sysnav_runtime.py
  camera_adapter.py
  observation_adapter.py
  depth_projection.py
  detector_adapter.py
  semantic_map_adapter.py
  motion_controller.py
  navigation_bridge.py
  runtime_node.py
  ros2_ws/
    src/
      tare_planner/
      semantic_mapping/
      strive_sysnav_bringup/

docs/
  real_robot_deployment.md
```

模块职责：

```text
contracts.py
  已定义平台无关 contract：
  RealObservation, DetectionFrame, ObjectNodeSnapshot, RoomSnapshot,
  SemanticMapSnapshot, NavigationIntent, MotionGoal, ViewpointGoal,
  NavigationStatus, ViewEvidence, RuntimeDecision。
  该层只依赖 Python 标准库，不引入 ROS、Habitat、numpy 或 detector 实现。

detector_vocabulary.py
  读取 SysNav objects.yaml，生成 DetectorVocabulary；
  记录 detector_name、label_space、prompt_space、is_instance 和 label provenance；
  只做 detector config 内的 canonical/prompt 字面匹配，不做自然语言 alias 推断。

sysnav_ros_adapters.py
  第一版 SysNav 复用层：
  RosDetectionResultAdapter 将 /detection_result 转为 DetectionFrame；
  RosObjectNodeAdapter 将 /object_nodes_list 转为 ObjectNodeSnapshot；
  RosRoomNodeAdapter 将 /room_nodes_list 转为 RoomSnapshot；
  RosWaypointController 将 MotionGoal 发布为 /way_point。

sysnav_runtime.py
  SysNavSemanticMapBridge 缓存 /object_nodes_list 和 /room_nodes_list；
  SysNavInstructionRuntime 将 SemanticMapSnapshot 交给高层策略并发布 waypoint；
  ViewpointEvidenceLoop 执行 ViewpointGoal 的异步证据采集和 final verifier；
  LatestObservationEvidenceProvider 从最新 RealObservation 和 crop provider 构造 ViewEvidence。

camera_adapter.py
  封装 Theta panorama 与 RealSense pinhole RGB-D 相机差异。

observation_adapter.py
  ROS topic buffer, timestamp sync, pose extraction。

depth_projection.py
  LiDAR point cloud -> panorama/pinhole sparse depth。

detector_adapter.py
  STRIVE detector / ROS detection result 的统一封装。

semantic_map_adapter.py
  将 real observation + detection 转成 mapper update 输入或 map snapshot。

navigation_bridge.py
  NavigationIntent -> MotionGoal / ViewpointGoal，读取 path/odom/stop 状态。

motion_controller.py
  定义 MotionController，并实现 RosWaypointController、ReplayMotionController
  等底层执行适配。高层不直接依赖 Habitat discrete action 或 ROS topic。

runtime_node.py
  实物模式主循环。

ros2_ws/src/tare_planner
  message-only SysNav 兼容包，提供 DetectionResult、ObjectNode、RoomNode 等消息。
  第一版不编译完整 TARE C++ planner，避免检测/建图迁移被局部规划依赖阻塞。

ros2_ws/src/semantic_mapping
  已迁入 SysNav detection_node 和 semantic_mapping_node。
  detection_node 订阅 /camera/image，发布 /detection_result；
  semantic_mapping_node 订阅 /detection_result、/registered_scan、/state_estimation，
  发布 /object_nodes_list。

ros2_ws/src/strive_sysnav_bringup
  启动 detection_node 和 semantic_mapping_node 的 launch-only 包。
```

## 10. 实施路线

### Phase 0: 离线 bag replay

目标：不接真车，先用 rosbag 或导出的 topic 文件验证接口。

输入：

```text
recorded /camera/image
recorded /registered_scan
recorded /state_estimation
```

输出：

```text
RealObservation 序列
debug projected_depth
debug object snapshots
```

验收：

```text
RGB、点云、pose 时间同步误差可记录。
投影 depth 和 RGB 对齐可视化正常。
mapper 不崩溃。
```

### Phase 1: 真实观测适配

目标：实现 `RealObservationAdapter`。

已完成的基础边界：

```text
real_robot/contracts.py
  定义 sensor, detection, semantic map, motion intent, viewpoint,
  navigation status, evidence, runtime decision 的统一数据契约。

tests/test_real_robot_contracts.py
  约束 contract 层保持平台无关，并验证 detection / viewpoint /
  verifier evidence 的基础语义。
```

验收：

```text
可持续输出 rgb_pano / pointcloud / pose。
所有 frame id 和 extrinsic 明确记录。
遇到缺帧时返回 wait，而不是阻塞 planner。
```

### Phase 2: 检测与对象图接入

目标：先复用 SysNav `/detection_result` 或 STRIVE detector 生成 `DetectionFrame`。

第一版直接复用 SysNav detector + semantic mapping：

```text
/camera/image
  -> SysNav detection_node
  -> /detection_result
  -> SysNav semantic_mapping_node
  -> /object_nodes_list
  -> RosObjectNodeAdapter
  -> SemanticMapSnapshot
  -> STRIVE instruction_adapter / concept matcher / final verifier
```

这条链路不把 STRIVE detector 迁移到 SysNav，也不让 STRIVE 重写 SysNav
semantic_mapping。STRIVE 只接收 SysNav 已经稳定维护的 object node / room node，
再做 prompt-first 指令解析、concept grounding、relation verifier 和 final verifier。

检测器词表处理：

```text
SysNav objects.yaml
  -> DetectorVocabularyAdapter
  -> DetectorVocabulary(label_space, prompt_space, is_instance)
  -> RosDetectionResultAdapter / RosObjectNodeAdapter metadata
  -> STRIVE concept grounding context
```

重要边界：

```text
adapter 不把 raw detector label 静默改成任务概念。
adapter 只记录 label_provenance：
  raw_detector_label
  detector_name
  config_path
  known_in_detector_vocabulary
  canonical_label
  prompt_labels
  matched_by
  is_instance

例如 detector 输出 "trash can"：
  ObjectNodeSnapshot.label 仍是 "trash can"；
  metadata.label_provenance.canonical_label 可以记录为 "garbage_bin"；
  是否满足用户说的 "bin" / "garbage can" 仍由 concept matcher / verifier 判断。
```

已实现模块：

```text
RosDetectionResultAdapter
  /detection_result -> DetectionFrame，并写入 detector_vocabulary 与 per-bbox label_provenance

RosObjectNodeAdapter
  /object_nodes_list -> ObjectNodeSnapshot，并写入 label_provenance

RosRoomNodeAdapter
  /room_nodes_list -> RoomSnapshot

SysNavSemanticMapBridge
  缓存 object/room list topic，并构建只读 SemanticMapSnapshot。
```

验收：

```text
ObjectNode uid 稳定。
同一对象不会频繁漂移。
目标/anchor matcher 可以消费真实对象。
```

### Phase 3: 高层输出 waypoint

目标：让 STRIVE 输出 `NavigationIntent`，通过 bridge 发布 `/way_point`。

已实现模块：

```text
SysNavInstructionRuntime
  SemanticMapSnapshot -> high_level_policy.decide()
  -> NavigationIntent
  -> MotionGoal
  -> RosWaypointController.send_goal()
  -> /way_point
```

核心边界：

```text
STRIVE 只输出语义 intent 和 waypoint。
SysNav localPlanner/pathFollower 继续负责局部避障、路径跟踪和速度控制。
```

验收：

```text
localPlanner 收到 waypoint。
pathFollower 能跟踪 path。
STRIVE 不直接控制 cmd_vel。
```

### Phase 3.5: MotionController 与异步 viewpoint 执行

目标：把 STRIVE 当前的同步 action loop 抽象为平台无关的 motion contract。

已实现模块：

```text
ViewpointEvidenceLoop
  ViewpointGoal
    -> goal.as_motion_goal()
    -> motion_controller.send_goal()
    -> poll NavigationStatus until reached / blocked / timeout
    -> evidence_provider.capture()
    -> final_verifier.verify()
    -> ViewpointResult

LatestObservationEvidenceProvider
  latest RealObservation + object crop provider
    -> ViewEvidence
```

伪代码：

```python
goal_id = motion_controller.send_goal(viewpoint_goal.as_motion_goal())
status = motion_controller.poll_status(goal_id)

while not status.is_terminal():
    status = motion_controller.poll_status(goal_id)

if status.succeeded():
    evidence = evidence_provider.capture(viewpoint_goal, status)
    decision = final_verifier.verify(evidence, context)
else:
    decision = {"decision": "motion_failed", "status": status.status}
```

关键原则：

```text
只有 motion layer 报告 reached 后，才采集 final verifier evidence。
blocked / timeout 不能伪造成成功视角。
VLM 判断语义与视觉证据质量；motion layer 判断是否到达和是否可执行。
```

验收：

```text
HabitatDiscreteController 可以包装原仿真 step loop。
RosWaypointController 可以发布 /way_point 并轮询到达状态。
ViewpointGoal 可以携带 look_at / target_uid / relation_edge_id。
ViewpointResult 可以记录 reached / blocked / timeout 和最终 evidence。
final verifier 只在 evidence acquisition 之后调用。
```

该阶段完成后，STRIVE 的 better-view 逻辑就不再依赖 Habitat discrete action；
实物模式只需要替换底层 controller 和 observation adapter。

### Phase 4: 目标确认闭环

目标：复用 instruction verifier、relation verifier、view-control。

第一版接口已经保留闭环入口：

```text
ViewEvidence.for_verifier()
  提供 image_ref、bbox、pose、target uid、anchor uid、relation edge id、
  view quality、verifier payload。

ViewpointEvidenceLoop.final_verifier
  可接现有 FinalInstructionVerifier 的薄封装，
  也可接 SysNav/CogNav 的独立 verifier node。
```

当前实现没有把现有 Habitat agent 的 `final_instruction_check()` 直接搬到实物
runtime。原因是该函数依赖仿真 agent 状态、日志路径和 mapper 内部对象。实物模式应
通过薄 wrapper 把 `ViewEvidence` 转成现有 verifier 需要的 candidate/evidence
payload，而不是让 real-robot runtime 继承 Habitat agent。

验收：

```text
red chair:
  错误实例会被 ledger 屏蔽。

book on shelf:
  anchor/target/relation edge 可验证。

cup:
  能使用 support region 或 anchor-first 策略去桌面/柜台区域搜索。
```

### Phase 5: 真车小范围测试

测试顺序：

```text
find chair
find table
find cup
red chair
book on shelf
cup on desk
```

每个测试必须保存：

```text
raw observation log
projected depth visualization
semantic map snapshot
NavigationIntent trace
VLM raw response
final verifier result
lower planner status
```

## 11. 关键风险

### 11.1 时间同步

真实系统中 RGB、LiDAR、odom 不会天然同步。必须记录：

```text
rgb_stamp
cloud_stamp
odom_stamp
max_sync_delta
```

超出阈值时应返回 `wait` 或降低该帧置信度。

### 11.2 坐标系

必须显式维护：

```text
map
vehicle/base_link
sensor/lidar
camera
```

禁止在业务逻辑里散落手写坐标转换。所有转换应集中到 adapter 或 geometry module。

### 11.3 Sparse depth

LiDAR projected depth 是稀疏深度，不能假设每个 RGB pixel 都有可靠 depth。

必须在 object reconstruction 中保留：

```text
valid_depth_ratio
point_count
geometry_confidence
```

### 11.4 VLM 调用延迟

实物模式必须使用缓存：

```text
candidate-instance verification cache
relation pair cache
final verifier cache
view-control pinned state
```

同一对象、同一关系、同一证据图不能重复调用 VLM。

### 11.5 安全控制

高层语义模块不承担急停职责。最低要求：

```text
manual joystick override
/stop topic
local obstacle check
planner heartbeat
watchdog timeout
```

## 12. 初始接口草案

### 12.1 已落地的 contract 边界

`real_robot/contracts.py` 目前只表达跨模块数据边界，不负责订阅 topic、
调用 detector、构图或控制底盘。后续任何实物 adapter 都应先把平台相关数据
转成这些 contract，再交给 STRIVE 高层模块：

```text
RealObservation
  一次同步观测，包含 robot_pose、CameraFrame、pointcloud_ref 和 frame id。

DetectionFrame
  某个 camera frame 上的 detector 输出，包含 bbox、label、confidence、
  track id 和可选 mask 引用。

SemanticMapSnapshot
  高层 planner 的只读地图视图，包含 ObjectNodeSnapshot、RoomSnapshot、
  ViewpointSnapshot 和 FrontierSnapshot。

NavigationIntent
  STRIVE planner 输出的语义动作意图，例如 go_to_object、go_to_anchor、
  improve_view、verify_relation、stop。

MotionGoal / ViewpointGoal
  navigation bridge 消费的运动目标。ViewpointGoal 额外携带 look_at、
  target uid、anchor uid 和 evidence requirements。

NavigationStatus
  下层运动控制返回的异步状态：running、reached、blocked、timeout、
  preempted、failed。

ViewEvidence
  到达 viewpoint 后采集的图像、bbox、pose 和质量信息，供 final verifier
  或 relation verifier 使用。

RuntimeDecision
  每轮实物 runtime 的可回放决策记录，连接 intent、motion goal、
  lower planner status 和 verifier decision。
```

该边界对应的职责划分是：

```text
VLM / verifier:
  判断 semantic_satisfied、relation_satisfied、view_evidence_quality。

Mapper / planner:
  维护对象、房间、frontier、候选 viewpoint 和目标状态。

Motion controller:
  判断 reached、blocked、timeout、path progress、collision feasibility。

Runtime:
  把上述证据组织成 RuntimeDecision 并落盘。
```

关键实现原则：

```text
contract 层不保存 numpy array 或 ROS message，只保存 image_ref、pointcloud_ref
和 JSON-friendly metadata。
```

这样可以让 live ROS、rosbag replay、离线日志和仿真 adapter 共用同一套
高层 planner / verifier 接口。

### 12.2 SysNav ROS adapter

`real_robot/sysnav_ros_adapters.py` 已实现第一版 SysNav 适配器。该模块的
设计目标是“复用 SysNav 已有 detector、semantic mapping 和 waypoint
controller”，而不是把 ROS 逻辑扩散到 STRIVE instruction planner。

当前 adapter：

```text
DetectorVocabularyAdapter
  输入：SysNav semantic_mapping/config/objects.yaml
  输出：DetectorVocabulary
  作用：向 concept grounding 暴露 detector label_space / prompt_space / is_instance。

RosDetectionResultAdapter
  输入：tare_planner/DetectionResult
  输出：DetectionFrame
  映射字段：track_id, x1/y1/x2/y2, label, confidence, inline image summary,
  label_provenance。

RosObjectNodeAdapter
  输入：tare_planner/ObjectNode 或 ObjectNodeList
  输出：ObjectNodeSnapshot
  映射字段：object_id, label, position, bbox3d, img_path, viewpoint_id, status,
  label_provenance。

RosRoomNodeAdapter
  输入：tare_planner/RoomNode 或 RoomNodeList
  输出：RoomSnapshot
  映射字段：id, centroid, neighbors, area, is_connected, room_mask reference。

RosWaypointController
  输入：MotionGoal / ViewpointGoal.as_motion_goal()
  输出：geometry_msgs/PointStamped on /way_point
  状态：返回 NavigationStatus，第一版可接入外部 status_provider。
```

SysNav 侧 topic 约定：

```text
/camera/image
  SysNav detection_node 订阅。

/detection_result
  SysNav detection_node 发布，semantic_mapping_node 订阅。

/object_nodes_list
  SysNav semantic_mapping_node 发布，STRIVE RosObjectNodeAdapter 订阅或离线读取。

/room_nodes_list
  SysNav room_segmentation 发布，STRIVE RosRoomNodeAdapter 订阅或离线读取。

/way_point
  STRIVE RosWaypointController 发布，SysNav local planner/path follower 执行。
```

关键边界：

```text
STRIVE 不直接订阅 /camera/image 做第一版实物检测。
STRIVE 不直接发布 /cmd_vel。
STRIVE 不修改 SysNav semantic_mapping_node 内部对象融合逻辑。
STRIVE 只消费 ObjectNodeSnapshot / RoomSnapshot，并发布 MotionGoal。
STRIVE adapter 不做 detector label alias；语义映射进入 concept grounding / verifier。
```

如果后续需要把 STRIVE detector 迁移到 SysNav，应作为替换
`detection_node` 的独立 ROS node，而不是塞进 `RosDetectionResultAdapter`。
adapter 层只做消息转换，不承载模型推理。

### 12.3 Runtime skeleton

```python
class RealRobotNavigator:
    def __init__(
        self,
        observation_adapter,
        detector_adapter,
        mapper,
        high_level_policy,
        navigation_bridge,
    ):
        ...

    def step(self, instruction=None) -> RuntimeDecision:
        observation = self.observation_adapter.read()
        if observation is None:
            wait_intent = NavigationIntent(
                mode=MotionGoalMode.WAIT,
                reason="waiting for synchronized observation",
            )
            return RuntimeDecision(timestamp=time.time(), intent=wait_intent)

        detections = self.detector_adapter.detect(observation)
        snapshot = self.mapper.update_real(observation, detections)
        intent = self.high_level_policy.decide(snapshot, instruction)
        motion_goal = intent.to_motion_goal()
        goal_id = self.navigation_bridge.send_goal(motion_goal)
        status = self.navigation_bridge.poll_status(goal_id)
        return RuntimeDecision(
            timestamp=observation.timestamp,
            intent=intent,
            motion_goal=motion_goal,
            navigation_status=status,
        )
```

`RuntimeDecision` 需要落盘，便于复盘：

```python
RuntimeDecision:
    timestamp: float
    intent: NavigationIntent
    motion_goal: MotionGoal | None
    navigation_status: NavigationStatus | None
    accepted_candidate_uid: str | None
    accepted_relation_edge_id: str | None
    verifier_decision: dict | None
    lower_planner_state: dict | None
```

### 12.4 已实现的最小闭环

当前最小闭环已经能用 fake ROS message / fake controller 在单元测试中跑通：

```text
SysNav object/room list
  -> SysNavSemanticMapBridge.build_snapshot()
  -> high_level_policy.decide(snapshot, instruction)
  -> NavigationIntent.to_motion_goal()
  -> RosWaypointController.send_goal()
  -> NavigationStatus
  -> RuntimeDecision
```

viewpoint 证据闭环：

```text
ViewpointGoal
  -> RosWaypointController / MotionController
  -> wait NavigationStatus.REACHED
  -> LatestObservationEvidenceProvider.capture()
  -> final_verifier.verify(ViewEvidence, context)
  -> ViewpointResult
```

这两条链路对应下一步 ROS live node 的最小实现：

```text
subscribe /object_nodes_list
subscribe /room_nodes_list
read latest RealObservation
call STRIVE policy / verifier
publish /way_point
write RuntimeDecision and ViewpointResult logs
```

### 12.5 本仓库内 ROS overlay

SysNav detector/mapping 已作为 vendor ROS overlay 迁入当前 workspace：

```text
real_robot/ros2_ws/src/tare_planner
real_robot/ros2_ws/src/semantic_mapping
real_robot/ros2_ws/src/strive_sysnav_bringup
```

构建：

```bash
bash scripts/build_real_robot_ros_ws.sh
```

运行：

```bash
export SYSNAV_DETECTOR_MODEL_TYPE=yoloe
export SYSNAV_DETECTOR_MODEL_PATH=/path/to/yoloe-26x-seg.engine
export SYSNAV_SAM2_CHECKPOINT=/path/to/sam2.1_hiera_base_plus.pt

bash scripts/run_sysnav_detection_mapping.sh
```

这条链路启动后，STRIVE 侧期望看到：

```text
/detection_result
/object_nodes_list
/room_nodes_list
```

其中 `/room_nodes_list` 仍取决于是否同时启动 room segmentation / local planner
相关节点。第一版 overlay 已保证 detector 和 semantic mapping 可在 STRIVE
workspace 内构建；完整 SysNav C++ local planner 后续应作为单独迁移阶段处理。

### 12.6 单镜像实物部署

实物部署不应依赖两个运行中的容器。当前仓库提供的实物镜像是单镜像方案：

```text
strive-real-robot:humble
  contains STRIVE high-level code
  contains real_robot adapters and runtime contracts
  contains vendored SysNav semantic_mapping overlay
  contains tare_planner ROS message definitions
  contains strive_sysnav_bringup launch package
```

仿真 benchmark 镜像和实物镜像的职责不同：

```text
strive-hm3d:local
  用于 Habitat / HM3D / OVON benchmark
  保留原始仿真依赖，避免被 ROS Humble / Ubuntu 22.04 依赖污染

strive-real-robot:humble
  用于真实机器人部署
  基于 ROS2 Humble，运行 SysNav detector/mapping 和 STRIVE 上层语义策略
```

也就是说，真机部署只启动 `strive-real-robot:humble` 一个容器；不需要同时启动
HM3D benchmark 容器。两个镜像只是开发阶段的不同运行目标，不是部署时的双容器架构。

构建单实物镜像：

```bash
IMAGE_TAG=strive-real-robot:humble \
INSTALL_LLM_DEPS=1 \
INSTALL_ML_DEPS=0 \
bash docker/build_real_robot.sh
```

`INSTALL_ML_DEPS=0` 是轻量验证模式，可用于确认 ROS overlay、topic adapter 和
launch 包是否完整。真机运行 detector/mapping 时需要安装或挂载相应模型运行依赖：

```bash
IMAGE_TAG=strive-real-robot:humble-runtime \
INSTALL_LLM_DEPS=1 \
INSTALL_ML_DEPS=1 \
bash docker/build_real_robot.sh
```

权重不建议写入镜像层，而应通过 volume 和环境变量传入：

```bash
export SYSNAV_DETECTOR_MODEL_TYPE=yoloe
export SYSNAV_DETECTOR_MODEL_PATH=/abs/path/to/models/yoloe-26x-seg.engine
export SYSNAV_SAM2_CHECKPOINT=/abs/path/to/models/sam2.1_hiera_base_plus.pt

IMAGE_TAG=strive-real-robot:humble-runtime \
bash docker/run_real_robot_sysnav_stack.sh
```

`docker/run_real_robot_sysnav_stack.sh` 会把上述权重文件所在目录只读挂载进容器，
并在容器内使用相同绝对路径读取模型。脚本也会透传 `LLM_PROVIDER`、
`ARK_API_KEY`、`LLM_MODEL`、`LLM_API_BASE_URL`、`MAP_PROVIDER`、`AMAP_KEY`
等运行时环境变量，因此真机部署不需要同时启动另一个 STRIVE 容器。

这种边界的原因是：Habitat 仿真栈和 ROS2 Humble 真机栈的系统依赖不同。强行把
Habitat、ROS、SysNav detector、SAM2、TensorRT 全部塞进同一个“万能镜像”，会让
CUDA、OpenCV、PCL、PyTorch 和 Python ABI 的冲突不可控。单实物镜像已经满足部署
需求；仿真镜像仅保留为 benchmark 开发和回归验证环境。

## 13. 当前结论

实物模式的核心路线是：

```text
STRIVE high-level semantic navigation
  + SysNav-style ROS sensor and motion stack
```

STRIVE 不需要重写底层局部规划，也不应该直接控制速度。它应输出可解释的语义子目标；真实机器人底层负责安全、连续、实时地到达该子目标。

`real_robot/contracts.py`、`real_robot/sysnav_ros_adapters.py`、
`real_robot/sysnav_runtime.py` 和 `real_robot/ros2_ws` 已完成第一版实物接口骨架
和 SysNav detector/mapping 本仓库迁移。
下一步建议实现一个最小 ROS runtime 或离线 bag replay：

```text
read /object_nodes_list and /room_nodes_list
  -> build_semantic_map_snapshot()
  -> instruction_adapter.decide()
  -> NavigationIntent.to_motion_goal()
  -> RosWaypointController.send_goal()
```

在这条链路稳定前，不建议迁移 STRIVE detector 或重写 SysNav semantic mapping。
