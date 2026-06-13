cmake_minimum_required(VERSION 3.8)
project(tare_planner)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# 核心：第一版只迁移 SysNav detector/mapping 所需消息接口。
# 完整 TARE C++ exploration/local planner 后续应作为独立包迁入，不能阻塞检测栈编译。
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectNode.msg"
  "msg/ObjectNodeList.msg"
  "msg/RoomNode.msg"
  "msg/RoomNodeList.msg"
  "msg/RoomType.msg"
  "msg/ViewpointRep.msg"
  "msg/DetectionResult.msg"
  "msg/RoomEarlyStop1.msg"
  "msg/VlmAnswer.msg"
  "msg/ObjectType.msg"
  "msg/NavigationQuery.msg"
  "msg/TargetObjectInstruction.msg"
  "msg/TargetObject.msg"
  "msg/TargetObjectWithSpatial.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
