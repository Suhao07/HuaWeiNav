# VLN 实物部署状态与 TODO

> 更新日期：2026-09-08
> 代码基线：本地 `realworld` 最新为 `ea37c38`（包含 `ad08c1a` 的低速 OCC 融合接受结论和 `3978b5c` 的 semantic mapping 合并索引修复）；`origin/realworld` 仍落后 6 个提交，未在本轮推送。新机器人工作区已用同一修复提交为 `0acbf3e` 并用于镜像重建，旧 `/home/orin26/HuaweiVLN` 保留为只读回滚工作区，未被覆盖。
> 目标平台：Orin-26、Intel RealSense D435i、Livox MID-360
> 当前安全状态：感知与影子控制验证；真实运动未批准

本文是实物部署的单一进度清单。设计接口见
[`real_robot_framework.md`](real_robot_framework.md)，实测证据保留在对应日期的
`real_robot_*_evidence_*.md` 文档中。

2026-09-21：复用 FWTW 松灵 rosbridge 的具体接入工作见
[`real_robot_agilex_execution_checklist.md`](real_robot_agilex_execution_checklist.md)。
该清单按 SysNav 原生下层路线拆分类型适配、遥测、模式联动、合同及分阶段验收；
仅为本地实现审计与执行计划，不改变本文既有实物验收状态。

### 2026-09-08 清洁工作区部署记录

- 新工作区：`/home/orin26/HuaweiVLN_deployments/realworld_2c0b783`；旧
  `/home/orin26/HuaweiVLN` 未修改。
- 旧工作区备份证据：`/home/orin26/HuaweiVLN_backups/HuaweiVLN_pre_clean_20260907T151320Z/`
  （staged/unstaged patch、Git 状态和未跟踪清单）。
- 迁移内容限定为共同 Git 基线、已确认的传感器/坐标/融合/控制 adapter 原理代码、候选
  标定文件和五个现有模型资产；未复制旧 logs、outputs、build/install 或缓存。
- 新镜像 `huawei-vln-realworld:orin-r36.5` 已重建，ROS overlay 的 7 个包编译成功。
- 外部 Livox/Point-LIO 仍由 `/home/orin26/code` 所有；通过项目包装器以 ROS 参数开启
  `publish.scan_publish_en` 和 `publish.scan_bodyframe_pub_en`，未编辑该工作区。
- LIO-only 诊断报告：`logs/diagnostics/lio_dds_20260907T154932Z.md`；四个 topic
  均收到真实消息。独立 60 秒测量约为 `/cloud_registered_body` 9.43 Hz、odom 99.5 Hz。
- semantic mapping 已在新容器中启动，输入明确为 `/cloud_registered_body`；对象节点
  `/huawei_vln/d435i_object_nodes_list` 约 0.4--0.5 Hz，样本 frame 为 `map`、坐标为米级，
  融合日志持续出现有效投影点；对象图像已正常写入 `output/object_images/`。
- shadow adapter 仅使用 `output_enabled=false`；合成 PointStamped 已完成 ego 转换日志，
  未创建 `/waypoint` publisher，也未发布 `/cmd_vel`。
- 本轮仍未批准真实底盘控制；当前实际配置的 controller contract 仍为
  `approval_status: unapproved`。

### 2026-09-08 实时验收续测

- Point-LIO/Livox：`/livox/lidar` 约 100 Hz、`/livox/imu` 约 200 Hz，
  `/cloud_registered_body` 约 9.4 Hz，`/cloud_registered` 约 100 Hz，
  `/aft_mapped_to_init` 约 100 Hz；60 秒 `tf2_echo camera_init -> base` 收到 57 个
  连续样本，除启动前两次 frame 尚未出现外没有 `future extrapolation`。
- D435i：RGB、aligned depth、CameraInfo 均有实时消息；RGB/Depth 约 15 Hz、
  `1280x720`，编码分别为 `bgr8`/`16UC1`，CameraInfo frame 为
  `d435i_camera_color_optical_frame`。该 frame 与 projection YAML 的历史别名不同，
  当前仍通过显式 OCC 投影配置处理，未广播伪 TF。
- Semantic mapping：发现对象合并索引缺陷并在 `3978b5c` 修复；容器内重建后连续运行
  超过 2 分钟无 `IndexError`、`Traceback` 或 `future extrapolation`。真实对象节点
  `/huawei_vln/d435i_object_nodes_list` 约 0.55 Hz，样本包含 `printer` 和合并后的
  SysNav object ID，坐标为 `map` 米制。
- `semantic_snapshot`：以自然语言 `Find the printer`（显式 `dataset_target=printer`）
  运行 `dry_run=true`，已记录 `InstructionPlan`、目标 UID `sysnav_object:3` 和地图坐标；
  因没有输入 `/viewpoint_rep_header`，`sysnav_viewpoint_bridge` 没有可发布的
  `ViewpointPose`，runtime 持续 `WAIT`，未生成 waypoint 或速度命令。
- Viewpoint bridge：节点已启动并订阅 `/viewpoint_rep_header`、`/huawei_vln/d435i_object_nodes_list`
  和 `/aft_mapped_to_init`；当前 `/viewpoint_rep_header` publisher 为 0，
  `/strive/sysnav/viewpoint_pose` 没有实际消息，因此真实 SysNav viewpoint 仍待外部
  SysNav 规划器启动后验收。
- 30 分钟资源/温度监控已完成；最终报告保留在
  `logs/diagnostics/resources/20260908T034523Z/`。

### 2026-09-08 监控收尾

- 30 分钟监控已完成：`/cloud_registered_body` 平均 9.441 Hz（窗口范围
  9.202--9.677 Hz），`/aft_mapped_to_init` 平均 99.661 Hz（最低窗口
  80.992 Hz）；Point-LIO 进程 CPU 54.7--62.2%，RAM 4.98--7.29 GB，
  `tj` 温度 61.5--72.7°C。监控目录内没有 `future extrapolation`、崩溃或 traceback。
  该监控的 `process.log` 以 Point-LIO 为跟踪对象，不能替代“修复后 detector + semantic
  mapping + Point-LIO 连续 30 分钟”的专门整栈 soak；整栈长测仍需单独安排。
- 语义合并索引修复已固化为镜像
  `huawei-vln-realworld:orin-r36.5-fix`（ID `27d904a21207`，7 个 ROS 包构建成功）；
  原 `huawei-vln-realworld:orin-r36.5` tag 保留未覆盖。临时检测/建图容器已停止，
  Livox、Point-LIO 和 D435i 外部进程保持运行。

勾选规则：只有代码、命令输出或真机记录能够直接证明的事项才标记为完成。单元测试、
bag replay、HIL 和影子 topic 不能替代真实底盘验收。

## 1. 当前结论

### 2026-11-03 先验地图合并部署（只读，无运动）

- 部署：`~/HuaweiVLN_deployments/realworld_7927a75`（HEAD `ce0fc8b`，工作区干净），镜像
  `huawei-vln-realworld:orin-r36.5-agilex-7927a75`；`ce0fc8b` 只改宿主机侧 `docker_en.sh`，
  镜像无需重建。旧目录 `realworld_2c0b783`、`~/HuaweiVLN` 与旧镜像 tag 未改动。
- 先验地图：手机地图包 `outputs/prior_bundles/phone-d655a5ce86-concepts100-conf1-labels-compact`
  （197 物体、0 房间）。容器内离线检查与本机逐字段一致：加载 0.6 s、初始化+4 次查询约
  1.9 s、RSS 117 MB；点对拟合残差约 1e-16。
- 正式入口：`docker_en.sh` 原先不挂载目录型地图包、也不透传
  `STRIVE_PRIOR_MAP_SOURCE/ALIGNMENT` 与 `LLM_OFFLINE`，已修复。修复后实测：runtime 加载
  场景 `d655a5ce86`，`alignment=unavailable → prompt_context_only`；detector 订阅 D435i、
  发布 `/huawei_vln/d435i_detection_result`（0.5 Hz，即 profile 的 2.0 s 节流）；
  `/way_point`、`/waypoint`、`/cmd_vel`、`/navflow_cmd_vel`、`/strive/test_way_point`、
  `/run_management/virtual_joy` 均无 publisher，runtime 本身无 publisher。
- LIO：开机自启的 `livox_odom` 在 NTP 校时跳变后不再输出（`/livox/lidar` 无消息）。经用户
  批准，用 `start_real_robot_sensors.sh orin26_livox_mid360_d435i start` 重启（项目脚本
  `start_orin_lio_for_strive.sh`），传感器检查 PASS。
- 整栈只读实测（`docker_en.sh`，语义建图 + runtime）：readiness 通过；语义建图融合出
  `cart` 并以 `frame_id=camera_init` 送达 runtime；先验 printer 排序分 1.098（语义 0.998 +
  exact 0.1，geometry 0），房间与 frontier 偏置为空，`alignment_mismatch=1`、无冲突；先验
  无 `cart`，不绑定（正确）；无 printer 在视野，决策 `WAIT`。live→先验的正向绑定尚未验证，
  需视野内出现先验中唯一标签的物体。
- 几何：手机地图坐标系 `map_z_up` 与 Point-LIO `camera_init` 之间没有配准（后者原点为
  每次 LIO 启动位姿），launch 默认值 `identity` 对该地图在几何上是错的；完成标定前必须用
  `STRIVE_PRIOR_MAP_ALIGNMENT=unavailable`（只作排序与提示上下文）。

### 先验地图 ↔ Point-LIO 对齐流程（2026-11-03 已在车上走通一遍）

`camera_init` 是 LIO 启动时重力对齐后的车体（`aft_mapped`，MID-360 内置 IMU，距雷达约
3 cm）位姿；定位模式（`location_mode`）下它是所载 PCD 的坐标系，因此“固定起点标记”和
“持久 LIO 地图”共用同一份标定。

1. 地面贴起点标记（车体位置 + 朝向），在手机地图（`map_z_up` 平面 x,y）上量出车体中心
   `base_xy` 和朝向（`--base-yaw-deg`，或车头正前方 ≥0.3 m 的一点 `--heading-xy`）。
2. 车放在标记上建图：`LIO_PCD_SAVE_EN=1 bash scripts/start_real_robot_sensors.sh
   orin26_livox_mid360_d435i start`，由人推车或底盘所有者遥控覆盖场景（本项目不发运动指令）。
3. `bash scripts/save_orin_lio_map.sh <dir>`：向 Point-LIO 窗格发 Ctrl-C（SIGINT 才会写
   `scans.pcd`；`kill-session` 会丢图），等新图写完后拷为 `<dir>/lio_map.pcd` 并写
   `lio_map.json`（sha256、建图会话时间），最后关闭 `livox_odom`，避免后续启动复用旧会话。
4. 生成对齐：`python3 scripts/build_prior_lio_alignment.py <prior_bundle> --base-xy x,y
   --heading-xy x,y --lio-map <dir>/lio_map.json --output <dir>/prior_alignment.json`。
   外参默认取静态 TF `aft_mapped→base`（x=-0.2，yaw=-1.5708）；脚本自检标记点车体在
   runtime 系应为 (-0.2, 0)，并记录先验地图 sha256 与 LIO 地图 sha256。
5. 之后每次启动：`LIO_LOCATION_MODE=1 LIO_MAP_PATH=<dir>/lio_map.pcd` 启动传感器。
   **定位模式必须收到一次 `/initialpose` 才会做 NDT+ICP 重定位**（`laserMapping.cpp`
   `initial_pose()` 无初值直接返回，不会默认单位阵），在此之前不发 `/aft_mapped_to_init`，
   `start_real_robot_sensors.sh` 的话题检查也因此不会 PASS。在标记上启动时发单位位姿：
   `ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped
   "{header: {frame_id: camera_init}, pose: {pose: {orientation: {w: 1.0}}}}"`（z 由回调取
   附近地图点高度；发布需用户批准；唯一订阅者是 `laserMapping`）。随后
   `STRIVE_PRIOR_MAP_ALIGNMENT=<dir>/prior_alignment.json` 启动 `docker_en.sh`。
   不用定位模式时，只要 LIO 在标记上启动，同一 JSON 也成立。

约束与保护：
- `start_orin_lio_for_strive.sh` 定位模式要求 `LIO_MAP_PATH` 为已存在的绝对路径，且与
  `LIO_PCD_SAVE_EN` 互斥；配置校验先于停旧会话，配置错误时不会把运行中的 LIO 杀掉。
- runtime 的 `identity`/`unavailable` 标签改为 `camera_init`；affine 对齐若遇到观测
  `frame_id` 与标定的 runtime 坐标系不一致，自动降级为 `prompt_context_only`。
- 无法自动发现的失效：LIO 不在标记上、且未用定位模式启动（坐标系名不变但原点变了）。
- 待实物验收：标记量测精度（真实手机地图）、远离起点时的重定位成功率、对齐后先验物体与
  live 物体的同标签位置残差（本次两侧检测器标签不重合，未测到）。

#### 2026-11-03 车上全流程试运行（静止车、D435i 当前视角建小图）

部署 `~/HuaweiVLN_deployments/realworld_17fe73f`，镜像 `huawei-vln-realworld:orin-r36.5-agilex-17fe73f`；
车全程静止，仅批准发布一次 `/initialpose`，无任何运动话题。

1. 建图模式启动 LIO（`pcd_save_en:=true`），采 5 帧 D435i RGB-D 与 `/aft_mapped_to_init`
   （位姿抖动 ≤1 cm，偏航 ≤0.08°）。先验坐标系取采集时 base 的水平位姿，
   `camera_to_map = inv(T_ci_P)·T_ci_body·T_body_lidar·inv(T_camera_from_lidar)`；本机
   Grounding DINO+SAM（`indoor_concepts`，max depth 4 m）建图得 5 个物体（staircase、
   trash bag×2、door、table），产物 `output/prior_bundles/d435i-20261103T214230`。
2. `save_orin_lio_map.sh output/lio_maps/20261103T214230`：79 万点，sha256 `1738e889…`。
3. `build_prior_lio_alignment.py --base-xy 0,0 --base-yaw-deg 0`：工具假设的标记位姿与采集时
   实测 base 位姿（(-0.195,-0.007)，-89.98°）相差 0.9 cm / 0.02°。
4. 定位模式重启：发单位 `/initialpose` 后 ICP 得分 0.457（阈值 0.5，余量小——地图只有
   静止一个视角，降采样后 6199 点），稳定后位姿 (0.004,-0.010) 与建图时相差约 0.3 cm。
5. 整栈 dry-run（`semantic_snapshot`，“Find the staircase”）：51 步对齐始终为 `affine_2d`、
   `runtime_frame_id=camera_init`、`alignment_mismatch=0`，未降级；top_object 为先验
   staircase；runtime 无发布者，`/cmd_vel` 等运动话题均不存在。runtime 中机器人在先验系
   (0.008,0.204)，与理论 (0.010,0.204) 一致。live YOLOE 只检出 shoe/cart，与先验标签不重合，
   未发生 live→先验绑定。
6. 几何残差（标签无关）：先验第 0 帧深度点经对齐变换后，对 10 s 在线 `/cloud_registered`
   和 LIO 地图做最近邻。近处物体掩码点残差中位数 5.5–7 cm（staircase、trash bag），3.6 m
   外的 door 35 cm；不对齐（直接当作 camera_init）时 48 cm。偏航网格搜索在 **-7°** 最优
   （对 LIO 地图中位数 19.1→8.3 cm），图像空间 LiDAR 投影左移约 110 px（≈7°）深度一致性
   最好，俯视图中先验墙面需转 -7° 才与 LIO 墙线平行。结论：标记→LIO 对齐链误差约 1 cm，
   **主要残差来自 D435i–MID-360 外参（candidate_20260826）约 7° 的偏航不一致**；该外参同样
   用于在线语义建图的 LiDAR→图像投影（运行时用雷达点投到 RGB 再按掩码取点，7°≈110 px，
   窄物体会取到背景点）。
7. 外参偏航修正（用户 2026-11-03 批准，不重标定）：用同一组数据（5 帧对齐深度逐像素中值 +
   LIO 地图，z-buffer 后 8×8 px 块级比较，相对误差 <5% 记一致），只扫一个参数——绕雷达 z
   轴、过相机光心的偏航。最优 **+6.25°**，一致率 31.4%→64.6%，相对误差中位数 8.4%→3.2%；
   左/右半幅分别 6.75°/5.75°。三轴旋转 ICP（64.1%）不更好，且左右半幅俯仰差 12.6°；
   六自由度 ICP 平移漂移约 30 cm（58.0%）——墙地为主的单视角约束不住俯仰与平移，故只修
   偏航。新外参 `real_robot/calibration/orin26_d435i_mid360_yawfix_20261103.json`，投影配置
   `projection_orin26_d435i_mid360.yaml` 升为 `calibration_version: 3`（注释里保留 v2 数值
   便于回退）。待验收：换朝向/场景静止采样做交叉检验；车上部署后看在线物体位置残差。
8. 车上复核（同日，部署 `realworld_7906fc9`，车未动、同一朝向）：用**实时**
   `/cloud_registered_body` 累积 5 s（37 帧、10.3 万点，位姿抖动 ≤1.1 cm）+ 5 帧对齐深度，
   直接调用运行时 `load_projection_config(..., require_calibrated=True)` 与 `project()`，
   指标同上：v2 一致率 31.7%（左/右 11.0%/45.8%）、中位数 7.8%；v3 69.3%（85.3%/57.9%）、
   中位数 2.9%。点云与修正所用的累计 LIO 地图相互独立，结论一致；但视角相同，换朝向交叉
   检验仍待做。先验链路（`camera_to_map` 回投深度点对 LIO 地图）残差中位数 19.1→9.1 cm，
   偏航最优点由 -6.5° 收到 -1.5°（8.4 cm，曲线平坦）。容器内确认加载 v3，整栈 dry-run 语义
   建图正常出 shoe 物体（框底 -0.05 m，邻近地面 -0.02 m），无运动话题。
   同时发现镜像缺 `lap`：ultralytics 仅在首次 `track()` 且联网检测通过时运行时 pip 安装，
   当次检测失败，`detection_node` 启动即崩；Dockerfile 已改为构建时补装（ab2b462）。车上
   `realworld_ab2b462` / 镜像 `huawei-vln-realworld:orin-r36.5-agilex-ab2b462` 已构建，镜像内
   `import lap` 为 0.5.14。注意：Orin 开机时钟停在旧时间，开机自启的 Point-LIO 在 NTP 校时
   跳变后停止输出（话题在、无消息，2026-11-04 再次复现），需重启 LIO 才恢复。
9. 手机先验地图 ↔ 20261103 LIO 地图免标记配准（2026-11-04，**否决**）：取两图离地
   0.3–1.5 m 墙带俯视点，全局 1° 步长相关 + 截尾 ICP，最优 yaw 352.2°、5 cm 内 81%，
   墙体主方向（mod 90°）也与之相差 ≤3°。但物体级核对不成立：按该对齐，车上 D435i 看到的
   上行楼梯/门/电动车落在手机图的办公工位区（显示器、电竞椅），同类物体最近 1.1–5.7 m；
   LIO 图有升到约 4 m 的上行楼梯、无低于车所在地面的点，而手机扫描单层（轨迹高差 0.17 m），
   唯一楼梯是下行梯（低于地面 0.24 m）。结论：车当前不在手机扫描区域内（大概率在其下一层
   楼梯间），墙带高重合是矩形几何的假匹配，候选对齐已删除。手机图起点标定与静止绑定复测
   需先把车放到手机扫描的楼层，或在车所在区域重新手机扫描。
10. 车周重扫 fbb15f81e4 → dry-run 目标规划（2026-11-04，车静止，无运动话题）：
    - 先验：`outputs/prior_bundles/phone-fbb15f81e4-concepts100-conf1-compact`（214 物体）。LIO
      在车当前位姿重启（会话 14:47:52，非存图模式），累积实时点云与手机点云配准：墙带候选
      106.4°/95.9° 由 3D 全点云区分，取 yaw 105.206°、t=(-1.679, 0.942)（5 cm 内 25%，中位
      23 cm——手机图远大于 LIO 局部图，比例仅作区分用）；LIO 原点附近手机点高度符合底盘，
      手机画面目视确认原点落在底盘。车体在先验系 (-1.626, 0.749)、yaw 15.2°，以此作起点标记
      生成 `alignment_camera_init_20261104T1447.json`。
    - 视点链三处缺陷已修：dry-run 路径不启动 viewpoint provider/bridge；首个视点早于语义建图
      就绪被丢；SysNav 物体合并后 uid 变化，视点可见性历史随之丢失（d00b8bb、cf8d331）。修后
      “Go to the cart” 解析出 cart（exact、terminal，中心 (-0.32,-1.39) `camera_init`），目标
      视点即车当前位姿（唯一看到 cart 的视点）。
    - 下层规划：隔离 domain 77 回放本会话 LIO 录包（`run_lower_planner_bag_replay.sh`，不启
      pathFollower）。目标距车 0.7 m 时只得单点停车路径（`goalBehindRange=0.8` 内目标在车后侧
      先停）；改为 1.0 m 接近点（离 cart 中心约 0.42 m）3.4 s 得有效路径（101 点、约 1 m），
      主路径偏约 50° 绕开直线上约 1.2 m 处、离地约 0.2 m 的 cart 边缘。
    - 先验绑定：先验词表无 cart（该处标为 staircase），matched=false 正确；在线 YOLOE 词表与先验
      标签交集仅 cabinet/chair/desk/keyboard/mouse/plant，当前视野无这些物体。正向绑定复测需
      车转向先验系约 (-1.0,-0.25) 的椅子。
    - 运动前置修复（d673298）：runtime/MotionServer/provider/bridge 的 `world_frame` 与 terrain
      输出 frame 原默认 `map`，profile 改为 `camera_init` 并由入口透传。
    - 新部署复测时视点 0 再次丢失：provider 收 5 帧点云（约 0.5 s）即建视点，早于 DDS 匹配
      mapping/bridge 订阅，volatile 事件丢失后静止车不再建新视点。provider 改为订阅数达
      `min_viewpoint_subscribers`（默认 2）后才建视点；车上 dry-run 复测 cart 再次解析成功。
    - 当前 LIO 会话非存图模式，完整 LIO 建图需在同一位置以 `LIO_PCD_SAVE_EN=1` 重启后推车。

### 2026-09-07 状态刷新

用户已确认 OCC 项目的 D435i--MID-360 外参与当前安装状态一致，并批准当前 candidate
用于低速 VLN 试运行。当前 profile 已写入 `calibration_status: calibrated`，外参文件为
`real_robot/calibration/orin26_d435i_mid360_candidate_20260826.json`（2026-11-03 起投影配置
改用其偏航修正版 `orin26_d435i_mid360_yawfix_20261103.json`，见上节第 7 条）。因此，**低速场景的
空间融合不再是部署阻塞项**。动态时间偏移和严格像素投影验收属于高动态/高精度增强项，
不再阻塞当前低速 VLN 验收。

最新 candidate 的证据摘要：

- 静态有效阶段：12/18（近距 6 个阶段因棋盘超出画面不可用）；
- 平面 P90：train 0.0105 m，held-out 0.0112 m；
- 深度 P90：train 0.0684 m，held-out 0.0692 m；
- held-out 深度内点率：79.11%；
- 像素代理 P90：train 11.46 px，held-out 11.27 px；
- 动态时间偏移：train -15 ms，held-out +20 ms，差异 35 ms，仍未验证。

因此当前允许启动 semantic mapping 做静态/低速数据链试运行；
`time_offset_status: unvalidated` 继续保留为事实记录，但不否定低速融合可用性。
真实 waypoint、`/cmd_vel` 和底盘运动仍受独立控制安全门禁约束。
Point-LIO 动态 TF 仍需连续窗口测试；出现 `future extrapolation` 时不能称为稳定部署完成。

当前已打通两条无运动链路：

```text
D435i RGB + MID-360/Point-LIO
  -> SysNav detector
  -> semantic mapping
  -> /huawei_vln/d435i_object_nodes_list

VLN /way_point (PointStamped, world frame)
  -> WaypointFormatAdapter
  -> /strive/test_waypoint_array (Float32MultiArray, ego frame)
```

第一条链路已经产生真实对象节点，第二条链路已经在影子 topic 上完成坐标与消息转换。
这些结果证明低速数据链和接口可运行；它们仍不能证明底盘能够安全执行 waypoint，
也不能替代高动态时间偏移/像素精度增强验收。

当前低速 VLN 仍缺三类硬条件：

1. 在修复后的 detector + semantic mapping + Point-LIO 组合上完成专门的整栈长时间
   soak，并把动态 TF、资源、掉帧和温度结果写入同一运行证据；
2. 接入并验收真实 SysNav viewpoint provider，完成不发布 waypoint 的任务级语义闭环；
3. 由底盘所有者批准 controller contract，完成 bench、低速和安全测试后，才能打开真实
   waypoint/底盘输出。

RGB--LiDAR 动态时间偏移和严格像素投影误差不再列为低速 VLN 的硬条件；若目标扩展到
高速或高精度融合，再单独开启增强验收。当前 `calibration_status: calibrated` 是经用户
确认的 OCC 低速运行决定，不应在后续清单中反复作为 P0 阻塞项。

本轮已经完成的**软件闭环**如下，但这些内容仍不等同于实物验收：

```text
SemanticMapSnapshot
  -> NavigationIntent
  -> SysNavGoalResolver
  -> MotionGoal / WAIT
  -> active goal tracking
  -> REACHED 后的新鲜 RGB / detection
  -> ViewEvidence / verifier 生命周期
```

其中 `SysNavGoalResolver` 只消费下层提供的 viewpoint pose，不再把对象中心或房间质心
伪装成执行位姿；当前迁移的 ROS 消息还没有携带完整 viewpoint pose，因此正式语义路径
在 bridge 尚未收到时间对齐的 viewpoint pose 时显式保持 `WAIT`。
`scripts/check_real_robot_acceptance.sh` 当前离线结果为 `113 passed`，该结果只证明软件
合同和状态生命周期，不代表真实控制器已接管底盘。

## 2. 已完成与证据边界

| 模块 | 当前状态 | 可证明内容 | 不能据此声称 |
|---|---|---|---|
| 平台无关 contract | 已完成 | `SemanticMapSnapshot`、`NavigationIntent`、`MotionGoal`、`NavigationStatus`、`ViewEvidence` 已实现并有单元测试 | 真车已运动 |
| SysNav detector | 真机数据链已验证 | D435i 图像进入 YOLOE，检测 topic 有真实消息 | 所有目标类别均稳定检测 |
| SysNav semantic mapping | 低速真机数据链已验证 | 2026-09-08 新工作区实测对象节点约 0.4--0.5 Hz，坐标为米级，含有效投影点 | 对象精度和长时间稳定性仍需任务级验收 |
| LIO 输入 | 60 秒真机数据链已验证 | `/cloud_registered_body` 约 9.4 Hz，`/aft_mapped_to_init` 约 100 Hz | 长时 TF、漂移和资源稳定性 |
| D435i--MID-360 标定 | 低速试运行已接受 | OCC 外参/current candidate 已导入，profile 为 `calibration_status=calibrated` | 高动态时间偏移和严格像素精度增强验收 |
| Instruction runtime | 软件闭环完成（离线） | snapshot、SysNav viewpoint resolver、active goal、REACHED 后新鲜取证和 verifier 生命周期有离线测试 | 已在 Orin 上运行真实自然语言任务 |
| SysNav viewpoint bridge | 软件 bridge 已实现，真机数据待验收 | `/viewpoint_rep_header` + 同时刻 odom -> `/strive/sysnav/viewpoint_pose`；object-viewpoint 观测历史、共视交集和无效时间/frame 的 WAIT 已有代码和测试 | Orin 上已验证 viewpoint 频率、时间偏移和 frame |
| 远程 LVLM | HTTP/schema 软件接口完成 | 商业 API、自部署 OpenAI-compatible 服务和 schema smoke 已实现 | 机器人网络上的 p95 延迟、超时恢复已验收 |
| Waypoint adapter | 真机影子验证 | `PointStamped -> Float32MultiArray`、world-to-ego、过期丢弃在影子 topic 验证 | 外部 `/waypoint` 已接收或底盘已执行 |
| SysNav 原生下层 | 代码迁移、离线/HIL 覆盖 | `localPlanner`、`pathFollower`、motion action、安全 mux 的软件链存在 | 它已成为 Orin 底盘的生产控制器 |
| 外部机器人控制器 | 只读接口审计 | 已观察 `/waypoint` 和 `/topoplan/reached_goal` 的历史接口 | blocked/timeout/cancel、急停和所有权已确认 |
| 真实先验地图 | 手机地图已部署（unavailable 对齐），几何对齐工具就绪、实物未标定 | `PriorMapData`、alignment 和 runtime context 可接入；起点标记/LIO 地图对齐流程见 §1 | 已有与真机坐标对齐的语义先验地图 |
| 房间节点 | adapter 已实现，producer 未确认 | `/room_nodes_list` 可选消费接口存在 | 当前 semantic mapping 会发布真实房间图 |

## 3. 与 SysNav 的边界

SysNav 参考实现的原生运动链为：

```text
/state_estimation + /registered_scan + /terrain_map + /way_point
  -> localPlanner
  -> /path
  -> pathFollower
  -> /cmd_vel
```

当前 Orin 上观察到的是另一条机器人自有链路：

```text
/waypoint (Float32MultiArray, 历史代码按 ego [x, y] 使用)
  -> 外部局部规划/PD controller
  -> AgileX bridge / mux
  -> 底盘
```

两条链路不能同时启用。迁移的 SysNav `pathFollower` 与外部控制器若同时发布速度，会造成
控制权不唯一。当前实施方案固定为：

- VLN 保持 `MotionGoal` 和 `/way_point` 为平台无关高层边界；
- Orin 第一阶段复用现有 `WaypointFormatAdapter` 接入机器人自有 `/waypoint`；
- 速度、局部避障、急停和人工接管继续由机器人自有控制链负责；
- 迁移的 SysNav 原生下层只保留为可选后端，在独立完成物理验收前不得与外部控制器并行启动。

这是当前证据下修改最少的路径。现阶段不新增通用 controller registry，也不再抽象一层
“万能底盘接口”；等控制器所有者给出真实消息和反馈合同后，只实现一个明确的 Orin
status adapter。

## 4. 当前待做清单

### P0：传感器、坐标与整栈运行稳定性

- [x] 新的机器人工作区和 `huawei-vln-realworld:orin-r36.5-fix` 镜像已建立；旧
  `/home/orin26/HuaweiVLN` 未覆盖，外部 `/home/orin26/code` 未修改。
- [x] Livox、Point-LIO、D435i 实时输入已恢复；当前 `/livox/lidar` 约 100 Hz、
  `/livox/imu` 约 200 Hz、`/cloud_registered_body` 约 9.4 Hz、
  `/aft_mapped_to_init` 约 100 Hz，且 Point-LIO 使用 `prop_at_freq_of_imu=false`。
- [x] 完成独立 60 秒 LIO/TF 检查；`camera_init -> base` 有连续样本，检查窗口未出现
  `future extrapolation` 或 `past extrapolation`。
- [x] D435i RGB、aligned depth、CameraInfo 的实时分辨率、编码和内参已记录；detector
  使用 BGR→RGB、CUDA、`imgsz=320`，与当前运行 profile 一致。
- [x] OCC 外参已按当前安装状态作为低速融合标定值使用；不再把严格动态时间偏移或像素
  误差作为低速 VLN 阻塞项。
- [ ] 在修复后的 detector + semantic mapping + Point-LIO **同时运行**时完成一段专门的
  30 分钟以上整栈 soak，记录各节点频率、掉帧、CPU/GPU、内存和温度。已有 30 分钟记录
  主要跟踪 Point-LIO，不能替代这项。
- [ ] 明确记录 `camera_init -> aft_mapped -> base` 的长时 TF 时间戳，以及下游若需要时
  `camera_init -> d435i_camera_color_optical_frame` 的实际连接方式；当前映射使用显式 OCC
  投影配置，禁止广播未经确认的伪 TF。
- [ ] 将本地 `realworld` 的最新完整 SHA（当前 `ea37c38`）推送到 `origin/realworld`，
  并让机器人部署 manifest 固定到同一完整 SHA；这属于可追溯性，不是传感器启动阻塞。

验收出口：整栈运行证据能对应到同一代码 SHA、镜像 ID 和 profile，且全程未发布真实
`/waypoint`、`/way_point` 或 `/cmd_vel`。

### 历史记录：2026-08-21 无运动实物执行记录

> 以下条目保留作历史证据，不覆盖 2026-09-08 的当前状态；其中当时的 LIO 无消息和旧镜像
> 结果已由后续清洁工作区部署和实时检查更新。

- Profile check：`orin26_livox_mid360_generic_rgb` 与 `orin26_livox_mid360_d435i` 均通过；两者均保持 `START_SEMANTIC_MAPPING=false`、`dry_run=true`、真实运动关闭。
- 镜像与软件：`huawei-vln-realworld:orin-r36.5`=`25048b859e0d`；ROS workspace 7 包编译成功；离线 acceptance 为 `113 passed`。
- GPU/模型 smoke：`torch cuda=True`，YOLOE detector init 成功；USB 相机只读 smoke 收到 `/camera/image` 和 `/camera_info`，分辨率 `1920x1080`、编码 `rgb8`、frame `default_cam`。
- LVLM/schema/verifier 影子测试：在 `--network none` 的最终镜像中运行结构化输出、非法 JSON 重试、保守 fallback、final verifier 和 deployment receipt 测试，`16 passed`；未进行真实远程 LVLM 延迟验收。
- runtime shadow：D435i profile runtime smoke 和 live `semantic_snapshot` dry-run 均保持 `WAIT`，原因是实时 `object_nodes`、pose、image 尚未同时到达；未发布 `/way_point`、`/waypoint` 或 `/cmd_vel`。
- waypoint adapter shadow：节点日志为 `output_enabled=False`；`/waypoint`、`/strive/test_waypoint_array`、`/way_point`、`/cmd_vel` 均未发现 publisher。
- LIO 只读诊断：报告为 `logs/diagnostics/lio_dds_20260821T133422Z.md`；`/livox/lidar`、`/livox/imu`、`/cloud_registered`、`/aft_mapped_to_init` 均未收到实际 header。Point-LIO 参数显示 `publish.scan_publish_en=False`，因此 semantic mapping 继续关闭；未重启或修改外部 LIO。
- 现场 bag 清单中暂未发现同时包含 `/viewpoint_rep_header` 与 `/object_nodes_list` 的 bag，因此 viewpoint replay 留待取得对应 bag 后执行。

### P1：真实对象图、viewpoint 与无运动任务闭环

- [x] 使用已批准 OCC candidate 从 profile 启动 semantic mapping，输入固定为
  `/cloud_registered_body`；对象节点已有真实样本，坐标为 `map` 米制。
- [x] semantic mapping 的对象合并索引缺陷已在 `3978b5c` 修复，并已重建到机器人镜像；
  修复后运行未再出现该 `IndexError`。
- [x] 完成 `SemanticMapSnapshot -> NavigationIntent -> SysNavGoalResolver ->
  MotionGoal/WAIT` 软件链路和 `REACHED -> fresh evidence -> verifier` 生命周期的离线测试。
- [x] `semantic_snapshot` 已在 Orin 上以 `dry_run=true` 读取真实对象节点；无可执行 viewpoint
  时保持 `WAIT`，没有发布任何 waypoint 或速度命令。
- [ ] 记录真实对象节点的位置稳定性、UID 连续性、误检和重复簇；至少覆盖桌椅、柜体和
  小物体，并保留对应 RGB/点云证据。
- [ ] 获取真实 SysNav viewpoint provider，使 `/viewpoint_rep_header` 有实际 publisher；
  当前 bridge 节点虽已启动，但输入 publisher 为 0。
- [ ] 在真实 viewpoint、object nodes、RGB 和 odom 同时存在时，完成不发布 waypoint 的
  `policy_mode=semantic_snapshot` 任务验收，记录 viewpoint 的 frame、频率、时间偏移和
  object-viewpoint 关联。
- [ ] 对远程 LVLM 做网络侧 instruction parse、concept grounding 和 final verifier 测试，
  记录 p50/p95 延迟、超时、服务不可用和非法 JSON 时的 `WAIT/HOLD` 行为；密钥不得写入
  仓库或日志。

验收出口：一条真实自然语言指令能在无运动条件下产生可追踪的
`InstructionPlan -> NavigationIntent -> ViewpointPose/WAIT -> ViewEvidence -> verifier`
证据；没有 viewpoint 时必须明确保持 `WAIT`。

### P1-增强：高动态/高精度融合（低速 VLN 非阻塞）

- [x] OCC D435i--MID-360 外参与当前安装状态一致，`calibration_status: calibrated` 已按
  低速 VLN 决定写入 profile。
- [ ] 若以后扩展到高速或高精度场景，再用独立连续片段复核 RGB--LiDAR 时间偏移的 held-out
  稳定性。
- [ ] 若以后需要像素级精度，再计算严格 LiDAR→图像边缘误差和更严格深度误差。

以上两项不阻塞当前低速 semantic mapping 或 VLN 试运行。`require_calibration: false` 仍是
诊断开关，不能代替 profile 中已记录的 candidate。

### P2：平台无关软件回归（已完成，持续维护）

- [x] `SemanticMapSnapshot -> NavigationIntent -> SysNavGoalResolver -> MotionGoal/WAIT`、
  `REACHED -> fresh evidence -> verifier`、`ViewpointPose` bridge 和 waypoint adapter 均有
  离线测试；最近一次 acceptance 为 `113 passed`。
- [x] waypoint adapter 当前保持 shadow：`output_enabled=false`，不创建生产 waypoint
  publisher，不发布 `/cmd_vel`。
- [ ] 每次修改 detector、mapping、viewpoint 或 runtime 后重新运行 acceptance，并把结果
  和部署 SHA 写入新的 run 目录。

### P3：确定并批准唯一的真实运动后端（当前最大阻塞）

- [ ] 由底盘控制器所有者书面确认 `/waypoint` 的消息类型、数组语义、单位、轴向、frame 和目标有效期。
- [ ] 确认 `/topoplan/reached_goal` 是否为生产反馈；补齐 blocked、timeout、cancel/preempt、heartbeat 和故障状态接口。
- [ ] 确认最终 `/cmd_vel` owner、速度/加速度限制、watchdog、急停断言/复位和人工接管流程。
- [ ] 完成机器人专属 controller contract，保持 `cmd_vel_direct_publish=false`，并由底盘
  所有者签字/现场确认后才允许改变。
- [ ] 根据确认后的实际反馈，只实现一个 Orin status adapter；不为尚不存在的反馈类型预写兼容分支。
- [ ] 明确生产后端是“外部 `/waypoint` 控制器”还是“SysNav 原生下层”，并在 launch 中保证互斥。

验收出口：`approval_status=approved` 只能由真实合同和现场验收记录支持，不能由单元测试或
历史源码推断产生。

### P4：分阶段安全运动验收（P3 批准后）

- [x] Shadow：保持 adapter `output_enabled=false`，已验证 frame、新鲜度、坐标变换和无
  `/cmd_vel` 所有权冲突；不得把 shadow 结果当作真实控制验收。
- [ ] Bench：轮子离地或电机硬件禁用，连接真实 `/waypoint`，验证正常、过期、取消和错误 frame。
- [ ] Low-speed：第二人持急停，在空场执行不超过 0.5 m 的前进、后退、转向、HOLD、blocked 和 timeout。
- [ ] Safety：实测急停、人工接管、通信中断、odom 过期和 controller heartbeat 丢失。
- [ ] Rollback：验证关闭 handoff、停止本部署容器后，底盘保持安全状态且不影响外部 LIO。

验收出口：每个状态都有底层反馈和时间戳证据；VLN 侧 `NavigationStatus` 与底盘实际状态一致。

### P5：完整实物 ObjectNav 闭环

以下三项已经具备离线软件实现，但仍需在真实传感器、SysNav viewpoint provider 和底盘
反馈上重新验收；因此不把它们作为实物完成项：

- [x] 软件层实现 `MotionGoal` 到达后再进入 `ViewEvidence`，`REACHED` 不直接等同于任务成功。
- [x] 软件层实现到达时间戳之后的新鲜 RGB、pose 和 detection 检查；不满足时保持同一 goal 的 `VERIFYING`，不重发 waypoint。
- [x] 软件层保留 `accept / need_better_view / reject_candidate` 的 verifier 决策边界，具体视角重规划仍依赖实际 SysNav viewpoint provider。

实物验收仍从以下条目开始：

- [ ] 先运行单目标显式指令，不启用 room prior 和复杂关系。
- [ ] 在真实底盘上验证 `MotionGoal` 到达后才采集 `ViewEvidence`；`REACHED` 不得直接等同于任务成功。
- [ ] 在真实传感器上验证 `REACHED` 后的 RGB、pose 和 detection 时间戳晚于到达事件；未满足时保持同一 goal 的 VERIFYING，不重发 waypoint。
- [ ] 在真实 SysNav viewpoint provider 和底盘反馈上验证 final verifier 的 `accept / need_better_view / reject` 会产生对应的 STOP、视角调整或重新规划。
- [ ] 再增加属性目标、anchor relation 和小目标搜索；每类至少保留成功和失败样例。
- [ ] 最后接入实物语义先验地图，完成地图 frame alignment 后再启用 room-level guidance。
- [ ] 记录成功率、路径长度、耗时、LVLM 调用次数、人工接管次数和安全中断原因。

验收出口：至少完成一组可重复的端到端场景，运行目录同时包含传感器摘要、对象图、指令计划、
运动状态、视觉证据和 final verifier 结果。

## 5. 工程实施约束

后续实现遵循以下边界：

- 不根据 topic 名称猜测消息语义；不确定项保持未完成并写明所缺证据。
- 复用已有 `MotionControllerProtocol`、`WaypointFormatAdapter` 和 `RosNavigationStatusProvider`，不再新增平行基类。
- 一个阶段只解决一个已确认接口缺口；标定、语义策略和底盘控制不混在同一修改中。
- 传感器和外部控制器输入非法时显式报错或保持 WAIT/HOLD，不把异常转换成成功。
- 不为理论上可能但现场未出现的 controller 版本编写兼容分支。
- 每次真机运行使用新的 run id；历史证据只读，不覆盖旧日志。
- LVLM 负责语义满足性，几何与下层控制器负责可达性、距离、碰撞和运动安全。

任务成功的控制语义保持为：

```text
motion_reached
AND instruction_satisfied
AND safety_state_allows_stop
```

其中 `motion_reached` 只说明机器人到达一个运动目标，`instruction_satisfied` 必须由绑定
原始指令和当前视觉证据的 verifier 给出；两者任何一个缺失都不能上报任务成功。

## 6. 下一次现场工作的最短路径

按以下顺序推进，期间保持真实运动关闭：

1. 用 `huawei-vln-realworld:orin-r36.5-fix` 重启一套干净进程，完成 detector + semantic
   mapping + Point-LIO 的整栈 30 分钟 soak，保存频率、掉帧、资源、温度和 TF 证据。
2. 启动真实 SysNav viewpoint provider，确认 `/viewpoint_rep_header` 有消息；再运行一条
   `semantic_snapshot` 自然语言指令，完成对象 UID、viewpoint、RGB 和 verifier 的无运动
   证据闭环。没有 viewpoint 时必须继续 `WAIT`。
3. 由底盘所有者确认并批准 controller contract，补齐状态 adapter；在批准前绝不发布真实
   `/waypoint` 或 `/cmd_vel`。
4. 依次执行轮子离地 bench、有人持急停的低速短距离和通信/急停故障测试，完成后才进入
   完整 ObjectNav。

高动态时间偏移、严格像素误差和深度精度保留为非阻塞增强项；只有目标场景升级时才重新
开启该验收线。
