# VLN 实物部署状态与后续现场验证

## 实物模式：已实现

已完成 2026-11-05 约定的 bridge-off 规划与速度输出验证；软件实现、部署镜像更新与实际运动验收分别记录。

按用户约定，正式一键入口、真实输入的前方 chair 接近点规划、限幅速度输出与默认关闭 bridge 均已完成。本轮无剩余软件部署阻塞项；实际行驶到达、未知空间探索和最终视觉 STOP 是后续独立现场验证范围，未标记通过。

后续参数修复（本地离线）：旧停止/到达阈值冲突已复现；最终按用户要求接入 0.5 m 停止、1.0 m 到达、1.0 m 前视/减速距离及入口检查。69 项回归通过，车端旧镜像尚未更新，bridge 未开启。详见 [参数手册中的停止距离诊断](real_robot_parameter_tuning.md#停止距离离线诊断)。

重要调参、迁移方法及参数配合风险见 [实物参数调节说明](real_robot_parameter_tuning.md)。以下正式入口记录优先于所有历史记录。

- [x] 已把 standoff、底盘位姿转换、原生规划/跟踪和安全速度复用器接入正式 runtime→Action→MotionServer 链路，不再用日志读取探针驱动目标。
- [x] Orin 一键入口：部署根目录 `bash scripts/run_real_robot_navigation.sh`。默认 preview，bridge 关闭，日志自动持久化。
- [x] 修复 Action 终止未撤销原生目标及 XY 容差未传递的问题；正式实测无进展后速度持续归零，没有误报 REACHED。
- [x] 56 项本地测试、最终镜像文件校验、真实输入及一次容器重启回归通过；UDP-only DDS 下同一 LIO 进程持续供数。
- [x] 已写清迁移参数、live 模式开关、控制合同、硬件状态和自动模式服务授权。live 未启动，车未受本轮软件驱动。

验收时镜像 `896fbd15b040...`，容器 `huawei-vln-navigation`。操作以 [正式实物模式一键入口](real_robot_navigation_entry.md) 为准。实际运动/探索/视觉 STOP 和历史混合实例问题保留为后续范围，未假定完成。

---

## 历史 shadow 验收结果（已升级到上方正式入口）

用户明确将本轮实物部署验收收敛为：前方 chair → 实物输入目标选择 → 合理规划轨迹 → 速度输出，默认关闭底盘 bridge。**此口径已通过**，不宣称实际行驶或未知空间探索已验收。

- [x] 真实定位/点云/在线对象输入，选择当前生产决策中的 `sysnav_object:2` chair，并按底盘方向检查在前方。
- [x] 修复融合消息写死 map 的坐标系标注；统一 camera_init；按已记录的底盘在雷达中 `[-0.2, 0, -1.5708]` 外参组合底盘 pose。
- [x] shadow 接近候选距椅子中心 0.8 m；原生 localPlanner 生成约 0.43 m 轨迹，最多 44 点。
- [x] 原生 pathFollower → SafetyVelocityMux 输出最终隔离速度；35 秒测试 640 个速度样本，639 个非零，线速度峰值 0.066 m/s、角速度峰值 0.113 rad/s。
- [x] bridge 未启动，四个物理控制话题发布者均为 0；测试子进程已清理，静止感知/定位继续运行。
- [x] 固化入口、候选镜像、原始失败/成功记录、轨迹图及迁移参数说明。

说明：目标 UID 来自生产 runtime；接近点是显式 shadow standoff 候选，不是已验证历史视点。当时该验收入口尚未接入生产目标解析；现已由上方正式入口集成 standoff resolver。这些结果不能用于推断任意指令/未知空间的自主导航已完成。

本轮不再继续的中间测试：重复全地图重放、为当前可见 chair 强制精确历史绑定、重复离线掩码审阅、为此验收追加豆包调用。原始证据保留，不删除。

后续仅在实际接底盘时开展：控制模式/故障码/watchdog 与接管核定、移动中定位及跟踪、实际到达与视觉 STOP、未知空间探索。历史碎片和原生跟踪器静止时转向切换保留为局限；安全 mux 已限制速度变化，不把它解释成运动闭环稳定性验证。

操作与参数见 [实物 shadow 部署说明](real_robot_shadow_deployment_20261105.md)。最终证据：`logs/diagnostics/shadow-acceptance/strive-shadow-final/{result.json,metrics.json,trajectory_and_speed.png}`。

---
以下为此前更严格的物理闭环交付清单及历史证据，不作为本轮 bridge-off 验收的阻塞项。

> 2026-11-05 离线掩码复核：历史实例混合/碎片与豆包解释错误同时存在；原请求图像配对无误。已在本地改为精确 masked-RGB 裁剪输入，33 项测试通过，未调用模型复测、未部署、未绑定。历史融合混合仍待修复，详见 [诊断报告](archive/real_robot_correspondence_mask_diagnosis_20261105.md)。


## 历史执行清单（不作为本次完成的阻塞项）

本节作为下一轮执行入口；下文历史记录和 AgileX A1–A9 保留作证据/接口参考，
其中旧的未勾选开发项不再直接当作当前待办。此次没有开启自主运动、发布速度、
切换选源。12:43 起按用户授权恢复了感知链路；采集收尾后已按实际车位发布一次
`/initialpose` 恢复保存地图定位，详见下方最新记录。

### 历史候选交付状态

| 关口 | 已有实际证据 | 尚未完成 |
| --- | --- | --- |
| G1 地图与定位 | 局部地图持久化；移动/重启后重新配准并恢复保存地图定位；静止连续性验证 | 目标路线覆盖与移动过程定位验证 |
| G2 感知与绑定 | 新镜像在线带 ID 椅子、点云融合、对象视点关联及 dry-run 目标；图像/框/位姿配对 | 判定/准入/ROS 接线已静止上线；真实模型与精确绑定未验收，保留碎片歧义 |
| G3 底盘控制 | 同一 API 转接实现；只读遥测/类型/发布者查询 | 核定模式/故障/硬件 watchdog；受限合同与现场授权；低速及异常停车实测 |
| G4 完整交付 | 当前候选镜像/资产/地图身份与静止启动回滚手册 | 配置真实视觉模型并验收新鲜视觉核验；真实行驶→视觉→STOP及失败/接管记录 |

当前运行 `huawei-vln-static-correspondence-candidate`，镜像 `3d97aec88b89...`；自主运动关闭。
待用户确认当前遥控器状态。视觉服务已指定沿用 Doubao，并找到本机 Ark 凭据；
用户已明确授权一次冻结图像判定；已完成 1 个真实 Doubao HTTP 请求，结果 fragments，未通过唯一绑定。
对应判定接线已完成本地及 ARM 断网测试并静止上线，但真实模型调用仍关闭，不能记为实物绑定通过。

底盘只读接口核对：`/robot_status` 类型 `robot_status_core/RobotStatus`，
`/ranger_status` 类型 `ranger_msgs/RangerStatus`，两者发布者均为 `/ranger_base_node`。
消息定义仅有灯/电机 ID 常量，没有 control_mode/base_state/fault_code 的枚举定义；
字段值差异仍待驱动/硬件规范或现场状态对照核定，不允许把 128 放入 ready_fault_codes。
证据 `logs/diagnostics/evidence-static-validation/chassis-schema.json`。

### 历史接续条件（一次冻结判定已完成）

用户已明确授权指定冻结证据的一次 Doubao 判定，已发送且仅发送 1 个 HTTP 请求。
此前的外发授权阻塞仅对本次调用解除，不扩展为任意新图像、额外模型调用或运动授权。
结果为 fragments，未通过唯一绑定；代码保留未绑定状态，没有修改车端地图或发送运动指令。

- 首次执行在 HTTP 发送前触发 SDK AttributeError。断网诊断定位为 openai 1.64.0 的
  beta raw 接口无 create；已将 Doubao 文本创建请求路由到标准 chat.completions，
  保留 beta 的 parse 路径。真实 SDK 的 MockTransport 回归及相关测试共 30 项通过。
  此修复目前仅在本地，车端镜像仍为 3d97aec88b89，后续服务启用前需同步。
- 修复后调用 `doubao-seed-2-0-pro-260215`，相同冻结包哈希
  `f36cc5c8a25f7bcd36d7c7c0c728cc6b03e5bc96fabe01aaaf8e4aeffbe94b70`，
  单次 HTTP 请求约 22.3 s。原始结果、前置错误及审查在
  `logs/diagnostics/instance-correspondence-doubao/`。
- 模型返回 fragments，原始响应虽带候选 UID，适配层按非唯一结果清空该 UID。
  3 条支持、8 条冲突观测只是模型判断。解释将部分 monitor/trash bag 标签视为其他物体，
  与此前逐张掩码的椅背/椅子局部观察有不一致，不能据此自动删除观测或修改地图。
- 尚需解决证据对应歧义并在当前 runtime 会话重新配对采集，才能验收静止精确绑定；
  旧包不能直接写入新会话。任何进一步外发需在用户授权范围内。
- 地图路线覆盖与移动定位、底盘模式/故障/watchdog 核定、低速跟踪/停止/取消/接管及
  真实行驶→视觉→STOP 仍未完成；自主运动继续关闭，运动仍须另行授权。

### 精确实例绑定的源证据复核（2026-11-05）

- 更正证据范围：此前输入包的 6 + 5 条是 compact bundle 为两候选保留的样本，
  不能作为完整融合历史。本机未压缩源的 manifest、prior_map、observations、build_receipt
  SHA256 与压缩回执一致；完整成员分别为 131、52 条，总源观测为 5382 条。
- 以原始 RGB-D/mask、原 5 cm voxel、6 m 深度上限及 confidence >= 1 重建固定成员，
  不重新运行检测、不重新分配 UID。恢复点数 25916/2509，与部署地图完全一致；
  质心误差 3.16e-6/2.73e-7 m。该结果验证源证据到地图几何的可追溯性，不是身份验收。
- 两云的原始点定向覆盖（最近距离 < 0.1 m）为大→小 0.5916、小→大 0.9753；
  与逐张掩码观察相结合，支持椅子整体与局部碎片并存，不能据此直接合并。
  这些覆盖值包含重复观测点权重，不等于去重后表面比例或身份置信度。
- 掩码核查：frame1640:detection9 虽标签 monitor，实际覆盖椅背；不能仅凭标签宣称
  显示器混入。frame1365:detection2/9 在桌侧边缘，和主椅子 mask 分离。
  构建回执明确记录 frame1365:detection2 UID 在 duplicate_instance_merges 中并入
  frame1350:detection1；具体错误准入仍需全关联历史重放，不能仅从最终合并反推原因。
- 下一步绑定判定必须允许“多个历史 UID 指向同一物体碎片/身份仍不唯一”，不能强制
  最近 UID 或自动把多个候选标成已验证。当前生产绑定接线仍未完成，未写入绑定、
  未替换车端地图、未启动运动。复核脚本与原始结果：
  `logs/diagnostics/instance-source-reconstruction/{reconstruct.py,result.json,review.json}`。

### 完整关联历史重放（已结束，非全图完全一致）

- 修复 `scripts/diagnose_prior_associations.py` 的重放一致性：读取原构建深度
  confidence 阈值并复用 `load_fusion_observation`；拒绝用 compact 证据冒充完整源；
  完整 baseline 增加末尾 duplicate merge 后的最终 UID 与点数/质心比较。5 项针对性测试通过。
- 5382 条源观测全部处理，进程 exit=0。5381 条最终 UID 一致；
  `frame685:detection14` 在两个显示器/计算机相关簇间不同，造成 2 个簇各差 25 点、
  最大质心偏差约 1.92 mm。差异原因尚未定位，不宣称全图确定性复现，也不归因为浮点误差。
- 目标椅子 183 条归属全部一致，两个实例点数 25916/2509 完全一致。
  合并前实际有三个簇：主椅子 80 条/16255 点，小碎片 52 条/2509 点，
  table 种子簇 51 条/9661 点（其中 48 条 chair/desk chair）。
- 进一步更正此前“桌子并入椅子”的简化描述：table 是初始种子标签，不是最终簇身份。
  主椅子与 table 种子簇按原 5 cm voxel 去重后双向覆盖为 0.9339/0.9693，
  满足现有 >0.85 的双向规则；主椅子与小碎片为 0.5503/0.9009，未满足双向规则。
  保留碎片是局部/整体保护的结果；没有充分证据将末尾重复合并判为算法错误。
- 不降低融合阈值、不手工合并 UID。当前首要软件缺口仍是有图像/掩码证据、能表达
  碎片及歧义的实例对应判定与生产写入接线。全图的单条复现差异另留作局限，
  不把重复全图重放设为当前椅子部署的必要中间测试。
- 结果位于 `logs/diagnostics/full-source-association/`：`baseline.json` 是全部关联记录，
  `target-merges.json` 是合并前去重表面比较，`summary.json` 保留差异及局部结论；
  `launch_receipt.json` 已标记完成并保留输入/代码哈希。未改变车端地图、绑定或运动设置。

### 实例对应判定接线（本地，未部署）

- 新增 `prior_map/instance_correspondence.py`，复用共享 Ark/OpenAI-compatible 客户端，
  独立结构化结果包含 unique/ambiguous/fragments/no_match/insufficient。
  检查文件哈希、图像/框时间配对、框边界、历史 mask 尺寸/非空及观测归属。
  同类、近距离和全场景相似不能替代实例判定；不强制从碎片中挑一个 UID。
- `PriorMapRealRobotRuntime.process_correspondence_package` 已由 ROS 推理入口调用，
  输入由 `correspondence_package` 参数指定（默认空，不调用服务）。准入核对加载地图
  ID/revision/hash、当前 runtime session、当前对象/track 及 120 s 时效；模型返回后
  再检查对象仍在。唯一且无冲突的真实客户端结果才调用 `mark_object_verified`，
  同一包只消费一次；注入客户端、离线、服务错误及非唯一结果均不写绑定。
- `prior_map/correspondence_session.json` 提供当前进程的 session_id 和地图身份。
  旧冻结包使用旧容器身份，且已过时，只能用于离线模型判定；不能改写旧 session 字段
  冒充新采集。车端更新后需要在当前会话重新配对采集并准备输入包。
- 接线测试覆盖写入后进入真实 prior query，另发现并修复无位置先验在 PNG 标注阶段
  的 None 坐标崩溃（只跳过不可绘制标注，不伪造坐标）。相关本地测试见后续结果。
- 证据读取进一步固定为一次读取的字节快照：哈希校验、图像尺寸检查和模型输入使用
  同一份 bytes，避免文件在校验后替换造成记录与模型所见不一致。文件替换回归测试通过；
  原冻结包哈希及 23 个图像块保持一致，本次检查没有网络请求。
- 真实服务固定使用此前的 `doubao-seed-2-0-pro-260215`，Ark 端点为
  `https://ark.cn-beijing.volces.com/api/v3`；凭据从本机原文件读取，不复制到仓库。
  第一次请求在执行前被自动审批拒绝，尚未外发。待授权内容为 1 张当前图、8 张历史图
  （11 条观测中重复使用）、11 张 mask 及框/ID；本地预检生成 23 个图像块。
  `logs/diagnostics/instance-correspondence-doubao/local-validation.json` 仅证明本地包校验，
  不是模型成功或实例绑定证据。

### 历史对应判定候选静止上线

- 镜像 `huawei-vln-realworld:orin-r36.5-yoloworld-correspondence-candidate`，
  ID `sha256:3d97aec88b898338ed74a4e4620819433053e6852c22caae2ef9972d892b9679`。
  在父镜像 `016e990ad481...` 上覆盖 8 个源码/launch 文件，不更新依赖。
- ARM 临时容器 `--network none`，不挂载设备或密钥：源码哈希、编译、schema、OpenAI
  导入、安装后的 node/launch 路径及真实证据包离线预检通过；19 项对应判定测试通过。
- 切换前将旧运行目录完整复制为宿主 `logs/pre-correspondence-switch-runtime`（127 MB），
  新容器 `huawei-vln-static-correspondence-candidate`，tmux `strive_static_correspondence`。
  LIO/相机未重启，地图未更换。`LLM_OFFLINE=1`、final verifier=false、correspondence_package
  默认空及全部运动开关关闭；没有外部视觉请求。
- 45.00 s 只读窗口：21 条检测均含带 ID 的 chair；22 条对象消息；odom 4197、body cloud
  401、RGB 335 条。四个运动话题发布者均为 0。视点事件未在本探针窗口重发，
  但实际 runtime 日志含 viewpoint 0、visible objects 1/2/3 及解析后的目标位姿。
- 决策快照 115 条：31 wait、84 go_to_object；目标 `sysnav_object:2`，reason 为
  `dry-run recorded go_to_object; no waypoint published`。这是目标/视点解析，不是路线或行驶验收。
- 新 runtime session 为 `5b3a84c4-a9b9-4272-89ec-f6b5be203da1`，旧输入包不能直接写入
  本次会话。`matched=false` 保持；真实 Doubao 调用仍待明确图像外发授权。
- 源码 manifest、启动脚本、ARM 检查结果、探针及决策在本地
  `logs/diagnostics/correspondence-candidate-build/`，宿主部署根 `correspondence-candidate-build/`。
  启动/导出/回滚步骤已更新至 `docs/real_robot_static_candidate_runbook_20261105.md`。

### 11:24 初始诊断（修复前快照）

- 本机 USB 网卡 carrier=1，地址 192.168.1.193；.101/.102 经该网卡直连。
- Orin .101 密钥认证失败后，使用用户提供的登录方式完成检查（凭据不留档）。
  11:24 CST 主机 ubuntu，uptime 46 分钟，说明已重新开机；部署 realworld_eb188e7
  HEAD=fe1037b。无运行容器，仅 livox_odom 会话，无 D435i 节点/会话。
  内存可用约 13 GiB、磁盘可用 56 GiB。LIO 的 pcd_save.pcd_save_en、
  location_mode、publish.scan_bodyframe_pub_en 均为 false。
  四话题逐个 5 秒 echo 均超时；再用 sensor_data QoS 同时订阅 8 秒，
  /livox/lidar、/livox/imu、/aft_mapped_to_init、/cloud_registered_body 均 0 样本。
  进程存在不等于数据链可用；断流原因尚未诊断，不能直接归因为历史 NTP 问题。
- 底盘 .102 的 ROS master 11311、rosbridge 9090、视频服务 8080 可达；master 为 Navis。
- 约 7 秒只读订阅：robot_status 8 条、ranger_status 7 条、BMS 7 条；
  mux selected 为 `/run_management/virtual_joy`，末次线/角速度为零。
  `/odom` 和 `/base_cmd_vel` 在本窗口没有收到样本，不能据此判断 Orin LIO 状态。
- 状态报告 fault_code=128，robot_status 文本为“过压保护”；ranger 电压 55.5 V，
  BMS 电压 51.9 V、SOC=100%。两种消息的 base_state/control_mode 也不一致，
  保留原始语义，不自行解释硬件枚举或确认故障解除。
- 结论：网络在线；尚不能确认可开始推车建图或完整静止绑定。登录已恢复，但传感器无样本、存图关闭、相机未启动。
  先确认车是否移动、恢复并复核传感器流，再核实故障及允许人工移动的状态。
  本次已发现重启，历史 alignment 的有效性需重新验证。

### 12:48 感知恢复与建图准备

- 校时顺序证据：开机约 30 秒启动 Livox/LIO；timesyncd 到单调时间 110.624 秒
  才首次同步，墙钟从 11 月 4 日 22:35 跳至 11 月 5 日 10:39。两进程最后输出停在
  校时前；已同步后重启即恢复七路真实消息。定位到开机校时与传感器启动的顺序问题，
  未通过人为跳时复现内部阻塞，也未修改雷达 SDK。
- 持久修复：orin26 的 crontab 原 `sleep 20; start_livox_odom.sh start`
  改为 `timeout 300 /usr/lib/systemd/systemd-time-wait-sync && start_livox_odom.sh start`。
  校时等待命令当前返回 0，crontab 写后校验一致；冷启动复验尚未执行，不为该项重启现场系统。
  备份/新值和原日志位于车端部署目录 `logs/diagnostics/sensor-recovery-20261105/`，
  已复制到本地同名目录。回滚用该目录 `crontab-before.txt` 恢复用户 crontab。
- 30 秒同进程采样：雷达 1729、IMU 1729、odom 1789、body cloud 276、
  RGB 410、对齐深度 411、CameraInfo 428 条，所有时间戳倒退计数为 0。
  body cloud 约 9.44 Hz，RGB/深度约 14.4 Hz；高频话题为探针实际收到的速率，
  不当作发布端频率。XY 范围约 0.95/1.03 cm、yaw 范围约 0.238°。
- 用户确认车自昨日配准后未移动。原位切到存图模式，实读
  `pcd_save.pcd_save_en=true`、`publish.scan_bodyframe_pub_en=true`、`location_mode=false`；
  七路话题再次全部收到消息。此次重启后位姿约 (0.001,-0.010,-0.012)m、yaw -0.012°，
  原起点对齐可用于本次测试，正向物体残差仍由 G2 核验。当前不要再重启 LIO，保存用 SIGINT。
- 静止绑定容器 `huawei-vln-static-binding-20261105` 已运行：
  镜像 `huawei-vln-realworld:orin-r36.5-agilex-1aea701`，车端 HEAD=fe1037b；
  先验 fbb15f81e4、格式 bundle、既有原位 alignment、目标 chair、规则解析、LLM_OFFLINE=1。
  检测/融合已连续产出 cart，viewpoint 0 已被 mapping 接收；当前视野尚无 chair，
  runtime 为 `No eligible observed candidate`，这是静止绑定测试就绪而非正向绑定通过。
- 所有运动开关关闭：dry_run=true、lower_controller=false、ALLOW_REAL_MOTION=false，
  AgileX bridge 未启动；ROS2 /cmd_vel 与 /cmd_vel/autonomy 查询为不存在。
  最终视觉 verifier 在本次静止准备中关闭，不支持任务 STOP 验收。
- 初次静止启动误将 source_format 设为 phone，导致 runtime 退出；纠正为 bundle 后成功。
  首次失败日志保留，传感器不受影响。
- **当前结论：** 感知、存图和静止绑定软件链已就绪。推车仍需现场确认底盘的过压提示及
  人工推行状态；目标椅子的正向绑定需人工摆位。存图会持续积累内存，若现场延期，
  应先保存当前扫描再结束存图会话，不把已经累积的地图直接丢弃。

### 12:50 现场延期后的最终运行状态

用户确认车未移动、仍连接充电器，要求先完成其他部分并保持无运动。
充电连接已确认；fault_code=128 的原因尚未证实，留待断开充电后现场复核。

- 感知恢复、自启校时顺序修复、静止绑定软件准备均已完成；实际椅子正向绑定尚待摆位。
- 为避免等待推车期间继续积累内存，经 SIGINT 保存当前静止扫描：
  车端 `output/lio_maps/static-recovery-20261105/lio_map.pcd`，2,274,880 点，
  72,796,413 bytes，SHA256 `1a085dcb1854bc9ee37c9c5f25e1b82197be3a16ebddb229246c81b57915fd89`，
  元数据哈希复核一致；仅为静止局部扫描，完整区域图尚未建立。
- 原位恢复普通感知模式（LIO_PCD_SAVE_EN=0），D435i 与静止 dry-run 容器保持运行；
  七路传感器再次 PASS。此前“存图已开启”为 12:45–12:49 会话状态，当前已关闭。
- 推车前：现场断开充电、确认可人工推行和底盘状态；车保持原位时再次开启存图，
  实读参数及实际消息后再推车。全程保持自主控制关闭。
- 不以充电器连接推断过压保护正常；不为完成本轮感知修复启用自主运动或发布初始化位姿。

### 15:04 现场小范围采集已开启

用户到车旁、已断充、车未移动，说明此前告警对应遥控器未开，计划使用原遥控器操作。
该解释记录为用户现场确认，尚未独立证明 fault_code 枚举含义。
Orin 在约 15:01 重启后 LIO 自启已有位姿输出，校时状态 yes。
原位以存图模式启动，实读 pcd_save.pcd_save_en=true，恢复 D435i，七路话题采样 PASS。
重启既有静止绑定容器，dry-run=true、远端桥接关闭、下层控制关闭；ROS2 /cmd_vel 不存在。
录包会话 `field_capture_20261105` 已订阅位姿、body cloud、RGB compressed、CameraInfo，
目录 `output/field_checks/tethered-20261105T1504`，数据库已写入约 50 MB。
当前存图与录包均在运行；需用户完成小幅移动并通知结束后正常保存，禁止直接重启 LIO。
网线限制下先做原位基线、10–20 cm 往返、停稳取景；范围只覆盖局部一致性与静止绑定，
不能记为完整区域建图或自主运动验收。检测器刚完成加载，正向物体绑定仍待实际观测。

### 15:08 首轮网线限制下的人工运动采集

用户报告第一轮完成并确认已回原位置/原朝向。录包正常结束，LIO 存图保持运行。
本轮约 226 秒（含等待）：odom 22,437 条、body cloud 2,138 条、RGB compressed 与
CameraInfo 各 3,377 条。首末各 5 秒位姿中位数的 XY 差 0.210 m、yaw 差 -23.19°，
最大起点位移 0.549 m；运动主要位于录包第 125–150 秒，之后估计位姿稳定在新位置。
起止相机画面明显不同：起点含左侧门口，结束偏向楼梯下滑板车/椅子。
因此当前结果为 inconclusive：不能把起止位姿差直接解释为回位误差或 LIO 漂移。
用户随后确认车头角度可能未完全对齐，实际回位姿态缺少独立真值；本轮验证了运动期间
数据采集，但回位精度不作结论。跳过重复往返，转入椅子取景/静止绑定。
本地分析及图片：`logs/diagnostics/tethered-20261105T1504/`；完整 bag 留在车端
`output/field_checks/tethered-20261105T1504/`。静止绑定尚无 eligible chair，不能记为成功。

### 15:15 两轮采集收尾

第二段录包正常结束：约 195.7 秒，odom 19,501、body cloud 1,850、RGB 2,929、
CameraInfo 2,928 条；对应 runtime 决策 196 次，89 次 No eligible observed candidate、
107 次 inference pending，motion_goal 为 0。最大右转视角中椅子清楚入镜，末帧则在右缘；
因此正向绑定未通过，下一步优先离线检查已录图像的检测/融合/候选过滤，不要求现场重复取景。
证据：本地 `logs/diagnostics/chair-20261105T1510/{analysis.json,rightmost_view.jpg,last_view.jpg}`。
完整录包保留车端 `output/field_checks/chair-20261105T1510/`。

地图已通过 SIGINT 保存到车端部署 `output/lio_maps/tethered-20261105T1503/`：
6,032,950 点，193,054,653 bytes，SHA256
`e5223d558032b77ab7066227584dae53a5566c33249d62cfe16a220217e21e5f`，实读哈希与元数据一致。
`prior_alignment.json` 已由原起点标定与本 PCD 哈希生成，仅适用于该 PCD 定位模式；
这是受网线限制的局部采集，不是完整场地地图，尚未进行定位模式初始化/重定位验证。

收尾状态：两段录包结束，静止绑定容器停止，LIO 保存后退出，D435i 相机会话保留。
当前车已经移动，禁止在新位置以普通建图模式重启后沿用旧 camera_init 对齐。
下一次可在保存地图的定位模式恢复，需以实际当前位置设置初始化位姿；本轮未发布 /initialpose。
全程未启用自主运动。Docker 历史日志导出的时间范围与本轮不一致，不能作为本轮完整标签计数，
本轮决策统计以 runtime_decisions.jsonl 的时间窗口和录包为准，后续诊断需保留该证据限制。

### 离线椅子诊断

离线椅子诊断更新：已确认并最小修复 detector 的 BGR/RGB 重复转换。
修复后真实 inference 在 97 帧全段采样和 38 帧清晰片段仍为 0 个带 ID 的 chair；
首个阻断在检测输出。后续复用已有 YOLO-World 权重，在相同 38 帧上取得 29 帧带 ID
的椅子输出，并通过实际 SAM2/点云融合产生 29 次椅子对象发布；最近邻传感器诊断
不等价于完整 ROS 回放，先验绑定/视点仍未验收。额外修复 SAM2 RGB 输入、YOLO-World
词表初始化和无图像路径时 ROS 序列化。下一项是负例核对及真实对象→先验→视点验证，
不重复现场采集或放宽候选过滤。默认模型未切换，修复尚未写入正式车端镜像。
实际对象→先验→目标解析器的离线审计又修复了同名标签误绑定 exact 历史实例的问题：
旧逻辑会把当前椅子匹配给约 5.01 米外的历史椅子；修复后保留未绑定状态。
29/29 次语义目标选择成功，但因无真实视点，运动目标仍为 0/29。
近邻两个历史候选残差约 0.137/0.180 米，仍有碎片身份歧义；下一项是实际 provider/bridge
的隔离回放和对象视点关联，不把标签同名或距离近当成已验证的具体实例绑定。
后续实际 ROS 隔离回放已产生 15 次对象发布、视点 0 及其椅子 ID 4 关联；正式 cache
和目标解析器可保留关联并生成该记录视点的目标，排除视点后正确返回 None。
这是冻结检测驱动的离线正例，未发布运动、未验收新路线或最终视觉 STOP。
完整检测并发资源、具体实例绑定、正式候选部署、保存地图定位和实车控制关口仍保留。
候选 profile 与离线 CLIP 挂载已补齐；另一段录包 111 帧中 28 帧椅子预测经视觉核对均
含实际椅子，起始无椅子帧正确无输出，尚不报告全量准确率。独立候选源码与 5 项资产
已校验，候选镜像已构建成功，image ID 为
`sha256:20594ab739c332743117f6ebaec338272809af8225cb9d69c0398bec566cf5b7`。
新镜像的实际 DetectNode 离线运行验证已通过：111 帧中 24 帧带 ID 椅子，
与既有同录包跟踪正例时间戳集合一致；无椅子首帧无输出。静止在线结果见后文。
用户确认采集后未移动/转向/重启，已用录包末尾位姿恢复保存地图静止定位。
详见 [离线诊断记录](archive/real_robot_chair_detection_diagnosis_20261105.md)。

### 保存地图静止定位复核（采集收尾后）

- 保存 PCD 哈希复核一致；实读 location_mode=true、pcd_save=false，加载上述局部地图。
- 根据用户确认的未移动/转向/重启状态，以录包末尾位姿发布一次 initialpose；
  Point-LIO 初始化成功，ICP 分数 0.005833。未启动控制器或发送速度。
- 44.88 秒收到 4,405 条 odom，时间戳倒退 0 次；排除前 5 秒后的 XYZ 跨度
  8.95/7.41/8.11 mm，yaw 跨度 0.191°。末尾与种子 XY 差约 2.3 mm、yaw 差 -0.084°。
  这些是静止稳定性/初值一致性，不是独立真值定位精度。
- 首末单帧世界点云分别仅 80/66 点，对保存地图的最近邻距离中位数 5.73/4.02 mm，
  20 cm 内比例 97.5%/96.97%；只支持局部重合核对，不代表整场地图覆盖验收。
- 本地证据 `logs/diagnostics/localization-20261105/{analysis.json,receipt.json,lio.log}`。
  LIO 保持定位模式。后续仍需在目标区域正常摆位中验证连续定位与在线绑定。

### 新候选镜像运行与在线静止核对

- 新镜像关键 detector/policy 修复文件哈希与本地一致；隔离容器断网加载本地 CLIP、
  YOLO-World，直接运行安装后的 DetectNode，不替换为外部推理实现。
- 固定独立录包 111 帧，41 prompts、BGR、640、conf=0.35、原 BoT-SORT：
  24 帧带 ID 椅子，与此前独立录包的跟踪正例集合一致；不把 24/111 当召回率。
  去掉首帧预热后推理中位数约 0.090 s、P95 0.129 s，不包含 SAM2/在线总延迟。
- 已启动独立候选容器 `huawei-vln-realworld-d435i-yoloworld-candidate`，
  tmux `strive_candidate_static`；新模型、SAM2 mapping、dry-run runtime 和 viewpoint bridge
  正常启动，加载与保存 PCD 绑定的 alignment。provider 按既有逻辑等待首个对象发布。
- 45 秒在线只读采样：odom 4,243、body cloud 403、RGB 335 条；检测结果/对象/视点均 0。
  detector 持续报告无 track ID；当前图片中椅子大部分在右侧画外，无法完成在线正例。
  这是当前取景下检测输出不足的证据，不能据此称融合或绑定失败，也不能称 G2 已通过。
- `/cmd_vel`、`/cmd_vel/autonomy`、`/way_point`、`/waypoint` 发布者均为 0；
  dry_run=true、lower_controller=false，底盘桥接关闭。未开启自主运动。
- 本地证据与可复现脚本：`logs/diagnostics/candidate-static-20261105/`。
  后续需在现场允许的小范围遥控摆位后验证实时正例；不再重复同一离线检测回归。

### 在线触发的新故障与现场移动/重启

- 首个对象触发 provider 启动后，候选栈异常退出，明确报错：
  `viewpoint input frame=camera_init; expected world_frame=map`。
  根因是宿主 Docker wrapper 未传递 `SYSNAV_WORLD_FRAME`，provider 使用默认 map。
  保留严格 frame 检查，修复 wrapper 的世界系和 provider odom/viewpoint 话题传递。
- 新增真实 shell wrapper 的 fake-Docker 参数回归，1 项通过；宿主修复已部署，SHA256
  `4e09cb4750deb2faeed040bafabb7c791c2e74255a1b8eb24d313d3a4ced98d6`。
  镜像 ID 不变；此宿主启动器变更是冻结源码包之后的显式部署差异，尚待在线触发复验。
- 用户随后确认已移动机器人并主动重启。重启后实读 location_mode=false、pcd_save=false；
  先前静止定位通过记录仅对重启前有效，不能把当前新 origin 与旧 alignment 混用。
  相机已恢复，`after-reboot.jpg` 显示椅背、座面与底座清晰入镜；传感器检查中
  RGB/深度/CameraInfo/Livox/IMU 有样本，body cloud 超时，尚未恢复完整定位感知链。
  候选感知/绑定容器当前停止；等待实际新车位说明后重定位，不自动沿用旧种子。
- 完整故障日志已归档 `logs/diagnostics/candidate-static-20261105/candidate-static-frame-failure.log`。
  继续保持自主运动关闭；相机可单独恢复，不能将普通 LIO 输出当作保存地图定位已恢复。

### 宿主配置传递补全与证据更正

- 进一步实测发现 wrapper 还漏传 `START_SEMANTIC_MAPPING`、`MAPPING_CONFIG`、
  `PROJECTION_CONFIG`、`DETECTION_TOPIC`、`OBJECT_NODES_TOPIC`。这会导致期望的关闭开关、
  D435i 标定与主题名在容器内回退默认值；已补齐并扩展 fake-Docker shell 回归。
- 因此，上述修复前“预期话题 45 秒零输出”不能作为实际检测/融合输出为零的证明。
  当时部分检测实际进入默认 mapping，且首个对象触发 provider 崩溃；该故障日志仍有效。
  当前图像取景变化的视觉事实有效，但不能仅据零消息计数将在线问题归因为取景。
- 用户确认有平移且已停稳。当前只启动检测，不启动先验绑定；容器内已实读确认
  mapping=false、正确 D435i 配置及 `/huawei_vln/d435i_detection_result` 话题。
  为只读重定位采样恢复普通 LIO 的 world/body cloud 发布，存图关闭，未发布定位初值。
  旧地图 alignment 仍不得用于当前普通 LIO 原点。
- 补全后在线采样 45 秒收到 20 条检测，20 条均有 chair、ID 始终为 2；实际阈值 0.35、
  分辨率 640、处理间隔 2 s。检测消息年龄中位数约 0.301 s，最大 1.530 s（含启动阶段）。
  同窗 odom 4,470、body cloud 424、RGB 594；四个运动话题发布者均为 0。
  当前仅检测，objects/viewpoints 为 0 是配置预期，不作为融合验收。
- 新车位静止点云采集 760 条、68,145 点；216 个位置/朝向初值及 12 个精配准候选
  收敛到近似位置，验证点集 96.6% 在图上最近邻 10 cm 内。已用该候选发布一次 initialpose；
  Point-LIO ICP=0.007914，45 秒 4,446 条 odom 无时间戳倒退，稳定后 XYZ 跨度
  6.43/7.81/10.63 mm、yaw 0.206°。末位姿约 (-0.228,0.163,-0.025)m、yaw -54°。
  当前已恢复保存地图定位；这是静止局部一致性，不代表路线覆盖或绝对精度。
  新证据在 `logs/diagnostics/relocalization-after-move/`；完整静止链路已重新启动，待采样验收。
  修复后的宿主启动器哈希与证据见 `logs/diagnostics/candidate-static-20261105/host-launcher-delta.json`。

### 新车位在线感知→视点→目标生成（修复后）

- 当前保存地图定位已恢复，容器中实际采用 D435i 标定/融合配置及 camera_init 世界系。
  45 秒：检测 22 条、对象 22 次、viewpoint pose 2 条；视点 0 的 observed_object_ids
  从空变为 [1,2,3]，其中 chair 为 sysnav_object:2。provider 未再因 frame 错误退出。
- 已归档 75 条 runtime 决策，其中 32 次 go_to_object；目标为 sysnav_object:2，
  解析到 viewpoint 0，坐标约 (-0.2274,0.1626,-0.0281)m、frame=camera_init。
  该视点来自当前实际观测车位，证明目标解析接线，不证明新路线可通行或实际行驶成功。
- 精确历史实例标注仍 matched=false，保留歧义；类别目标可生成不能替代实例绑定。
  最终 verifier 关闭，candidate bbox 元数据仍有 NaN，最终视觉证据路径未验收。
- 四个运动话题发布者均为 0，dry-run 记录 no waypoint published；底盘控制没有开启。
  证据：`logs/diagnostics/static-online-after-fix/{probe.json,runtime_decisions.jsonl,summary.json}`。

### 最终视觉证据入口审计

- `candidate_from_object` 对缺失二维框执行 `np.array(None, dtype=float)`，生成 `[NaN]`。
  已在本地改为缺失/非有限/非四元/退化框输出空列表；有效 xyxy 保留。
  不用虚构框补齐，也不改变候选 UID 或语义选择。此修复尚未进入当前运行镜像。
- 使用现有 Python 3.10 / pydantic 环境执行 `test_scene_query.py` 与
  `test_real_robot_final_verifier.py`，10 项通过。默认 Python 环境缺 pydantic，
  其 verifier 回退导致的两项失败不归因为本次修改；未更改环境或绕过断言。
- 最终 verifier adapter 使用当前 capture 的框覆盖候选地图框，因此 `[NaN]` 本身
  不证明视觉核验必然失败；仍需检验现场 capture 的图像/检测时间戳配对与实例归属。
  当前 ROS cache 将检测关联到 latest RGB，而检测存在处理延迟；下一步应对这条
  配对链做实际取证核对，不能把静止检测稳定或模拟 verifier 测试当作最终视觉 STOP。

### 检测图像、框与位姿配对修复

- `RosObservationCache.update_detection_result` 不再无条件借用 latest RGB：优先保留
  检测消息携带的输入图像，空内联 header 使用检测 envelope 时间戳；非空但矛盾的
  时间戳不接受。无内联图像时只查找同时间戳 RGB 缓存，不拼接另一帧。
- 保留有界位姿历史，证据图像配对最近实测位姿（最大偏差 0.25 s）；REACHED 时间屏障
  同时约束图像/检测和配对位姿。没有可配对图像时返回新整帧而不套用旧框。
  同类但不同 track ID 不得替代已选实例。runtime 已接入该配对入口。
- 缺失/无效 bbox 修复及图像配对、ID 区分、时间屏障、runtime 回归共 47 项通过，
  环境为 `/home/ubuntu/anaconda3/envs/conceptgraph/bin/python`，没有真实运动或 VLM 请求。
- 实时隔离订阅验证成功：sysnav_object:2、track 2，检测/证据时间戳均
  1793869929.2047157；位姿偏差约 -0.00418 s；原始检测图像和落盘图像字节哈希均为
  `5ab9d56d3970bfc8c2415b6c67ef2be7c1efcf91f6903cbe2c5f617929bc6338`。
  实测用 IDLE 状态加时间屏障验证传输，不伪造 REACHED，也没有产生 STOP。
- 证据 `logs/diagnostics/live-evidence-check/`。当前运行节点尚未替换；三文件独立候选
  镜像已构建并完成容器内哈希/导入/安装接线验证，ID 为
  `sha256:016e990ad48182ca54cb6d8ab4f802a2bb5be505e9d8c316912a6725e3320745`，
  tag=`huawei-vln-realworld:orin-r36.5-yoloworld-evidence-candidate`。尚未切换运行服务。
  基线/文件哈希、构建配方与验证回执在 `logs/diagnostics/evidence-candidate-build/`。
  这些检查只证明证据配对，不证明精确先验身份、真实视觉判定或最终闭环完成。

### 证据修复候选已上线及底盘只读复核

- 当前运行镜像已切换为 `016e990ad481...`，容器 `huawei-vln-static-evidence-candidate`；
  旧决策已在停止旧容器前保存，相机/LIO 未重启。新镜像 45 秒采样有 21 条检测、22 次对象。
  本窗口错过一次性 viewpoint 事件，runtime 仍保存有效视点 0：192 条决策中 156 次
  go_to_object，目标 sysnav_object:2，bbox_2d=[]；严格 JSON 序列化通过。
  四个运动话题发布者均为 0，未启用自主运动。
- 新的 10 秒底盘只读订阅：robot_status/ranger_status 各 485 条、BMS 484 条、selected 2 条。
  selected 仍为 `/run_management/virtual_joy`，反馈速度为 0；ranger 模式/状态为 0/2，
  robot_status 为 1/0，两者 fault_code=128。robot_status 文本为过压保护，BMS 电压 52.3 V、
  ranger 电压 55.5 V。未知枚举不按正常处理，也不以用户过去的遥控器解释覆盖当前样本。
  已询问当前遥控器状态；本次没有 advertise、速度 publish 或选源 service call。
- 原始证据：`logs/diagnostics/evidence-static-validation/`。
  启动、停止、回滚与版本边界见 [静止候选运行手册](real_robot_static_candidate_runbook_20261105.md)。

### 精确实例绑定的当前缺口与核验输入

- 全仓调用审计确认：`PriorMapMemory.mark_object_verified` 仅有测试调用，生产 runtime
  没有已接入的跨地图身份判定→写入路径。因此 matched=false 不只是检测质量问题；
  需要真实身份判断及其 provenance 接线，不能靠调宽类别/距离阈值消除未绑定状态。
- 新镜像目标快照的椅子到两个最近历史实例 XY 残差为 0.0727/0.1230 m，第三个 1.895 m。
  只比较 XY，未验证 Z 对齐。前两者不能凭最近距离唯一接受。
- 检索全部历史支持观察：detection1 有 6 条（desk chair 2、chair 1、table/shelf unit/monitor
  各 1）；detection6 有 5 条（trash bag 3、desk chair/chair 各 1）。这是标签混杂的信号，
  不是仅凭标签便断言不同实物，也不足以自动合并历史实例。
- 新容器会话内已重采当前椅子证据，使用镜像原生 cache：图像/检测时间戳一致，位姿偏差
  约 0.000042 s，原图字节哈希与落盘一致；未伪造到达或发送 STOP。
  不能把旧容器数字 ID=2 自动当作新会话同一身份，输入包显式绑定 container ID/started_at。
- 完整输入包：`logs/diagnostics/instance-correspondence-inputs/inputs.json`，含 11 条原始
  观察、图像、掩膜、框与 26 项文件哈希；decision=not_performed、binding_written=false。
- 当前项目/进程环境未发现可用视觉模型服务配置，运行仍 LLM_OFFLINE=1。
  已询问沿用的现有服务配置路径/地址和模型名，不在聊天收集密钥；此时不得将静态图像
  输送检查或 mocked verifier 的通过记为真实视觉判定通过。

### 已核对的底盘转接接口

用户指定核对 FW2W ICRA2027；正式源码位于
`/home/ubuntu/WorkSpace/research/code/ICRA2027/FWTW/tools/agilex_api/`。
`ros2_cmd_vel_to_agilex_ws.py` 将 ROS2 Twist 经 .102:9090 rosbridge 发布到
`/navflow_cmd_vel`；`keyboard_mux_switch.py` 通过 `/mux_vel/add`、`/mux_vel/select`
管理自动源与 `/run_management/virtual_joy` 人工源。physical 需要明确使能。
STRIVE 已在 `agilex_platform.py` 集成同一转接协议，并从安全仲裁输出提取 Twist，
核对选源回执和硬件状态合同；不再另启 FWTW bridge，避免双发布者。
重新核对本地 `agilex_platform.py::_worker_session`：STRIVE 已每 2 秒发送应用层
`op=ping`，此前“未见对应实现”的记录有误，不应新增重复心跳实现。
G3 仍需验收本车实际断链停车行为；源码中的心跳与断连撤销授权不能代替实车停车证据。
接口只负责控制转接，不能恢复 Orin 的 LiDAR/LIO 数据流，也不能证明故障已解除。

### 历史物理闭环口径的四个交付关口

- [ ] **G1 持久地图与定位**：局部 PCD、哈希、alignment 及保存地图静止初始化已完成。
  剩余是在实际任务覆盖区域和后续绑定摆位中验证连续定位；网线限制下的小范围采集
  不能当作完整场地图。既有尺度真值豁免仍保留，不补做卷尺精度实验。
- [ ] **G2 静止绑定与视点**：用先验约 (-1.0,-0.25) 的椅子做正向绑定，
  在正常摆位过程中取不同视角，确认同一实物 UID、坐标残差和可执行 viewpoint；
  顺带保留一个同类干扰/不匹配样例。全程 dry-run、远端控制关闭。
  已有 cart 负例可复用，不单独重复运行。记录现有匹配阈值，不能只凭 matched=true 判通过。
- [ ] **G3 实际底盘控制验收**：故障状态和底盘 watchdog/急停语义明确后，完成受限控制合约、
  唯一输出归属与限速；取得现场运动授权再验证短距离跟踪、停止、取消、人工接管、
  断链/传感器过期停止及重连不恢复旧命令。按不同停止机制覆盖，合并重复场景；
  停车时间/距离阈值在试验前固定，实际停车证据不可用零速日志替代。
- [ ] **G4 真实任务与交付**：自然语言→先验→在线绑定→视点→实际行驶→新鲜视觉核验→STOP；
  至少一个完整成功任务及失败/接管记录。把整栈连续 30 分钟资源/TF/丢帧观测并入现场会话，
  不再另开空转长测；固定部署版本、地图/标定哈希、配置、启动/停止/回滚步骤。
  单例闭环只支持受限场景交付，不作为泛化成功率结论。

### 从当前待办移除的重复或非阻塞工作

- 不重跑已通过的同一录包 1 m 路径规划、101 点路径及 HOLD 全零回放；
  仅在相关代码/配置变化或现场出现新问题时做对应回归。
- 不重复实现已落地的 bridge、类型适配、viewpoint 订阅就绪和 frame 修复；
  已有本地/fake 测试不作为新阶段独立任务。
- 不另做完整 HM3D 对照、仿真 sofa 修复、碎片几何阈值搜索或 tv/monitor 模型优化；
  已知局限保留，若现场目标受影响再定向处理。
- 不把“三个起点各一次”等建议采样数当硬性门槛；围绕不同失效机制采样，
  固定实际尝试分母和失败，不以删测试缩减控制安全与完整闭环验收。

**启动注意：** `start_real_robot_sensors.sh ... start` 会复用已有 livox_odom 会话，
仅设置 `LIO_PCD_SAVE_EN=1` 再调用 start 不保证切换存图。必须先检查当前参数、
保护未保存数据，再按现场许可有序重启并复核参数；禁止直接 kill-session 丢图。


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

## 1. 历史结论（以文首正式入口状态为准）

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
    - 隔离容器（`--network none`）回放同一录包，terrainAnalysis `terrainFrame=camera_init` +
      localPlanner `useTerrainAnalysis=true`（无 pathFollower）：`/terrain_map` frame 为
      `camera_init`，1.0 m 接近点 6.3 s 得有效路径（101 点）。顺带修回放脚本 `timeout_s`
      默认值须为浮点（原 `30` 会被 probe 拒绝）。
    - 同一隔离方式跑完整 `sysnav_lower_stack.launch.py`（terrain/world frame=`camera_init`，含
      pathFollower、MotionServer、SafetyVelocityMux）并循环回放录包，发 `camera_init` 动作目标：
      目标被接受，`/way_point` frame `camera_init`，`path_valid=true`；mux 全程 HOLD
      （autonomy 未使能），551 条 `/cmd_vel` 全为零，目标以 `goal_timeout`（“safety mux
      remained in hold”）结束。该回放证明 HOLD 链路有效；真实运动仍需当前清单 G3 的合同、底盘状态与现场控制验收，不能仅放行 autonomy。
    - 16:26:43 在原位以存图模式重启 LIO（重启前位姿 (-0.005,0.010)，偏航差约 0.3°），
      对齐 JSON 继续适用。当日推车推迟；存图模式静止时内存约 +1.4 GB/h，16:45:29 原位恢复
      非存图会话（车未动）。推车前需原位再以 `LIO_PCD_SAVE_EN=1` 重启。
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
