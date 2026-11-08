# 2026-11-05 椅子录包离线诊断

> 归档说明（2026-11-08）：历史“当前/下一步”和测试数仅对当时版本有效，非最新部署指令；原文及负结果保留，代码块中的历史路径按当时工作目录解释。当前入口见 [文档门户](../README.md)。

最新结论：部分修复，替代模型已取得离线融合正例。首个可复现阻断发生在检测输出前；
已修复重复颜色通道转换，但原 YOLOE 模型在这段实物图像上仍不能稳定产生带 ID 的 chair。
后续使用既有 YOLO-World 权重获得连续检测并完成对象发布（详见末节）；正向先验绑定未通过验收。
不能据此判定融合、先验匹配或候选过滤已经正常，也不能将问题全部归因于颜色错误。

## 输入与隔离边界

- 录包：车端部署 `/home/orin26/HuaweiVLN_deployments/realworld_eb188e7/output/field_checks/chair-20261105T1510/`，195.7 秒、2,929 帧 RGB。
- 离线容器：`strive-chair-offline-20261105`，`--network none`，录包/模型只读挂载。
  未运行底盘接口、未发布自主运动命令、未重启 LIO 或改变地图对齐。
- 原镜像：`huawei-vln-realworld:orin-r36.5-agilex-1aea701`；YOLOE `yoloe-11s-seg.pt`，
  Ultralytics 8.4.66；完整 41 个提示词，正式 imgsz=320、conf=0.35、half=True。
- 使用录包中压缩 RGB 解码所得 BGR；非精确重放当时 callback 调度。未保存原始检测/语义节点消息，
  历史 Docker 日志时间范围也不完整，因此这些是固定输入复现实验，不是线上逐帧追溯。

## 固定分母结果

| 实验 | 输入/修改 | 阈值通过的椅子帧 | 带 ID 的椅子输出 |
| --- | --- | --- | --- |
| 全段均匀采样 | 首帧后每隔至少 2 秒，97 帧；原配置 | 0/97（跟踪调用结果） | 0/97 |
| 全段真实 inference | 同 97 帧；仅修正 BGR 输入 | 未单独计数 | 0/97 |
| 全段诊断配置 | 同 97 帧；BGR、640、完整词表增加 office chair | 原始 predict 0/97，最高 0.3057 | 0/97 |
| 清晰片段原配置 | 时间戳 1793862792–1793862812 秒，每隔至少 0.5 秒，共 38 帧 | 原始 predict 0/38，最高 0.1688 | 未单独运行 |
| 清晰片段最小修复 | 同 38 帧，仅修正 BGR | 原始 predict 1/38，0.3540 | 真实 inference 0/38 |
| 清晰片段诊断配置 | 同 38 帧，BGR、640、office chair | 原始 predict 2/38，最高 0.4282 | 未单独运行 |

上述原始 predict 使用 conf=0.05 仅观察阈值以下输出，统计通过率仍按正式 0.35。
原配置全段的 `.track()` 结果无法严格等同跟踪前的原始检测；清晰片段原始 predict
和全段诊断配置原始 predict 提供了跟踪前的独立证据。两套采样有重叠，不合计为 135 个独立样本。
单帧最清楚画面在 BGR+640+office chair 时可达 0.4282，但不能代表持续检出。

## 阶段定位与最小修复

1. **确定的输入合同错误**：CvBridge 明确输出 BGR，`inference()` 又手动反转为 RGB；
   实际 Ultralytics `BasePredictor.preprocess()` 对 numpy 输入再执行 BGR→RGB，造成模型收到错误通道。
   修复为直接传入 BGR numpy 数组，不改变尺寸、阈值、词表或 tracker。
2. **剩余检测瓶颈**：清晰片段在正确通道下仍仅孤立一帧越过 0.35；真实 BoTSORT 跟踪后
   没有带 ID 的椅子。提高分辨率/加提示也未形成充分的持续检出证据，故不作为正式修复。
   序列中出现过 ORB GMC 匹配不足/回退警告，但没有证据表明它解释了跟踪前的低置信度。
3. **融合不是本次首个阻断点**：修复后两套采样传给后续模块的 chair 数为零。
   ObjMapper 自身另有 confidence>=0.30、有效点数等门槛；此次没有真实 chair 输入穿过它，
   不宣称投影、分割、聚类、先验绑定全部通过，也不改这些门槛。
4. **候选过滤尚未被正向验证**：`No eligible observed candidate` 是末端表现，
   本次没有证据支持通过放宽过滤来修复。后续必须从真实带 ID 的椅子追踪到点云、UID、绑定和视点。

## 验证与证据

已在隔离容器内提取并执行本地修复后真实 `inference()` 方法，使用真实 YOLOE、BoTSORT、原始词表
及录包输入；只绕开 ROS 节点初始化，不替换检测/跟踪实现。结果如上。
新增红蓝不对称输入回归测试，检查交给后端的 BGR 内容、连续内存、原图不变以及原门槛保留。

本地命令：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_navigation_root_causes.py -k real_detector
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_detector_vocabulary.py
git diff --check
```

结果分别为 1 passed、4 passed、无 whitespace 错误。关闭 pytest 第三方自动插件是为了避开
本机无关 Dash/Flask/Jinja2 插件冲突，不涉及业务运行配置。

逐帧时间戳、框、置信度、跟踪 ID、实际后端预处理源码及 SHA256 清单位于本地
`logs/diagnostics/chair-detection-20261105/`；其中 `continuous_ablation.json`、
`fixed_inference_tracking.json` 是本次更密集采样与最小修复验证的主要证据。
车端同名诊断目录保留原始实验 JSON 与 `detection_node_fixed.py`。

## 交付边界与下一步

修复已进入本地源码，未替换正式车端镜像。未降低阈值、未绕过跟踪或伪造候选；
当前不能宣布“实物椅子可用目标”恢复，也不能进入自主运动验收。
下一项只保留检测稳定性：用这批固定帧约束后续检测方案，补充椅子以外负例衡量误检，
先获得持续真实 chair 输出，再做一次完整离线融合→候选链路验证。
不要求现场重复往返或重复已通过的路径规划测试。

## 后续推进：跟踪输入、全帧排查与 SAM2 输入修复

- 保持最终输出门槛 0.35，仅将 tracker 输入门槛降为 0.10，以允许 BoTSORT
  的低分关联阶段获得输入：38 帧和 97 帧仍均为 0 个带 ID 且过 0.35 的 chair。
  该诊断参数没有写入正式配置，证据为 `tracker_input_ablation.json`。
- 对同一 20 秒清晰片段的全部 300 帧运行真实 track，BGR320 原词表和
  BGR640 加 office chair 两组均为 0 个带 ID 的椅子。因此在该片段中，单纯取消
  2 秒采样不足以恢复有效轨迹。证据为 `fullrate_ablation.json`；track 返回值
  受跟踪关联影响，不能将其当作原始 predict 的检测数量。
- 发现独立的融合入口格式错误：`detection_callback` 用 `bgr8` 解码，直接交给
  SAM2 `set_image`，而 vendored SAM2 明确要求 RGB。最小修复仅在 SAM2 调用处
  使用 `cv2.cvtColor(..., COLOR_BGR2RGB)`，后续映射原图仍保留 BGR。
  此错误位于检测之后，不用它解释上游没有 chair；也尚无证据宣布目标融合通过。
- 对真实生产方法运行两项输入合同测试，分别验证 Ultralytics 接收 BGR、SAM2
  接收 RGB，以及两条路径均不改写原图；结果 `2 passed`。
  命令：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_navigation_root_causes.py -k 'real_detector or real_mapper'`。
- 转接接口复核纠正了清单：本地和车端 `agilex_platform.py` 均已有每 2 秒
  `op=ping` 心跳，不新增重复实现。真实断链停车验收仍未完成。

## 替代模型对照与真实融合输出

复用工作区已有 `yolov8l-world.pt`，没有下载新模型或改变生产默认配置。
本地、车端临时副本 SHA256 均为
`8bdfaef999116760247d6fb0b0f8fca064b43e94598b3d4a807ebae9bcf0cdd5`。
使用原 41 项词表（不加 office chair）、BGR、imgsz=640、half=True、输出 conf=0.35，
BoTSORT 配置不变。CLIP 使用部署中已有 ViT-B-32.pt，通过隔离容器自己的缓存路径加载。

| 固定输入 | 原始 predict 椅子过 0.35 | 跟踪后椅子输出 |
| --- | --- | --- |
| 清晰片段 38 帧 | 31/38 | 29/38；ID 4 出现 27 帧、ID 1 出现 2 帧 |
| 全段采样 97 帧 | 未单独测量 | 44/97；ID 1 出现 37 帧、ID 5 出现 7 帧 |

这是有无输出的统计，不是标注后准确率或任务成功率。全段可能含多个椅子，不能仅靠 ID
断言同一实物身份。模型 track 耗时中位数（去掉首帧）分别为 0.0721/0.0963 秒，
不含 SAM2、融合和在线队列，不能当作整栈频率。

使用清晰片段真实 track 输出，运行实际 SAM2、CloudImageFusion、ObjMapper 与 ROS
ObjectNodeList 序列化。仅 29 帧有有效跟踪输出进入融合；29 次发布均含 chair，最终
chair 对象合并 ID `[1, 4]`，位置约 `[-1.2792, -0.4758, 0.1227]`（录包 LIO 坐标系）。
这不是先验坐标残差或目标匹配成功证据。

诊断边界：使用按 bag 时间选取的最近邻 body cloud 与 odom，每帧最多 10,000 点；
最大时间差分别为 0.0533 秒和 0.0060 秒。使用正式 v3 投影、无 mask erosion、关闭深度跳变过滤，
RGB 在 SAM2 入口转换为 RGB。与生产的缓存窗口/里程计插值/节流不完全等价。
viewpoint_id=-1，不伪造可执行视点；关闭图像落盘，不提供图像证据供最终视觉验收。

此次实际调用暴露并修复两个接口问题：

1. YOLO-World 分支原本未执行 `set_classes`，后续却按自定义词表解码类别索引。
   现在两个开放词表后端都初始化同一提示词顺序；固定类别 YOLO 与 engine 不改变。
2. `best_image_path` 尚无值时为 None，直接赋给 ROS 字符串字段导致对象发布断言失败。
   发布时将缺省值规范为空字符串；修复后 29 次实际 ROS 序列化成功。

新增真实构造器回归覆盖 YOLOE/YOLO-World 提示词顺序；颜色合同与构造器测试 4 passed，
词表及 ROS adapter 测试 27 passed。未声称全仓测试通过。

证据：`world_model_probe.json`、`world_tracking.json`、`world_fusion_probe.json`；
车端另存 `world_objects_*.cdr`；可复现诊断脚本 `replay_fusion_probe.py` 与哈希清单位于
同一 `logs/diagnostics/chair-detection-20261105/` 目录。之前失败实验未删除。

下一步先审阅图像/负例并将这些真实对象接入先验匹配，验证同一实物 UID、残差和视点。
默认模型仍为 YOLOE；替代权重、CLIP 缓存和修复代码尚未打包成正式车端版本。
完整 ROS 链路、定位恢复、控制与任务闭环的交付关口保持未完成。

## 真实对象到先验与目标解析器的审计

将 29 次实际 ObjectNodeList 发布经 `RosObjectNodeAdapter` 转为快照，使用录包 odom 的
`camera_init` 坐标、原先验地图与该 LIO 会话的 alignment，运行实际 prior runtime、
semantic snapshot policy 和 SysNavGoalResolver。所有步骤均为离线函数调用。

- 29/29 次选择了观测到的 chair，产生语义 GO_TO_OBJECT intent；它不是运动命令。
  UID 序列包含 `sysnav_object:1`、`sysnav_object:4` 各 1 次，合并后的
  `sysnav_object:1:4` 共 27 次；不能将原始 track ID 变化当作持久 UID 已无条件稳定。
- 29/29 次快照均无可执行视点，解析器返回 None，运动目标 0/29；没有用对象中心
  伪造机器人目标位姿。当前不能宣布视点链通过。
- 最后一个椅子中心变换到先验平面约 `[-0.8841, -0.1675]`。最近两个历史 chair
  分别为 `frame1350:detection1` 和 `frame1350:detection6`，平面残差约 0.1373/0.1797 米，
  第三个约 1.9627 米；29 次的最近候选残差范围为 0.0818–0.3140 米。
  该残差包含分割/点云中心变化，不是定位误差真值，也不使用未经对齐的 z 值作比较。
- 核对两张当前检测图和两个历史候选支持图，均包含带设备的黑色办公椅；历史两个候选
  存在同一椅子不同部位/碎片的可能性。保留两个候选身份，不凭近邻距离直接合并或验证绑定。
  当前画面还出现 shoe/cabinet 等类别误判；没有据此宣称模型整体准确率。

在同一个真实末帧上复现了先验注释缺陷：HEAD 中 `_match_object_prior` 的标签回退
将当前 chair 匹配为历史 `frame250:detection0`（`exact=True`），平面残差 5.0054 米，
且 `matched_runtime_uid=None`。这只能证明同类别，不能证明历史实例身份。

最小修复：具体 historical instance 只接受已有 UID/显式对应；标签回退仅允许唯一的
非 exact 类别先验，多个类别假设不任取第一个。注释增加 `match_scope` 区分类别先验与
历史实例。修复后的真实快照注释为 matched=false，同时仍可选择在线椅子作为类别任务目标。
没有为本例添加距离特判，也没有调用 `mark_object_verified` 冒充正向绑定成功。

新增回归覆盖同名 exact 拒绝、多个类别假设歧义拒绝、显式 runtime 对应正确选择；
`test_prior_map_policy_adapter.py` 与 `test_prior_map_real_robot_integration.py` 共 14 passed。
证据位于 `logs/diagnostics/chair-binding-20261105/`：`world_snapshots.json`、
`binding_audit.json`、`adapter_before_after.json`、检测/历史图像及可复现 `check_binding.py`。

后续重点是带真实 provider/bridge 的隔离 ROS 回放，获得对象与视点的关联并验证 UID
合并后的关联继承；同时保留历史实例歧义，不以 matched=true 作为验收替代品。

## 实际 ROS 建图与视点回放

在 network=none、ROS_DOMAIN_ID=77 内运行实际 mapping、viewpoint provider 和 bridge，
不启动控制器。首次脚本用了错误的 bridge 可执行名，播放前退出，失败记录保留在车端
`chair-viewpoints-20261105/attempt1/`；纠正为注册入口后，三个节点完成回放，无异常退出。

回放 1793862792–1793862814 秒窗口，0.5 倍速，共 2,196 条 odom、207 条 body cloud、
38 帧冻结 YOLO-World 跟踪结果及原图。未录 world cloud，因此以最近记录 odom 对 body
点云做实际刚体变换，发布 `/offline/world_cloud` 给 provider；没有仅修改 frame 名。
mapping 使用正式缓存窗口、2 秒处理间隔、点数上限、投影与 SAM2 路径。

- 15 次实际 ObjectNodeList 发布，1 个 provider 视点事件（ID 0）。
- 2 次 bridge 位姿更新：初始无对象，随后 observed_object_ids=[4]。
- mapping 在视点周期发布 chair/viewpoint_id=0；后续自由运行更新 viewpoint_id=-1。
  正式 scene cache 保留历史关联，没有被后续更新清空。
- 使用实际 CDR 对象和记录的位姿字段做回放后解析：语义目标为 sysnav_object:4，
  解析出的视点 0 位置约 [-0.51249,-0.21517,-0.02513]，frame=camera_init。
  位姿来自 provider 时间戳对应 odom，不是椅子中心；排除视点 0 后正确返回 None。

这支持“冻结检测→实际 ROS 建图→provider→bridge→目标解析”的离线正例。
回放后 cache 检查使用记录视点作为 robot_pose 占位，不能代表现场当前车位；
它不影响此 provider 的身份交集选择，但不构成到达证据。目标是当时观测位置，未发布执行。
没有证明新路线可达、完整 detector 并发负载、具体历史实例绑定或最终视觉 STOP。
本地 `logs/diagnostics/chair-viewpoints-20261105/` 保存 receipt.json、resolved_goal.json、
objects_*.cdr、节点日志、replay_ros_viewpoints.py 与哈希清单。

正式默认模型和部署镜像仍未替换。下一步准备候选部署及未参与调参的负例检查，
保留具体历史实例歧义；保存地图定位恢复及实车控制仍需要现场条件和运动授权。

## 独立录包核对与候选构建

冻结上述模型、41 项词表、640/BGR/0.35/BoTSORT 参数，使用未参与此前调参的
`tethered-20261105T1504` 录包：首帧后每隔至少 2 秒取样，共 111 帧。
28 帧有 chair 输出，其中 24 帧带 ID。查看全部 28 张预测图，均可见带设备的黑色办公椅；
框有时包含周围杂物。起始帧没有椅子，输出为空，是一个明确负例。
这是同场景的另一段录制，不是跨场景泛化测试；未逐帧标注所有漏检，不报告召回率或整体准确率。
证据：`logs/diagnostics/chair-heldout-20261105/{receipt.json,review.json,contact_sheet.jpg}`。

新增独立候选 profile `orin26_livox_mid360_d435i_yoloworld_candidate.env`，继承原传感器配置，
使用独立镜像和容器名，全部运动开关仍关闭，原默认 profile 不变。Bash/Zsh 均成功加载，
shell 语法与 diff 检查通过。正式 runner/smoke 补充 CLIP 权重只读挂载到 Ultralytics
默认的 `/workspace/STRIVE/weights/clip/`，保留原 CLIP 默认缓存挂载。

全新 network=none 容器中验证了 41 项词表初始化和真实起始负例推理，无需下载，退出 0。
首次 smoke 的诊断命令漏传 NVIDIA 环境变量而看不到 GPU；保留失败记录，按既有 runner
配置传入 NVIDIA_VISIBLE_DEVICES/DRIVER_CAPABILITIES 后通过，不修改宿主驱动。

已冻结当前 HEAD=3a3db20 加工作区修复的 612 个源码文件，并在车端逐文件核对哈希。
源代码包 SHA256 为 `71f340c5601b26b10e30db724f86f39e02ddcc67ecdbba78f4511b3be772ec41`。
独立候选目录：`/home/orin26/HuaweiVLN_deployments/yoloworld_candidate_71f340c5`；
5 项模型依赖已单独复制并验证，不跨部署建立符号链接。源码与资产 manifest 保留在本地
`logs/diagnostics/yoloworld-candidate-20261105/`。

镜像构建使用已验证基镜像 `huawei-vln-realworld:orin-r36.5-agilex-1aea701`，
`docker/Dockerfile.real_robot`、`--network none`、SKIP_BASE_DEPS=1、INSTALL_ML_DEPS=0、
INSTALL_LLM_DEPS=0，目标 tag 为 `huawei-vln-realworld:orin-r36.5-yoloworld-candidate`。
构建正在运行，不能记为完成；尚未用新镜像启动正式服务。后续从该目录 build.log 和实际
镜像 ID 核验完成，再做该镜像的 smoke，不复用旧容器成功冒充新版本通过。

用户确认采集结束后未移动、未转向、未重启。已读取末条 odom 准备定位种子：
camera_init 下位置约 [-0.150503,0.041775,-0.023260]、yaw=-0.280879 rad。
这只是种子；现场 Point-LIO `/initialpose` 会根据地图另选 z 并执行 NDT+ICP，仍需检查
收敛、点云重合和位姿连续性。当前没有发布 initialpose 或运动命令。

## 候选镜像在线配置边界补充

新镜像固定 111 帧实际 DetectNode 回归得到 24 帧带 ID 椅子，与前述独立录包的
跟踪正例集合相同。在线首次运行暴露了宿主 Docker wrapper 漏传参数：不仅包括
`SYSNAV_WORLD_FRAME`，也包括 mapping 开关、标定配置、检测/对象话题名。
首对象触发 provider 时出现 `camera_init` 对 `map` 的严格校验失败；保留校验，
补齐环境传递，并用运行真实 shell wrapper 的 fake-Docker 回归验证。
因此，修复前预期话题上的零消息计数不能证明实际 detector 没有输出。

用户移动、转向并重启后，新的完整椅子取景下，补全配置的纯检测节点在 45 秒
收到 20 条结果，全部包含 chair、ID 始终为 2；实读阈值 0.35、imgsz 640、
配置处理间隔 2 秒。该结论仅覆盖该静止取景，未把它当作融合/绑定/运动验收。
原始记录见 `logs/diagnostics/candidate-static-20261105/detector-only-probe.json`，
宿主启动器版本差异见同目录 `host-launcher-delta.json`。
