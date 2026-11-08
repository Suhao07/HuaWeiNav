# IntentionNav 接入与验收

> 归档说明（2026-11-08）：历史“当前/下一步”和测试数仅对当时版本有效，非最新部署指令；原文及负结果保留，代码块中的历史路径按当时工作目录解释。当前入口见 [文档门户](../README.md)。

**目标：在完整下载的单场景先跑通 STRIVE 闭环，场景下载独立推进。**

## 当前状态（2026-11-05，单episode评分后）

**105 的初始化/渲染阻塞已通过独立安装并固定 Isaac Sim 6.0.1 解决。STRIVE tool-agent 已完成一轮有界运行、真实 STOP 和官方评分，但导航失败：SR=0、SPL=0，60次重复查询后预算耗尽，未发生平移。不能宣称有效导航闭环或全量 benchmark 已验收。**

| 必要步骤 | 状态 | 证据与下一步 |
|---|---|---|
| 冻结需求组 | 完成 | IntentionNav 500任务×2措辞；LangMap 1,000任务，分别统计 |
| 选择完整场景 | 完成 | `SEL_001` / `kujiale_0262`，875个资产已校验；本次再次核对入口 USD 的冻结哈希 |
| RGB-D共享协议适配 | 单场景运行验收通过 | 768×768 RGB/光学Z深度、实际位姿/内参及三视角重投影通过；复用现有导出代码 |
| Isaac真实渲染 | **6.0.1通过** | 官方空场景30次更新与正式首帧入口均退出0；4.5保留，仍有原生崩溃 |
| STRIVE任务、感知、运动闭环 | 已接线，有效导航仍待验收 | 实际TaskSession、DINO/SAM、共享融合、MotionSupervisor及STOP；最新完整运行未提交移动 |
| 单episode结果 | **已有失败结果** | SEL_001：SR=0、OSR=0、SPL=0，终点距离2.3m，路径0m；正常预算终止不等于导航成功 |

## 首次有界运行与官方评分（2026-11-05）

- Run ID：`strive-SEL_001-tool-agent-isaac6-20261105-221624`。
- 105原件：`/mnt/nvme0/haosu/IntentionNav/results/strive-SEL_001-tool-agent-isaac6-20261105-221624`。
- SEL_001 / kujiale_0262 / formal，官方原始指令与起点未改；策略输入仅包含selection_id、scene_id、style、instruction、start_position、start_rotation_quat_wxyz六字段。目标类别、实例、位置、房间及照片不从GT注入策略；评估私有元数据在策略退出后写入。
- Isaac实际版本 `6.0.1-rc.7+release.42383.32955d8d.gl`，GPU7渲染、GPU5真实DINO/SAM感知，复用已有Qwen服务；没有重启模型服务或改驱动。
- 模式为 `strive_tool_agent`，不是默认 `baseline_guided`，也不是官方Isaac4.5等价复现；动作是freemap检查后的运动学相机位移，不是动态物理机器人。

| 项目 | 实际结果 |
|---|---|
| 官方SR / OSR / SPL | 0 / 0 / 0.0 |
| 终点 / 最小目标XY距离 | 2.3m / 2.3m |
| 实际XY路径 / 官方最短路 | 0.0m / 3.897m |
| 停止原因 / 任务完成 | `decision_budget_exhausted` / false |
| 决策调用 / 全部模型调用 | 60 / 66 |
| 移动命令 | 0；60次工具调用全部为 `query_scene`，均无合格候选 |
| RGB-D | 7帧，768×768，深度与实际位姿审计通过 |
| 评分steps / 最终轨迹行 | 7 / 9；评分后shutdown额外追加一条无位移STOP，steps不是移动次数 |
| 策略耗时 / 退出码 | 407.07秒；策略和模拟器均为0 |
| 停止回执 | STOP及shutdown均确认zero_velocity |

独立审计复算官方 `nav_metrics` 与落盘评分一致，并检查六字段输入、容器挂载、模型输入中的私有ID、源码哈希、逐段碰撞及真实RGB-D。零移动轨迹的碰撞检查不构成非零运动能力验收。191项本地回归通过；语义完成、GSR及目标可见性成功未验收。

证据：[结果与完整日志包](../../../STRIVE-deploy-105/evidence/strive-SEL_001-tool-agent-isaac6-20261105-221624-results.tar.gz)、[官方评分](../../../STRIVE-deploy-105/evidence/strive-SEL_001-tool-agent-isaac6-20261105-221624/evaluation.json)、[轨迹](../../../STRIVE-deploy-105/evidence/strive-SEL_001-tool-agent-isaac6-20261105-221624/trajectory.jsonl)、[工具决策](../../../STRIVE-deploy-105/evidence/strive-SEL_001-tool-agent-isaac6-20261105-221624/tools.jsonl)、[审计](../../../STRIVE-deploy-105/evidence/strive-SEL_001-tool-agent-isaac6-20261105-221624/audit.json)。日志包含完整模型请求/响应、动作、感知、场景快照、7帧RGB-D和源码哈希清单；源码快照保留在105原件中。

此前两个异常尝试分别保留在 `20261105-220708`（freemap原始编码0/1/2误按二值校验）与 `20261105-220855`（实际移动2次后模型使用错误control_epoch而退出）对应run目录，均未伪装成完整评分结果。已按官方仅 `==1` 可行走语义修正加载；错误任务/控制版本现在返回拒绝回执，不代改参数、不放宽运动所有权检查。最新运行没有提出移动，因此没有在真实运行中触发该拒绝恢复路径。

**下一步先解决无候选时反复查询、不探索的策略停滞，再在同一完整episode内验收有效移动和目标验证；不能仅因评分程序正常返回就启动或宣称全量benchmark验收。**

## 共享观测协议

复用现有地图协议：米制、右手系、世界Z向上、光学相机右/下/前。

\[
T^{map}_{optical}=T^{map}_{USD}\operatorname{diag}(1,-1,-1,1),\qquad
p_{map}=T^{map}_{optical}[zK^{-1}(u,v,1)^\top,1]^\top.
\]

位姿直接读取USD相机实际变换，Gf行向量矩阵转置为共享协议的列向量矩阵。
深度使用 `distance_to_image_plane`，无效值转0并保留有效比例；整帧无效直接报错。
目标GT不进入导出的帧与指令；start pose仅用于初始化。

渲染入口：`scripts/capture_intentionnav.py`，使用Isaac自带 `python.sh`。
官方环境封装仅增加显式GPU参数，禁用多GPU渲染，当前指定GPU0。

## 4.5 阻塞诊断（历史证据，保留）

105硬件为RTX3090，驱动595.99.02。Isaac 4.5实际运行和GDB定位均在SimulationApp初始化阶段失败：

```text
SIGSEGV
librtx.scenedb.plugin.so
libcarb.scenerenderer-rtx.plugin.so
libomni.hydra.rtx.plugin.so
```

**已确认4.5的RTX库原生崩溃；在宿主驱动和审计库不变的条件下，6.0.1已通过相同GPU上的空场景及真实场景验证。版本切换是已验证的解决方案，但不足以唯一归因于驱动或审计库。**
既有外部线索：[IsaacSim issue 912](https://github.com/isaac-sim/IsaacSim/issues/912)；本次未独立核实该issue内容，不将其作为根因或验收证据。

远端原始日志：

```text
/mnt/nvme0/haosu/IntentionNav/logs/strive-SEL_001-render-001.log
/mnt/nvme0/haosu/IntentionNav/logs/strive-SEL_001-render-gdb.log
```

保留4.5与失败产物。用户已确认并行安装新版。当前选择Isaac 6.0.1，保留4.5与整机驱动。
整机驱动变更会影响其他GPU任务，需要维护窗口。

## 指标口径

固定官方评测代码 `eval/aggregate/compute_metrics.py::nav_metrics` 使用二维终点距离与2米半径：

\[
S=\mathbf1[\|x_T-p_{GT}\|_{xy}\le2],\qquad
SPL=S\frac{L^*}{\max(L^*,\sum_t\|x_t-x_{t-1}\|_{xy})}.
\]

其中L*来自官方episode元数据。可见性/语义接地指标单独报告，避免将几何到达等同于目标辨认。
该口径与LangMap的目标视点测地距离<0.25米不同，**两个数据集各自统计**。

## Isaac 6.0.1 并行验证

官方6.0.1要求页列出的Linux测试驱动为595.58.03，与105的595.99.02同分支；实际3090渲染仍以验收为准。
[官方版本要求](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/requirements.html)

- 新目录：`/mnt/nvme0/haosu/IntentionNav/isaac-sim-6.0.1`。
- 安装包：官方 `https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.0.1-linux-x86_64.zip`。MD5：`65e2c2e83e2461ce0f33b0732d0ee4a3`；本地SHA256校验记录另存。
- 启动入口：`/home/haosu/Workspace/Navigation/IntentionNav/validate_isaac6.sh`，顺序执行下载、校验、解压、真实首帧采集；使用GPU0。
- 总状态：`/mnt/nvme0/haosu/IntentionNav/logs/isaac6-validation.status`。下载/校验状态：同目录 `isaac6-install.status`。
- 流水线日志：同目录 `isaac6-validation.log`。渲染日志：`strive-SEL_001-render-isaac6-001.log`。
- 输出：`/mnt/nvme0/haosu/IntentionNav/results/strive-SEL_001-render-isaac6-001`。
- **安装和自动首帧已完成。** 原流水线状态 `capture_complete_pending_geometry_review` 仅代表首帧里程碑；随后独立完成了下面的图像、深度、几何和默认入口验收，并非修改该状态冒充通过。

### 6.0.1 验收结果（2026-11-05 20:48 CST）

安装包MD5通过；归档SHA256为 `348013c4d3ac398d40162566d0edfe4f3837a0eb5efa358ec87065f81aa68dd7`，见[校验记录](../../../STRIVE-deploy-105/evidence/isaac6-archive.sha256)。实际VERSION为 `6.0.1-rc.7+release.42383.32955d8d.gl`。

| 检查 | 实际结果 | 本地证据（远端原件保留） |
|---|---|---|
| 官方空场景 | GPU0，禁用多GPU；完成30次更新，正常关闭，退出0，17.01秒 | [回执](../../../STRIVE-deploy-105/evidence/isaac6-qoder-empty-receipt-20261105.json)、[日志](../../../STRIVE-deploy-105/evidence/isaac6-qoder-empty-20261105.log) |
| 原场景自动首帧 | 原SEL_001起点，真实RGB和光学Z深度；首次冷启动采集180.06秒 | [manifest](../../../STRIVE-deploy-105/evidence/strive-SEL_001-render-isaac6-001/manifest.json)、[RGB](../../../STRIVE-deploy-105/evidence/strive-SEL_001-render-isaac6-001/rgb/000000.png) |
| 三视角几何 | 原位姿、偏航+8°、沿原相机右向平移0.10米；实际位置/光轴与请求一致，三帧均768×768且深度有限 | [验收回执](../../../STRIVE-deploy-105/evidence/strive-SEL_001-isaac6-geometry-001/acceptance.json)、[原始帧索引](../../../STRIVE-deploy-105/evidence/strive-SEL_001-isaac6-geometry-001/frames.jsonl)、[日志](../../../STRIVE-deploy-105/evidence/isaac6-qoder-geometry-20261105.log) |
| 固定默认入口复跑 | 读取runtime.env，不临时覆盖版本；退出0，进程31.31秒，采集27.35秒，有效深度99.8827% | [回执](../../../STRIVE-deploy-105/evidence/isaac6-qoder-default-entry-receipt-20261105.json)、[RGB](../../../STRIVE-deploy-105/evidence/strive-SEL_001-render-default-isaac6-001/rgb/000000.png)、[深度](../../../STRIVE-deploy-105/evidence/strive-SEL_001-render-default-isaac6-001/depth/000000.npy)、[日志](../../../STRIVE-deploy-105/evidence/isaac6-qoder-default-entry-20261105.log) |

现有几何诊断使用stride=8、阈值0.05米；预设验收为每对投影点>1000、全投影一致率>90%、可见点绝对残差中位数<0.02米，三对全部通过：

| 帧对 | 投影点 | 全投影一致率 | 可见点残差中位数 / p95 | 前向冲突（远离深度边缘） |
|---|---:|---:|---:|---:|
| 0→1 | 7928 | 99.6720% | 0.493 / 9.181 mm | 14（2） |
| 0→2 | 6767 | 99.5271% | 0.172 / 5.538 mm | 4（0） |
| 1→2 | 5951 | 99.0758% | 1.207 / 9.787 mm | 12（1） |

原始RGB已逐帧查看；落盘RGB非恒定、float32深度有限且非负、内参/正交右手位姿均复核。三视角是交叉一致性检查，不是绝对物理标定，也不保证无残余冲突。

最终身份核验：
- 原入口USD为11,968,797 bytes，SHA256 `a365ad55ea935142de42856f38a19650c016b4ca2e835c2a3d2626cb0fbc8720`，与固定revision的[场景清单](../../../IntentionNav-deploy-105/scene_manifest.jsonl)一致。
- 本地与105的runtime.env SHA256相同：`9e2ca95875a5a6e1e0844ba564973b37777a4c1e32332a83453a5dbb9a6e5636`；相对备份仅修改ISAACSIM_ROOT的版本目录。
- 采集入口、RGB-D导出和几何诊断源码的本地哈希与验收回执记录的远端哈希一致；本次未修改这些生产Python文件。
- 4.5安装保留，NVIDIA驱动仍为595.99.02，系统审计库仍加载；没有修改整机驱动或审计配置。新版不再崩溃，因此未触发条件性的容器/管理员排查。

### 已固定入口与边界

105的 `/home/haosu/Workspace/Navigation/IntentionNav/runtime.env` 和本地[部署配置](../../../IntentionNav-deploy-105/runtime.env)均默认指向6.0.1。105可在STRIVE目录执行以下命令，输出目录必须是未存在的新目录；不要重跑带固定输出路径的安装流水线：

```bash
source /home/haosu/Workspace/Navigation/IntentionNav/runtime.env
export CUDA_VISIBLE_DEVICES=0
"$ISAACSIM_ROOT/python.sh" \
  /home/haosu/Workspace/Navigation/STRIVE/scripts/capture_intentionnav.py \
  --data-root "$INTENTIONNAV_DATA_ROOT" --selection-id SEL_001 --gpu 0 \
  --output "$INTENTIONNAV_DATA_ROOT/results/strive-SEL_001-render-$(date +%Y%m%d-%H%M%S)"
```

远端原配置备份为 `runtime.env.before-isaac6-20261105`，可供版本回退参考，但回到4.5会回到已知失败路径。

剩余限制：
- 场景日志仍有 `limpopo_golf_course_4k.hdr` 缺失/上传失败，以及多Physics Scene步进、PhysX读写重叠等警告/错误；没有将其过滤或当作已修复。RGB-D验证通过不能代替物理仿真验收。
- 首次冷启动RGB的地面呈蓝色网格，后续热启动三视角与默认复跑显示地毯；存在材质就绪差异，尚未系统定位或修复，不能声称冷/热启动图像等价。
- 仅验收SEL_001/kujiale_0262，不外推全部176场景；资源覆盖量未在本次刷新。
- 已接入STRIVE并取得一轮失败评分，见本文单episode结果；反复查询、不探索的停滞仍未解决，有效移动和目标验证需在同一完整episode内验收。必须标注Isaac6.0.1，不能与官方4.5环境等同。

### 4.5 清洁环境最小对照（历史记录）

2026-11-05，在105清除继承的 `PYTHONPATH`、`PYTHONHOME`、`LD_LIBRARY_PATH`、`LD_PRELOAD`，使用Isaac 4.5自带 `python.sh` 直接创建官方 `SimulationApp(headless=True, active_gpu=0, multi_gpu=False)`。**仍在 `_wait_for_viewport` 初始化发生SIGSEGV**。该过程不导入STRIVE、不加载IntentionNav场景，排除本项目适配和场景资产作为必要触发条件；当时尚待6.0.1对照，现已完成，结果见上文。

远端证据：`/mnt/nvme0/haosu/IntentionNav/logs/isaac45-clean-startup.log`。当时新版下载与自动渲染流水线仍在运行。

### 105 独立复验（2026-11-05 20:13 CST，历史快照）

**该时刻仍未修复，后续6.0.1验收见上文。** 主机为 Ubuntu 24.04.4、内核 7.0.0-38、256 逻辑 CPU；GPU0 空闲。
内核模块与实际加载的 NVIDIA 用户态库均为 595.99.02，未发现这些库混用驱动版本。
以下测试均直接运行官方空场景，不导入 STRIVE；清除继承的四项 Python/链接器环境变量，
设置 `CUDA_VISIBLE_DEVICES=0`、active_gpu=0、physics_gpu=0、multi_gpu=false。
每次只改变表中一组参数，设 90 秒超时，不修改安装、驱动或全局配置。

| 对照 | 结果 | 日志后缀 |
|---|---|---|
| `/plugins/carb.tasking.plugin/threadCount=32` | 约11秒，初始化 SIGSEGV | threads32 |
| 仅 NVIDIA ICD，禁用 Mesa 设备选择层 | 约12秒，同类崩溃 | nvidia-icd |
| `taskset` 限本进程到16个逻辑CPU | 约10秒，同类崩溃 | affinity16 |
| `/rtx-transient/scenedb/maxInstancesLimit=1048576` | 约13秒，设置实读生效仍崩溃 | instances1m |
| `/rtx/sceneDb/tlasInstanceList/useElementArray=false` | 约12秒，同类崩溃 | tlas-array-off |

独立 GDB 不再继承 Isaac 的 Python 环境，获得了没有旧 GDB Python 错误干扰的原生栈。
故障仍在 `librtx.scenedb.plugin.so`，指令 `lock add %r12d,(%rcx)`，`rcx=0x3380`，
表现为空指针加偏移后的近零地址访问。GDB 自身退出0不代表被调试程序成功；程序实际收到 SIGSEGV。
缩小 TLAS 容量也未解决，不能仅凭崩溃前的 TLAS 日志归因为显存不足。

**清洁环境的边界：** `/etc/ld.so.preload` 仍全局加载 `libeagleaudithook.so`，
GDB 确认其存在。清除环境变量不等于无系统注入；目前没有证据证明该审计库导致崩溃，
未禁用或修改它。若需要排除此因素，应由管理员安排受控兼容性验证。

六份日志保留在远端原 logs 目录，本地副本位于 `../../STRIVE-deploy-105/evidence/`，
文件名为 `isaac45-qoder-<后缀>-20261105.log`；原生栈后缀为 `native-backtrace`。

**处理顺序：** 保留现有 6.0.1 自动安装/渲染任务，不另起竞争任务。20:13 下载约52%，
状态仍为 installing/downloading，没有新版渲染结果。下一步应对照新版官方空场景及同一场景首帧，
再验 RGB-D/几何；同属 R595 不等于 595.99.02 已受官方验证。若新版仍失败，再考虑管理员协调的
官方容器对照以区分宿主用户态环境和驱动因素；容器共享宿主驱动，不能用换容器冒充驱动对照。
整机驱动变更留给维护窗口；不采用跳过驱动版本校验作为本次原生崩溃的修复。
