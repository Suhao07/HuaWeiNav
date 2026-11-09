# HM3D Docker 评测操作指南

从零安装、数据和权重准备、两层 Docker 构建统一见 [交付 README](../README.md#仿真器构建流程)。本文维护简单任务包制作、评测输入、运行命令、结果与排障，不再维护第二套 CogNav 基础镜像安装流程。模型先按 [LVLM 部署指南](lvlm_server_deployment.md) 验证。

## 1. 环境与路径

命令在 STRIVE 根目录执行。`STRIVE_DATA_ROOT` / `HM3D_DATA_ROOT` 优先于 `configs/strive_weights.yaml`；迁移时不要依赖开发机的 CogNav 绝对目录。容器挂载为：

| 宿主内容 | 容器路径 | 约束 |
|---|---|---|
| STRIVE 源码及输出 | `/workspace/STRIVE` | 正式运行应使用冻结副本 |
| 数据根目录 | `/workspace/data` | 只读 |
| SAM / GroundingDINO 权重 | `/weights` 下对应文件 | 只读 |

`--dataset_path` 和 `--episode_manifest` 由容器内 Python 读取，必须使用容器可见路径。数据放在挂载之外时，仅设置宿主绝对路径不会使容器看到它。

```bash
bash docker/preflight.sh
```

成功的 preflight 只验证依赖，不证明模型服务、指令理解或导航成功。

## 2. 单例 smoke 与 Agent 入口

配置模型变量和私有密钥后：

```bash
bash docker/run_hm3d_baseline.sh \
  --benchmark hm3d_objectnav \
  --eval_episodes 1 --start_episode 0 \
  --agent_max_model_calls 100 --max_steps 500 \
  --vlm self_hosted --save_dir hm3d_agent_smoke
```

- 最终公开入口仅使用 Agent：`run_agent_episode` 负责高层工具决策，增强 ObjNav 服务提供多视角感知、搜索/接近选项与执行能力。
- 不再接受 `--decision_owner`、`--observation-mode` 或 `--observation_mode`；删除旧命令中的这些参数，而非改成 `agent`/`multi`。启动脚本名中的 `baseline` 不表示 guided 决策；内部 guided/legacy 实现仅供历史回归，没有生产入口。
- 初始语言编译与 Agent 决策共享 `agent_max_model_calls`；感知与验证调用另列。
- `--max_steps` 是环境步预算；达到预算、模型预算耗尽、工程错误和验证完成是不同终止原因。
- `LLM_OFFLINE=1` / `STRIVE_LLM_FALLBACK=1` 不属于正式评测配置；不得用离线默认值绕过严格完成条件。

单场景调试可用：

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 --object_category tv --episode_rank 0 \
  --vlm self_hosted \
  --save_dir hm3d_scene_smoke
```

该入口筛选并物化单 episode，适合调试；正式数据来源应通过显式 benchmark / split 固定，见 [Benchmark Provider](benchmark_providers.md)。

### 观测模式与对照口径

当前交付仅保留增强多视角 ObjNav：Habitat 自适应扫描每批4方向、最多12个朝向，候选或覆盖条件满足时可提前结束；RGB-D 以对应位姿投影并融合地图、稳定实例和历史证据。它不是只增加发给 VLM 的图片张数，也不是旧自由 Agent 的固定12方向全景循环。

| 后端能力 | Agent 的决定 | 不能混淆的边界 |
|---|---|---|
| 自适应扫描与地图/实例融合 | 显式选择 `scan_scene` 补充证据 | `observe` 只读取缓存状态，不转向、拍摄或重新验证 |
| 房间与 coverage frontier、可达性/路径信息 | 选择探索目标与运动 | 排序和推荐是选项，不是 guided 自动调度 |
| approach/recheck 视点与定向取证 | 选择接近、复核或换候选 | 可接近、运动到达均不表示语义接受 |
| 作用域反馈与最终验证 | 根据失败/未知继续搜索、验证或 `propose_finish` | 反馈按任务版本、实例、谓词/对象对及观测依赖绑定，不能把局部失败扩大为全类拒绝 |

Agent 是高层决策所有者，唯一运动所有者负责低层执行、版本准入和安全条件。完整原始指令仍是最终语义权威；结构化解释和搜索假设只是执行辅助，最终完成还需当前完整原指令验证、数量/顺序进度与运动静止，不能用局部候选接受替代。临时缓存清理不等于地图清空；无有效可通行几何时保留 `observation_geometry_unavailable` 等真实失败，不伪造访问、运动或成功。

上面的命令已是多视角 Agent 入口，不需再加模式参数，见 [交付 README](../README.md#3-agent-决策与增强多视角能力)。GOAT 连续父 episode 使用 `run_agent_episode` 执行子任务，父 driver 独占 `SUBTASK_STOP` 与全局 STOP，保留连续位姿及地图记忆；语言派生单任务入口也不提供旧模式开关。无任务 `collect_hm3d_patrol.py` 仍保留全景采集，详见 [GOAT 协议边界](benchmark_providers.md#goat-连续父任务的决策与终止)。

运行身份与配置从 `run_manifest.json`、`harness/episode_identity.json`、`harness/termination.json` 核对；自适应扫描的 `episode-*/observation_cycles.jsonl` 记录视角、起止 step 与完成状态。`cycle_complete=true` 只表示一批处理结束，不等于 `complete` 所表示的全景覆盖，节点已访问也不等于360°已扫描。帧数与转向成本须结合 RGB/轨迹和模型调用产物统计。

跨版本比较固定完整任务身份（尤其起点与朝向）、模型/权重、预算、先验图与评分协议，分别冻结源码；不同决策策略不称为纯观测消融。除 SR/SPL 外，核对实际转向/动作、处理帧数、感知/验证/Agent 调用及耗时；扫描本身消耗转向步。旧结果按其冻结源码解释，不能补写成当前 Agent/multi，也不能重标历史 single 记录。本次文档更新不声明新导航验收、验证集 SR 或恢复旧成功率。

#### 2026-11-09本机功能验证

以下保留旧 single/multi 版本的历史实验描述、失败与回归统计，不是当前 Agent 多视角版本的验收，也不提供旧模式的生产启动方法。

使用 RTX 3090、`strive-hm3d:local`、本机 Qwen2.5-VL-3B（BF16、32K上下文、vLLM 0.17.1），图像不外发。测试服务只将 `guided_json` 转为 vLLM 新版 `structured_outputs.json`，保留原 JSON schema，关闭离线与解析回退；该服务不是生产 Qwen3.5 服务。源码、权重、镜像标识及失败尝试保存在本机忽略目录 `logs/view_mode_smoke_20261109/`，以 `source_freeze_bounded80.json`、`execution_bounded80.json`、`acceptance_summary.json` 为80步配对依据，不提交运行图片或模型请求。

场景为 `wcojb4TFT35`，类别 `tv`，rank 0，两组起点、朝向及源码哈希一致。单场景入口的 `auto` 实际选中了本机 `hm3d_ovon/val_seen_complex_balanced_2k` 分片，来源 episode 为 `instr_0418_begin_your_journey_by_entering_the_entra_base_6951`，运行指令按类别适配成 `Find tv`；这不是官方 ObjectNav 全量评测，也不是原复杂语言指令评测。复现实验须显式绑定相同来源，不能只靠 `auto`。两组预算均为80环境步、20次高层模型调用。

| 观测模式 | 终止 | 环境步 / 行程 | 建图采集帧 / 额外采集动作 | 全部模型调用 | 基准 success / SPL |
|---|---|---|---|---|---|
| single | `episode_limit`，未接受目标 | 80 / 11.25m | 6轮各1帧 / 0步 | 8 | 0 / 0 |
| multi | 内部 `verified_finish`，已发STOP | 40 / 5.00m | 1轮8帧 / 7步 | 6 | 0 / 0 |

single 到达了5个 frontier，6轮采集均满足 `start_step == end_step`，地图继续增长，没有用额外转向补齐输入。multi 最终距基准目标10.84m，画面为镜面/椅子区域，内部模型接受不能算成功；single 距目标0.85m也因没有合法STOP而仍为失败。这里的采集帧数不含运动和定向复核帧，实际 RGB 轨迹分别为80/40帧；本入口没有完整逐动作类型账本，不从这些数字推算全部转向次数。进程耗时分别约143/73秒，包含加载和产物保存，仅供本机排障，不作效率排名。

此前120步同源码对照中，single 在周期复议时收到修改不可变 ObjectNav 目标的模型提案，被 `Only language interpretations may be revised` 拒绝；multi 在52步内部接受并STOP，但基准 success/SPL 仍为0。更早的无约束解释超长、8K上下文不足和主动停止的预跑也保留，未混入80步结果。80步补测只覆盖有界观测/运动链路，不消除120步暴露的模型协议问题或视觉误接受，后续正式评测仍需解决并复验。

667项定向回归全部覆盖：主环境658通过、9项Open3D用例跳过；9项均由现有真实CPU Open3D环境补测通过。回归还覆盖锚点复核后发现新目标、临时缓存清空后的连续任务，以及有效深度与饱和深度混合时保留原始投影证据。Isaac与实机效果不在本次仿真验收范围内。

## 3. 自然语言与数据集目标

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 --object_category tv \
  --enable_instruction_adapter \
  --custom_instruction "find the television in the scene" \
  --vlm self_hosted \
  --save_dir hm3d_instruction_smoke
```

不开 adapter 的 ObjectNav 按数据集目标确定性编译。公开 Agent 自然语言入口使用有界初始解释，保存完整原文和编译诊断，不在失败后悄悄改成数据集类别；原文始终保留最终语义权威。`rules` 是历史数据集兼容路径，不是通用自然语言规则解析器。任务义务、动态检索词和当前验证的职责见 [指令合同](instruction_adapter.md)。

## 4. 固定清单与批量重跑

标准 HM3D val 的规模不足以凭 `--eval_episodes 2000` 自动产生2,000个独立任务。需先准备具有足够独立身份的实际 split / 任务包；不要通过重复采样补数。

### 4.1 制作简单2000候选包

现有制作入口是仓库同级部署工具 [run_hm3d200.py](../../STRIVE-deploy-105/run_hm3d200.py)。这是105专用脚本：源码、数据及批次根目录由脚本的 `SOURCE`、`DATA`、`BATCH` 常量固定，不读取本页运行入口的环境变量。先确认部署机具备 `DATA/objectgoal_hm3d/train/content/*.json.gz`、对应 HM3D 场景文件及 `SOURCE` 源码；它不负责下载原始数据。将脚本部署到该服务器后，在脚本所在目录执行：

```bash
python run_hm3d200.py prepare \
  --count 2000 --split train \
  --batch-name hm3d-train2000-rebuild
```

批次名必须是不存在的新目录名，不能覆盖旧批次；不要以 `python -O` 关闭脚本断言。`prepare` 只制作数据、记录分片哈希并复制四份源码快照，不启动导航、GPU任务或模型请求。不要把旧脚本的 `run` 子命令及内置 GPU/端口分配当作当前运行授权。

制作规则：

1. 读取 train 原始分片，校验六类别映射一致、对应场景文件存在，保留目标定义。
2. 以场景、起点、起始朝向、目标类别去重，**不靠不同 episode_id 补足任务数**。
3. 固定 `seed=20261105`，按来源身份的 SHA256 排序；六类别配额为334、334、333、333、333、333，同类别按固定场景顺序轮转选取。不足2,000个独立任务时报错。
4. 保留原始 `source_episodes`、分片路径/索引与采样哈希；另存按 Habitat 加载顺序重编号的 `runtime_episodes`，固定执行身份及 worker 分配。

产物位于源码目录同级的 `STRIVE-batches/<batch-name>/`：`dataset/tasks.json.gz`、`dataset/manifest.json`、`source_hashes.json` 和 `sources/w0` 至 `sources/w3`。manifest 记录来源分片哈希、任务包 SHA256、去重/抽样规则、类别/场景分布及原始/运行身份；源码快照哈希不能替代任务包哈希。

已有候选包为 [hm3d-train2000-20261105/tasks.json.gz](../../STRIVE-deploy-105/evidence/hm3d-train2000-20261105/tasks.json.gz)，配套 [manifest.json](../../STRIVE-deploy-105/evidence/hm3d-train2000-20261105/manifest.json)。它包含2,000个独立任务、145个场景：bed/chair各334，plant/sofa/toilet/tv_monitor各333；任务包 SHA256 为 `a081efbfadea563dcbcb4672286880127247d6ba18c440d57d07884920dc93d2`。

2026-11-08制作核验已确认该包哈希、2,000个不依赖 episode_id 的唯一身份，以及逐项 source/runtime 绑定；本次没有重算全部原始分片哈希或重新验收全部场景。**这是 train 类别目标任务包，不是2,000句独立自然语言，也不是 val 成绩。** 当前 [交付配置](../configs/benchmarks/acceptance_4500.json) 的简单组仍为 `pending_unique_HM3D_OVON_selection`：候选包存在不等于正式验收身份选择已确认，本文不擅自将其提升为最终交付集。

### 4.2 使用固定身份运行

已准备好任务包与身份清单、确认交付选择并部署到容器可见位置后，示例：

```bash
bash docker/run_hm3d_baseline.sh \
  --benchmark hm3d_objectnav \
  --dataset_path /workspace/data/datasets/hm3d_simple_2000/tasks.json.gz \
  --episode_manifest /workspace/data/datasets/hm3d_simple_2000/manifest.json \
  --dataset_episodes 2000 --eval_episodes 2000 --start_episode 0 \
  --agent_max_model_calls 100 \
  --max_steps 500 --vlm self_hosted --save_dir hm3d_simple_2000_new
```

上述路径是部署示例，不会自动生成或下载数据。清单是包含 `runtime_episodes` 数组的 JSON，每项使用 [episode_identity](../benchmark/episode_selection.py) 的完整身份；不要将需求筛选的 JSONL 清单直接传给这个参数。reset 后入口保存并核对场景、原始 episode、起点、朝向和目标类别。

**区间语义：**入口循环是 `range(start_episode, eval_episodes)`，后者是结束索引而非“从起点再跑多少例”。例如从100跑到199应设置 `--start_episode 100 --eval_episodes 200`，并固定原始 `--dataset_episodes` 与同一清单。

正式比较要求：

1. 冻结完整源码（含需要交付的未提交文件时，另存清单）、镜像 ID、数据/权重哈希、模型版本、参数与种子。
2. 新旧运行目录分离；完整重跑2,000条不等于只补跑旧批次缺失条目。
3. 优先安排旧批次已有结果的身份以尽早配对；其余任务仍保留在总清单，不重复、不遗漏。
4. 并发 worker 领取不相交的身份集合，保存每例命令、退出回执、完整身份、轨迹和评分。
5. “文件已生成”“进程已结束”“评分可用”分别核验。先报告完成数/异常数/待完成数，再计算具有有效评分的子集指标并明确分母；最终交付总指标不得通过剔除工程失败美化。
6. 旧代码若跨多个版本，按版本分层；配对只比较身份相同且双方评分可用的样本。并发耗时不直接作为延迟提升证据。

当前交付的分组和困难子集定义见 [验收口径](acceptance_4500.md)，不将实时进度数字写进部署说明。

## 5. 运行产物与评分

输出默认在 `logs/<save_dir>/`：

| 产物 | 用途 |
|---|---|
| `run_manifest.json`、`benchmark_spec.json` | 运行参数、数据来源与成功阈值 |
| `metrics.csv` | Habitat 指标、内部验证与模型调用统计 |
| `episode-*/episode_identity.json` | reset 后实际任务身份 |
| `episode-*/initial_interpretation.jsonl` | 初始语言解释及有界修正（发生时） |
| `episode-*/instruction_adapter/` | 任务合同、兼容投影与进度诊断 |
| `episode-*/harness/` | 决策、工具、终止回执 |
| `episode-*/lvlm_calls/` | 实际模型请求和响应追溯 |
| `episode-*/final_verifier/`、`evidence_blobs/` | 验证结果与归档图像 |
| `episode-*/rgb/`、视频、`frontier_map/` | 观测和路径诊断 |

`success` / `spl` 是外部 evaluator 结果；`instruction_success`、`final_stop_success` 是内部任务/验证结果，不能相互替代。`accepted_distance_to_target` 是接受实例的证据距离，不等同于 GT `distance_to_goal`。`lvlm_call_count_by_type` 记录调用与缓存命中；`lvml_call_count_by_type` 是历史拼写别名。

GOAT 语言独立评分另见 [运动证据与评分](benchmark_providers.md#运动证据与独立评分)，普通 ObjectNav 不自动生成那套证据。模型请求、图像及响应可能含敏感数据，应限制输出目录访问，不提交 Git 或复制进文档。

每次使用新 `save_dir`。`--clean_save_dir` 会删除旧目录内容，不用于正式批次的续跑或配对复核。

## 6. 常见问题

- **Docker socket / GPU 权限**：由部署者确认 Docker 与 NVIDIA runtime 权限，不改业务代码绕过。
- **缺权重**：检查 YAML 与环境覆盖；确需下载时显式使用 `STRIVE_DOWNLOAD_WEIGHTS=1`，不要在每个 worker 中重复下载。
- **模型401 / 错误模型名**：检查变量是否存在，以及模型名与 endpoint 是否匹配，不打印密钥。

```bash
python - <<'PY'
import os
names = ("ARK_API_KEY", "LLM_MODEL", "LLM_API_BASE_URL",
         "VLN_LVLM_API_KEY", "VLN_LVLM_MODEL", "VLN_LVLM_BASE_URL")
print({name: bool(os.getenv(name)) for name in names})
PY
```

- **连接正常但结构化失败**：分别排查传输/服务端错误和实际非法响应，保留原始原因。不能增加静默字段过滤或自动接受来让批次“跑完”。
- **导航进程异常**：保留退出状态和最后一次动作/请求，不把不完整记录伪装为正常预算终止。

进入同一镜像调试：

```bash
bash docker/run_hm3d_baseline.sh bash
```
