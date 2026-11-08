# IntentionNav 部署、公开输入与独立评分

从零部署总入口见 [README](../README.md)，模型见 [LVLM 部署](lvlm_server_deployment.md)，交付条件见 [验收口径](acceptance_4500.md)。本页维护当前启动器与评分合同；2026-11-05安装、渲染验收、Isaac4.5崩溃和单例失败全过程移至 [历史记录](archive/intentionnav_integration_20261105.md)。

## 1. 环境与交付边界

- 准备完整 IntentionNav 官方任务与场景资产、官方评测源码、独立 Isaac Sim 安装。
- 已验证渲染的版本是 Isaac `6.0.1-rc.7+release.42383.32955d8d.gl`，不是官方 Isaac4.5 的等价复现。
- 渲染使用 Isaac 自身 `python.sh`；策略感知在 `strive-hm3d:local` 容器中运行，模型服务独立部署。不要将三套 Python/CUDA 依赖混装。
- 当前启动器模式为 `strive_isaac_guided`，动作是 freemap 校验后的运动学相机位移，不是动态物理机器人，也不等价于 Habitat 默认 `baseline_guided`。
- 分配空闲渲染/感知 GPU 与端口，不覆盖运行中的服务，不通过改全机驱动修复单任务。

在 STRIVE 根目录准备以下环境，路径替换为已准备资源：

```bash
export STRIVE_ROOT="$PWD"
export INTENTIONNAV_DATA_ROOT="/path/to/IntentionNav/data"
export INTENTIONNAV_ROOT="/path/to/IntentionNav/official-source"
export ISAACSIM_ROOT="/path/to/isaac-sim-6.0.1"
export HF_HOME_HOST="/path/to/huggingface-cache"
export SAM_CHECKPOINT="/path/to/sam_vit_h_4b8939.pth"
export GROUNDING_DINO_CHECKPOINT="/path/to/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
```

预先注入 `VLN_LVLM_API_KEY`，不在文档或命令历史保存密钥。当前启动器连接同机回环服务，模型名固定为 `huaweivln-qwen3.5-4b`；`--model-port` 选择端口，不是任意远端 endpoint 参数。启动前核对模型发现与图像/schema smoke。

## 2. 先验收渲染与观测

先创建输出父目录；每次运行使用其中不存在的新子目录，启动器不会覆盖已有结果。

```bash
mkdir -p outputs/intentionnav
"$ISAACSIM_ROOT/python.sh" scripts/capture_intentionnav.py \
  --data-root "$INTENTIONNAV_DATA_ROOT" \
  --selection-id SEL_001 --gpu 0 \
  --output outputs/intentionnav/capture-new
```

输出目录使用新路径。资产齐备、进程正常结束或首帧保存都不等于导航验收；应核对 RGB、有效深度、内参、实际相机位姿及跨帧重投影。

共享观测采用米制、右手 Z-up 世界系、光学相机右/下/前。USD 位姿转为共享列向量合同：

\[
T^{map}_{optical}=T^{map}_{USD}\operatorname{diag}(1,-1,-1,1),\qquad
p_{map}=T^{map}_{optical}[zK^{-1}(u,v,1)^\top,1]^\top.
\]

深度是 `distance_to_image_plane` 的光学 Z 深度，不是径向距离；无效值转0并记录有效率，整帧无效报错。共享帧不注入目标类别、目标位置或目标照片。

历史三视角检查只是内部一致性证据，不证明绝对标定、所有场景覆盖或机器人安全。冷/热启动材质差异和 Physics Scene 警告按实际日志保留，不以过滤日志当作修复。

## 3. 两种任务输入

官方任务与风格：

```bash
python scripts/launch_intentionnav_episode.py \
  --selection-id SEL_001 --style formal \
  --sim-gpu 0 --policy-gpu 1 --port 9466 --model-port 18182 \
  --max-moves 30 --max-model-calls 60 \
  --output outputs/intentionnav/official-new
```

已冻结公开输入：

```bash
python scripts/launch_intentionnav_episode.py \
  --public-episode /path/to/public_episode.json \
  --sim-gpu 0 --policy-gpu 1 --port 9466 --model-port 18182 \
  --max-moves 30 --max-model-calls 60 \
  --output outputs/intentionnav/public-new
```

`--selection-id` 与 `--public-episode` 互斥；公开文件不能再被 `--style` 覆盖。公开输入只允许六字段：

```text
selection_id
scene_id
style
instruction
start_position
start_rotation_quat_wxyz
```

从实际官方任务导出，不写入目标真值。启动器将输入一次性保存为 `public_episode.json`，记录 SHA256，将**同一文件**交给模拟器与策略；修改外部原文件不会改变冻结输入。

六字段逐项与官方任务核对。场景、原文、风格、起点或朝向不匹配时，该运行可以作为非官方输入实验保留，但 `evaluation_binding` 为 unavailable，相关指标为 null，不能继承官方目标与最短路评分。策略容器不挂载官方目标元数据或场景真值，仅评测端读取。

### 观测模式

`launch_intentionnav_episode.py` 与直接策略入口 `run_intentionnav.py` 均默认 `--observation-mode single`。每轮只做一次当前 RGB-D observe 和感知；不先转向、不恢复朝向后重复采集，当前图像也会投影地图里历史对象的可见部分。历史地图持续积累，目标定向取证仍保留。

多视角对照需显式指定 `--observation-mode multi --panorama-views 4`。静止决策观测按4方向感知后恢复原朝向，再发布一帧当前投影，所以是 **4次感知 / 5帧图像**，并非4帧；N可调整但 multi 要求N≥2。运动跟踪刷新和定向取证仍用单视角；single 模式忽略方向数量配置，以实际记录的帧数为准。

```bash
python scripts/launch_intentionnav_episode.py \
  --selection-id SEL_001 --style formal \
  --sim-gpu 0 --policy-gpu 1 --port 9466 --model-port 18182 \
  --max-moves 30 --max-model-calls 60 \
  --observation-mode single \
  --output outputs/intentionnav/view-single

python scripts/launch_intentionnav_episode.py \
  --selection-id SEL_001 --style formal \
  --sim-gpu 0 --policy-gpu 1 --port 9466 --model-port 18182 \
  --max-moves 30 --max-model-calls 60 \
  --observation-mode multi --panorama-views 4 \
  --output outputs/intentionnav/view-multi
```

两条命令顺序运行，使用相同六字段公开输入、冻结源码/模型、freemap及预算，输出目录必须不同且未存在。模式是运行参数，不写入六字段任务文件，不改变官方评分绑定。启动和策略回执保存配置；`scene_snapshots.jsonl` / `perception.jsonl` 可核对实际图像与感知次数。比较 SR/SPL 同时列出视角/转向、移动、模型调用与耗时，不能将运动学相机结果解释为实体机器人安全或效率。

当前完成的是 CPU 协议与回归验证；2026-11-09本机缺少完整 Isaac 运行时与数据，未运行新的 IntentionNav 仿真。同期 Habitat 功能对照不能替代这里的 single/multi 导航效果验收，原历史结果不重标、不覆盖。

## 4. 运动与评分

[启动器](../scripts/launch_intentionnav_episode.py) 冻结源码与镜像 ID，保存命令及公开输入哈希；运动、STOP和退出后评分保持分离。freemap 按官方 `==1` 表达可行走，不把全部非零值当可行走。错误任务/控制版本拒绝执行，不替模型修改参数。

[评分适配](../benchmark/intentionnav.py) 使用官方 `nav_metrics`。几何 SR 为终点到官方目标的二维距离不超过2m，SPL 使用官方最短路：

\[
S=\mathbf1[\|x_T-p_{GT}\|_{xy}\le2],\qquad
SPL=S\frac{L^*}{\max(L^*,\sum_t\|x_t-x_{t-1}\|_{xy})}.
\]

实际 STOP 轨迹哈希与评分绑定；记录公开输入、轨迹、场景与 freemap 哈希。G相关指标保留官方3m配置，缺少语义/可见性证据时保持 null。几何到达不等于辨认正确实例，内部 task_complete 也不能替代官方分数。

| 产物 | 作用 |
|---|---|
| `source/`、`source_manifest.json` | 冻结源码与内容清单 |
| `public_episode.json`、`execution.json` | 公开输入、模型/镜像配置与身份绑定 |
| `simulator.log`、策略日志、`trajectory.jsonl` | 观察、动作和终止过程 |
| `evaluation.json` 及运行结果 | 独立评分、缺失原因与工程退出状态 |

## 5. 验收顺序

先完成场景/渲染 → 实际模型请求 → 同一 episode 的有效移动与目标验证 → STOP及独立评分，再运行已授权的固定回归集合。30次移动、60次模型调用是示例预算，不代表完整交付协议；同批比较必须固定预算。

历史2026-11-05单例60次查询后预算耗尽、路径0、SR/SPL均0，保留为失败；后续代码变更不能把历史结果改称成功。新版本是否改善只能由新冻结运行判断，不能把代码接线、RGB-D检查或评分程序返回当作全量导航通过。
