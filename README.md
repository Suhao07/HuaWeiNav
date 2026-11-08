# VLN

VLN 是面向 HM3D / Habitat、IntentionNav / Isaac 与实物机器人的语义导航框架。

本文是交付的**从零部署统一入口**；数据、权重、镜像的原有部署步骤继续保留，新增模型服务、指令合同和评测流程在下文接入。命令默认在 STRIVE 仓库根目录执行，路径与 GPU 分配由部署者设置。

- Habitat 默认使用 `baseline_guided`；`legacy` 和自由 `agent` 是显式对照模式。
- 自然语言编译为不可变 `TaskSpec`；候选检索、任务验证、运动回执和外部评分各自保留职责。
- IntentionNav 使用独立 Isaac 入口，不等同于 Habitat 默认模式或官方 Isaac 4.5 复现。
- 实物模式已实现，默认 bridge-off；调试参数见文末速查，实际行驶与到达另行验收。


## 目录

- [文档目录与产物边界](docs/README.md)
- [LVLM 接入与部署基础](docs/lvlm_server_deployment.md)
- [HM3D 语义先验地图与对比评测](docs/hm3d_prior_map_evaluation.md)
- [推荐环境](#推荐环境)
- [仿真器构建流程](#仿真器构建流程)
- [运行 HM3D 仿真评测](#运行-hm3d-仿真评测)
- [新增能力与交付验收](#新增能力与交付验收)
- [本地源码安装备选方案](#本地源码安装备选方案)
- [环境变量](#环境变量)
- [输出结果](#输出结果)
- [实物机器人入口](#实物机器人入口)

## 推荐环境

运行仿真或实物导航前，先完成 LVLM 接入。项目支持直接调用商业 API，也支持将模型
部署在独立 GPU 服务器后由机器人通过 OpenAI-compatible HTTP API 访问；配置、验收和
公网 HTTPS 反向代理要求见 [LVLM 接入与部署基础](docs/lvlm_server_deployment.md)。

推荐使用 Docker 跑仿真，原因是 Habitat-Sim、Habitat-Lab、MMDetection、CUDA、PyTorch 和 HM3D 数据路径之间的版本耦合较强。

默认项目路径：

```bash
cd "/home/ubuntu/WorkSpace/project/Huawei Nav/Code/STRIVE"
```

本仓库通过 `configs/strive_weights.yaml` 记录本机数据和权重路径；启动脚本会优先
使用 `STRIVE_DATA_ROOT`/`HM3D_DATA_ROOT` 环境变量，再回退到 YAML 中的 `data_root`。
当前工作区默认复用 CogNav_ObjNav 的数据目录。迁移到其他机器时修改 YAML，或显式
覆盖数据根目录：

```bash
export STRIVE_ROOT="/home/ubuntu/WorkSpace/project/Huawei Nav/Code/STRIVE"
export STRIVE_MODELS_DIR="$STRIVE_ROOT/models"

export STRIVE_DATA_ROOT="/path/to/CogNav_ObjNav/data"  # 可选，优先级高于 YAML
mkdir -p "$STRIVE_MODELS_DIR"
```

当前本机配置为：

```text
configs/strive_weights.yaml:data_root
-> /home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav/data
```

需要提前准备：

- NVIDIA GPU 和 Docker GPU runtime。
- 一个已经可导入 Habitat-Sim / Habitat-Lab 的基础镜像，例如 `habitat-hm3d:local`。
- HM3D 场景数据：`$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2`，其中
  `$STRIVE_DATA_ROOT` 为环境变量或 YAML 中的 `data_root`。
- HM3D ObjectNav 回合数据：对应数据根目录下的
  `objectnav_hm3d_v2/val/val.json.gz` 或 `objectgoal_hm3d/val/val.json.gz`。
- SAM 权重：`$STRIVE_MODELS_DIR/sam_vit_h_4b8939.pth`。
- GroundingDINO Swin-L 权重：`$STRIVE_MODELS_DIR/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth`。

## 仿真器构建流程

### 1. 准备 HM3D 场景数据

HM3D v0.2 场景数据使用 Matterport 官方公开 tar 链接下载，不需要 MP3D 的账号密码流程。官方入口：

- `https://github.com/matterport/habitat-matterport-3dresearch`

VLN 需要每个 split 的四类文件：

```text
hm3d-<split>-glb-v0.2.tar
hm3d-<split>-habitat-v0.2.tar
hm3d-<split>-semantic-annots-v0.2.tar
hm3d-<split>-semantic-configs-v0.2.tar
```

从 0 部署时直接运行仓库脚本。默认下载 `val` 和 `minival`，tar 包保存在
`$STRIVE_DATA_ROOT/downloads/hm3d_v0.2`，解压后的标准目录是
`$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2`：

```bash
chmod +x scripts/download_hm3d_v02.sh
HM3D_SPLITS="val minival" bash scripts/download_hm3d_v02.sh
```

如果要补齐 train split：

```bash
HM3D_SPLITS="train" bash scripts/download_hm3d_v02.sh
```

也可以手动下载。以 `val` split 为例：

```bash
mkdir -p "$STRIVE_DATA_ROOT/downloads/hm3d_v0.2"
cd "$STRIVE_DATA_ROOT/downloads/hm3d_v0.2"

curl -L -O https://api.matterport.com/resources/habitat/hm3d-val-glb-v0.2.tar
curl -L -O https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar
curl -L -O https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar
curl -L -O https://api.matterport.com/resources/habitat/hm3d-val-semantic-configs-v0.2.tar
```

完成后应看到下面的目录结构：

```text
$STRIVE_DATA_ROOT/
  scene_datasets/
    hm3d_v0.2/
      hm3d_annotated_basis.scene_dataset_config.json
      val/
        00802-wcojb4TFT35/
          wcojb4TFT35.basis.glb
          wcojb4TFT35.semantic.glb
          wcojb4TFT35.basis.navmesh
      train/
      minival/
```

如果实验室内网已经提供了下载好的 tar 包，也可以直接交给脚本处理；要求最终仍然整理成同样的 `$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2`：

```bash
mkdir -p "$STRIVE_DATA_ROOT/downloads/hm3d_v0.2"
cp /path/to/hm3d-*-v0.2.tar "$STRIVE_DATA_ROOT/downloads/hm3d_v0.2/"
HM3D_SPLITS="" bash scripts/download_hm3d_v02.sh
```

校验场景数据：

```bash
test -f "$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
find "$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2/val" -maxdepth 2 -name "*.basis.glb" | head
find "$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2/val" -maxdepth 2 -name "*.basis.navmesh" | head
find "$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2/val" -maxdepth 2 -name "*.semantic.glb" | head
```

### 2. 准备 HM3D ObjectNav 回合数据

ObjectNav 回合数据只保存 episode、scene id、起点和目标类别，不保存 VLN 运行时的在线地图。推荐目录：

```text
$STRIVE_DATA_ROOT/
  objectnav_hm3d_v2/
    val/
      val.json.gz
      content/
        wcojb4TFT35.json.gz
        ...
```

如果你的数据版本使用旧命名，也可以放在：

```text
$STRIVE_DATA_ROOT/
  objectgoal_hm3d/
    val/
      val.json.gz
      content/
```

整理示例：

```bash
mkdir -p "$STRIVE_DATA_ROOT/objectnav_hm3d_v2/val"
cp /path/to/objectnav_hm3d_val/val.json.gz "$STRIVE_DATA_ROOT/objectnav_hm3d_v2/val/val.json.gz"
cp -r /path/to/objectnav_hm3d_val/content "$STRIVE_DATA_ROOT/objectnav_hm3d_v2/val/content"
```

校验回合数据：

```bash
test -f "$STRIVE_DATA_ROOT/objectnav_hm3d_v2/val/val.json.gz" || \
test -f "$STRIVE_DATA_ROOT/objectgoal_hm3d/val/val.json.gz"
```

### 3. 准备视觉模型权重

权重默认放在 `$STRIVE_MODELS_DIR`：

```text
$STRIVE_MODELS_DIR/
  sam_vit_h_4b8939.pth
  grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

如果本机还没有权重，可以手动下载到 `$STRIVE_MODELS_DIR`：

```bash
wget -O "$STRIVE_MODELS_DIR/sam_vit_h_4b8939.pth" \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

wget -O "$STRIVE_MODELS_DIR/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth" \
  https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

如果权重放在其它位置，也可以显式设置路径：

```bash
export SAM_CHECKPOINT="/path/to/sam_vit_h_4b8939.pth"
export GROUNDING_DINO_CHECKPOINT="/path/to/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
```

镜像和数据也准备好之后，还可以让运行脚本自动补齐缺失权重：

```bash
STRIVE_DOWNLOAD_WEIGHTS=1 bash docker/preflight.sh
```

### 4. 构建 HM3D 仿真 Docker 镜像

HM3D 仿真链路是 Habitat / Python / CUDA 环境，不是 ROS2 运行环境。它分成两层镜像：

```text
habitat-hm3d:local   基础镜像：CUDA、Conda、PyTorch、Habitat-Sim、Habitat-Lab
strive-hm3d:local    VLN overlay：MMDetection、SAM、pathfinding 和本项目运行依赖
```

先从 0 构建 Habitat 基础镜像：

```bash
chmod +x docker/build_habitat_base.sh
IMAGE_TAG=habitat-hm3d:local \
bash docker/build_habitat_base.sh
```

默认基础镜像配置：

```text
CUDA base image  = nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
Conda env        = strive
Python           = 3.9
PyTorch          = 2.0.1 + CUDA 11.8
Habitat-Sim      = 0.3.2
Habitat-Lab ref  = v0.3.2
```

如果需要固定 Habitat-Lab 的源码来源或分支，可以显式覆盖：

```bash
HABITAT_LAB_REPO=https://github.com/facebookresearch/habitat-lab.git \
HABITAT_LAB_REF=v0.3.2 \
IMAGE_TAG=habitat-hm3d:local \
bash docker/build_habitat_base.sh
```

基础镜像构建完成后，校验 Habitat 组件：

```bash
docker run --rm --gpus all habitat-hm3d:local python - <<'PY'
import habitat
import habitat_sim
import torch
print("habitat:", getattr(habitat, "__version__", "unknown"))
print("habitat_sim:", getattr(habitat_sim, "__version__", "unknown"))
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY
```

然后构建 VLN overlay 镜像，默认输出镜像名为 `strive-hm3d:local`：

```bash
BASE_IMAGE=habitat-hm3d:local \
bash docker/build.sh
```

VLN overlay 镜像会在基础 Habitat 环境上补齐：

- `mmengine`
- `mmcv`
- `mmdet`
- `segment-anything`
- `pathfinding`
- `bresenham`
- `supervision`

构建完成后检查镜像：

```bash
docker images strive-hm3d:local
```

### 5. 构建实物 ROS2 Humble Docker 镜像

实物机器人链路使用 ROS2 Humble，入口是 `docker/Dockerfile.real_robot` 和 `docker/build_real_robot.sh`。这条链路与 HM3D Habitat 仿真镜像分离，不使用 ROS1 Noetic。

默认输出镜像：

```text
huawei-nav-real:orin
```

构建命令：

```bash
IMAGE_TAG=huawei-nav-real:orin \
INSTALL_ML_DEPS=0 \
INSTALL_LLM_DEPS=1 \
bash docker/build_real_robot.sh
```

如果需要把 detector/mapping 的重依赖也装进镜像，可以显式打开：

```bash
IMAGE_TAG=huawei-nav-real:orin \
INSTALL_ML_DEPS=1 \
INSTALL_LLM_DEPS=1 \
bash docker/build_real_robot.sh
```

### 6. 检查权重和数据挂载

运行 preflight；如果本地缺少 SAM / GroundingDINO 权重，可加 `STRIVE_DOWNLOAD_WEIGHTS=1` 自动下载：

```bash
STRIVE_DOWNLOAD_WEIGHTS=1 bash docker/preflight.sh
```

检查内容包括：

- HM3D 场景和 ObjectNav 回合数据是否存在。
- SAM / GroundingDINO 权重是否存在。
- Habitat / Habitat-Sim 是否可导入。
- PyTorch CUDA 是否可用。
- MMDetection `DetInferencer` 是否可初始化。
- SAM builder 是否可用。
- HM3D config 和相机内参是否能生成。

成功时会输出：

```text
preflight OK
```

### 7. 进入仿真容器调试

```bash
bash docker/run_hm3d_baseline.sh bash
```

进入容器后：

```bash
cd /workspace/STRIVE
python docker/preflight.py
```

容器内关键路径：

```text
/workspace/STRIVE          当前仓库，可写
/workspace/data            HM3D 场景和 ObjectNav 回合数据，只读挂载
/weights                   SAM 和 GroundingDINO 权重
```

## 运行 HM3D 仿真评测

### 1. 配置模型客户端

先按 [LVLM 接入与部署基础](docs/lvlm_server_deployment.md) 验证实际图像与结构化请求。商业 API 可沿用下方 Ark 示例；已部署的 Qwen3.5 服务使用：

```bash
export LLM_PROVIDER=self_hosted
export STRIVE_LLM_CLIENT=self_hosted
export VLN_LVLM_BASE_URL="http://127.0.0.1:18182/v1"
export VLN_LVLM_MODEL="huaweivln-qwen3.5-4b"
export VLN_LVLM_GUIDED_JSON=1
export VLN_LVLM_TIMEOUT_S=180
export VLN_LVLM_TRANSPORT_RETRIES=0
export VLN_LVLM_PARSE_RETRIES=1
export LLM_OFFLINE=0
export STRIVE_LLM_FALLBACK=0
```

`VLN_LVLM_API_KEY` 通过私有环境注入，不写入命令历史或仓库。上述回环地址仅适用于模型与使用 host 网络的容器在同一台机器；跨机使用经过认证的 HTTPS endpoint。`GUIDED_JSON=1` 只表示客户端发送 JSON schema，请求是否被约束解码取决于服务端，不能据此宣称启用了 XGrammar。

### 2. 运行一例真实模型 smoke

```bash
bash docker/run_hm3d_baseline.sh \
  --benchmark hm3d_objectnav \
  --eval_episodes 1 \
  --start_episode 0 \
  --decision_owner baseline_guided \
  --agent_max_model_calls 100 \
  --max_steps 500 \
  --save_dir hm3d_self_hosted_smoke \
  --vlm self_hosted
```

初始指令解释与 Agent 调用共用 `agent_max_model_calls`；感知、关系及最终验证调用仍分别统计，不把它当成全部模型调用总上限。ObjectNav 类别输入可以确定性编译，不需要额外语言解析。

离线检查先运行 preflight 和回归测试。`LLM_OFFLINE` / `STRIVE_LLM_FALLBACK` 是显式兼容选项，不保证当前严格语义链路能完成导航；缺少验证证据不能自动接受目标，离线 smoke 也不能替代真实模型或导航验收。

### 指定场景和目标物

推荐使用单场景入口：

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --save_dir hm3d_wcojb4TFT35_tv \
  --vlm ark
```

选择同一场景、同一目标物下的第 N 个回合：

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --episode_rank 1 \
  --save_dir hm3d_wcojb4TFT35_tv_rank1 \
  --vlm ark
```

快速调试时可限制步数：

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --save_dir hm3d_debug_max_steps \
  --vlm ark \
  --max_steps 20
```

### 启用自然语言指令适配器

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id wcojb4TFT35 \
  --object_category tv \
  --save_dir hm3d_instruction_adapter \
  --vlm ark \
  --enable_instruction_adapter \
  --custom_instruction "find the television in the scene"
```

普通 ObjectNav 不需要开启 adapter；`--instruction_adapter_backend rules` 是旧的数据集目标兼容路径，不是任意自然语言的规则解析器，也不代表运行时不调用模型。当前非 legacy 指令入口使用有界初始编译，解析失败不得悄悄改成数据集目标。配置 self-hosted 后，将上述命令的 `--vlm ark` 改成 `--vlm self_hosted`。

批量数据选择、身份清单、重跑配对与日志诊断统一见 [HM3D 评测操作指南](docs/hm3d_docker_benchmark_guide.md)。每次使用新输出目录，不通过清空历史结果来混合新旧版本。

HM3D 语义先验地图的生成、批量构建、运行接入和有/无先验对比评测见
[HM3D 语义先验地图与对比评测](docs/hm3d_prior_map_evaluation.md)。

## 新增能力与交付验收

### 指令、检索与完成合同

- 原始任务义务保存在 `TaskSpec`，活动/材料/资源需求不会被搜索设施假设替代；检索词更新与任务语义修订分别计版本。
- 动态检索词进入现有 DINO/SAM 感知链路；模型允许接近某候选，不等于该实例已被验证。
- 关系判断分别检查目标身份、参照身份和有方向的关系；功能条件需候选绑定的图像证据与观测 ID，不能靠常识猜测完成。
- 当前图像、稳定实例、几何依赖与任务/控制版本进入准入；任务进度只经 reducer 更新，运动到达、任务接受和外部评分分别记录。

具体接口见 [指令与证据合同](docs/instruction_adapter.md) 和 [Harness 运行合同](docs/harness_refactor_design.md)，历史实现和小样本诊断移入 [文档归档](docs/archive/README.md)。

### 评测入口与范围

| 用途 | 当前入口 | 结果边界 |
|---|---|---|
| HM3D 简单指令 | [HM3D 操作指南](docs/hm3d_docker_benchmark_guide.md) | 同一冻结清单重跑，报告 evaluator SR/SPL 和工程状态 |
| 需求指令 / Isaac | [IntentionNav 部署与评分](docs/intentionnav_integration.md) | 公开输入与官方任务精确匹配后才给官方评分 |
| 交付条件与困难子集 | [4,000 条验收口径](docs/acceptance_4500.md) | 简单2,000 + 需求2,000；困难500是需求组子集 |
| GOAT / LangMap 研究 | [Benchmark 协议](docs/benchmark_providers.md)、[LangMap 接入](docs/langmap_integration.md) | 保留研究入口，不计入当前交付条件 |
| 实物机器人 | [正式一键入口](docs/real_robot_navigation_entry.md) | 已实现；bridge-off 规划/速度输出已验证，实车行驶独立验收 |

GOAT 语言派生任务新增基于冻结运行 NavMesh 与实际运动轨迹的独立评分；IntentionNav 新增公开输入快照及官方身份绑定。代码接入、数据冻结、纯 Python 回归通过都不等于批量导航通过。

### 交付验证顺序

1. 固定源码、镜像 ID、数据/权重版本与模型服务配置，不仅记录镜像标签或 Git HEAD。
2. 完成容器 preflight、模型图像/结构化 smoke；确认错误和超时不会变成目标接受。
3. 运行一例完整 episode，核对身份、当前图像、动作/STOP 回执和独立 evaluator。
4. 通过固定回归集后，再按授权运行批量清单。分母保留失败与异常，未知分数不伪装成0或排除。
5. 新旧版本只对同一身份的已完成样本配对；没有完整结果时不发布全量 SR/SPL。

CPU 回归可在安装好测试依赖的开发环境执行，例如：

```bash
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -B -m pytest -q -p no:cacheprovider --capture=sys \
  tests/test_episode_selection.py tests/test_lvlm_smoke_client.py
```

这些测试不启动真实模型或仿真器；导航与实物验收仍需对应运行证据。

## 本地源码安装备选方案

如果不使用 Docker，可以本地安装。但这条路径更容易受 CUDA、Habitat、MMDetection 版本影响，推荐只用于开发调试。下列 Python 3.12 开发环境不是上文 Python 3.9 Docker 镜像的等价替代；正式复现实验应使用冻结镜像 ID，并记录实际 Habitat/Lab/Sim 版本。

### 1. 创建 Conda 环境

```bash
conda create -n strive python=3.12 -y
conda activate strive
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 Habitat-Sim 和 Habitat-Lab

本项目在上游 Habitat 上有少量补丁和兼容修正，建议使用以下 fork，并切到 `v0.3.2` 分支：

- `https://github.com/zwandering/habitat-sim`
- `https://github.com/zwandering/habitat-lab`

安装完成后设置：

```bash
export HABITAT_LAB_PATH="/path/to/habitat-lab"
```

### 4. 安装 Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 5. 安装 MMDetection / GroundingDINO

```bash
mim install mmengine
mim install mmcv==2.1.0
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

下载 GroundingDINO Swin-L 权重：

```bash
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

## 环境变量

不要把真实 key 写进仓库文件。建议只在 shell 中导出，或写入本机私有配置。

```bash
export HABITAT_LAB_PATH="/path/to/habitat-lab"
export SAM_CHECKPOINT="/path/to/sam_vit_h_4b8939.pth"
export GROUNDING_DINO_PATH="/path/to/mmdetection"
export GROUNDING_DINO_CHECKPOINT="/path/to/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth"
export HM3D_DATA_PATH="/path/to/HM3D_v2"
export MP3D_DATA_PATH="/path/to/MP3D"
```

真实 LLM 测试示例：

```bash
export LLM_PROVIDER=ark
export ARK_API_KEY="<你的-ark-api-key>"
export LLM_MODEL=doubao-seed-2-0-lite-260428
export LLM_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
```


检查关键变量：

```bash
python - <<'PY'
import os
keys = [
    "HABITAT_LAB_PATH",
    "SAM_CHECKPOINT",
    "GROUNDING_DINO_PATH",
    "GROUNDING_DINO_CHECKPOINT",
    "HM3D_DATA_PATH",
    "MP3D_DATA_PATH",
]
print({k: bool(os.getenv(k)) for k in keys})
PY
```

## 输出结果

仿真输出默认保存在：

```text
logs/<save_dir>/
```

常用文件：

```text
metrics.csv
run_manifest.json
episode-*/rgb/
episode-*/frontier_map/
episode-*/episode_identity.json
episode-*/instruction_adapter/
episode-*/initial_interpretation.jsonl   # 发生初始语言编译时
episode-*/harness/
episode-*/lvlm_calls/
episode-*/final_verifier/
episode-*/evidence_blobs/
episode-*/prior_map/
episode-*/evaluation/                   # GOAT 语言独立评分证据
```

先验地图运行产物、动态 BEV、room/frontier 语义选择和 A/B 对比指标见独立文档。

## 实物机器人入口

**实物模式：已实现。** 正式一键入口已接入真实感知、底盘位姿、目标接近点、Action、原生规划/跟踪和安全速度输出；2026-11-05 已完成 bridge-off 验证。默认 `preview`、底盘 bridge 关闭；真实行驶、到达与最终视觉 STOP 另行现场验收，不改变软件功能已实现的标记。

### 实物调试参数速查

下表是当前源码默认值，不是重新读取的车端配置；距离单位为米。完整语义、迁移方法和故障排查见 [参数手册](docs/real_robot_parameter_tuning.md)。

| 调试项 | 环境变量 | 默认值与注意事项 |
|---|---|---|
| 跟踪停止距离 | `NAV_FOLLOWER_STOP_DISTANCE_M` | **0.5**，相对当前路径跟踪点，不是直接到物体的距离 |
| Action 到达容差 | `NAV_GOAL_TOLERANCE_M` | **1.0**，相对 standoff 接近点，不是物体中心/表面 |
| 前视 / 末端减速距离 | `NAV_FOLLOWER_LOOKAHEAD_M` / `NAV_FOLLOWER_SLOWDOWN_DISTANCE_M` | **1.0 / 1.0** |
| 对象接近距离 | `NAV_STANDOFF_M` | **0.8**，到对象中心；接近点已在到达容差内时不派发 Action |
| 车体长 / 宽 | `NAV_VEHICLE_LENGTH_M` / `NAV_VEHICLE_WIDTH_M` | `0.6 / 0.6`，按实际外廓核定，同时检查路径库碰撞假设 |
| 雷达到底盘外参 | `NAV_BASE_X_M` / `NAV_BASE_Y_M` / `NAV_BASE_YAW_RAD` | `-0.2 / 0 / -1.5708`，最后一项为弧度，换车必须标定 |
| 线 / 角速度上限 | `NAV_MAX_SPEED_MPS` / `NAV_MAX_ANGULAR_RPS` | `0.15 m/s / 0.35 rad/s`；follower 的 `NAV_FOLLOWER_YAW_DEG_S=20.0` 使用度/秒 |
| 线 / 角加速度上限 | `NAV_MAX_ACCEL_MPS2` / `NAV_MAX_ANGULAR_ACCEL_RPS2` | `0.15 m/s² / 0.3 rad/s²` |
| 指令 / 无进展超时 | `NAV_COMMAND_TIMEOUT_S` / `NAV_NO_PROGRESS_TIMEOUT_S` | `0.25 / 12.0` 秒；不能通过放宽保护掩盖停更或无运动授权 |

四项停止/到达/前视/减速距离必须是有限正数，且 `stop < goal_tolerance`、`stop < lookahead`。先在仓库根目录检查配置（**不启动 Docker、ROS 或 bridge**）：

```bash
STRIVE_EXECUTION_MODE=preview \
NAV_FOLLOWER_STOP_DISTANCE_M=0.5 NAV_GOAL_TOLERANCE_M=1.0 \
NAV_FOLLOWER_LOOKAHEAD_M=1.0 NAV_FOLLOWER_SLOWDOWN_DISTANCE_M=1.0 \
bash scripts/run_real_robot_navigation.sh --check
```

`--check` 输出四项距离与模式，不校验整车几何、在线 ROS 参数或硬件状态。新增距离参数已完成本地接线及离线验证；此前车端镜像尚未包含修复，必须更新镜像、重建 ROS 安装目录并核对实际参数，不能只设置环境变量。

传感器与定位就绪、镜像更新后，使用正式入口启动 bridge-off 链路：

```bash
STRIVE_EXECUTION_MODE=preview bash scripts/run_real_robot_navigation.sh
```

正式链路为 `InstructionRuntime → StandoffGoalResolver → ExecuteWaypoint Action → MotionServer → localPlanner → pathFollower → SafetyVelocityMux`；只由 MotionServer 发布原生目标，Action 终止先取消目标并 hold。当前正式入口 `LLM_OFFLINE=1`、最终视觉 verifier 关闭；不因实现状态更新而启用模型或实物运动。

启动/停止、日志、live 授权与控制合同见 [正式一键入口](docs/real_robot_navigation_entry.md)；部署记录见 [实物状态](docs/real_robot_deployment_todo.md)，整体接口见 [实物框架](docs/real_robot_framework.md)。
