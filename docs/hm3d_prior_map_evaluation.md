# HM3D 语义先验地图与对比评测

本文说明如何从 HM3D 场景构建可迁移的语义先验地图，并通过严格的有/无先验对照
实验评估其对 ObjectNav 搜索效率的影响。本文不介绍 Docker 从零安装；环境、数据和
模型准备见 [README](../README.md) 与
[HM3D Docker benchmark guide](hm3d_docker_benchmark_guide.md)。

## 1. 设计边界

先验地图是搜索排序上下文，不是运行时真实物体身份，也不是运动控制地图。

```text
HM3D semantic scene + scene geometry
  -> offline semantic prior map
  -> room / topology / semantic BEV context
  -> frontier and room candidate ranking
  -> online detector, mapper, verifier
  -> waypoint and final stop decision
```

先验地图可以影响：

- 房间和 frontier 的搜索优先级；
- 目标概念对应的区域先验；
- LVLM 读取的静态房间布局和语义 BEV 上下文；
- 运行过程中的 prior/runtime 对齐诊断。

先验地图不能：

- 创建当前回合不存在的目标实例；
- 替代在线检测、SysNav `ObjectNode` 或运行时对象 UID；
- 直接生成底盘速度或 Habitat 离散动作；
- 绕过关系验证、物理停止约束或 final verifier。

正式先验地图不包含 Habitat NavMesh 导出的 `.npy` 几何栅格。NavMesh 只在离线构建
阶段辅助恢复房间边界和连通性，最终交付的是结构化语义 JSON 与语义 BEV 图像。

## 2. 数据与坐标约定

设置场景数据根目录和目标场景目录：

```bash
export STRIVE_DATA_ROOT=/path/to/CogNav_ObjNav/data
export SCENE_ROOT="$STRIVE_DATA_ROOT/scene_datasets/hm3d_v0.2/val/00802-wcojb4TFT35"
export SCENE_ID=wcojb4TFT35
```

场景目录至少应包含：

```text
$SCENE_ROOT/
  <scene_id>.basis.glb
  <scene_id>.semantic.glb
  <scene_id>.basis.navmesh
  <scene_id>.semantic.txt
```

FloorPlan 风格输出采用二维米制坐标。Habitat 的水平面 `(x, z)` 会映射为
floorplan 的 `(x, -z)`；输出 JSON 会记录 `frame_id`、`source_frame_id` 和变换元数据，
避免把图像坐标、先验坐标与机器人运行时坐标混用。

## 3. 生成语义 BEV 与 room layout

### 3.1 单场景构建

该入口只需要 Habitat 场景和语义数据，不需要 SAM、GroundingDINO 或 ObjectNav 回合
文件：

```bash
COGNAV_ROOT=/path/to/CogNav_ObjNav \
IMAGE_TAG=strive-hm3d:local \
bash docker/run_hm3d_floorplan_layout.sh \
  "$SCENE_ROOT" \
  --scene_id "$SCENE_ID" \
  --output "logs/prior_maps/${SCENE_ID}_floorplan.json" \
  --quality_output "logs/prior_maps/${SCENE_ID}_floorplan_quality.json" \
  --topdown_resolution 0.25
```

不要使用 `docker/run_hm3d_baseline.sh` 代替该入口。baseline 启动器会额外检查
检测器权重和 ObjectNav episode，而布局构建只需要 Habitat-Sim 及 HM3D 场景文件。

### 3.2 输出文件

```text
logs/prior_maps/<scene>_floorplan.json
logs/prior_maps/<scene>_floorplan_quality.json
logs/prior_maps/prior_map_bev.png
logs/prior_maps/prior_map_bev.svg
logs/prior_maps/prior_map_bev_markers.json
logs/prior_maps/prior_map_manifest.json
```

`floorplan.json` 保存 `levels -> regions -> boundaries / center / connectivity` 结构。
`prior_map_bev.png` 和 JSON 同源，包含房间区域、拓扑和语义标签，用作 LVLM 的静态
地图上下文；PNG/SVG 是可视化结果，不是运动目标。

质量报告至少应检查：

- 房间边界覆盖率；
- 房间中心和房间拓扑边数量；
- 语义 BEV 是否成功生成；
- 坐标系和变换元数据是否完整；
- `object_instances_omitted` 是否符合当前 layout-only 配置。

同一 semantic region 内的 NavMesh 样本会先投影到 BEV 再提取连通组件，避免不同
高度采样生成重复房间。`component_N` 是几何组件标识，不是自然语言房间类别。

### 3.3 批量构建

```bash
bash docker/run_hm3d_baseline.sh bash -lc '
  cd /workspace/STRIVE
  python scripts/build_hm3d_floorplan_layouts.py \
    /workspace/data/scene_datasets/hm3d_v0.2/val \
    --output_root logs/prior_maps/hm3d_val_floorplans \
    --topdown_resolution 0.25
'
```

批处理会为每个场景写出 `floorplan.json`、质量报告、语义 BEV 和 manifest，并在输出
根目录保存批处理 `manifest.json`。批量结果属于 `logs/` 运行产物，不应提交到 Git。

## 4. 构建 canonical HM3D ground-truth prior map

如果需要将语义场景、房间拓扑和离线物体先验统一为 canonical contract，可运行：

```bash
bash docker/run_hm3d_baseline.sh bash -lc '
  cd /workspace/STRIVE
  python scripts/build_hm3d_groundtruth_prior_map.py \
    /workspace/data/scene_datasets/hm3d_v0.2/val/00802-wcojb4TFT35 \
    --scene_id wcojb4TFT35 \
    --output logs/prior_maps/wcojb4TFT35_groundtruth_prior_map.json \
    --alignment_output logs/prior_maps/wcojb4TFT35_groundtruth_alignment.json \
    --topdown_resolution 0.25 \
    --min_room_area_m2 0.25 \
    --mask_dilation_radius_m 0.35
'
```

输出：

```text
logs/prior_maps/<scene>_groundtruth_prior_map.json
logs/prior_maps/<scene>_groundtruth_alignment.json
```

典型 metadata：

```text
source_format = hm3d_groundtruth_semantic_scene
frame_id      = habitat_world
authority     = semantic_scene_plus_navmesh
```

该文件记录的是离线场景先验，不代表当前 episode 的在线对象 UID。运行时仍由
mapper 或 SysNav `ObjectNode` 提供对象身份；先验对象只能参与候选排序和对齐诊断。

## 5. 内容验证与可视化

### 5.1 JSON 内容检查

```bash
python - <<'PY'
import json

path = "logs/prior_maps/wcojb4TFT35_groundtruth_prior_map.json"
with open(path, encoding="utf-8") as stream:
    payload = json.load(stream)

print("scene:", payload["scene_id"])
print("rooms:", len(payload["rooms"]))
print("objects:", len(payload["objects"]))
print("edges:", len(payload["topology_edges"]))
print("tv:", [item for item in payload["objects"]
              if item["label"].lower() == "tv"][:1])
PY
```

### 5.2 运行时可视化产物

启用先验地图的 episode 会在对应运行目录写出：

```text
logs/<save_dir>/episode-*/prior_map/floorplan_global.png
logs/<save_dir>/episode-*/prior_map/floorplan_global.svg
logs/<save_dir>/episode-*/prior_map/floorplan_global_markers.json
logs/<save_dir>/episode-*/prior_map/floorplan_step_*.png
logs/<save_dir>/episode-*/prior_map/floorplan_chosen_frontier_*.png
logs/<save_dir>/episode-*/prior_map/som_global.png
logs/<save_dir>/episode-*/prior_map/search_prior_result_*.json
logs/<save_dir>/episode-*/prior_map/chosen_frontier_*.json
```

`floorplan_chosen_frontier_*.png` 用于检查房间边界、房间拓扑、先验物体、目标先验
物体、frontier、选中 frontier、运行轨迹及在线检测匹配是否一致。

## 6. 接入 ObjectNav 运行

先验地图启用后仍需使用普通 ObjectNav 入口：

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id "$SCENE_ID" \
  --object_category tv \
  --save_dir "hm3d_${SCENE_ID}_prior_on" \
  --vlm ark \
  --enable_prior_map \
  --prior_map_path "logs/prior_maps/${SCENE_ID}_groundtruth_prior_map.json" \
  --prior_map_source canonical_json \
  --prior_map_alignment "logs/prior_maps/${SCENE_ID}_groundtruth_alignment.json"
```

先验地图只改变 room/frontier 搜索排序，不直接生成运动目标，也不绕过在线目标确认、
关系验证或 final verifier。未完成 prior/runtime 坐标标定时，BEV 只能作为静态语义
上下文，不能绘制未经变换的机器人、轨迹或 frontier 位置。

## 7. 有/无先验 A/B 对比

### 7.1 实验原则

每组对比必须固定：

- scene、episode rank、目标类别或自然语言指令；
- Habitat、检测器、LVLM、模型权重和最大步数；
- `save_dir` 为两个全新的目录；
- 随机种子及其他运行参数。

唯一变量是 `--enable_prior_map` 及其对应的 prior map/alignment 输入。不要复用旧
日志目录，否则历史 metrics、verifier trace 和当前运行会混在一起。

### 7.2 开启先验地图

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id "$SCENE_ID" \
  --object_category tv \
  --save_dir "hm3d_${SCENE_ID}_prior_ab_on" \
  --vlm ark \
  --enable_prior_map \
  --prior_map_path "logs/prior_maps/${SCENE_ID}_groundtruth_prior_map.json" \
  --prior_map_source canonical_json \
  --prior_map_alignment "logs/prior_maps/${SCENE_ID}_groundtruth_alignment.json" \
  --clean_save_dir
```

### 7.3 关闭先验地图

```bash
bash docker/run_scene_object_nav.sh \
  --scene_id "$SCENE_ID" \
  --object_category tv \
  --save_dir "hm3d_${SCENE_ID}_prior_ab_off" \
  --vlm ark \
  --clean_save_dir
```

### 7.4 对比指标

```bash
python - <<'PY'
import csv

for run in ("hm3d_wcojb4TFT35_prior_ab_on",
            "hm3d_wcojb4TFT35_prior_ab_off"):
    path = f"logs/{run}/metrics.csv"
    with open(path, newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    print("\n", run)
    for key in (
        "prior_map_enabled", "success", "spl", "distance_to_goal",
        "Episode Steps", "travel distance", "Found Goal",
        "final_stop_accept_step", "prior_map_top_object_label",
        "prior_map_top_object_score",
    ):
        print(key, row.get(key, ""))
PY
```

建议将以下量作为主要结果：

- Habitat success 和 SPL；
- 导航步数、行走距离和距离目标的最终距离；
- instruction-level success、final-stop decision 和 accept step；
- 先验候选命中、房间/frontier 选择变化；
- LVLM 调用次数、缓存命中率、超时和 schema repair 次数。

先验地图的增益应体现为更早进入包含目标的区域、更少的无效 frontier 探索或更少的
导航距离，而不是静态先验直接提高成功标志。最终停止仍必须由在线视觉证据、物理停止
约束和统一 final verifier 共同决定。

## 8. 产物归档

一次完整评测建议保留以下结构：

```text
logs/<run_id>/
  run_manifest.json
  metrics.csv
  episode-0/
    prior_map/
    instruction_adapter/
    final_verifier/
    detection/
```

提交或共享实验结果时只保留 manifest、指标和必要的可视化摘要。原始图像、视频、模型
权重、API key 和包含敏感路径的完整回执留在受控存储中。
