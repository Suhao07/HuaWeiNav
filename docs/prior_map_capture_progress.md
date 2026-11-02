# HM3D 先验采集首轮实现记录

> 2026-11-01整理：本文件保留历史实施、临时实验及失败证据，文中的“下一步”反映记录当时判断。当前执行清单统一以 [设计方案的必要交付](prior_map_mode.md#距离交付仍需完成的必要工作) 为准，历史测试无需逐项重跑。

状态：P1 协议基础、P2 有预算采集、P3 投影诊断已实现；P4 逐帧检测与共享实例融合、P5 导出和跨进程重载检索已跑通。P3/P4 的几何及实例质量仍在验收，P6–P8 尚未完成。RGB-D capture 与预测地图分别报告，预测对象不作为真值或导航完成证据。

## 实现范围

- `prior_map/observations.py`：无损 RGB PNG、float32 米制深度 NPY、相机内参与右手 Z-up 位姿；纯文件记录，未引入数据库。
- `prior_map/bundle.py`：现有 canonical PriorMapData 的地图包读写，验证证据引用和两种视点；移动目录后的加载测试通过。实际预测对象由 `capture_builder.py` 导出。
- `backends/habitat/capture.py`：独立 Habitat-Sim 采集，按 NavMesh island 取样、障碍间距过滤、覆盖增益选点、路径邻近排序和水平视角渲染。采集不依赖 ObjectNav episode、目标或模型服务。
- `prior_map/geometry_diagnostic.py`：跨帧深度重投影，分别记录一致点、遮挡候选和前方空间冲突；属于一致性诊断，不证明绝对标定正确。
- `scripts/collect_hm3d_observations.py`、`scripts/diagnose_prior_capture.py` 与 `docker/run_hm3d_capture.sh`：运行入口。

初版候选选择使用欧氏最远点扩散；当前已升级为路径距离覆盖增益选择，详见本文末尾的多区域对照。初版未实现自动按缺口补采、俯仰扩展、基于视野的拒绝重采与连续运动。采集完成时 `coverage_requirement_met=null`，不会把有预算扫描标记为全面覆盖达标。尚未给出全场景表面覆盖率。

## 实际环境与数据

- 镜像：`strive-hm3d:local`，镜像 ID `9d834e22fa15`。
- Habitat-Sim：0.2.4；镜像 Python：3.9.23；GPU：RTX 3090。
- 代码 HEAD：`dddfed1da68164b00f06a17b20f43a33677077c5`，工作区含已有改动和本轮新增代码；不是冻结提交。
- 场景：HM3D v0.2 `00802-wcojb4TFT35`，island 4，NavMesh 面积约 40.81 平方米。
- 该 island 是连通区域，不等同于一个语义房间。检查图包含楼梯、通道及储物区域。
- 数据源：`/home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav/data`。
- 场景和 NavMesh SHA256、采集器哈希及配置见各次 manifest。

## 固定输入的覆盖对照

两组固定 seed=17、候选采样数=256、独立检查点数=256、间距=1m、障碍间距=0.25m、8 个朝向、320×240、HFOV=90°、相机高度=1.25m。只改变位置预算。

| 运行 | 位置 | 帧 | 1m 路径半径覆盖 | 采集函数耗时 |
| --- | ---: | ---: | ---: | ---: |
| island4-control | 4 | 32 | 61/256 = 23.83% | 1.98s |
| island4-dense | 24 | 192 | 234/256 = 91.41% | 3.70s |

耗时包含 simulator 初始化、位置规划和文件写入，不含 Docker 启动与本次代码中的资产哈希阶段，也不含语义检测、融合、地图构建或导航。192 帧目录约 72.64 MiB（含诊断 JSON），不能作为轻量运行时语义地图的大小。

## 几何诊断

- 早期 32 帧跨位置样本同位置旋转配对的残差中位数约 1.53mm。
- 192 帧使用 stride=16、容差 5cm，生成 8,422 个存在投影的帧对。
- 跨位置且至少 50 个有效投影点的帧对有 5,708 个。
- 这些帧对累计 1,233,191 次投影，731,771 次为遮挡候选，29,718 次为前方空间冲突。
- 去除遮挡候选后，各帧对绝对残差中位数的中位数约 6.19mm。

投影点在不同帧对重复计数；遮挡候选是深度关系判断，不是经人工确认的遮挡真值。不能仅凭过滤后的低残差宣称几何验收完成。需要检查空间冲突集中在深度边缘、薄结构还是异常视角，再冻结融合输入。

## 复现

在 STRIVE 根目录执行（输出目录须不存在）：

```bash
STRIVE_DATA_ROOT=/home/ubuntu/WorkSpace/research/code/Navigation/CogNav_ObjNav/data \
  bash docker/run_hm3d_capture.sh \
  /workspace/data/scene_datasets/hm3d_v0.2/val/00802-wcojb4TFT35 \
  --output outputs/prior_capture/wcojb4TFT35-island4-repeat \
  --island 4 --samples 256 --max-positions 24 --headings 8

/home/ubuntu/anaconda3/envs/UrbanNav/bin/python \
  scripts/diagnose_prior_capture.py \
  outputs/prior_capture/wcojb4TFT35-island4-repeat --stride 16

/home/ubuntu/anaconda3/envs/UrbanNav/bin/python -m pytest \
  tests/test_prior_capture.py tests/test_prior_map_contracts.py \
  tests/test_prior_map_loaders.py -q
```

16 项测试通过，含无损深度、轴约定、地图移动后引用解析、错误引用和故意位姿偏移的重投影检测。静态语法与 shell 检查通过。

## P4 前的具体接入问题

1. `get_object_entities()` 旧单帧路径调用 ObjectNode 时缺少当前构造函数要求的一项参数。当前 baseline 主要使用全景路径；不能直接认为单帧路径可用。
2. `initialize()` 内全景坐标表固定 640×480，角度固定 ±30°；本次采集为 320×240、45° 朝向间隔。需要从实际传感器与分组参数构造输入，禁止直接喂入不匹配缓存。
3. mapper 存在轴交换反射与初始位置平移；新增采集坐标已单独定义，融合接入必须显式转换并回归投影。
4. 原始 RGB-D 已可独立重放；下一阶段提取现有实例生成和融合调用边界，保留 mask 与 observation ID，完成真正的预测地图导出，再运行查询。

以上问题来自静态源码核对，尚未证明是用户历史失败运行的根因。原有实验文件与未提交修改保持原样；本轮未提交 Git、未启动导航或豆包请求。


## P4/P5：共享融合与预测地图（2026-10-30）

**当前有效集成产物为 `outputs/prior_bundles/wcojb4TFT35-control-rgb/`；早期 8/32 帧结果与中断的 dense 运行用于问题诊断。**

### 代码与职责

- `cv_utils/sam.py`：`target=None` 时执行显式词表检测与 SAM 分割；保留有目标任务的检测入口。词表支持单个概念及更长列表，沿用原来的最多四批提示词分组。类别来自调用参数，不读取场景 semantic GT 或 episode 目标。
- `mapping/object_fusion.py`：从 mapper 提取原实例关联函数。在线 mapper 通过包装函数传入目标与匹配谓词，离线构建传入无任务语义。已将函数体与原 HEAD 做 AST 对照，任务上下文替换之外的算法一致。
- `prior_map/capture_builder.py`：读取统一 RGB-D、按 mask 反投影、调用已有点云降采样/聚类和共享融合、保存证据并导出 PriorMapData。此入口不构造假的 mapper 或 env。
- `scripts/query_prior_bundle.py`：独立进程加载地图，调用已有 PriorMapQueryService 返回概念候选与证据。当前是类别/别名检索，不包含自然语言任务编译、视觉特征检索或运动。
- `prior_map/evidence_audit.py`：按实例审计多类别证据与同帧分离掩码，输出人工复核候选，保持地图不变。
- `docker/run_prior_capture_builder.sh`：独立复用已有镜像和权重配置，不依赖 ObjectNav 数据集、在线模型凭证或导航生命周期；以宿主 UID/GID 写文件。

当前保持原始全景分组记录，但采用**每帧独立检测再三维融合**。旧三视图拼接中的固定分辨率/角度假设尚未迁移；本轮不能称为与旧全景感知过程等价。旧单帧 ObjectNode 参数缺失的入口也没有被调用或宣称已修复。

### 几何与融合原理

统一反投影：

\[
p_M = T^M_C \begin{bmatrix}d(u,v) K^{-1}[u,v,1]^\top \\ 1\end{bmatrix}.
\]

沿用原融合的双向近邻覆盖率，距离阈值为 0.1 m，关联阈值为 0.25：

\[
s(P,Q)=\frac{\sum_{p\in P}\mathbf{1}[d(p,Q)<0.1]+\sum_{q\in Q}\mathbf{1}[d(q,P)<0.1]}{|P|+|Q|}.
\]

同类别累积置信度沿用权重 \(w_i=c_i^2 n_i/\sum_j c_j^2 n_j\)，并使用 \(c'=\sum_i w_i c_i\)。这是原工程规则，不是经过校准的类别概率。新入口的单帧点云先以 0.05 m 体素采样，并使用 6 m 最大融合深度；保存的原始证据深度保持不变。旧融合过程持续追加点云，尚未实现融合后体素压缩，其资源增长需要单独测量。

对象 UID 绑定首次建立该预测实例的 capture frame/detection；融合保留 UID。最新导出使用 `exact=True` 表示具体的**预测实例身份**，正确性与任务满足由后续证据判定。早期 8/32 帧试运行将此字段写为 False，属于导出语义未收敛的中间产物；检测与融合计数仍可用于固定输入诊断。

### 通道修复前的小样本诊断结果

| 输入 | 预测实例 | 有效 mask 证据 | 构建耗时 |
| --- | ---: | ---: | ---: |
| 第一个位置的 8 帧 | 30 | 82 | 25.54 s |
| 4 个位置、32 帧 | 86 | 343 | 89.59 s |

耗时含模型加载、检测、融合及导出，不含 Docker 启动；这两次早期试运行尚未计算权重哈希。原始采集耗时与建图耗时分开统计。

32 帧地图包约 11.61 MiB，核心 `prior_map.json` 约 39.29 KiB。当前复制全部有效证据，尚未进行代表性证据精选。运行时查询不加载全部点云与 RGB-D，但地图包总成本包括这些证据。

构建进程退出后，已将 32 帧包复制至 `/tmp/strive-prior-control-reload`，用独立进程重载、检索 `cabinet`，并验证 `washing_machine` 无概念匹配。原查询服务最多返回 12 个候选，返回数量不等于地图中的该类别总数。查询 authority 仍为 `ranking_only`，没有当前 frontier，没有发生导航或 STOP。

### 已知质量问题和证据

- 32 帧地图中，57/86 实例有多帧证据；34/86 关联了多个检测类别；23/86 具有同帧 IoU < 0.05 的掩码对。后二者是复核标记，不能直接作为错误率：例如柜体与搁板可能是整体/部件关系。
- 检测拼图显示门框、柜体、搁板的重叠检测与类别混淆。现有融合主要按几何覆盖关联，尚未建模整体/部件关系或实例排斥。
- 增加视图不等于提升实例精度。必须结合固定观察下的融合审计，区分重复检测、实例碎片和错误合并；目前没有实例级真值精度结论。
- 192 帧深度诊断累计 1,305,025 次投影、30,237 次前方冲突；其中 30,147 次在源或目标 3×3 深度邻域极差超过 0.1 m 的位置。分层统计保留全部样本，不用边缘过滤掩盖错误。
- 剩余 90 次冲突中，10 次落在源/目标图像边界；80 次位于图像内部，最大约 10.43 cm。全体剩余冲突的最大 25.28 cm 出现在目标图像顶边。对应像素已保存，但尚未通过独立渲染或网格交点确认全部原因，因此几何验收保持开放。

诊断产物位于 `outputs/prior_diagnostics/`：`detection_contact.jpg`、`control_evidence_audit.json`、`control_relocated_query.json`、`control_absent_query.json`、`interior_conflicts.json`、`interior_conflicts.jpg`。这些是输出文件，未纳入 Git。

### 可复现入口

```bash
bash docker/run_prior_capture_builder.sh \
  outputs/prior_capture/wcojb4TFT35-island4-control \
  outputs/prior_bundles/wcojb4TFT35-control-repeat \
  --classes chair table cabinet door window stairs lamp plant sofa bed toilet shelf

/home/ubuntu/anaconda3/envs/UrbanNav/bin/python scripts/query_prior_bundle.py \
  outputs/prior_bundles/wcojb4TFT35-control-repeat cabinet \
  --output outputs/prior_diagnostics/cabinet_query.json

/home/ubuntu/anaconda3/envs/UrbanNav/bin/python scripts/audit_prior_bundle.py \
  outputs/prior_bundles/wcojb4TFT35-control-repeat \
  --output outputs/prior_diagnostics/evidence_audit.json
```

纯 CPU 协议、几何、查询与审计测试 22 项通过；维护镜像内共享融合/无目标检测/单概念词表/RGB-BGR 桥接测试 4 项通过。CPU 环境没有 Open3D，感知集成测试在那里显式 skip，在维护镜像中实际执行。


## 检测颜色协议根因修复与复测

通过读取维护镜像内实际 MMDetection 源码确认：`InferencerLoader` 对 ndarray 调用 `LoadImageFromNDArray`，后者保持通道顺序；当前 GroundingDINO 配置启用 `bgr_to_rgb=True`。原共享入口直接把 Habitat RGB 传入该路径，导致模型最终接收红蓝交换后的输入。SAM 的 `set_image()` 默认接收 RGB，因此不能在整条分割链路入口统一交换颜色。

根因修复放在 `MMDINO_Grounded_SAM._run_grounding()`：RGB 转连续 BGR 数组后交给 MMDetection，SAM 继续读取原始 RGB。新增红色像素契约测试验证转换和输入不变性。修复会改变在线共享检测器的输入；本轮验证了检测接口与实际离线建图，尚未进行导航 episode 回归。对象融合函数体保持原算法。

发现问题后停止了 192 帧 dense 建图，最终完成 151 帧处理，**未完成地图导出**。状态写入 `outputs/prior_bundles/wcojb4TFT35-dense/run_status.json`，日志保留于 `logs/prior_builder_dense.log`。原始 192 帧采集与深度诊断不经过检测器，因此这些数据继续有效。后续扩大建图前，先完成修复后的小样本实例质量检查。

固定同一组 32 帧复测：

| 项目 | 结果 |
| --- | ---: |
| 预测实例数 | 98 |
| mask 证据数 | 395 |
| 多帧支持的实例数 | 64 |
| 存在多个检测类别的实例数 | 35 |
| 存在同帧分离掩码的实例数 | 29 |
| 构建耗时（含初始化、权重哈希与导出） | 93.27 s |
| 核心 JSON | 44.96 KiB |
| 完整地图包 | 11.35 MiB |

数量变化不是精度改善证明。修复后仍有类别混淆和实例关联复核项，尚未满足高质量语义地图验收。检测来源、mask、深度、内参和位姿可追溯，支持下一步定位问题。

构建进程已退出。复制到 `/tmp/strive-prior-control-rgb-reload` 后，全部 **460 个文件** SHA256 一致，实例 ID、位置与核心数据一致；独立进程检索 `cabinet` 返回 12 个候选（服务返回上限），检索 `washing_machine` 返回 0 个匹配。新地图中 `exact=True` 仅表示具体预测实例身份；查询仍为 ranking_only。

- 构建记录：`outputs/prior_bundles/wcojb4TFT35-control-rgb/build_receipt.json`，包括本轮源码、模型权重和模型配置 SHA256。
- 重载记录：`outputs/prior_diagnostics/control_rgb_reload_receipt.json`。
- 检索记录：`outputs/prior_diagnostics/control_rgb_relocated_query.json` 与 `control_rgb_absent_query.json`。
- 实例证据审计：`outputs/prior_diagnostics/control_rgb_evidence_audit.json`。

下一步先用上述实例与空间冲突的对应证据完成 P3/P4 质量诊断，再扩大多区域采集和融合。当前未实现精选证据预算、导航视点生成、重定位、连续巡视或历史地图指令导航；未给出这些能力的验收结论。

## 多区域采集与实例根因诊断（2026-10-30，续）

本轮属于**部分完成**：补齐 P2 覆盖需求点选点、空间邻近采集顺序、独立检查点和缺口记录；完成同场景全部连通区域的采集扩展。P4 增加独立标注评测，尚未通过语义地图质量验收。P6 的预测融合扩展、P7/P8 仍待完成。

### 覆盖选点与真实采集

旧版本以候选相机位置之间的最远距离作为停止条件，清障过滤后的候选集不能代表全部可导航区域。改为在原始导航采样点集 Q 上进行贪心覆盖，相机位置仍从满足 clearance 的候选集 V 中选择：

\[
v^*=\arg\max_{v\in V}\left|\{q\in Q_{\mathrm{uncovered}}:d_{\mathrm{nav}}(q,v)\le r\}\right|.
\]

达到位置预算或覆盖增益为零时停止。选择完成后用路径距离近邻顺序组织渲染。独立评测点不参与选点，避免用验收集反复调采集位置。coverage.json 保存检查点坐标、最近路径距离、未覆盖点索引和不可达数量；每帧记录有效深度比例、深度分位数与 0.25 m 内深度占比。这些统计不等同于视野质量已验收。

固定同一场景、seed=17、每区域 samples=256、位置预算上限 24、半径 1 m、8 个朝向和原传感器参数。两次运行的各区域独立检查点坐标逐项相同。

| island | 路径最远点选样位置数 | 覆盖增益选样位置数 | 前者覆盖点 | 后者覆盖点 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 256/256 | 256/256 |
| 1 | 14 | 17 | 177/256 | 176/256 |
| 2 | 8 | 9 | 235/256 | 237/256 |
| 3 | 17 | 23 | 221/256 | 243/256 |
| 4 | 20 | 23 | 220/256 | 218/256 |
| 5 | 1 | 1 | 239/256 | 255/256 |
| 6 | 1 | 1 | 245/256 | 256/256 |
| 合计 | 62 | 75 | 1593/1792 | 1641/1792 |

合计按每区域 256 点等权计算，不是场景表面积覆盖率。改善伴随位置数量增加，不能称为同采集成本下的效率提升。区域 1 和区域 4 未改善，候选池、清障范围与剩余缺口的关系仍需进一步定位。

- 最远点对照：`outputs/prior_capture/wcojb4TFT35-all-islands-geodesic/`，496 帧，采集函数约 7.13 s，178.80 MiB。
- 当前覆盖增益版本：`outputs/prior_capture/wcojb4TFT35-all-islands-coverage/`，600 帧，采集函数约 8.70 s，216.22 MiB。
- 最低有效深度比例约 90.44%，所有帧的 0.25 m 内深度占比为 0；尚未据此声明视野无碰撞或全面覆盖。
- 各区域包含不同高度及楼梯连接，NavMesh island 仍不等同于语义房间或楼层。

这是 geometry-assisted coverage_scan，采集段间直接设置姿态；没有机器人连续行走证据，也尚未对 600 帧运行预测融合。

### 聚类假设与点云增长

新增 `scripts/diagnose_prior_clusters.py`，在修复后 32 帧地图保存的 395 个 mask 上重放原深度裁剪、0.05 m 体素和 DBSCAN 参数。

- **0/395** 次选中了 DBSCAN 噪声簇。因此，当前样本没有支持“最大簇误选噪声是主要原因”的假设，本轮保持聚类算法。
- 重放后全部 **98/98** 个实例的累计点数与导出记录一致。
- 累计 155,974 个点对应各实例合计 60,499 个占用体素，约 2.58 点/体素。该计数按实例分别累计，不是场景唯一几何体素数。
- 原融合直接拼接点云，重复观测增加存储密度。这个结果支持后续进行融合后体素压缩的固定证据对照；尚未验证压缩对实例关联、精度和耗时的影响，因此本轮没有直接替换融合规则。

产物：`outputs/prior_diagnostics/control_rgb_cluster_audit.json`。

### 独立语义标注评测：定位错误发生在哪一层

新增 `prior_map/semantic_audit.py` 和 `scripts/diagnose_prior_instances.py`，只在离线评测中读取 HM3D semantic 资产。预测地图保持只读；这些报告标记 `oracle_diagnostic_only`、`semantic_gt_used=true`，不作为建图器或导航的输入。

实际镜像中的 HM3D 语义 ID 使用纹理。仅启用语义/深度传感器时，本次探针返回全 0；加入颜色传感器以启用纹理后，语义实例 ID 正常。没有将全 0 输出误计为模型失败。

使用保存的 camera_to_map 还原相机，原 32 帧的重新渲染深度与保存深度**逐像素完全一致**。这证明本次评测的相机复原及渲染对齐，不替代此前全部跨视角残差的分析。

评测结果：

- 395 个 mask 中，287 个有超过完整 mask 一半的像素支持同一标注实例。Unknown 像素留在分母中。
- 98 个预测实例中，87 个获得这种多数支持；其中 17 个关联到了多个不同标注实例。标注的整体/部件粒度仍需核对，不能直接把 17/98 当作通用错误率。
- 105 个预测为 door 的多数支持 mask 对应标注 wall。进一步查看原检测框：其中 **99 个框内属于名称包含 door 的标注类别的像素不足 10%**；没有一个框达到 50%。

因此，至少在这个固定子集上，问题在检测框/类别阶段已经存在，不能全部归因于 SAM 分割或后续三维融合。105 个是跨视图 mask 计数，不是 105 个独立物体；door 与 closet door 等类别按本次诊断明示的标签规则统计，未冒充开放词汇任务准确率。

候选机制包括短提示词表的语义竞争、低置信度检测准入、纹理/标注粒度差异。下一项最小实验应固定同一图像、模型权重、解码与阈值，仅检验加入通用结构类别对原 door/window/stairs 检测的影响；在该证据形成前，不修改融合阈值来掩盖上游误检。

产物：

- `outputs/prior_diagnostics/control_rgb_semantic_audit.json`：第一轮独立评测。
- `outputs/prior_diagnostics/control_rgb_semantic_spatial_audit.json`：框与 mask 的分别统计，以及评测代码、标注资产、预测地图 SHA256。
- `outputs/prior_diagnostics/door_wall_localization.json`：固定的 door/wall 子集定位记录。

复现（在维护镜像中，场景资产挂载至 /workspace/data）：

```bash
python scripts/diagnose_prior_instances.py \
  outputs/prior_bundles/wcojb4TFT35-control-rgb \
  --scene-dataset-config /workspace/data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json \
  --output outputs/prior_diagnostics/instance_audit_repeat.json
```

本轮新增行为包含隔墙路径距离、清障候选与区域需求点分离、语义多数支持/冲突保留的测试。预测地图仍是上一轮已经重载验证的产物，本轮没有将真值结果回写进去。

## 固定提示词对照与手机输入适配（2026-10-30，续）

### 结构类别提示词未解决主要误检

固定原 32 帧、检测权重、RGB/BGR 桥接、score > 0.2 和 NMS IoU=0.8，仅比较一个提示词组：

- base：door / window / stairs。
- structure_context：上述词组后追加 wall / floor / ceiling。

语义缓存与图像 SHA256、相机/深度对齐及模型指纹独立校验。缓存仅供离线评分；词表按通用结构类别预先固定，模型不读取标注 ID。

| 诊断项 | base | structure_context |
| --- | ---: | ---: |
| NMS/面积过滤后的 door 框 | 160 | 164 |
| 框内门类标注像素不足 10% | 120 | 118 |
| IoU≥0.5 一对一匹配的可见标注门实例 | 18/45 | 23/45 |

45 是逐帧可见、至少 100 像素的门类实例计数，同一个物体可跨帧出现。门类定义为标注名称包含 door（包括 closet door）；该诊断不等同于完整 mapper、开放词汇准确率或导航成绩。

结构上下文改善了这组样本的框匹配，但低门像素支持的框仍很多，不能认定问题已解决。本轮保留原默认建图词表，没有部署按该单场景调出的规则。新增 `scripts/diagnose_detector_vocabulary.py`，报告保留全部框、像素支持、每帧匹配以及两种条件。

主要产物为 `outputs/prior_diagnostics/vocabulary_context_contrast.json`；增加实际模型指纹检查后的复核运行单独保存到 `vocabulary_context_contrast_verified.json`。语义评分缓存位于 `control_rgb_oracle_cache/`，与预测地图隔离。

源码检查还确认：原 baseline 的 B 类背景对象直接进入融合；只有 C 类目标候选进入 `refine_candidates()` 的 VLM 复核。因此不能把所有背景实例理解为已经通过视觉语义验证。下一步需区分预建图的对象假设和任务时的证据验证，并在固定实例上验证相应边界。

### 手机导入器已接入共同协议

已下载并核对官方 Stray Scanner 源码，新增 `prior_map/stray_scanner.py`、`scripts/import_stray_capture.py` 和对应测试。关键修正是遵循上游已经转换为光学轴的导出四元数，避免再次应用 Habitat 的相机轴翻转。

支持逐帧内参、带表头旧版会话内参、RGB/深度分辨率对应、米制深度、时间戳和原始 confidence 保存。地图包增加可选 confidence_ref 验证与复制，核心地图协议保持一致。带畸变 LUT 的输入明确要求先矫正，不静默当作针孔模型。

详见 [手机输入适配说明](prior_map_stray_scanner.md)。目前是工程转换测试通过，真实手机数据与实物导航尚未验收。原有预测地图和失败诊断产物均保留。


## 精确距离查询优化与历史视点解析（2026-10-30，续）

### 保持融合定义，减少无关距离计算

共享 `mapping/object_fusion.py` 继续使用原双向近邻重叠率：

\[
s(A,B)=\frac{\sum_{a\in A}\mathbf 1[\min_{b\in B}\|a-b\|<0.1]
+\sum_{b\in B}\mathbf 1[\min_{a\in A}\|b-a\|<0.1]}{|A|+|B|+10^{-6}}.
\]

融合准入阈值保持 0.25。本轮改变距离查询实现：cKDTree 使用 `eps=0` 精确查询；包围盒距离下界先排除重叠率为零的点云对：

\[
d_{box}(A,B)=\|\max(l_A-h_B,l_B-h_A,0)\|_2,
\qquad d_{box}\ge 0.1\Rightarrow s(A,B)=0.
\]

浮点计算与稠密求解器的运算顺序不同，数学等价不代表任意输入逐位相同；固定输入对照验证实际分配与输出。没有改变点云采样密度、聚类或关联阈值。

| 固定 395 个已接受观测 | 稠密距离 | 索引方案 | 结果 |
| --- | ---: | ---: | --- |
| 仅替换为 KD-tree | 14.95 s | 24.04 s | 分配一致，但变慢；保留失败对照 |
| AABB 下界筛选 + KD-tree | 15.134 s | 3.568 s | 分配与对象记录一致，98 个实例 |

后一项原稠密分配也复现了保存地图的分配；计时针对融合重放，不包含检测模型。报告分别为 `outputs/prior_diagnostics/fusion_distance_contrast.json` 与 `fusion_bounded_distance_contrast.json`。

完整 32 帧检测、融合、导出重跑得到：

- 新产物：`outputs/prior_bundles/wcojb4TFT35-control-indexed/`。
- 98 个预测对象、395 条观测，与 `wcojb4TFT35-control-rgb` 相同。
- 去除地图名称前缀后实例分配相同，所有 mask 字节一致，各实例累计点数一致。
- 对象中心最大差异约 0.0000105 m；保留这个浮点差异，不宣称完全逐位一致。
- 构建记录耗时从 93.272 s 到 74.465 s，约降低 20.2%。这是两次进程运行的观察值，尚无重复运行方差，不能当作通用性能收益。
- 新进程重载后 cabinet 查询正常；这验证索引链路，不验证 cabinet 类别正确率。

产物：`outputs/prior_diagnostics/indexed_build_parity.json`、`indexed_cabinet_query.json`；日志 `logs/prior_builder_control_indexed.log`。

共享距离实现也被在线 mapper 调用。当前验证覆盖离线固定观测和对象融合测试，尚无本次变更后的导航 episode 回归成绩。

### 历史观察位姿与运动候选分开

新增 `backends/habitat/prior_viewpoints.py` 和薄入口 `scripts/resolve_prior_viewpoints.py`。解析器读取捕获时已知的相机安装高度，恢复底盘位置与水平朝向，检查当前起点、目标可导航性及路径连通；地图保持只读。

本采集器相机仅有竖直安装偏置，统一 Z-up 地图中：

\[
p_{base}^{map}=p_{camera}^{map}-[0,0,h]^T.
\]

此式适用于本 Habitat 采集配置。手机手持相机需要地面及机器人可执行位姿解析，不能直接使用该高度公式。

32 帧地图中有 30 个带接受观测的历史视点。实际 NavMesh 检查得到：

| 当前起点 | 路径可达候选 | 拒绝候选 |
| --- | ---: | ---: |
| 原采集连通区域起点 | 30 | 0 |
| 另一断开的区域起点 | 0 | 30 |

**旧 NavMesh 没有保存构建参数。** Habitat 加载日志为 `NavMeshSettings aren't present, guessing that they are the default`。因此返回的半径、高度等值只是库报告值，`build_profile_verified=false`；所有候选标记 `path_reachable_profile_unverified`。路径连通检查不能证明对应机器人尺寸可通过。

当前有效报告为：

- `outputs/prior_diagnostics/historical_views_connected_profile_unverified.json`
- `outputs/prior_diagnostics/historical_views_disconnected_profile_unverified.json`
- 日志 `logs/prior_viewpoint_resolver_profile.log`

早先无 `_profile_unverified` 后缀的两个探索报告已被以上报告替代，不能据其参数名称推断构建配置已验证。解析器没有提交运动目标，也没有根据历史类别更新任务完成状态；P8 尚未完成。

### 验证与后续准入

- 主机协议、采集、手机适配与查询测试：32 项通过。
- 维护容器中的检测桥接、融合、精确近邻和包围盒测试：6 项通过，见 `logs/prior_fusion_index_tests_final.log`。
- 语义误检及实例冲突仍沿用前述诊断结论；距离计算优化没有解决类别质量问题。

下一步需要记录明确机器人配置、场景身份与输出哈希的 NavMesh 构建证据，并在该配置下检查历史目标和连续行走。随后才经现有运动所有者接入历史视点执行。多区域 600 帧预测融合、DualMap 同输入对照、真实手机几何验收与新指令导航仍为待完成项。


## 显式 NavMesh 构建与多区域建图启动（2026-10-30，续）

新增 `backends/habitat/navmesh_build.py` 与 `scripts/build_prior_navmesh.py`，要求显式传入半径、高度、跨阶与坡度；构建记录保存全部生效的 Recast 参数、场景/网格/源码 SHA256 和维护 Habitat 版本。历史视点解析器新增可选 `--build-receipt`：只有构建完成、网格哈希和采集场景哈希一致时，才接受重建网格及其参数。旧网格路径继续保留参数未验证状态。

本轮使用明确的**实验配置**：半径 0.10 m、高度 1.50 m、最大跨阶 0.20 m、最大坡度 45°。它用于建立可追溯的仿真检查，不代表已经核实实际机器人的尺寸。输出 `outputs/prior_navmesh/wcojb4TFT35-r010-h150/`，7 个连通区域，可导航面积约 127.949 m²。

同一个采集起点、同一份 32 帧地图，重建网格检查得到 **8 个路径可达视点、22 个拒绝视点**（7 个底盘位置在网格外，15 个与当前起点不连通）；原网格是 30 个可达。这说明同场景和相同的库报告默认值不足以证明网格等价，必须保留逐目标的可导航/路径检查。详见 `outputs/prior_diagnostics/historical_views_rebuilt_profile.json`。构建记录可验证后，候选仍需运行时机器人配置匹配和当前状态复查；尚未提交运动。

协议及相关主机测试目前 33 项通过，包含构建记录的网格替换、场景替换与未完成状态拒绝。

同时启动 `wcojb4TFT35-all-islands-coverage` 的全部 600 帧预测建图，输出目标为 `outputs/prior_bundles/wcojb4TFT35-all-islands-indexed/`，日志 `logs/prior_builder_all_islands_indexed.log`。沿用 32 帧实验的 12 类固定词表、权重、0.05 m 体素与 6 m 深度上限。本节记录启动事实；完成状态以构建进程退出及最终 `build_receipt.json` 为准，当前不报告完整地图数量或总耗时。


## DualMap 同输入对照接入（2026-10-30，续）

官方源码下载到工作空间 `DualMap/`，固定提交 `157235ec49e6a1f439babbc571c4c02ad1f06aa9`。本机已有 `dualmap` Conda 环境；修复实际导入失败所缺的 `filelock==3.16.1` 和 `pympler==1.1` 后，官方 `Dualmap` 核心模块导入通过。环境仍有 MobileCLIP 旧版 torchvision 依赖声明冲突；目前导入通过，模型前向尚未验证，不据此宣布环境完整验收。

新增 `prior_map/dualmap_input.py` 与 `scripts/build_prior_dualmap.py`：直接消费共同采集协议，以 RGB uint8、HxWx1 的 float32 米制深度、原始内参和 camera_to_map 构造官方 DataInput。保持 Z-up 世界坐标，不应用第一帧相对化、不再次翻转光学轴、不把深度转成整数 PNG。没有时间戳的采集不伪造采集时间；此次使用逐帧处理，不使用时间关键帧筛选。

维护环境中实际构造了 32 个上游 DataInput，逐帧深度与抽样反投影坐标完全一致。结果见 `outputs/prior_diagnostics/dualmap_input_parity.json`；这是输入协议验证，尚不是建图质量或运行速度对照。

对照入口要求显式权重路径、新输出目录和固定词表；保存解析后的配置、模型/输入哈希、上游提交与差异。使用顺序模式处理全部选中帧，以先验证完整输入与实例结果。该配置不代表上游并行模式吞吐。固定词表位于 `configs/prior_map/comparison_classes.txt`，来自前述 STRIVE 32 帧构建记录，不读取场景标注。

上游 `utils/object_detector.py` 的四处模型初始化异常原来记录日志后返回，留下半初始化实例。本地源码仅将这些异常分支改成重新抛出；运行记录保存对应 upstream.patch。STRIVE 运行代码不依赖这份修改。

YOLO 与 MobileSAM 权重已在已有工作空间找到；MobileCLIP-S2 与 FastSAM 权重下载已启动。下载、模型前向、完整 32 帧建图、输出适配和质量对照仍待完成。当前不把适配器测试计为 DualMap 复现成功。


## 全区域几何诊断与 DualMap 颜色接口核验（2026-10-30，续）

600 帧原始采集已完成 stride=8、残差阈值 0.05 m 的跨视角深度诊断，共 63,503 对有效投影记录。汇总按同一区域和跨区域分开，避免大量跨区域遮挡掩盖局部问题。

| 区域 | 投影样本数 | 前表面冲突 | 非深度边缘冲突 |
| --- | ---: | ---: | ---: |
| 0 | 4,535 | 9 | 0 |
| 1 | 2,622,224 | 2,871 | 88 |
| 2 | 660,228 | 13,895 | 86 |
| 3 | 4,798,497 | 63,125 | 571 |
| 4 | 4,775,027 | 128,745 | 212 |
| 5 | 4,535 | 10 | 0 |
| 6 | 4,535 | 5 | 0 |
| 跨区域 | 28,061,769 | 6,480 | 67 |

同区域累计仍有 957 次非边缘冲突；同一个像素可参与多对投影，以上不是独立物体计数。冲突集中于深度边缘只支持进一步检查栅格化/遮挡边界，不能证明全部残差由边缘引起。尚未据此宣布绝对几何或地图质量通过。

产物：采集目录的 `geometry_diagnostic.json` 与 `outputs/prior_diagnostics/all_islands_geometry_summary.json`。语义融合与几何诊断独立运行。

DualMap 的 MobileCLIP-S2、FastSAM 权重下载完成。检查上游 dataset 的 RGB 入口、CLIP 的 PIL 输入以及本机 Ultralytics 8.3.103 预处理后，确认 YOLO、SAM、FastSAM 的 ndarray 输入按 BGR 解释。上游直接传 RGB 会再次翻转通道。本地修复只在三个 Ultralytics 调用处做 RGB→BGR，保留 DataInput、CLIP 与点云颜色的 RGB 约定。对应红色像素边界测试覆盖三个调用和输入不变性。该变体将明确标记为协议修正版 DualMap，对照报告保留源码差异，不宣称原样复现上游。

对照入口同时关闭上游固定绝对高度 2.5 m 的演示天花板过滤；当前坐标原点不保证在地面，继续使用该阈值会引入不适用的删点。启用检测证据保存，并将交互动作配置隔离到本次输出目录。其余配置由 resolved_config.yaml 留档。

完整模型运行安排在当前 600 帧 STRIVE 构建结束后，避免 GPU 竞争影响建图耗时记录。下载完成、输入契约通过与实际模型/地图验收分别记录。


## 非边缘冲突根因定位（2026-10-30，续）

新增 `diagnose_prior_conflicts.py` 保留全部原始非边缘冲突，比较整数最近像素与亚像素逆深度插值。对于针孔相机观察到的平面，\(1/z(u,v)=a u+b v+c\)，因此逆深度插值可用于检验平滑表面的取样误差；不改变原始深度、位姿或验收阈值。

| 固定原始冲突集合 | 全部区域对 | 同一区域 |
| --- | ---: | ---: |
| 原始非边缘冲突 | 1,024 | 957 |
| 有效内部插值 | 734 | 667 |
| 插值后仍超出 0.05 m | 161 | 139 |
| 无内部插值（图像边界） | 290 | 290 |

同区域的 528 次冲突在插值后落入阈值；剩余最大约 0.353 m，跨区域最大约 3.484 m。未把插值结果覆盖原始诊断，也未删除失败点。

独立检查使用原场景 GLB 的双面三角形射线交点，不读取语义标注，不参与预测建图。第一版 Open3D 资产坐标未对齐 Habitat，保存为 `all_islands_ray_conflicts.json`，**该报告无效，仅保留排错历史**。实际读取维护 Habitat 的 stage template，确认资产 up=(0,0,1)、front=(0,1,0)、scale=(1,1,1)，见 `logs/prior_scene_orientation.log`；修正坐标后重跑。

有效报告 `all_islands_ray_pixel_conflicts.json` 中：

- 全部 1,024 条源像素射线命中原几何，交点深度与采集深度的最大差异约 1.3 mm，中位约 0.006 mm。这个结论只覆盖所检查射线，支持本次源深度和位姿链路。
- 161 次插值后仍失败的冲突中，112 次的**目标整数像素射线**与目标渲染深度在 5 cm 内一致，支持细结构亚像素取样导致差异的解释。3×3 深度梯度筛选没有覆盖所有细结构。
- 另 49 次目标整数像素的双面几何交点与渲染不同，交点三角形法向与射线同向，即背面命中。两个最大残差案例属于这一组，支持表面朝向/渲染可见性差异；没有据此修改预测地图。
- 161 次中，160 次的目标连续射线交点与投影点深度在 5 cm 内一致；剩余一次差异约 5.18 cm，保留为未充分解释项。

因此，当前证据不支持为了消除这些计数而整体修改相机坐标或地图位姿。后续几何验收应保留图像边界、像素取样、背面可见性与未解释误差的分层；对象类别与实例融合质量仍单独评测。

可视化为 `outputs/prior_diagnostics/all_islands_conflict_examples.png`。逆深度取样及偏置保留测试通过，采集相关测试共 12 项通过。完整原始冲突和全部投影报告均保留。


## DualMap 权重加载与证据导出（2026-10-30，续）

已有 dualmap 环境成功在 CPU 加载 MobileCLIP-S2（99,131,089 参数，含官方重参数化）、YOLOWorld、MobileSAM 与 FastSAM。日志 `logs/dualmap_cpu_model_load.log`；这验证权重和导入链路，尚未证明模型推理成功。GPU 推理保持排在 600 帧 STRIVE 构建完成后，避免两者竞争。

源码核对发现，上游 BaseObject 的 pickle 序列化没有保存 observations，重载时会把 observations 设为空。只转存这些 pickle 无法满足统一地图的原始证据关联要求。新增 `prior_map/dualmap_export.py`，在官方 end_process 后、对象仍在内存时导出其保留的观测：

- 保留原 UUID、预测类别、中心位置、点数及类别概率。
- 按 selected-frame index 映射回原始 frame_id，支持非连续帧号。
- 保存对象对应的原始 RGB、深度、mask、框、内参、相机位姿和观测视点。
- 导出器不补造被上游删除的观测；证据范围明确为 mapper 当前保留的列表。
- 跨目录移动后的重载测试验证 UID、中心、frame_id=25 的证据绑定和 mask 不变；两项输入/导出测试通过。

对照入口在新输出目录下生成 `bundle/`，保存完整构建记录和采集 manifest，供已有加载、查询与历史视点解析使用。上游地图 pickle 和检测中间结果另行保留。完整运行、导出及实际重载仍以待运行的单帧/32 帧产物验收，未将合成导出测试算作真实建图完成。


## 600 帧构建完成与独立质量审计（2026-10-30，续）

构建进程成功退出，`outputs/prior_bundles/wcojb4TFT35-all-islands-indexed/build_receipt.json` 状态 complete：600 帧，919 个预测实例，7,771 条观测，593 个带证据观察视点，构建耗时 1,776.09 s（约 29.6 分钟）。其中 7 帧没有接受观测，原始 capture 仍保留全部 600 帧。

地图包 233,380,765 bytes（约 222.6 MiB），包含核心对象/证据/视点索引 6,469,245 bytes（约 6.17 MiB）；全量原始采集约 216.22 MiB 单独报告。没有把图像外置成本排除在总成本之外。一次加载约 1.02 s，属于当前文件缓存条件下的单次测量。

复制至 `/tmp/strive-prior-all-islands-reload` 后，8,962 个文件 SHA256 全部一致。新进程重载并查询 cabinet，结果与原路径逐字段一致；返回 12 条受服务上限约束的候选，不代表全图只有 12 个 cabinet。

运行中资源采样：容器内存约 2.71 GiB，整卡显存 9,324 MiB、GPU 利用率 98%。这是采样值，包含整卡其他占用，不是峰值显存或独占性能基准。构建期间进行了 CPU 诊断，GPU 模型对照保持串行。

| 区域 | 帧数 | 帧处理耗时合计（s） | 接受观测 |
| --- | ---: | ---: | ---: |
| 0 | 8 | 16.02 | 54 |
| 1 | 136 | 288.12 | 1,250 |
| 2 | 72 | 169.77 | 760 |
| 3 | 184 | 595.66 | 3,175 |
| 4 | 184 | 649.99 | 2,384 |
| 5 | 8 | 24.47 | 75 |
| 6 | 8 | 24.12 | 73 |

各区域顺序处理，共享累计融合地图；不能仅凭区域耗时把差异归因于面积或某一种算法。完整记录保留每帧候选数和耗时。

内部证据审计：673 个实例有多帧支持，453 个包含不同观测类别，181 个存在同帧不相交 mask。上述是审查线索，部件/整体关系可能合法，不直接当作错误率。

独立 HM3D 语义审计覆盖全部 593 个带证据视图，其重渲染深度与保存深度逐像素一致。7,771 个 mask 中 6,804 个获得完整 mask 超过一半像素支持同一标注实例；890 个预测实例获得强标注支持，其中 264 个包含不同标注实例的强投票。标注的整体/部件粒度需核对，这不是已校准的实例错误率。

类别问题仍明显：769 个预测 door 的多数支持 mask 对应 wall，222 个 plant 对应 window。计数是逐帧 mask，不是独立物体数或完整分类准确率。**本轮确认工程流程扩展到全采集并可重载；地图语义质量未通过。** 真值只存在 `outputs/prior_diagnostics/all_islands_semantic_audit.json`，未回写地图。

主要报告：`all_islands_bundle_cost.json`、`all_islands_build_regions.json`、`all_islands_evidence_audit.json`、`all_islands_semantic_audit.json`，均位于 `outputs/prior_diagnostics/`。

### DualMap 实际运行状态

串行等待脚本曾因 zsh 的只读变量 status 失败；600 帧任务已成功完成，没有重启。直接续跑 DualMap 后，第一轮隔离运行明确报错 mobileclip 缺失。此前 CPU 加载使用了用户 site-packages 引入的旧源码路径；已将当前上游固定的 MobileCLIP 子模块 `1140b8d197e4ed7d56b3a92216ded98bb1c2ac87` 安装到既有环境，之后继续使用 PYTHONNOUSERSITE=1。

`outputs/prior_dualmap/wcojb4TFT35-smoke-isolated/` 的真实单帧流程成功退出，耗时记录约 4.28 s，最终 0 个实例、0 条证据。它证明检测/生命周期/导出可以走通，不能证明建图有效。零实例结果保留，未为通过测试修改上游稳定性规则。

随后启动完整 32 帧固定输入对照：`outputs/prior_dualmap/wcojb4TFT35-control/`，日志 `logs/prior_dualmap_control.log`。输入 RGB 改为可写拷贝，解决实际 PyTorch 对只读数组的警告，像素内容不变；适配器源码和 MobileCLIP 提交均加入运行记录。32 帧结果以最终构建记录和进程状态为准。


32 帧 DualMap 进程随后成功退出，但最终 **0 实例、0 证据**，构建记录约 6.21 s。日志包含大量空检测。这是无效建图结果，不作为速度收益，也不算质量对照完成。固定 frame 4 的直接检测探针在 CPU 能返回 YOLO/FastSAM 框，正在核查实际设备、模型调用路径与过滤。第一版同一模型对象依次传 cpu/cuda 会复用 Ultralytics predictor，不能作为设备对照；新的探针对每个设备重新构造模型并记录 predictor.device，保留两版日志以区分证据效力。


### 零地图根因定位更新

进一步核对完整 INFO 日志与固定输入复验后，**早先由终端空检测警告推测检测阶段导致零地图的方向未获支持**。新建预测器的实际 CPU/GPU 单帧结果一致到正常数值误差；完整 Detector 和五帧完整调用路径也能产生检测。原流程的 INFO 日志显示结束处理开始时已有对象，随后逐步清空。

保持输入、模型与配置不变，仅增加 `before_finalize.json` 的复验输出到 `outputs/prior_dualmap/wcojb4TFT35-control-lifecycle-audit/`：结束处理前 44 个对象，44 个均 observed_num=1，44 个均 is_stable=false；结束后 0 个对象。`stable_num=8`、`active_window_size=10`、`max_pending_count=5` 均保持官方配置。

上游 `LocalObject.update_status()` 对离开活动窗口的对象进行稳定性判断；未稳定对象进入 pending，超过次数后删除。`end_process()` 推进无新观测的虚拟帧以完成该状态转换。因此，本输入下零输出的直接原因是没有形成重复关联观测并被生命周期清理，而非完全没有检测。

尚未确定观测次数不足主要来自稀疏全景采集、实例匹配条件还是二者共同作用。下一步检查逐帧匹配并使用更连续的观测进行对照；不降低稳定性阈值或把清理前的单帧假设冒充最终地图。32 帧对照仍判为不适配当前输入、没有有效最终地图，不计算速度优劣或导航增益。


## 跨帧匹配对照与实际连续采集（2026-10-30，续）

增加可选匹配调试日志，不改关联算法。官方局部关联为
\[
s(o,m)=s_{spatial}(o,m)+\cos(f_o,f_m),\qquad s(o,m)>1.2.
\]
空间项是当前点云中距离参考点云最近点小于 downsample_voxel_size=0.02 m 的比例；空间项计算前还受包围盒重叠筛选。使用独立 debug handler 保存逐候选分数，避免上游 INFO handler 丢弃诊断。

- 原 32 帧：77 次候选关联，最大综合分数约 0.959，0 次超过 1.2；九成候选最大空间重叠为零。
- 同区域 192 帧密集离散扫描：545 次候选关联，5 次超过阈值，最大约 1.527。最终仍为 0 实例；清理前剩余 49 个对象全部 observed_num=1。更早曾发生关联，不应把“清理前全为 1”解释成整个运行从未关联。

### 连续采集器

新增 `backends/habitat/walk_capture.py` 与 `scripts/collect_hm3d_walk.py`，复用统一 observation writer 和场景绑定的 NavMesh 构建记录。参考采集只提供无任务的候选位置与传感器配置；先筛选当前网格可导航且同连通区域的位置，再按路径距离排序。

初始状态仅设置一次，其后每帧来自真实 `sim.step`。移动步长 0.25 m、转角 10°；greedy follower 执行前进/转向，观察扫描也用真实转向动作完成。保存 actions.jsonl 的前后帧、实际平移、碰撞及阶段；预算结束时保留不完整扫描组并标记 budget_exhausted。该采集器是 geometry-assisted 跟随器，不是 STRIVE baseline 探索，也不代表 P8 指令导航。

实测 `outputs/prior_capture/wcojb4TFT35-continuous-control/`：93 个动作、94 帧，其中 88 次左转、5 次前进，0 次碰撞，累计平移约 1.25 m。当前起点连通区域仅有 2 个可用参考位置，两个位置均到达并完成扫描。没有将请求的 3 个位置写成实际完成 3 个。

动作核验通过：frame_count=action_count+1；sim_step 连续；每条动作引用相邻帧；由保存底盘位姿差计算的平移与动作记录一致，最大步长约 0.2500003 m。报告 `outputs/prior_diagnostics/continuous_capture_action_audit.json`。跨视角深度诊断已完成，但本轮没有将小区域采集认定为全场景覆盖。

### 连续观测仍未让 DualMap 留下稳定地图

94 帧连续输入保持相同模型与关联/稳定性阈值。192 次候选关联中只有 3 次超过 1.2；最大空间重叠约 0.851、最大综合分数约 1.488。最终仍为 0 实例；结束前 30 个对象各有 1 次观测。因此“仅增加时间连续性即可解决零地图”的假设未获得支持。

尚需检查分辨率/点云间距与关联半径的尺度匹配：针孔图像在深度 z 处横向像素间距约 z/f_x；当前 f_x=160，z=4 m 时约为 2.5 cm，已大于默认 2 cm 近邻半径。这个量级关系是待验证解释，不能直接据此放大阈值或归因全部失败。后续应保留具体对象/点云匹配证据并做固定输入对照。

同一 94 帧输入的 STRIVE 共享建图已启动，输出 `outputs/prior_bundles/wcojb4TFT35-continuous-control/`，日志 `logs/prior_strive_continuous.log`。它用于验证采集复用和同输入建图对照；完成状态以实际记录为准。P7 的 baseline 探索对照、P8 历史地图指令执行仍未完成。

### 连续输入建图完成与点采样消融

STRIVE 上述运行已完成：94 帧、78 个预测实例、1,183 条观测，227.51 s。独立进程重载后 cabinet 查询返回 11 个带证据的预测候选，记录为 `outputs/prior_diagnostics/continuous_strive_cabinet_query.json`。这些数量表示产物规模，不表示语义正确率或导航成功率。

进一步核对上游：DualMap 默认 `pcd_sample_ratio=0.04`，随机采样后再 DBSCAN，关联半径为 2 cm。官方 Replica 输入配置为 1200×680、fx=600；本次是 320×240、fx=160。保持比例采样会使低分辨率输入的空间点密度显著降低。

新增 `scripts/diagnose_prior_sampling.py`：使用已保存的 193 个有效 mask，对同一观测独立抽样两次，计算
\[
s(A,B)=\frac{1}{|A|}\sum_{p\in A}\mathbf{1}[\min_{q\in B}\|p-q\|<0.02].
\]
4% 采样的平均空间重叠 0.2969，58.03% 的 mask 超过 0.2；20% 为 0.6834、89.64%；100% 自身对照均为 1。该实验隔离采样影响，省略聚类、包围盒和外观，属于同观测重复性诊断。不能把它写成实际跨帧关联成功率。详见 `outputs/prior_diagnostics/continuous_sampling_repeatability.json`。

随后使用完整官方建图链路，固定输入、随机种子 17、模型、词表、关联阈值和稳定性参数，仅改变采样比例：

| 采样比例 | 候选关联数 | 综合分数 > 1.2 | 最终实例 | 最终证据 | 构建时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.04 | 192 | 4 | 0 | 0 | 10.07 s |
| 0.20 | 192 | 68 | 1 | 16 | 10.96 s |

结果支持点采样密度是本输入关联不足的重要因素，尚未证明所有零地图问题均由它造成。随机种子固定改善可复现性，不保证所有 GPU 算子按位确定。构建入口新增显式 `--pcd-sample-ratio`，省略时沿用上游配置；20% 是消融设置，未设为生产默认。

产物为 `outputs/prior_dualmap/continuous-seeded-004/` 和 `continuous-seeded-020/`，对照汇总为 `outputs/prior_diagnostics/continuous_sampling_ablation.json`。后者保留一个标注 shelf 的预测实例，尚未进行语义真值验收。导出后将地图复制至 `/tmp/strive-dualmap-continuous-seeded020-reload`，53 个文件逐一 SHA256 相等；新进程成功重载并查询 shelf，记录 `dualmap_continuous_relocated_query.json`。这首次验证了实际非空 DualMap 产物的证据导出与跨目录重载，不代表实物重定位。

本轮相关协议与输入测试 14 项通过。首次 pytest 启动被环境中无关 Dash/Flask 插件的 Jinja2 兼容问题中断；关闭第三方插件自动发现后执行项目测试，无需修改项目或环境依赖。共享建图的在线导航回归、P7 baseline 探索对照、P8 历史视点执行及真实手机验收继续保持待完成状态。

### 连续历史视点与独立语义验收

核对两种采集器均使用 `[0, sensor_height, 0]` 刚性相机安装后，`backends/habitat/prior_viewpoints.py` 允许 `continuous_walk` 进入原有解析链路。没有新增运动所有者或改变底层导航。94 帧相机位姿转换得到的底盘位置与保存的 `agent_position_habitat` 完全一致（最大位置误差 0 m）；维护中的 Habitat 环境使用已绑定场景哈希的 NavMesh 核验，94 个观察点全部从记录起点可达。报告为 `continuous_base_pose_check.json` 与 `continuous_viewpoints.json`。这是坐标和路径准入检查，尚未执行历史目标导航。相关 12 项采集测试通过。

DualMap 20% 采样产物的独立语义诊断发现：16 帧深度对齐全部通过，16 条 mask 均多数对应同一个 GT 实例 794；但预测标签是 shelf，标注类别为 curtain。空间关联稳定与语义正确是两个不同指标。本次采样实验改善了前者，尚未解决后者。报告 `dualmap_continuous_semantic_audit.json` 保持独立，未修改预测地图。

同输入 STRIVE 地图：94 帧深度对齐通过，1,183 个 mask 中 1,129 个具有多数标注支持；78 个预测实例中 34 个存在多个 GT 实例投票，需逐例检查合并/分割与部件语义。显著类别对应包括 door→wall 387 条、shelf→wall 113 条、window→curtain 56 条。均为逐帧观测数，不能作为实例错误率。报告 `strive_continuous_semantic_audit.json`。

当前 12 类词表只是固定对照词表，缺少大量环境类别。后续词表消融采用 DualMap 仓库原有 `config/class_list/gpt_indoor_general.txt` 全部 100 类，而非按当前标注添加词语。固定 94 帧、采样比例 0.2、种子 17 和关联/稳定性参数；输出 `outputs/prior_dualmap/continuous-seeded020-indoor100/`。该消融只检验通用词表覆盖对本输入的影响，不能直接与 12 类 STRIVE 的运行成本比较，也不改变默认建图器。

该词表消融已完成，耗时 11.14 s，最终 4 个预测实例、65 条证据。独立诊断 `dualmap_indoor100_semantic_audit.json` 的深度对齐通过，65 个 mask 均有多数标注支持。四个预测实例分别对应两个 curtain、一个 door、一个 desk；每个预测 UID 的全部多数投票都落在单一且不同的 GT 实例上（30、19、8、8 条）。地图最终标签为 curtains、curtains、door、desk，与标注语义相符。

逐帧类别仍有波动：窗帘证据中还包含 blinds、unknown、curtain rod、towel；桌子有 cabinet，门有 wardrobe。不能把最终投票标签正确表述为所有观测分类正确。也没有完整对象召回率、其他房间和跨场景验证，因此本轮只证明该小区域内四个保留实例的一致性。

下一步优先将相同的任务无关通用词表用于共享 STRIVE 建图对照，隔离检测词表与融合算法；再在已有更大覆盖输入上验证采样设置和词表是否可迁移。历史导航继续遵循当前取证与任务完成验证，不使用本节 GT 标签作运行时输入。

### 词表协议根因修正：unknown 占位项

进一步源码核对发现，以上 12 类 DualMap 对照存在接入缺陷：上游 `Detector.unknown_class_id = len(classes)-1`，FastSAM 未分类 mask 直接使用这个 ID。12 类输入的末项是 shelf，因此部分未知区域被编码为 shelf。之前将 shelf→curtain 单独解释为类别覆盖不足的结论撤回；这些运行同时混入了未知类别索引错误。采样消融的数值记录保留，但它证明的是旧接入配置下的关联变化，需在修正后的配置复验。

上游 `gpt_indoor_general.txt` 实际包含 101 个非空条目：100 个具体概念和末尾 unknown。先前称为“100 类”是具体类别数，不是实际输入条目数。该通用词表运行的 unknown 位置正确，因此四个最终实例的独立诊断证据仍成立；与错误的 12 类接入对比不能归因于词表覆盖这一单因素。

修复 `prior_map/dualmap_input.py` 的词表桥接：明确在具体类别后保留唯一 unknown；运行入口将有效词表写到输出目录 `classes.txt`，记录完整 `effective_classes` 与 `unknown_class_id`，原词表保持只读。测试覆盖末尾真实类别 shelf 与已有 unknown 两种输入，3 项输入/导出测试通过。修正后的 12 类、20% 采样复验输出为 `outputs/prior_dualmap/continuous-012-unknown-corrected/`。

复验已完成 94 帧，保留 1 个实例、16 条证据，最终标签为 unknown（索引 12），而非之前的 shelf（索引 11）。这直接验证了占位协议修复的效果；不能将 unknown 计为已识别目标。该运行与 STRIVE 共享 GPU，耗时不用于性能比较。

STRIVE 新增互斥的 `--classes-file` 入口，并通过整行多词类别传参检查。当前 `wcojb4TFT35-continuous-indoor100` 运行直接读取上游原始 101 项文件；它属于原始词表输入诊断，包含 unknown 提示词，不能称为纯 100 个具体概念的最终对照。后续正式共享词表应为具体概念列表，由 DualMap 适配器独立添加 unknown 状态项。运行产物保留，不回写配置伪装为另一实验。

共享词表已生成于 `configs/prior_map/indoor_concepts.txt`，保留上游 100 个具体概念的顺序，仅移除最后一个 unknown。来源、提交、源文件哈希与转换后哈希见 `indoor_concepts.provenance.json`，未使用场景标注补充词语。

### 连续采集预算边界实测

`outputs/prior_capture/wcojb4TFT35-budget-boundary/` 使用维护镜像、同场景与同一已绑定 NavMesh，实际执行 2 个转向动作。产物包含 3 帧 RGB-D，`termination=budget_exhausted`，全景组为 `[0,1,2]` 且 `complete=false`。逐项检查索引引用、帧数和动作数均通过，报告 `outputs/prior_diagnostics/walk_budget_boundary.json`。采集文件写完的 status=complete 与路线完成由不同字段表达；该样本没有完成预定路线或空间覆盖。

### STRIVE 原始词表诊断完成

原始 101 项词表（100 个具体概念加 unknown）的 STRIVE 运行已完成：94 帧、97 个预测实例、1,563 条观测，342.06 s，最终 unknown 标签实例为 0。由于运行中与短时其他 GPU 任务重叠，这个耗时只作本次构建记录，不作为严格性能对照。

`strive_raw101_semantic_audit.json` 深度对齐通过；1,465 个 mask 有多数标注支持，97 个预测实例中 36 个有多个 GT 投票。door→wall 仍有 462 条观测；增加词表没有消除这一问题。drawer→desk 98 条、shelf unit→bookshelf 60 条同时提示部件/整体和类别粒度差异，需要逐项评审，不能把所有非同名类别都计为错误。

正式共享具体概念列表的运行已启动：`outputs/prior_bundles/wcojb4TFT35-continuous-concepts100/`，日志 `logs/prior_strive_concepts100.log`。本轮只移除 unknown 状态提示词，保持原始采集、几何与融合配置。最终结论待完成记录和独立诊断，不将前一组产物改名冒充该对照。

### 历史查询到实例视点的绑定

历史视点解析新增 `object_uid`，CLI 对应 `--object-uid`。输入为现有地图查询返回的历史实例 ID；解析仅保留该实例的 observation_id，再执行原有底盘转换和 NavMesh 检查。输出继续标记历史来源，不将它注册为当前可见对象，不产生完成事件。

实际 desk 查询的 UID 在原 100 类通用词表 DualMap 地图中对应 8 条证据，解析后得到 6 个可达视点；集合核验确认 8 条目标证据全部保留、其他对象证据全部排除。报告 `dualmap_desk_viewpoints.json`、`desk_viewpoint_evidence_check.json`。这完成查询→关联证据→可达视点，尚未完成运动提交和到达取证。

运行接口核对发现：先验包是右手 Z-up，现有 Habitat mapper 使用 `[x,z,y]` 加初始相机平移偏置的局部坐标；运动入口要求 `mapper_local`。现有 `navigation/action_controller.py::habitat_waypoint` 还使用当前底盘高度，不能直接代表历史视点的跨层高度。下一步绑定应显式转换坐标并核验现有局部导航器的高度能力，不能仅把 frame_id 改成 mapper_local 后发送。

### 扩展连续路线

使用已有全场景无任务位置清单作为参考，在当前起点的连通区域内按路径距离选择 5 个位置。`outputs/prior_capture/wcojb4TFT35-walk-expanded/` 实际完成 239 个动作、240 帧、5 个完整全景，累计移动 4.250001 m、0 碰撞，`planned_route_finished`。这是几何引导路线，尚不是 baseline 自主探索，也不证明多房间覆盖。

逐动作核验前后帧、sim_step 与底盘位置差。首次用 float64 重算时，相对采集器 float32 范数最大差 1.25e-8 m；按 Habitat 原生 float32 重算后逐项相等。报告 `walk_expanded_action_check.json` 明确记录计算精度。相关 12 项采集测试通过。

### 历史底盘位置进入现有运动接口

`navigation/action_controller.py` 与 `backends/habitat/motion.py` 增加显式 `map_z_up` 位置支持。转换为 Habitat 世界坐标为 `(x_map, z_map, -y_map)`，保留历史底盘高度；对应 geodesic 起点从当前模拟器底盘状态读取。原 mapper_local 分支保持已有语义。两种坐标最终都交给同一个原局部 planner 和 MotionSupervisor，没有新增运动所有者。

新增运动测试检查：当前 Habitat 高度为 1 m 时，历史底盘目标 `(4,7,5)` 必须作为 `(4,5,-7)` 传给路径查询和 planner；waypoint 到达仅报告执行完成，不发送任务 STOP。尚未验证实际跨层轨迹、最终观察朝向或任务完成，不把此接口测试称为 P8 验收。

在已有 UrbanNav 测试环境执行运动、guided navigation 和导航根因回归共 35 项通过。默认宿主 Python 缺 pydantic，维护 Habitat 镜像缺 pytest，因此未向它们安装额外依赖。两项旧融合测试原来 AST 提取 mapper 同名转发方法，迁移后产生同名递归；现改为测试共享融合函数，保留图像与框同步的原断言。guided 测试替身接受新增 frame_id 关键字。现有 Python 依赖仅有 Pydantic 配置弃用警告。

正式 100 个具体概念 STRIVE 构建已完成，94 帧、97 个实例、1,561 条观测、339.34 s。耗时包含与短时采集任务共享 GPU 的时段，作为本次记录。语义诊断输出 `strive_concepts100_semantic_audit.json`，以实际报告为准。

诊断已完成且深度对齐通过：1,464 个 mask 有多数标注支持，97 个预测实例中 36 个有多个 GT 投票。door→wall 仍为 462 条，shelf unit→wall 为 91 条。与原始 101 项诊断相比，移除 unknown 没有解决主导误识别，因此后续重点转向具体 mask、框与原始图像的一致性及融合关系；继续增加词表不足以解释或修复当前问题。

### door→wall 的分阶段证据

已查看 frame 3 / detection 1 和最高分的 frame 26 / detection 0 的原图及 mask。前者为远处木饰面窄条，后者为图像右缘的木墙区域；分割基本沿检测区域展开。462 条多数 mask 对应 wall 的观测中，458 条检测框本身也有超过一半像素对应 wall，因此主要错误在检测/类别阶段已经存在，不能只归因于 SAM 扩张或后续点云融合。报告 `door_wall_stage_diagnosis.json`。

同一输入的反例检验：错误组 255/462 个框在 2 像素边界内，正确 door→door 组 56/80 个也贴边；错误组全部置信度低于 0.5，正确组也有 79/80 条低于该值。直接删边缘框或提高至 0.5 会损失大量正确证据，本轮未采用这些阈值调整。以上为逐帧观测统计，不代表独立物体错误率，也不把标注类别粒度当作唯一视觉真值。

扩展路线采用修正后的 unknown 协议、同一 100 具体概念词表、20% 点采样：`outputs/prior_dualmap/walk-expanded-concepts100/` 已完成 240 帧、13 个实例、442 条证据、25.18 s。质量诊断单独输出 `dualmap_expanded_semantic_audit.json`，完成状态以实际报告为准。

扩展诊断已完成：深度对齐通过，440/442 条 mask 有多数标注支持；13 个预测实例中 3 个有多个 GT 投票。具体表现为 lamp/ceiling fan 交叉投票及 bed/blanket 部件混合。另有 3 个 pillows 预测 UID 都对应同一个 GT pillow 实例 311，提示拆分；一个 window 对应 wall，一个 light switch 对应 wall（需进一步视觉核验标注粒度）。两个独立 window、两个 bedside cabinet 等实例具有一致标注支持。因此，小样本四实例的良好结果没有消除扩展后的合并、拆分和类别问题；保留实例证据供当前取证使用，尚未判为全场景语义验收通过。

### 历史实例视点的真实运动集成测试

新增 `scripts/check_prior_motion.py`，从预测地图的实例 UID 解析可达视点，通过 `HabitatExecution` / `HabitatMotionController` / `MotionSupervisor` 执行实际 `sim.step`，保存每步 RGB-D、内参、相机位姿、底盘位姿和动作。该脚本使用 Habitat-Sim greedy follower 的接口适配进行运动集成检查，不是完整 STRIVE ObjectNav driver；没有任务解析、Agent 决策或最终语义验证。

从预测地图查询 nightstand 得到两个候选，选择第一个查询结果的实例 UID `808b2086-e252-455e-84df-e170dfa84513`。为覆盖真实平移，诊断明确选择该实例距起点最远的可达历史视点（2.183 m 路径距离）；该选择只用于测试，不替代运行时视点排序。

- `outputs/prior_motion/nightstand-integration/`：12 个动作、约 2 m 实际路径、0 碰撞，终态 reached，距目标视点 0.1909 m；未对齐朝向。
- `outputs/prior_motion/nightstand-aligned/`：同一起点和目标，位置到达后再通过同一运动所有者执行 3 次转向，共 15 个动作；最终朝向误差约 -0.000162°，距视点 0.1909 m。保存 16 帧新 RGB-D。

已经查看到达后的第 15 帧，画面包含床及床头柜。该人工视觉检查不是自动任务成功判定；receipt 明确 `semantic_verification=false`。视点到达没有调用任务 STOP。当前实验证明历史实例→视点→实际运动→朝向对齐→新观测可执行，尚未证明独立新指令的完整 P8 闭环、baseline 局部导航行为等价或实物能力。

### 新指令编译与当前证据验证

`scripts/query_prior_bundle.py` 增加显式 `--compile-instruction`，调用原 `instruction.compiler.compile_instruction`，再通过原 runtime_plan 与 PriorMapQueryService 检索。普通概念查询默认保持离线；自然语言分支不使用 dataset_target 或 benchmark 元数据。实际豆包关闭 thinking 的调用将 `Find a nightstand.` 编译为一个 min_count=1、无附加约束的 TaskSpec，查询到两个历史 nightstand。记录 `compiled_nightstand_query.json`。

将实际运动后的第 15 帧原始 RGB-D、内参、位姿重新封装为 `outputs/prior_capture/nightstand-arrival-current/`，manifest 绑定运动 receipt 的 SHA256。原 STRIVE 检测/分割/投影生成一帧当前候选包 `outputs/prior_bundles/nightstand-arrival-current/`：1 个候选、1 条证据，使用独立 UID `nightstand-arrival-current:frame15:detection0`，没有继承历史实例身份。此处重新检测沿用任务词 nightstand，不读取 GT。

新增 `scripts/check_prior_arrival.py` 用原 FinalInstructionVerifier 与 TaskProgress reducer 核验当前证据。该诊断显式限于单个无附加约束的对象任务，避免把缺失的关系/数量执行能力包装成完整支持。图像框和距离来自同一当前 mask；平面最近点距离沿用 baseline 的度量形式，支持点为本次 mask 的有效深度（非历史融合点云）。停止距离显式传入 1.0 m；它是本次诊断合同，不是对 HM3D evaluator 成功的替代。

豆包实际视觉判断支持 nightstand 语义，当前距离为 2.5267 m，硬距离条件不满足，最终结果 `need_better_view`。reducer 的 count=0、completed=false，模式为 `better_view_for_verified_pair`，反馈包含继续接近并保持可见。输出 `outputs/prior_verification/nightstand-arrival/verification.json`；未发送任务 STOP。

这一串行诊断复用了既有运动测试的到达帧，编译步骤发生在该次运动之后；因此它验证组件的数据衔接和失败分支，不宣称已完成“新指令触发的一次完整独立导航 episode”。下一步仍需把编译后的任务、当前候选接近和再次取证纳入同一连续运行过程，并保持单一运动所有者。

### 当前候选接近提案与实际图像尺寸

复用 `planning/viewpoint_policy.py::build_check_again_viewpoints` 时发现其投影范围固定为 640×480，而当前采集为 320×240。现增加显式 image_width/image_height，并将同一尺寸传给原 score_projection；在线调用方从当前 RGB 读取尺寸。算法、权重、路径插值和排序保持原实现。新增测试给出三个已投影点，其中一个位于 320 像素右边界之外，验证 visible_ratio=2/3 且剩余点中心评分为 1。维护感知环境的 7 项测试通过，UrbanNav 环境的 35 项运动/guided/根因回归通过。

新增薄绑定 `scripts/resolve_prior_approach.py`：读取一帧当前候选的真实 mask/depth、运动结束底盘位置，显式转换 canonical map 与 baseline mapper 轴向，查询 NavMesh 路径，再调用原补拍提案及排序函数。几何不来自历史对象位置；motion receipt 哈希用于确认当前候选来源。当前路径使用同楼层底盘高度，尚未提供跨层接近策略。

`nightstand_approach_proposal.json` 的选中提案：路径距离 1.8463 m，预测对象平面距离 0.7708 m，到达容差 0.2 m，满足 d_object+rho≤1.0 m；但 visible_ratio≈0.0501，score≈0.2456。原排序优先选择距离可完成提案，这不代表当前视图已通过最终视觉验证。

当前采集相机高 1.25 m，目标低矮；水平观察的近距离提案可能把主体投影到画面下方。修复尺寸后该问题显式暴露，不能通过恢复 640×480 范围把不可见点计为可见。后续应验证俯仰取证或其他可见视点；本轮尚未执行该接近提案，也未接受任务完成。

### 接近提案执行与俯仰取证

`check_prior_motion.py` 增加显式当前接近提案输入和 look-at 点。诊断使用原 current-candidate baseline 提案，朝向对齐后以 Habitat 实际 look_up/look_down 动作调整相机，全部动作仍经同一 MotionSupervisor。俯仰误差按当前相机前向与当前对象点方向计算，停止容差为离散 10°动作的半步。

`outputs/prior_motion/nightstand-approach-pitched/` 已完成 20 个动作、约 1.7588 m 路径、0 碰撞，距接近视点 0.0868 m；航向误差约 0.3105°，俯仰误差约 -4.8432°。保存 21 帧 RGB-D，最终图像可见床头柜主体。这是从上一段结束位置重新初始化的独立接近诊断，尚未合并为同一进程的连续任务 episode；不将两段相加后称为完整自主导航轨迹。

新增 `extract_prior_arrival.py` 复用统一帧写入协议提取最后实际观测，并绑定运动 receipt SHA256。当前新帧经原 STRIVE 检测/分割生成 `outputs/prior_bundles/nightstand-approach-current/`；独立候选由当前 mask 支持 5,034 个有效深度点，平面最近距离 0.8572 m，满足显式 1 m 距离条件。

新的豆包视觉复验被自动审批拒绝：该审核要求对这张新图像补充明确的外发授权。已向用户请求授权，未重试或绕过拒绝。`check_prior_arrival.py --prepare-only` 将具体候选图像和输入保存到 `outputs/prior_verification/nightstand-approach-review/`，没有模型请求、语义结果或任务 STOP。当前只能报告几何条件满足，不能将先前远距离图像的语义判定用于本帧完成。相关 16 项运动/采集测试通过，脚本编译和 diff 检查通过。

### 等待外发授权期间的本地验证与 P7 准备

`nightstand_pitched_alignment_audit.json` 在本机 Habitat 按保存相机位姿重渲染：最大深度差 0.001276 m，超过 1 cm 的像素比例为 0，alignment_valid=true。当前 mask 的多数标注为 bedside cabinet，与预测 nightstand 对应。报告严格保留在独立离线诊断路径，未回写运行时验证器或 reducer，也未用作绕过豆包外发授权的任务成功证据。

P7 的房间优先探索选择器移除对非空 agent.task 的强依赖：明确 task=None 的巡视状态时，将 plan=None 传给原确定性房间距离策略。原指令模式继续传原 task.plan，房间切换与局部选点算法不变。新增测试验证当前房间耗尽后，在无指令且关闭语义房间选择的条件下调用原距离策略，未调用模型策略。36 项运动/guided/根因测试通过。

此项只完成巡视的策略选择入口准备；传感器采集、mapper 任务无关初始化、动作记录与真实 baseline 巡视对照尚未接通，P7 保持未完成。新候选图像的外发授权仍待用户回复。

### P7 感知入口契约与剩余任务依赖

核对 `ImagePerceiver.perceive` 后移除面积过滤的裸 except：合法空检测保持零长度张量，类别、框和 mask 索引错误直接传播到采集驱动，保留失败原因。巡视的 `target=None` 只返回词表检测，目标候选组为空。新增两项测试覆盖无任务词表检测、合法空观察与错误契约的区分；维护中的 `strive-hm3d:local` 容器执行该文件 9 项测试全部通过，Python 编译及 git diff --check 通过。测试使用合成张量，不调用模型服务。

当前真实巡视接入仍存在以下具体依赖：`HM3D_Objnav_Agent.reset` 从 episode.object_category 生成目标并读取 distance_to_goal；`update_trajectory` 将 ObjectNav 指标与几何更新绑定；自适应观察读取 task.spec.revision，并使用目标出现作为提前停止条件。下一步应让巡视使用明确的预算/覆盖终止条件，沿用全景观察与原房间距离策略，并将采集记录与任务指标分开。当前尚未增加巡视运行入口，也未启动 baseline 巡视实验。感知错误传播修复只证明异常可定位，不证明建图语义质量提升。

### P7 无任务 baseline 巡视入口与首次真实运行

新增 `backends/habitat/patrol_runtime.py` 与薄入口 `scripts/collect_hm3d_patrol.py`。原 agent 的 `reset(..., patrol=True)` 使用 `target=None` 初始化词表检测，全景观察复用原 mapper 与对象融合；轨迹更新在巡视模式跳过 ObjectNav 指标。新增观察记录回调将实际传感器的 RGB、米制原始深度、K 和相机位姿写入统一 `capture/` 协议。模拟器 scene/start 从冻结 episode 文件读取，TASK.SENSORS 与 TASK.MEASUREMENTS 显式清空；场景起点来源写入 manifest。当前入口针对维护中的 Habitat 0.2 配置。

运行循环复用当前房间优先与确定性房间距离选择，通过同一 HabitatExecution/MotionSupervisor 执行。到达并完成全景观察后才标记节点覆盖；一次全景需要 12 次转向，剩余预算不足时终止并保留未观察节点。返回 frontiers_exhausted、step_budget 或具体 motion 终态，task_complete 始终为 null。巡视停止不发送任务 STOP。

首轮 `baseline-smoke-001` 完成全景检测后因空候选精修仍读取 task.spec 退出。原错误日志完整保留。`refine_candidates` 现对空候选返回空证据，避免初始化不存在的任务上下文；有效候选仍走原流程。新增测试直接验证无 task 的空候选调用。

断网容器重跑 `outputs/prior_patrol/baseline-smoke-002/` 成功：40 步预算实际记录 29 帧（初始帧加 28 个动作），一次完整全景、一次 frontier 到达，行走 3.2499996 m。到达后只剩 11 步，结束于 step_budget，当前节点未标记完成观察。29 帧连续编号及 RGB/depth 文件引用检查通过。几何诊断生成 225 对跨视角统计，仅作为一致性诊断，尚未判为完整几何验收。

相关控制回归 41 项通过；维护感知容器的 11 项集成测试通过，包含真实 reset/轨迹更新在没有 current_episode 和任务指标条件下完成初始观察的检查。新的 80 步测试用于验证到达后的再次建图和下一轮选点，最终结果追加记录。当前结果证明巡视入口运行，正式同传感器、同起点、同预算的采集过程对照仍待完成；未据此声明地图语义质量提升。

80 步真实测试 `outputs/prior_patrol/baseline-smoke-003/` 已退出成功，耗时 110.97 s：80 帧（79 个实际动作）、3 次完整全景、3 个 frontier 提交，其中前两个到达并完成再次观察；第三个在预算结束时由 finally 取消，行走 7.2499995 m。终止为 step_budget。帧编号和原始文件引用检查通过；含几何诊断的 capture 目录为 121,465,789 字节。几何诊断包含 1,486 对跨视角统计。该次 decisions 日志没有保存最后取消终态，运行器随后补充取消回执记录，并通过预算取消回归测试；保留本次原日志，不追补虚构回执。更新后的巡视/运动 9 项针对性测试通过。

目前 P7 已有真实 baseline 连续巡视，下一步为其与几何引导路线建立相同场景、起点、传感器及预算的正式对照，并用同一建图器比较覆盖与实例质量。当前 640×480、272 类 baseline 感知配置与之前 320×240、100 类独立建图输入存在差异，因此现有两组结果只证明各自链路可运行，不能直接作为策略优劣结论。

### P7 匹配采集配置与对照检查

`collect_hm3d_patrol.py` 增加显式 seed、完整 Habitat 配置落盘、实际运行 NavMesh 导出以及 `capture_profile.json`。profile 冻结相机分辨率、HFOV、高度、K、深度截断范围、起始位置/旋转、动作步长、扫描方向、到达半径与动作预算。运行网格回执明确 scope 为 exported_active_mesh_with_runtime_agent_dimensions，记录实际机器人尺寸与网格哈希，不把它标为全参数重新构建的 NavMesh 回执。

`collect_hm3d_walk.py --capture-profile` 消费同一 profile：强校验场景和 NavMesh 哈希、相同动作预算，加载实际起点/朝向，保持相同深度截断及完整扫描预算规则。全景步数由 360/turn_degrees 计算，要求转向步长整除一周；默认几何采集仍保留原 10°、36 次转向设置。

新增 `compare_prior_captures.py` 做离线匹配检查：profile 相等后核对真实首帧位姿、K、RGB 和深度，再统计统一 stride/voxel 下的观测表面体素。深度处于 min/max 截断端点时不计入表面。该统计是采样表面覆盖，不能解释为房间地面面积或对象召回率。21 项协议/匹配/巡视测试通过。

`matched-baseline-001` 完成 80 帧、3 次全景、7.25 m，但首版 profile 遗漏了 planner 到达半径；已补充实际 0.15 m，重跑 `matched-baseline-002`。几何对照原默认半径为 0.2 m，因此不将旧配置视为完全匹配。当前起点来自冻结 episode，目标传感器和评测指标禁用；几何路线使用独立覆盖点和 NavMesh，其几何辅助来源保留在 manifest。

匹配采集已完成：`matched-baseline-002/capture` 与 `prior_capture/matched-geometric-001` 使用同一 active NavMesh、640×480/79°/0.88 m、0.25 m 前进、30°右转扫描、0.15 m 到达半径和 79 动作预算上限。首帧实际 camera_to_map/K/RGB 完全匹配，深度最大差 0 m。baseline 80 帧/79 动作/7.25 m，几何采集 77 帧/76 动作/4.00 m；几何采集剩余预算不足完整全景而停止。baseline 的第三个导航目标在预算退出时有 preempted 回执。

统一 stride=16、voxel=0.1 m 的采样表面统计分别为 17,819 和 12,144 个体素，交集 8,552、并集 21,411。报告 `outputs/prior_diagnostics/matched_patrol_geometric_001.json`。这是同预算上限的单场景采集对照，不是严格相同已花动作数量，也不支持一般化的策略优劣或对象质量结论。

复放建图前确认 Habitat-Lab 的 min/max 截断端点代表量程限制，不能直接反投影成实例表面。`capture_builder` 根据已记录的 capture_profile 排除这些端点，保留原深度文件；无该截断元数据的既有原始 Habitat-Sim 采集沿用原有效深度规则。receipt 增加 depth_policy。两组已按同一 100 类词表、5 cm 融合分辨率串行启动 STRIVE 离线建图，产物为 `prior_bundles/matched-baseline-concepts100` 与 `matched-geometric-concepts100`，日志分别为 `logs/prior_matched_baseline_build.log`、`logs/prior_matched_geometric_build.log`。建图完成后还需重载、证据审计和实例质量对照，当前不提前宣告通过。

### 同条件地图复放与独立实例审计

baseline 的 80 帧已完成统一 100 类词表复放：200 个预测实例、2,003 条观察证据、337.33 s。新进程 `query_prior_bundle.py` 重载并完成 chair 概念查询，返回 12 个带证据的假设；这不是 12 个真实 chair 的证明。`matched_baseline_evidence_audit.json` 报告 164 个多帧实例、107 个多标签实例、43 个关联同帧不相交 mask 的实例。多标签可能来自整体/部件表述，需进一步诊断，不能直接计作错误率。

新增轨迹对照图 `outputs/prior_diagnostics/matched_capture_paths.png`，两栏统一坐标范围；灰色为保存深度采样表面。该例 baseline 进入相邻区域，几何路线主要在起始区域补充观察。几何采集完成 4 组全景，baseline 完成 3 组全景；扫描密度与区域覆盖的差异需结合后续实例质量判断。

审计适配：`diagnose_prior_instances.py --scene` 接受重定位后的场景资产路径并校验 capture SHA256；新巡视 manifest 记录 scene_path。审计器根据 capture_profile 对渲染深度应用实际 min/max 截断后比较，不将传感器截断误判为位姿不一致。独立 oracle 诊断只写单独报告，预测地图保持只读。旧 raw Habitat-Sim 输入继续比较原始渲染深度。16 项采集/匹配测试及编译检查通过。几何组建图仍在运行，完成后再生成并列质量结论。

### P7 首个匹配场景的完整结果

两组统一建图均结束；classes、voxel_m、max_depth_m、depth_policy、源文件哈希和模型哈希逐项一致。两份地图在新进程重载并执行 chair 概念查询，文件引用检查通过。汇总为 `outputs/prior_diagnostics/matched_capture_map_comparison.json`。

| 指标 | baseline 巡视 | 几何路线 |
| --- | ---: | ---: |
| 保存帧数 | 80 | 77 |
| 预测实例 | 200 | 123 |
| 观察证据 | 2,003 | 1,506 |
| 离线建图时间 | 337.33 s | 272.65 s |
| 地图包大小 | 124,365,097 B | 117,671,209 B |
| 有多数 mask 标注支持的预测实例 | 199 | 123 |
| 跨 GT 实例投票的预测实例 | 41 | 25 |
| 至少一次多数 mask 支持的 GT ID | 94 | 64 |
| 与多个预测 UID 关联的 GT ID | 47 | 34 |

语义诊断单独保存，预测地图保持只读。两组都通过既有深度对齐门槛（每帧超过 1 cm 的像素比例 ≤0.1%），实际最大比例均约 0.00326%。仍保留稀疏大误差：baseline 最大 3.6421 m，几何组最大 2.2715 m；超过 1 cm 的像素总计 24/27，其中深度边缘附近 22/21、非边缘 2/6。报告为 `matched_baseline_semantic_edges.json` 和 `matched_geometric_semantic_edges.json`。门槛通过不代表每个像素完全一致，非边缘残差尚未逐点解释。

baseline 的跨 GT 投票为 41/199（约 20.6%），几何组 25/123（约 20.3%）；这属于实例关联复核指标，不能直接解释为对象错误率。两组仍有 plant→window（分别 172/52 条强支持 mask 观测）、drawer→table 等标签对应；前者需视觉检查，后者还包含整体/部件粒度差异。GT ID 统计包括结构和物体标注，不能作为任务目标召回率。

本例 baseline 进入相邻区域，观察的表面及有标注支持的 ID 更多；几何路线在起始区域做更多全景。结果支持保留 baseline 巡视作为可执行采集方案，尚不支持一般化的策略优势结论，也没有证明融合质量问题已经解决。下一步优先为地图实例对应与轻量证据保留设计可验收修复，并补充其他场景/更长预算；P8 连续新指令闭环及真实手机/机器人验收仍未完成。

### 固定证据上的实例关联根因与两项未采纳消融

使用实验因果诊断流程，选择首帧中同 UID 的 window/wall 观察。`diagnose_prior_associations.py` 从保存的 mask/depth 重建相同点云并调用原共享融合函数，不加载检测器或 GT。首帧 39 条观察、首轮全景 13 帧的 298 条观察均完整复现原 UID 分配。

首帧 window（739 点）和 wall（1,031 点）mask IoU≈0.0000268，分别有独立 GT 402/404 支持。wall→window 的 0.1 m 近邻比例为 0.1678，反向为 0.3681；按点数合并后的重叠为 0.2514，越过原 0.25 合并门槛。因此这个具体误合并在纯空间邻近度门槛中已发生，不依赖后续导航器或模型最终判断。

保持 298 条原始观察和关联算法固定，只增加一个实验候选准入条件。两项条件仅在诊断脚本生效，生产 `associate_object_entities` 保持原样：

| 条件 | 预测 UID | 跨 GT 混合 UID | 多预测 UID 的 GT ID | 观察对 precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原算法 | 72 | 10 | 17 | 0.849 | 0.497 | 0.627 |
| 同帧 mask IoU≥0.05 | 85 | 9 | 21 | 0.905 | 0.371 | 0.526 |
| 可见参考点 mask 支持≥0.25 | 86 | 6 | 19 | 0.872 | 0.376 | 0.525 |

第二项将参考点投影到当前图像，只统计当前有效量程内、距深度表面 0.1 m 内的点，计算落在当前 mask 内的比例；没有可见点记为 unknown，继续沿用原几何关联。window 点在当前 wall mask 内的支持为 0，因此可拆开该例。但整体观察对召回下降，单例改善不能作为上线依据。观察对指标条件于多数 mask GT 支持，用于固定样本关联消融，不是开放词汇类别准确率。汇总 `outputs/prior_diagnostics/matched_association_ablation_13frames.json`，两项负结果保留，未作为正式修复。

为统一不同建图器的实际深度支持，新增 `metric_depth_validity`，STRIVE、DualMap 输入及覆盖比较共同排除已记录的传感器截断端点；原始证据文件保持不变。20 项输入/采集/对比测试通过。此修改统一的是传感器观测语义，与上述实验关联约束分开。

现继续以相同 matched capture 对照 DualMap。baseline 输入已完成 80 帧、25 个稳定实例、499 条保留证据、24.13 s；这是包含上游稳定性过滤的结果，不能由数量少直接判断质量高。几何组与独立语义审计继续执行。

### DualMap 在匹配采集上的成本与质量边界

20% 点采样、seed=17、相同 100 具体概念（另加 upstream unknown sentinel）的独立建图结果：

| 输入 | 帧数 | 稳定实例 | 保留证据 | 建图耗时 | 跨 GT 投票实例 | 多数 mask 支持的 GT ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matched baseline 巡视 | 80 | 25 | 499 | 24.13 s | 4/25 | 25 |
| matched 几何路线 | 77 | 25 | 453 | 22.94 s | 4/25 | 29 |

两组独立语义审计通过现有深度对齐门槛，GT 仅用于审计。与 STRIVE 的 200/123 个预测实例相比，DualMap 有更低的本例计算成本，但存在稳定性筛选和检测器差异；保留证据支持的 GT ID 也更少。这个对照改变整套建图器，不能将差异单独归因于 CLIP、YOLO 或关联公式，也不能直接将 4/25 与 STRIVE 的 41/199 当成同一分母的准确率比较。

两个地图包仍分别占 106,684,961 和 115,692,471 字节，原始 RGB-D 证据占据主要空间。计算轻量已经获得样例证据，证据保留的存储轻量化尚未完成。

补测同一 80 帧上游默认 4% 点采样：14 个稳定实例、223 条证据、16.55 s。相对 20% 更快，但保留实例明显减少。该消融保持词表、模型和随机种子一致，语义审计另存 `dualmap_matched_default_sampling_semantic.json`。当前保留 20% 作为已验证实验配置，上游默认不改；分辨率、有效像素密度与稳定性门槛的关系仍需在其他输入上检验。

默认 4% 审计已完成：深度对齐通过，14 个保留实例均有多数 mask 支持，0 个跨 GT 投票，覆盖 12 个有多数 mask 支持的 GT ID。更少的混合同时伴随更少的保留对象/证据；尚不能由此选择统一默认采样比例。


### 证据精简：实现、实测与验收限制

新增 `prior_map/compact_bundle.py` 和薄入口 `scripts/compact_prior_bundle.py`，将完整预测地图导出为独立证据子集。对象 UID、标签、位置和拓扑保持原值；每个检测标签至少保留一条观察，对存在同帧分离 mask 的实例保留一对复核证据，再按相机方向差异补充。每实例 3 条是期望预算，必要复核证据可超额。选取过程使用预测证据，不读取 GT。

视角补充准则为 `argmax_c min_s (1 - d_c · d_s)`，其中 d 是对象中心到相机位置的单位方向。该准则选择位置差异，尚未建模图像清晰度与遮挡质量。新证据集合产生新 revision；已有导航视点的 checked_map_revision 保持原值，执行时重新检查。

| 指标 | STRIVE 完整→精简 | DualMap 完整→精简 |
| --- | ---: | ---: |
| 预测实例 | 200→200 | 25→25 |
| 观察证据 | 2,003→599 | 499→94 |
| 有引用的 RGB-D 帧 | 80→79 | 70→45 |
| 完整地图包字节 | 124,365,097→120,593,887 | 106,684,961→68,118,269 |
| 包体积下降 | 3.03% | 36.15% |
| 保留多标签复核实例 | 107/107 | 19/19 |
| 保留同帧分离 mask 复核实例 | 43/43 | 12/12 |
| 独立 GT 冲突仍可见的实例 | 40/41 | 4/4 |
| 必要证据超过 3 条的实例 | 35 | 10 |

独立验证核对 canonical 地图完全相等、每个选中文件与源文件 SHA256 相同，并在新进程完成加载及 chair 查询：STRIVE 返回 12 个假设，DualMap 返回 0 个。这些是标签/别名检索结果，不是语义准确率。19 项相关测试通过，覆盖搬移后重载、源文件不变、分歧证据优先及采集输入协议。

STRIVE 的 `matched-baseline-concepts100:frame51:detection8` 在完整证据中的跨 GT 投票，在选取后仅剩一个 GT 投票。对象几何与标签完全未改，此项下降表示反例丢失，不能解释为融合改善。GT 只用于独立诊断，未用于挑选运行时证据。故当前精简包保持实验用途，完整包仍作为质量审计依据，尚未完成轻量地图验收。

存储瓶颈来自共享整帧 RGB-D：减少同帧 mask 数量不会释放仍被其他对象引用的深度与图像。STRIVE 几乎保留全部帧，因此下一步应优先评估保持数值与标定不变的无损深度存储，再考虑改变帧选择；实例关联质量另行解决。原始采集与完整地图继续保留，以上体积只计各自地图包，不代表项目总存储下降。

汇总回执：`outputs/prior_diagnostics/compaction_matched_validation.json`。实验包：`outputs/prior_bundles/compact-strive-matched-baseline`、`outputs/prior_bundles/compact-dualmap-matched-baseline`。本轮不涉及外部模型请求，也未完成 P8 连续导航或实物验收。


### 完整证据的无损深度存储：已验证

针对上述删选收益有限与反例丢失问题，新增 `prior_map/depth_storage.py` 和 `scripts/compress_prior_depth.py`。导出复制完整包，将被引用的 NPY 深度以 gzip level 1 编码；每个文件解压后逐字节比对，再记录压缩文件与原始文件 SHA256。源包保持不变，全部观察、RGB/mask、标定、核心地图和视点保留。`load_depth` 统一读取 `.npy` / `.npy.gz`，语义审计、关联诊断、接近规划和到达验证的地图深度入口已接入。采集 writer 继续输出原始 NPY。

| 指标 | STRIVE | DualMap |
| --- | ---: | ---: |
| 保留观察 | 2,003/2,003 | 499/499 |
| 保留深度帧 | 80/80 | 70/70 |
| 原始深度字节 | 98,314,240 | 86,024,960 |
| 压缩深度字节 | 60,088,336 | 52,622,833 |
| 原始完整包字节 | 124,365,097 | 106,684,961 |
| 无损完整包字节 | 86,173,408 | 73,308,532 |
| 包体积下降 | 30.71% | 31.29% |

STRIVE 80 帧单次内存编码测量：压缩合计 1.22 s，解压+NumPy 读取合计 0.34 s（约 4.20 ms/帧）。此测量不含磁盘冷读取，并非实物平台性能承诺。两份导出包的新进程验证逐项检查全部观察记录（仅深度引用扩展名变化）、RGB/mask/confidence 哈希、全部深度字节和 stride=16 反投影坐标；核心地图、两种视点和完整内部证据审计完全一致。STRIVE 新进程 chair 查询仍返回 12 个假设。

22 项相关测试通过，包含搬移重载、共享深度引用、浮点位模式、损坏压缩文件错误传播及原采集兼容。压缩包的全部观察内容与源包一致，因此完整证据中的冲突仍然存在；本轮没有重跑 Habitat GT 渲染，也未声称实例质量改善。源码入口编译检查通过，尚未在压缩包上重跑完整连续导航 episode。

产物：`outputs/prior_bundles/lossless-strive-matched-baseline`、`outputs/prior_bundles/lossless-dualmap-matched-baseline`。回执：`outputs/prior_diagnostics/lossless_depth_validation.json`、`depth_lossless_measurement.json`，各包内含 `depth_storage_receipt.json`。相对删选方案，后续运行时地图优先采用这一保留完整证据的无损导出路径。P6 多场景质量、P8 连续新指令闭环与真实手机/机器人验收仍未完成；本轮未调用外部模型。


### P6 两个新场景：覆盖不足与空地图负结果

维护中的 Habitat 0.2.4 镜像离线采集 `TEEsavR23oF`、`HaxA7YrQdEC`，固定 seed=17、320×240、HFOV=90°、高度1.25m、128个候选/独立检查点、1m覆盖半径、8朝向。此为 NavMesh 辅助离散观察，未执行连续导航，连通岛不能当作房间或楼层。

| 配置 | 位置/帧 | 主岛路径距离覆盖 | 采集耗时 |
| --- | ---: | ---: | ---: |
| TEEsav，每岛最多4位置 | 9/72 | 19/128=14.84% | 1.88s |
| HaxA，每岛最多4位置 | 7/56 | 18/128=14.06% | 1.66s |
| TEEsav，每岛最多16位置 | 21/168 | 61/128=47.66% | 2.93s |

TEEsav 两预算的独立检查点逐项一致，其他采集参数一致。主岛采样高度跨度分别约3.13m、3.00m，提示需分高度检查；这不直接证明楼层覆盖。小连通岛另有0.66–1.00覆盖，不能与主岛简单平均来掩盖大区域缺口。

跨视角诊断保留遮挡与冲突：三组非深度边缘前方冲突分别30、233、995个投影样本；按已有平面逆深度插值诊断后仍超5cm者分别2、212、901。图像边界无法插值的样本另保留。此为成对投影统计，不能解释成独立表面错误率；残差尚未全部归因，几何验收保持待确认。详细列表保存各采集目录 `interior_conflicts.json`。

DualMap 对 TEEsav 72帧进行100概念、20%点采样、seed17建图，耗时11.63s，最终0实例/0证据。`before_finalize.json` 显示结束前65候选、0稳定实例、最大 observed_num=6，而 stable_num=8。直接机制是所有剩余候选未达稳定门槛；检测、跨帧关联和离散路线对计数的具体贡献仍需分开诊断。保留上游阈值，空结果作为失败记录，不以降低门槛宣布修复。

同时发现并修复 `semantic_audit.py` 空证据的真空通过：此前空帧集合可得到 alignment_valid=true；现在输出 status=insufficient_evidence、alignment_valid=null、summary=null，并显式抛错，在启动模拟器之前终止。14项空地图/采集相关测试通过。旧审计结果保留为问题证据，新回执为 `multiscene_TEEsavR23oF_empty_audit.json`。

总回执 `outputs/prior_diagnostics/multiscene_capture_001.json`。P6 获得跨场景执行与失败诊断证据，尚未达到多场景质量验收。下一步应在固定场景检查剩余几何残差，并区分覆盖预算与稳定实例所需重复观测；P8 与实物验收继续保留。


### P6 几何残差进一步定位：扫描网格的正反面可见性

继续固定原始深度和位姿分层分析。HaxA 的212个插值后非边缘前方冲突中210个跨连通岛、1个同位置旋转；TEEsav 16位置组901个冲突中13个跨岛、5个同位置旋转，主要来自主岛内跨位置观察。不能简单以排除小岛消除所有问题。

新增 `scripts/diagnose_prior_mesh_visibility.py`，在独立诊断中读取源GLB三角形，以Open3D计算双面最近交点和正面最近交点。射线 `d=R K^-1[u,v,1]` 未归一化，因此交点参数t就是optical-Z米制深度；按三角形绕序法线n，`n·d<0` 为正面。原始HM3D GLB与map_z_up的坐标对应通过采样深度一致性检验，未将网格输入预测地图。

每帧固定25个内部像素：TEEsav共4200射线，4177条具有正面交点及正渲染深度，全部在1cm内一致；HaxA为1400射线中的1391条，同样全部一致。其中分别7/8条存在比渲染深度近5cm以上的背面交点。无正面交点/无有效渲染的射线保留在报告，不当作通过。

具体冲突 frame108→120 的视觉检查及原始像素复核：frame108的墙面像素渲染深度0.702479m，网格正面交点0.702483m；frame120对应附近原始像素渲染深度7.962310m，最近双面交点2.106976m、最近正面交点7.962456m。该射线明确存在被单面可见性跳过的更近表面。报告为 `TEEsav_conflict_pair_ray_check.json`，图像为 `multiscene_conflict_pair.png`。

结论边界：这是源扫描网格的可见性诊断，支持部分大残差来自正反面/扫描表面条件，排除了该采样集上的整体坐标变换错误解释。尚未逐一解释全部残差，不能据此宣称所有位姿正确或直接放宽几何验收。图像与深度保持原样；后续应把可见性冲突与采集位置、表面方向联合记录，避免把背面穿透当作可靠自由空间。该机制仍需与实物RGB-D的实际遮挡模型区分。

新增回执 `outputs/prior_diagnostics/TEEsav_mesh_visibility.json`、`HaxA_mesh_visibility.json`；本轮没有修改融合阈值、渲染模式或导航完成判定。


### P6/P7 新场景采集过程对照：连续观察恢复实例保留

保持 TEEsavR23oF、320×240/HFOV90°/1.25m传感器、DualMap模型/100概念/20%点采样/seed17/稳定阈值8不变，先运行扩展离散采集，再复用现有连续路线采集入口。显式重建 radius=.1m、height=1.5m、climb=.2m、slope=45° 的NavMesh，其SHA与原场景NavMesh完全一致。连续路线从同一起始参考位置开始，实际执行150动作、4.75m、3组完整全景、151帧，planned_route_finished。此为几何辅助采集，不是新指令导航成功。

| 采集 | 帧 | 保留实例 | 证据 | 建图耗时 |
| --- | ---: | ---: | ---: | ---: |
| 离散扫描，最多4位置/岛 | 72 | 0 | 0 | 11.63s |
| 离散扫描，最多16位置/岛 | 168 | 1 | 9 | 22.72s |
| 连续动作，3个到达位置 | 151 | 33 | 707 | 23.26s |

扩展离散组的单个实例9条证据均有多数GT支持且无跨GT投票。连续组679/707条mask有多数GT支持，33实例均有支持，5实例出现跨GT投票；内部审计另有26多标签实例、10同帧分离mask实例。均为复核指标，非开放词汇准确率。连续地图新进程chair查询返回1个带证据假设，尚未据此执行任务。

上游稳定规则先要求observed_num≥8，再检查最多类别计数>observed_num/3或类别后验收敛；源码注释写1/2但实际比较为1/3，本次未改上游。规则未要求相机基线或独立视角数量。因此稳定实例数增加不能等同几何/语义正确性增加。

结果支持采集必须提供充分的重叠观察，单纯增加离散空间位置可能仍不能满足在线融合的稳定要求。连续组同时改变转角密度（10°对45°）、位置集合与处理顺序，不能把提升单独归因于连续移动。后续最小对照应固定这151帧，通过明确抽帧控制观察密度，结合实例保留、关联冲突与覆盖报告判断，而非直接降低稳定阈值。

产物 `outputs/prior_capture/multiscene-TEEsavR23oF-walk001`、`outputs/prior_dualmap/multiscene-TEEsavR23oF-walk001/bundle`。汇总 `outputs/prior_diagnostics/TEEsav_capture_process_comparison.json`，独立语义审计 `multiscene_TEEsavR23oF_walk_semantic.json`。P6/P7获得另一场景的可运行采集—建图—重载链路，但质量仍未验收；P8和实物仍待完成。


### 固定连续轨迹的观察密度对照

新增 `build_prior_dualmap.py --frame-stride`，默认1保持原路径；`selected_capture_frames`供输入与导出共同选择原始帧，先抽帧再应用max_frames。上游idx是选中序列索引，导出回到原始frame_id。回执记录stride、max_frames、源frames.jsonl哈希；5项相关测试通过，包含非连续ID及搬移后的图像/证据对应。

在原151帧连续capture中固定从frame0开始每2帧取1帧，得到76帧（最后仍为150）。保持原模型、100概念、20%点采样、seed17、稳定阈值8。

| 结果 | 全帧 | stride2 |
| --- | ---: | ---: |
| 输入帧 | 151 | 76 |
| 稳定实例 | 33 | 16 |
| 保留证据 | 707 | 203 |
| 建图时间 | 23.26s | 13.41s |
| 跨GT投票实例 | 5 | 0 |
| 有多数mask支持的GT ID | 26 | 14 |

两组深度/位姿独立审计通过现有门槛。隔帧组203条mask均有多数GT支持，但有支持的GT ID减少12个。GT ID包括不同标注粒度，以上不等于任务目标召回率；无跨GT投票也不能等于语义完全正确。

因此暂不将stride2设为默认，也不把冲突下降算作实例融合修复。抽帧改变输入密度、上游以处理帧计的窗口物理跨度、以及随机点采样消费顺序；此为完整采样策略对照，尚未将这些机制单独隔离。保留完整连续capture作为后续质量诊断基准。

回执：`outputs/prior_diagnostics/TEEsav_sampling_density_comparison.json`、`multiscene_TEEsav_walk_stride2_semantic.json`；实验地图 `outputs/prior_dualmap/multiscene-TEEsavR23oF-walk-stride2/bundle`。当前证据支持继续完善实例关联与有效多视角证据判定，而非通过减少输入或降低门槛来宣布地图可用。


### 多地图身份根因修复

多场景接入检查发现：DualMap 导出使用 root.name，所有 `.../bundle` 均写入 map_id=bundle；STRIVE 同样使用目标目录名，独立构建只要同名即冲突。该实现不满足设计中的地图身份约定。

在 bundle 中增加单次新建 UUID 分配入口，两个 builder 新建时调用；已持久化地图加载、复制、压缩沿用原ID。STRIVE对象UID沿用原来的map_id+原帧/检测索引组成方式，因此新构建中的对象命名同时摆脱目录重名。地图身份不表示内容相等或场景不同，内容可复现性继续由源/模型/参数哈希记录。

23项关联测试通过，包含同场景同名输出目录的两次独立导出拥有不同ID、搬移保持身份、证据删选/无损压缩保持身份与原始证据。既有实验产物未改写，旧 map_id=bundle 的包需按明确路径与回执辨识，不能将它们视为同一地图。这是接口身份修复，未重跑检测融合，也未增加地图质量或导航成功主张。


### P8 查询回执绑定到历史视点执行

查询脚本现记录map_id、map_revision、prior_map.json SHA256。`prior_map/query_binding.py`在选择最高分假设前验证三项一致，并检查对象存在；不匹配或空查询显式报错。核心哈希还用于区分既有同名map_id的实验包。旧查询回执需要重新生成，未用缺省字段绕过检查。

`check_prior_motion.py --query-receipt`直接消费该回执，解析其历史对象证据的当前可达视点，并采用路径距离最小者。直接指定UID的原诊断仍选择最远点用于运动检查。两者均通过已有HabitatExecution/MotionSupervisor；不改变任务进度。5项新测试通过，覆盖搬移后绑定、地图ID/版本/核心哈希不匹配、空查询和未知对象。

TEEsav连续地图的chair概念查询产生一个假设，未调用语言编译器或外部模型。起点为采集末端时，最近历史视点与起点重合，只执行6次旋转；该结果保留为零平移诊断。随后从原采集起点重新初始化独立模拟运行：35动作、1.9999995m平移、0碰撞、motion_status=reached、heading_alignment=true，最终路径距离0，保存36帧当前RGB-D。它不是前次执行的连续片段，也不是完整自然语言任务episode。

运动产物 `outputs/prior_motion/TEEsav-chair-bound-query-from-start`，查询 `outputs/prior_diagnostics/TEEsav_chair_bound_query.json`。最终图像本地查看为客厅座椅场景，但没有绑定当前检测对象与历史实例，也未调用最终验证器/reducer或发送任务STOP。此项只证明查询到实际到达链路；P8仍需在同一运行中接当前检测、接近/补拍和完成判定，不能将reached作为task_success。


### P8 同一模拟器会话内的到达后感知

将原 `extract_prior_arrival.py` 的逻辑提取到 `prior_map/arrival.py`，独立复放入口与运动入口共用最终帧打包函数。`check_prior_motion.py --arrival-classes` 在位置到达和定向结束后，保持原 simulator 打开，调用 `build_capture_map` 处理当前最终帧；生成arrival_capture、arrival_bundle及arrival_perception.json。类表来自显式文件，未用历史预测标签替代当前检测。没有新增外部模型请求，也未写任务进度或发送任务STOP。

TEEsav chair查询实测：35动作、1.9999995m平移、0碰撞、第35帧。当前100概念检测28框，27条观察经共享融合产生16个预测实例，其中1个chair标签实例。当前单帧建图耗时10.10s（包含模型初始化等），帧内处理约3.02s。前后模拟器位置一致，arrival_capture绑定的运动receipt SHA256与终态文件一致。

13项测试通过，覆盖最终帧ID、位姿、float32深度、运动回执哈希、非reached终态拒绝取证及现有查询/运动行为。此次仍未匹配历史/当前实例，未生成新的接近/补拍动作，未通过最终验证器和reducer，所以是同会话运动→感知的接入证据，不是完整指令episode。产物 `outputs/prior_motion/TEEsav-chair-same-session-perception`。


### P8 当前接近提案的共享入口

对同会话到达感知的chair候选核查：当前chair中心与历史查询chair中心距离0.19025m；两个desk chair候选分别1.06621m、5.05243m。现有概念查询将chair评分1.25置于首位，desk chair为0.905。此为候选选择依据，不是身份确认，尚未融合历史与当前UID。

将 `resolve_prior_approach.py` 的实际几何代码提取为 `backends/habitat/prior_approach.py::resolve_current_approach`，CLI保留为薄入口。继续校验当前证据与运动receipt哈希、NavMesh构建记录，复用现有点云投影、baseline视点生成与排序；没有新增阈值或类别特例。

真实保存输入产生10个可达提案。选中点的预测目标距离0.59178m、路径距离1.83759m、可见率0.33942；既有stop_feasible规则将0.2m到达容差计入1m距离限制。提取前已提交脚本与新函数分别运行，全部输出字段与选中结果完全一致，36项相关回归测试通过。

结果 `outputs/prior_diagnostics/TEEsav_current_chair_approach.json`。尚未执行该提案，预测stop_feasible不能作为当前任务完成依据；下一步将共享入口接入同会话的再次运动和当前取证。P8完成判定仍未验收。


### P8 同会话两段运动与再次取证

`check_prior_motion.py --approach-current`复用一次simulator初始化及同一个HabitatExecution。历史视点到达后当前感知，通过现有概念查询选择当前候选，调用共享baseline接近提案，再执行第二段运动并重新检测最终帧。当前UID独立保留，不被历史UID覆盖；该选取只是候选，不是历史身份确认。单概念诊断范围明确，不作为任意复杂TaskSpec的执行器。

实测 `outputs/prior_motion/TEEsav-chair-continuous-approach001`：第一段35动作/2.00m，第二段16动作/1.75m，两段均reached、零碰撞，第二段终点距提案0.09174m。总51动作，最终第51帧检测21框、19条观察形成10个预测实例。当前chair mask有12038个有效深度点，平面最近距离0.67772m；最终图像本地查看为近距离扶手椅。该距离属于当前预测mask，不是数据集目标距离或任务成功证明。

帧35在第一段终点与第二段起点的位姿和RGB字节完全一致；两份arrival_capture与各自不可变运动receipt哈希匹配。14项相关测试通过。执行后补充了当前候选的运动元数据与后续失败单独写session_receipt的逻辑，避免失败改写已经被证据引用的前段回执；此补充尚未单独重跑同一完整仿真。

未执行最终VLM/reducer/STOP。已询问是否允许将本次第51帧及框/裁剪发送到豆包，等待新图像授权；此前另一组新图像曾被自动审批以授权不明确拒绝，旧授权不自动扩展到这组图像。下一步需将最终验证真正接回同会话任务完成；当前仍不能标记P8完成。


### P8 已授权最终图像的真实豆包验证

用户明确允许TEEsav第51帧及目标框/裁剪发送豆包后，实际调用现有 `FinalInstructionVerifier`，模型doubao-seed-2-0-pro-260215、thinking disabled、证据复用关闭。使用明确的单目标chair合同（source=explicit_concept_diagnostic），没有宣称额外完成自然语言编译。

验证结果：decision=accept、semantic_satisfied=true、view_sufficient_for_stop=true、confidence=0.99；模型解释框内为深棕色皮质躺椅。代码硬约束采用当前mask深度距离0.67771997m≤1m。现有TaskProgress reducer计入当前UID，count=1、completed=true。证据、判断和进度保存在 `outputs/prior_verification/TEEsav-chair-final-authorized/verification.json` 与receipt.json。

将原check_prior_arrival逻辑提取为 `prior_map/arrival_verification.py::verify_arrival`，供CLI与下一步会话编排复用。接口接显式TaskSpec、当前候选UID和运动绑定；prepare_only不会实例化外部验证器。7项到达/验证测试通过，注入判断仅验证协议与reducer行为，不作为模型性能结果。

真实验证发生于连续轨迹结束后的保存证据上，原simulator已关闭；task_stop_sent=false。故此证明了该样本“当前证据→真实最终验证→任务进度”的链路，尚不证明同一会话的验证完成事件及STOP，也不等于HM3D官方成功率/SPL。P8仍待整合验收。新图像授权已明确收到，本组调用已完成，先前等待状态解除。


### P8 单目标同会话最终验证与STOP闭环

实测产物 `outputs/prior_motion/TEEsav-chair-live-verification001`。同一个simulator与现有HabitatExecution/MotionSupervisor依次执行历史视点到达、当前检测、当前候选接近、再次检测、真实豆包最终验证及STOP。显式TaskSpec为“Find a chair.”，source=explicit_concept_diagnostic；此次覆盖单目标合同执行。

`session_receipt.json`记录status=complete、simulator_initializations=1、total_actions=52、task_completed=true、task_stop_sent=true。`session_actions.json`总移动3.74999921m、零碰撞；最后action=0，translation_m=0。最终第51帧框图SHA256为 `0dd6496efec4558b89456528edc3af99ec5d766a109f008b20797e0ff371d78f`，与用户已授权保存图像一致；外部临时运行器在真实模型调用前校验图像哈希，验证结果由实际请求产生。

豆包返回accept、confidence=0.99、semantic_satisfied=true、view_sufficient_for_stop=true。当前预测mask深度的平面最近距离0.67771997m，满足1m停止约束；该距离仍是预测实例距离。reducer接受当前frame51候选后completed=true，调用方通过运动所有者执行STOP。验证服务receipt中的task_stop_sent=false描述服务本身不发送停止；最终停止事实以session_receipt与session_actions为准。

终止后端为habitat_sim_task_adapter：STOP结束任务会话，保持位姿并读取当前传感器，不产生移动。official_objectnav_metrics_evaluated=false，因此本次没有官方SR/SPL。17项到达、验证、查询绑定及运动测试通过，最新元数据与失败回执逻辑也已随本次真实会话执行。

P8现已取得一个显式概念任务的完整执行闭环证据；自然语言复杂合同、历史与当前实例对应、多样本泛化及官方评测仍待完成。地图融合质量、覆盖缺口和真实手机/机器人重定位维持原有待验收状态。


### P8 历史实例—当前mask—停止距离的独立空间核查

对 `TEEsav-chair-live-verification001` 的两份当前预测地图运行既有 `diagnose_prior_instances.py`，仅离线渲染HM3D语义标注，不将GT送入查询、建图或验证器。核对历史/当前预测地图SHA256、同一semantic mesh与labels哈希，以及最终语义缓存与RGB哈希。第35/51帧重渲染深度最大误差分别0.000430m/0.000341m，两帧超过1cm的像素均为0，空间对齐检查通过。

| 阶段 | 预测实例的标注支持 | 结论范围 |
| --- | --- | --- |
| 历史chair | 26条mask观察；25条过50%多数支持，唯一强支持ID为266（armchair） | 含1条仅33.61%支持的低质量观察，不能宣称所有历史证据纯净 |
| 当前到达，第35帧 | 99.3175% mask像素属于266 | 本帧候选与历史强支持对应同一实例 |
| 接近后，第51帧 | 99.9003% mask像素属于266 | 最终候选仍为同一实例 |

进一步复算停止距离：对最终mask内满足原验证器有效深度条件的12038点，使用同一公式 `d=min ||p_map,xy - camera_map,xy||_2`。最小值0.6777199749643563m来自像素(u=193,v=157)，该像素GT ID=266；仅保留属于266的12029点后，最小距离完全相同。其他9个有效深度点的最小距离0.70316946m。本例停止距离由椅子表面支持，并非mask边缘污染点制造的虚假近距离。

汇总与来源哈希：`outputs/prior_diagnostics/TEEsav_chair_live_identity_distance.json`；逐mask报告为 `TEEsav_chair_live_arrival_semantic.json` 和 `TEEsav_chair_live_final_semantic.json`。这是一例离线GT空间归属诊断，不是运行时实例关联算法，也不是官方ObjectNav任务成功判定。其他当前实例仍有混合GT：到达地图2个、最终地图1个，地图融合质量问题仍存在。

后续重点：当前 `current_concept_candidate` 每阶段独立按概念排序，尚未携带历史实例约束；简单“找一把椅子”允许同类替代，但指代某个历史实例或属性消歧任务需要独立的关联与拒绝判定。应在固定的相似实例/不可见对象案例上验证该边界，不能把本例一致性推广为通用关联正确。


### P8 编译任务与当前候选查询的语义绑定

源码审计发现，`query_prior_bundle.py --compile-instruction` 的concept字段保存原始自然语言，而同会话接近阶段直接将该字段传给概念查询；最终验证另读verify-task文件。因此接近和验证原先没有共享同一任务合同，也没有在运动前检查合同一致性。

在 `prior_map/query_binding.py` 增加query_execution_task，复用已有TaskSpec。编译查询以回执中的合同为准，另传合同必须与该版本一致；纯概念查询允许绑定同名目标的显式合同。单目标诊断的数量/约束范围在运动前明确检查。`check_prior_motion.py` 仅解析一次，接近和最终验证均使用同一task的目标名称，保留属性与其他合同字段供最终验证使用。此修改不新增实例关联算法。

21项相关测试通过，覆盖自然语言输入解析为当前目标概念、属性及版本保留、不同任务版本拒绝、纯概念兼容、多目标范围检查，以及现有到达/运动/验证回归。语法检查与git diff --check通过。

实际保存数据回放：`compiled_nightstand_query.json` 的“Find a nightstand.”解析为nightstand，在nightstand-arrival-current和nightstand-approach-current两份地图中均选择既有nightstand候选。旧原始句子路径在这两例也能选中相同候选，因为现有概念评分支持文本匹配；本次修复证明合同来源一致，尚无这两例选择准确率提升的证据。回放报告 `outputs/prior_diagnostics/compiled_task_binding_replay.json` 保存查询与地图哈希，无模型调用、无新增运动。最新任务绑定代码尚未重跑完整仿真闭环。

另一个待处理的边界：PriorMapMemory._match_prior_object在标签匹配后返回首个对象，现有代码不足以提供多实例身份关联保证；TaskSpec目前也没有显式历史实例引用字段。应先固定实例指代的合同，再联合空间与视觉证据验证，不能把prior对象exact=true解释成用户指定了该实例。


### P8 在线记忆中的类别与实例身份分离

先写行为测试复现：现有 `_match_prior_object` 遇到同名对象直接返回列表首项，mapper与snapshot均会把新观察记到第一个历史对象；已有显式verified对应也未被优先使用。查询评分同时把同类标签当作每个具体历史实例的live_match。新增测试在修复前6失败/1通过。

修复限定在memory/query：具体历史实例使用持久化UID或已有显式verified对应；类别/概率先验仅在标签与房间筛选后唯一时更新；多候选保留未确定状态。观察记录仍保存当前UID和类别。具体实例的live_match依据身份，类别先验仍可使用标签支持。mark_object_verified说明调用方须先建立实例对应，类别任务验证通过本身不提供该身份依据。没有增加几何距离阈值，也没有新增模型调用或更改建图融合。

91项相关测试通过，覆盖先验协议、查询、记忆、仿真/实物适配、到达、任务验证和运动接口。真实保存数据回放将最终10个当前对象送入TEEsav历史地图：旧逻辑自动绑定2对，离线GT核查这2对均正确；新逻辑保留10个观察、自动跨图绑定0对。结果说明本例没有错误关联收益，新增限制需要后续空间/视觉关联服务补全召回。报告 `outputs/prior_diagnostics/TEEsav_memory_instance_binding_replay.json` 记录旧新代码及地图哈希。

这一步修正身份所有权与证据含义，通用在线关联、相似实例消歧与不可见对象重新搜索仍待实现；没有重跑导航或产生新SR/SPL。


### P8 跨地图表面对应诊断：同类标签之外的空间依据

新增离线脚本 `scripts/diagnose_prior_correspondence.py`。在同一Habitat资产、simulator_gt世界位姿与坐标协议下，从两份地图的保存mask、深度、K和相机位姿重建体素表面。使用5cm体素中心、10cm最近邻半径、6m最大深度；保留原采集量程截断规则。输出每对对象的方向性覆盖 `f(C,H)=mean_p[min_q ||p-q||<r]`、反向覆盖、按点数加权的合并覆盖和当前点中位距离。其距离统计与已有STRIVE融合原则一致，但重建表面没有复现建图器内部采样/聚类/过滤，不能称为原融合精确重放。

该脚本不读取GT、不改变地图或身份状态；记录所有mask/depth、观测索引、构建回执及核心地图哈希。相同frame_id并非跨次实物对齐证明，因此脚本明确限定同一仿真世界。3项算法测试通过：部分视图的方向非对称性、分离表面零覆盖、共同刚体变换不变性。

实际运行：历史33个对象，对第35帧16个对象产生528对；对第51帧10个对象产生330对。独立GT评估事后读取冻结报告，未参与打分。

| 当前阶段 | 有单一GT支持且历史存在对应：最高分正确 | 历史没有单一强支持对应 | 当前混合或证据不足 |
| --- | --- | --- | --- |
| 第35帧 | 3 | 11 | 2 |
| 第51帧 | 4 | 5 | 1 |

这些是同一场景两帧的对象计数，含重复真实实例，不能作为7个独立任务成功样本。chair两帧正确历史对象的当前点覆盖均1.0；次优分别0.02145和0.02817，合并覆盖0.96620/0.95450。

发现的限制：当前table与历史shelf unit实际同属GT267，当前点覆盖0.9943，说明严格同类别门控会漏掉真实对应；当前混合table候选包含GT336/351，仍对历史cabinet351产生0.8849当前覆盖及0.9382合并覆盖，说明高覆盖不能排除对象混合。oven与stove两个当前候选都指向GT348，历史也有类别不同的同实例碎片，不能直接把最高分转换成唯一实例计数。

产物：`TEEsav_arrival_historical_surface_pairs.json`、`TEEsav_current_historical_surface_pairs.json`、`TEEsav_surface_correspondence_evaluation.json`，均在outputs/prior_diagnostics。下一步应检查逐观察对应的一致性，再制定候选关联与拒绝规则；当前没有把离线分数或本例阈值接入运行时验证。


### P4/P8 混合实例的逐观察证据与原融合复放

将跨地图诊断扩展为 `--current-observations`：逐mask重建表面，保留observation_id到原object_uids的映射，保持预测地图不变。最终19条观察对33个历史对象输出627对，4项测试通过（新增同一融合UID内两个独立mask保持分离的测试）。报告 `TEEsav_final_observation_surface_pairs.json`。

逐观察显示，混合UID detection8包含ironing board（GT336占85.75%）、drawer（GT351占93.22%）、table（GT351占71.74%）。三者对历史cabinet351的当前表面覆盖分别0.67778、1.0、0.88353。因此逐观察最高分一致，也不能直接确认同一实例：相邻表面在10cm半径内互相支持。

随后使用维护中的strive-hm3d:local运行已有diagnose_prior_associations.py，对保存的19条accepted观察重放原STRIVE融合。assignment_parity=true，输出10个实例，保存UID归属逐条一致；证据为 `TEEsav_final_fusion_trace.json`。错误合并实际发生顺序如下：

1. detection8 ironing board先形成独立90点实例。
2. detection13 drawer的51点与它的合并近邻覆盖为0.333333，两个方向分别0.392157/0.300000，超过原0.25阈值，被并入detection8。两个当前mask交集为0像素。
3. detection14 table的249点与已混合141点实例的合并覆盖0.717949，方向分别0.558233/1.0，再次并入。

直接mask核查：table覆盖99.03%的ironing board区域及100%的drawer区域；同一个较大的检测mask把两个分离区域同时纳入，强化已发生的混合。报告 `TEEsav_final_mixed_mask_intersections.json`。这里既有近邻关联误合并，也有跨对象的宽泛mask；不能把原因全部归于跨地图关联或类别名称。

对应数学边界：当前融合计算的是带半径容差的表面邻近 `P(d(p,H)<0.1m)`，它并不是“同一物理实例”的概率。相邻物体、包含关系和宽mask均可得到高值；混合点云并集又会进一步提高后续覆盖。当前证据支持在融合中保留逐观察归属与冲突，而不是直接提高/降低一个全局阈值。此前co-view IoU与投影支持消融已暴露召回损失，未将这些规则作为默认修复。

本轮取得可重复的具体根因；未修改生产融合阈值、未宣称实例质量修复成功。下一步应围绕同帧独立区域和跨对象宽mask构建关联对照，联合报告错误合并与同实例拆分，避免只优化单个混合case。


### P4 同帧混合case的约束对照与统一关联评估

固定最终第51帧全部19条accepted观察，在维护容器中分别运行现有coview IoU≥0.05及可见mask支持≥0.25两种实验约束，原深度、位姿、检测与融合阈值保持一致。生产融合未修改。独立语义审计中19条mask均达到多数支持门槛，覆盖10个GT实例。

| 变体 | 预测实例 | 混合实例 | 被拆分GT实例 | TP/FP/FN观察对 | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原融合 | 10 | 1 | 1 | 11/2/9 | 0.8462 | 0.5500 | 0.6667 |
| coview≥0.05 | 11 | 1 | 2 | 10/1/10 | 0.9091 | 0.5000 | 0.6452 |
| visible≥0.25 | 11 | 1 | 2 | 10/1/10 | 0.9091 | 0.5000 | 0.6452 |

两种变体均将drawer从ironing board拆开，但后来的宽table mask仍归入ironing board，混合未消除，并损失了drawer与table之间的同实例关联。该结果与之前13帧消融中的精度/召回权衡一致，不能将两种约束提升为默认修复。

新增 `prior_map/association_evaluation.py::evaluate_association` 统一计算观察对指标：TP为同时同预测UID/同GT的对数，FP为同预测UID/不同GT，FN为同GT/不同预测UID。利用组内组合数计算，避免枚举全部观察对。报告支持与未支持mask数量，分母为当前固定输入中通过GT多数支持的观察；指标不代表mask纯度或导航成功。3项测试覆盖误合并与拆分并存、输入顺序不变性、无多数支持及缺失/错位标注。

结果及来源哈希见 `outputs/prior_diagnostics/TEEsav_final_fusion_ablation_evaluation.json`；两份trace为 `TEEsav_final_fusion_coview005.json` 与 `TEEsav_final_fusion_visible025.json`。下一步候选方案应显式处理宽mask跨越多个已分离区域的歧义，同时检查其对部件/整体关系的影响；当前尚未选出有效的生产关联修复。


### P2/P6 覆盖预算对照与独立审计随机流修复

沿用TEEsav场景资产、原NavMesh、种子17、320×240、HFOV90、相机高1.25m、8朝向、1m测地覆盖半径，完成三组新增离散RGB-D采集。只增加位置预算，主区域从16位置增至28后停止（未用满128预算）；再增加候选样本至512，主区域达到47位置。此后以相同512候选预算将clearance从0.25m降至0.1m做独立对照。

发现评估协议缺陷：原采集器先在同一随机流中产生候选样本，再产生检查点；当候选预算增加，旧检查点可能进入新候选池。先前在旧128检查点上得到的90.625%/99.21875%只保留作诊断，不作为独立覆盖结论。新增 `scripts/evaluate_prior_coverage.py`，固定资产/NavMesh哈希，用独立种子17001、每个连通区域1024个检查点重评全部四组，核验各组审计点逐项相同。

| 变体（候选数/位置上限/clearance） | 实际总位置/帧 | 主区域覆盖 | island1覆盖 | 采集耗时 | 保存大小 |
| --- | --- | --- | --- | --- | --- |
| 128/16/0.25m | 21/168 | 47.4609% | 62.8906% | 2.93s | 66.19MB |
| 128/128/0.25m | 33/264 | 67.7734% | 62.8906% | 4.07s | 100.54MB |
| 512/128/0.25m | 52/416 | 86.5234% | 64.0625% | 6.44s | 158.33MB |
| 512/128/0.10m | 59/472 | 92.5781% | 100% | 8.52s | 178.91MB |

这是几何辅助离散扫描的位置覆盖，不是表面覆盖、语义房间覆盖或连续导航成功率。clearance=0.1m组有2帧超过半幅图像处于0.25m近距离内，全体平均近距离像素占比1.3256%；其他三组该占比为0。相机采集clearance也不能代替机器人尺寸和运动可达性验收。因此当前不将0.1m作为默认配置。

原采集器 `CaptureConfig` 新增audit_seed与audit_samples，将候选和审计按连通区域分别设种子；默认审计种子17001、1024点，候选数量不影响其他区域或审计流。选择算法、几何传感器与默认clearance保持原样。16项采集/比较测试通过；另在真实Habitat用32×24、每位置1朝向、每区域最多1位置完成128/512候选两组小样本，所有纳入区域的32个审计点完全相同。新随机流规则会改变旧实现中第二个及后续岛的候选序列，已有采集保留其原始配置与回执。

产物：新增采集目录 `multiscene-TEEsavR23oF-128`、`multiscene-TEEsavR23oF-s512`、`multiscene-TEEsavR23oF-s512-c010`；汇总 `outputs/prior_diagnostics/TEEsav_coverage_budget_comparison.json`，各变体独立评估报告以 `_independent_coverage.json` 结尾。新的416/472帧数据尚未进行对象融合、语义质量评估或多层逐层验收。


### 2026-10-31：416帧DualMap质量审计与STRIVE同输入对照启动

用户恢复推进，顺序为新增地图审计→同输入STRIVE构建→根据质量决定融合改进。已核对DualMap实际处理416帧，生成24个预测实例、391条证据，构建回执耗时60.1616s。当前DualMap提交dd9045c相对原上游的差异与旧168帧实验保存的upstream.patch逐字节相同；模型与词表沿用已有配置，pcd_sample_ratio=0.2、seed=17。

独立Habitat语义审计完成，深度对齐通过；391条mask中367条有超过50%的单一标注支持。23个预测UID有强支持，2个UID跨GT实例混合，1个GT被多个预测UID支持。所有受支持UID涉及26个GT实例，其中21个GT出现在单一GT支持的预测UID内。旧168帧地图仅1个GT支持实例，因此新增观察确实增加了可支持对象，但覆盖、视角数与上游稳定性筛选同时变化，不能单独归因于覆盖率一个变量。

混合UID：door关联GT107门、106墙、108/109玩具；sofa关联GT132沙发与165毛绒玩具。保存证据显示同帧或多帧跨对象观察仍被计入同一UID，实例身份质量尚未通过。26个GT中含墙体等对象，该数量不是导航目标召回率。报告 `outputs/prior_diagnostics/TEEsav_s512_dualmap_semantic.json`，旧新比较 `TEEsav_dualmap_coverage_quality_comparison.json`。

STRIVE对照已启动，输入完全相同416帧、100概念词表，原SAM+DINO、5cm体素、6m深度上限、共享原融合；输出 `outputs/prior_bundles/multiscene-TEEsavR23oF-s512`，日志 `outputs/prior_diagnostics/TEEsav_s512_strive_build.log`。启动时GPU为RTX3090，读取设备必须在允许设备访问的环境，沙箱内nvidia-smi失败不能代表驱动故障。

已为此次实际运行容器1eec8dcaf980安排成功后的独立语义审计，报告目标 `TEEsav_s512_strive_semantic.json`，审计日志 `TEEsav_s512_strive_audit.log`。等待器先检查docker wait返回的真实退出码；失败时不启动审计。本文记录启动安排，不代表构建或后续审计已经完成，最终以进程与回执为准。

对照判读约束：两种建图器的检测模型、实例稳定性过滤不同，即使使用相同原始观察，accepted mask分母仍不同。分别报告强支持、混合、重复实例、证据与资源成本；不能把预测实例数量较少直接解释为融合更准确。完整对照结果出来前不修改生产融合或扩大导航验证。


### 2026-10-31：等待416帧对照期间的已有证据与改进边界

完整416帧STRIVE构建及语义审计尚未完成；禁止将其中间UID数量与DualMap最终稳定实例直接比较。已重新读取并核对此前相同80帧的完整对照：STRIVE 200预测UID/2003观察，199UID有强支持、41混合UID、94受支持GT、47个GT有多个UID、70个GT出现在单一GT支持UID内，耗时337.33s；DualMap 25UID/499观察，25UID有强支持、4混合UID、25受支持GT、4个GT有多个UID、18个GT出现在单一GT支持UID内，耗时24.13s。

新增 `scripts/compare_prior_map_quality.py`，检查完整采集manifest与处理帧数一致、语义资产哈希一致、审计绑定到当前地图且覆盖全部观察。输出同时保留预测UID、观察分母、混合、拆分、支持GT、单一GT支持UID、耗时和存储。实际80帧结果复核成功，产物 `outputs/prior_diagnostics/matched80_map_quality_comparison.json`。该对照比较整套系统，检测器和稳定性筛选不同，无法孤立证明某个融合模块更好。

源码边界核对：STRIVE以当前mask的几何准入及重叠合并后直接导出全部entities；DualMap使用stable_num=8，随后类别众数>observed_num/3或类别收敛判定稳定。源码注释中的1/2与实际1/3不一致，分析采用实际表达式。重复帧计数不等于独立观察基线，该稳定性规则也未消除最新地图中的混合对象。

后续改进应分开验证：
1. 关联提交：宽mask同时覆盖多个已有独立区域时保留观察与候选关联，避免立即将点云并入一个身份；必须保留部件/整体关系的不确定性。当前证据支持该问题存在，具体判定规则尚需固定观察对照。
2. 实例输出：区分暂定候选与已获多视角一致支持的实例，检索保留暂定候选；验证独立视角证据而非照搬8帧硬门槛。验收同时检查支持GT损失和错误关联。
3. 分裂诊断：部分视图与大历史点云的合并重叠会受点数分母影响，应检查方向性和当前可见支持；已有简单IoU/投影门槛实验失败，不能直接启用。
4. 运行成本：每候选遍历所有实例并重复计算AABB/构建最近邻索引是源码可见开销；先测组件耗时，再验证AABB/空间索引缓存是否保持逐观察关联等价。当前逐帧日志不能单独区分检测与融合耗时。

以上为待验证方案，不代表已修复生产融合。仍以当前完整建图→语义审计→同输入质量报告作为下一道判定依据。


### 质量对照报告的验证补充

新增test_prior_quality_comparison.py，验证混合UID涉及多个GT时不会增加单一GT支持计数，以及拒绝深度对齐失败、地图哈希不符和观察集合缺失的审计。连同关联评估测试共7项通过。此为评估工具验证，不代表当前416帧建图或语义审计完成。


### 2026-10-31：416帧STRIVE建图、语义审计及完整系统对照完成

原STRIVE建图会话与自动审计会话均以退出码0完成。STRIVE回执status=complete，416帧、782个预测实例、7516条观察，elapsed_s=2616.0438（43.60分钟）。DualMap同一416帧回执24实例、391观察、60.1616秒。两者均包含模型初始化与构建/导出计时，但构建流程不同，不能将速度差完全归因于融合模块。

两套审计均alignment_valid=true，各审计帧超过1cm的深度误差像素比例最大值为0。完整输入manifest、处理帧数、语义资产、地图哈希、观察集合已由compare_prior_map_quality.py检查通过。正式报告 `outputs/prior_diagnostics/TEEsav_s512_map_quality_comparison.json`。

| 指标 | STRIVE | DualMap |
| --- | --- | --- |
| 完整输入帧 | 416 | 416 |
| 带保存证据的帧 | 416 | 204 |
| 预测UID | 782 | 24 |
| 观察mask | 7516 | 391 |
| 多数GT支持mask | 6611 | 367 |
| 有GT支持UID | 756 | 23 |
| 跨GT混合UID | 206 | 2 |
| 有支持的不同GT | 341 | 26 |
| 出现在单一GT支持UID中的不同GT | 232 | 21 |
| 对应多个预测UID的GT | 198 | 1 |
| 建图时间 | 2616.04s | 60.16s |
| 地图包大小 | 167.45MB | 79.43MB |

GT多数支持表示mask超过50%的像素属于该标注实例；单一GT支持UID仍可能有像素污染或错误预测类别，不能称为类别准确率。两种系统使用不同检测器与稳定性筛选，表中分母差异很大。

#### 共同对象控制

两者共同支持24个GT，STRIVE独有317个、DualMap独有2个（plush toy165、toy109）。共同GT在STRIVE涉及99个预测UID，其中32个UID全局存在混合，20个GT有多个UID；DualMap涉及23个预测UID、2个混合、1个GT有多个UID。共同GT中有单一GT支持UID的数量分别22/21。因此，在保留共同对象时STRIVE仍有明显碎片问题，差异并非只来自总对象范围。该子集是事后诊断，不是独立测试集。报告 `TEEsav_s512_common_gt_comparison.json`。

STRIVE的206个混合UID中，41个涉及同一标注类别的不同实例，165个涉及不同标注类别；最多一个UID涉及6个GT（plant、couch、table、basket等）。仅做类别一致性门控既无法解决同类多实例，也可能错拆检测器标签不同但实际相同的对象。

#### 输出稳定性筛选的负结果

保持所有预测归属不变，分别模拟输出数量筛选，结果如下：

| 筛选 | 保留UID | 有支持UID | 混合UID | 支持GT | 单一GT支持UID涉及的GT |
| --- | --- | --- | --- | --- | --- |
| 全部 | 782 | 756 | 206 | 341 | 232 |
| 至少8条观察 | 342 | 340 | 158 | 238 | 103 |
| 至少2个采集位置 | 538 | 529 | 193 | 288 | 159 |

混合占有支持UID的比例从27.25%升至46.47%/36.48%。这是输出筛选消融，没有复现DualMap完整稳定性算法，也没有重新融合。结论是直接删除低频UID会丢失大量单一实例支持，同时保留已有混合。报告 `TEEsav_s512_strive_support_filter_ablation.json`。

#### 融合改进的优先级

首要问题是关联归属，而非输出数量。令a(o)表示观察o的预测实例归属，n_i=sum_o 1[a(o)=i]；错误关联本身就会增加n_i，所以观察次数不能作为独立的身份验证。

下一版先在冻结观察上试验可回溯的归属决策：保留原始mask/位姿、候选关联及冲突证据，宽mask跨越已有独立区域时保留未确定归属，避免立即并入一个UID；关联得到新证据后再决定合并或拆分，点云几何由已确定的观察重建。优先复用已有observations与空间查询，不引入第二套地图框架。

分别在固定坏例（相邻熨衣板/抽屉/宽table mask）与正常多视图/部件整体例上评估；同时报告FP误合并、FN错拆、GT支持损失和未确定观察数。现有IoU/可见支持门槛及本轮次数筛选均未达标，保留为负结果。稳定实例分级置于关联修复之后；效率优化单独测量并要求关联结果保持一致。

完整对照已完成，但可靠融合修复仍待实现和验证；当前不扩大导航评测，也不将地图质量标记为验收通过。


### 延迟同帧歧义关联：实现、对照与未通过结论

在既有诊断脚本增加 `--defer-coview-bridges`（要求显式min-coview-iou）。保留原0.25几何关联准入；当前观察同时匹配至少两个候选，且候选已有同帧mask互相低IoU时，记录assigned_uid=null和全部冲突候选UID，不改变任何实例的点云或类别计数。原始观察仍在源bundle中。该实现只保存未确定关联，没有宣称具备后续消歧。生产mapping/object_fusion.py保持不变。

评估器将每条deferred观察作为独立单例参与观察对评估：同GT的未完成关联仍计入FN，不能通过把所有观察推迟而虚增分数。报告增加deferred_observation_count，supported_predicted_instances只统计已分配实例。4项CPU评估测试通过；维护容器没有pytest，直接执行2项无fixture断言测试通过；8项评估/质量比较回归测试通过。

| 固定输入 | 变体 | 混合UID | deferred观察 | TP/FP/FN | F1 |
| --- | --- | --- | --- | --- | --- |
| TEEsav最终19观察 | 原融合 | 1 | 0 | 11/2/9 | 0.6667 |
| TEEsav最终19观察 | coview≥0.05+延迟桥接 | 0 | 2 | 9/0/11 | 0.6207 |
| 既有13帧298观察 | 原融合 | 10 | 0 | 650/116/657 | 0.6271 |
| 既有13帧298观察 | coview≥0.05+延迟桥接 | 7 | 17 | 422/29/885 | 0.4801 |

两组都减少错误关联，但未确定观察和错拆增多，整体F1下降。因此只能保留为诊断机制，未作为可靠融合修复。第二组有293条观察达到GT多数支持，5条另计证据不足；17条deferred为全部输入中的数量。报告 `outputs/prior_diagnostics/coview_bridge_deferral_evaluation.json`，保留两组trace与审计哈希。

进一步核对两个deferred案例：table观察主要支持GT351 kitchen cabinet，冲突候选分别是预测drawer/GT351与预测ironing board/GT336 worktop；另一个floor观察主要支持GT289，与GT289/292两个地面区域发生歧义。这里“ironing board”等是检测标签，实际标注是worktop，不能当作物理类别真值。标注实例粒度与部件/整体关系需要明确，不能把所有空间邻近或不同标注ID都解释为同一种物理实例错误。

下一阶段必须补全歧义处理闭环：保留观察→获得额外空间/视觉或任务粒度证据→重新确定归属；只做延迟、只加观察次数和简单同帧门槛均不足以实现已验收地图。需要使用独立正常样本验证可恢复关联的召回，同时维持完整错误/未确定分母。


### 复用DualMap外观特征的冻结候选诊断

核对DualMap实际实现：局部关联使用空间相似度+视觉余弦相似度；视觉表示为normalize(0.7×image_feature+0.3×class_text_feature)，上游sim_threshold=1.2，空间定义与STRIVE并不相同。全局/地图内合并还有不同路径。因此本实验检验外观信息能否辅助STRIVE，不宣称复现DualMap完整关联。

新增extract_prior_appearance.py，直接复用DualMap Detector.compute_clip_features_batched和本地MobileCLIP-S2权重，对保存的RGB、bbox进行上游20像素padding裁剪；不重新检测，不发送图像到外部服务。19+298条观察全部实际编码，保存image/text/weighted三个归一化特征、观察ID、源图像和权重哈希；显式类别不执行unknown类别替换。实际检查特征全为有限值，19例最大归一化误差1.2e-7。

新增diagnose_prior_appearance.py，按原关联trace聚合特征，保持原几何与身份决策。仅在候选已有单一GT多数支持时，使用GT判断当前观察与候选是否同实例；混合/证据不足候选另计且不进入AUC。原trace每观察最多保存3个空间候选，所以该统计是条件候选诊断，不能当作全局匹配准确率。UrbanNav环境无sklearn，诊断使用已有依赖的dualmap环境运行，未安装新依赖。

| 冻结输入 | 全部候选对/可判定对 | 图像AUC | 图像+类别加权AUC | 几何AUC | 几何+加权外观AUC |
| --- | --- | --- | --- | --- | --- |
| 最终19观察 | 36/32 | 0.5333 | 0.6000 | 0.7917 | 0.7875 |
| 13帧298观察 | 686/565 | 0.5906 | 0.6538 | 0.9143 | 0.8688 |

在已知存在正确候选的观察子集中，19例条件首选正确数为8/8→8/8；13帧为197/201→195/201。该指标排除了无正确候选和不可判定关系，不能用来证明完整关联成功。

坏例局部结果：宽table与drawer加权相似度0.4059，高于与ironing board的0.3425；但floor同GT对应0.2739，低于不同GT floor的0.2914。整体分布重叠，不能从单个有利例子设置外观门槛。原几何分数与外观直接相加没有改善两组候选排序，因此尚不接入生产融合。

产物目录 `outputs/prior_diagnostics/TEEsav_final_appearance`、`matched13_appearance`；诊断报告 `TEEsav_final_appearance_diagnosis.json`、`matched13_appearance_diagnosis.json`。后续需区分padding背景干扰、局部视图变化、类别文本对不同实例的同质化，以及原错误关联对聚合特征的污染；当前结果不能推广为所有视觉特征无效。

## 2026-11-01：观察归属、延迟消歧与局部重建首轮实现

实现：只读几何评分和显式更新分开；新 `PriorAssociation` 统一建图与重放；暂缓观察保留原始证据；
候选版本变化触发重评；提供带原因的撤销与局部重建入口。原在线策略继续使用共享旧算法。
`--deferred-association` 显式启用，默认配置保持原融合。自动错误发现/撤销规则尚未验收。

### 工程验收

- 维护镜像 `strive-hm3d:local`：13 项 unittest 通过，包括新增协调器、原融合几何回归、
  固定检测输出下的真实几何构建/导出/重载。该集成测试使用固定检测器结果，未验证模型推理。
- UrbanNav CPU 环境：27 项 pytest 通过，覆盖评测、bundle、深度存储和质量比较。
- `py_compile` 与 `git diff --check` 通过。
- 未新增依赖，未调用外部图像模型，未改在线导航决策。

### 冻结观察质量结果

| 输入 | 方法 | TP/FP/FN | F1 | 最终暂缓 | 原有支持 GT 集合保留 |
|---|---|---|---:|---:|---:|
| 最终帧19观察 | 原融合 | 11/2/9 | 0.666667 | 0 | 100% |
| 最终帧19观察 | 新协调器 | 10/1/10 | 0.645161 | 1 | 100% |
| 13帧298观察 | 原融合 | 650/116/657 | 0.627110 | 0 | 100% |
| 13帧298观察 | 新协调器 | 621/116/686 | 0.607632 | 5 | 100% |

19观察 FP 减半，但 F1 下降 0.021505，超过冻结容差0.02。
298观察 FP 没有下降；F1 下降0.019478；对象保留通过。
因此总体质量门槛未通过，停止扩大建图规模，保留显式实验入口。

298观察中14项曾暂缓，9项最终归属（64.29%）：6项最终同组GT一致，1项与纯净参考组GT不同，
2项参考组已混合而无法确定。这里是最终组一致性，不是消歧时刻的因果正确率。
- `frame7:detection27` GT457归入`frame5:detection7` GT428，是明确错误；原因从候选分差转为唯一候选。
- `frame0:detection12` 的参考组涉及GT415/416；`frame0:detection39` 涉及GT402/404；
  两项不能仅凭最终混合组断言消歧当时选错。

结果表明队列机制已工作，但累计几何分差及单个独立观察仍不足以确定身份。
下一步应检查唯一候选为何是错误实例、其原始观察的共同可见区域与空间冲突。
优先改接纳证据，保持覆盖；不通过调宽验收门槛或删除待定观察宣称成功。
局部重建目前由明确纠错调用触发；自动触发与完整导航收益仍待验证。

### 产物与边界

- `outputs/prior_diagnostics/TEEsav_final_owned_association_v1.json`：0.2283秒，10个实例。
- `outputs/prior_diagnostics/matched13_owned_association_v1.json`：16.8851秒，72个实例。
- `outputs/prior_diagnostics/owned_association_evaluation_v1.json`：指标、集合保留、门槛与输入哈希。
- 上述时间仅为新实现重放时间；未完成同环境配对时间与峰值内存验收。
- 本轮没有重跑检测器、完整场景地图或导航episode；查询/可执行视点保留尚未验收。
- 前期无v1后缀的产物保留；最初“3项消歧错误”的描述已细化为1项明确错误、2项不可判定。

## 2026-11-01：误接纳根因定位与固定准入的双向排序对照

上一轮产生了实现与冻结结果，本轮据此继续定位 `frame7:detection27`，没有扩大到全场景。
独立审计：该观察91%像素属于window GT457，9%属于armchair GT428。
错误候选`frame5:detection7`的已接纳成员全部是armchair428；另一候选`frame4:detection0`成员全部是window457。
因此本例可明确区分错误对象与同类部件粒度问题。

### 机制证据

原评分是按点数加权的双向近邻覆盖：
\[
s_{pool}=\frac{N_P c(P,Q)+N_Q c(Q,P)}{N_P+N_Q+\epsilon}.
\]
第7帧错误armchair候选：总分0.483871，观察到参考覆盖0.233618，参考到观察0.878652。
重评时：总分0.587500，两个方向0.254986/0.903794。
正确window候选：总分0.462868，但观察到参考覆盖为1.0，参考到观察0.401306。
累计参考点数可让局部匹配候选获得更高权重；候选分差变大不意味着观察身份得到独立验证。

新增单因素诊断开关 `--deferred-association --balanced-coverage`，固定原pool>=0.25的准入，
仅将候选排序与分差改为 `s_balanced=(c(P,Q)+c(Q,P))/2`。原始成员支持条件保持不变。
新单元测试验证参考点均匀重复时等权分数不变、原评分变化，以及准入集合保持不变。
维护镜像12项协调器测试通过，含固定检测器结果的导出重载。默认配置仍为pool排序。

### 对照结果

| 298观察方法 | TP/FP/FN | F1 | 混合UID | 最终暂缓 |
|---|---|---:|---:|---:|
| 原融合 | 650/116/657 | 0.627110 | 10 | 0 |
| 首版协调器 | 621/116/686 | 0.607632 | 10 | 5 |
| 等权准入+排序（混杂对照，未保留为入口） | 1058/204/249 | 0.823667 | 9 | 8 |
| 固定原准入，仅等权排序 | 626/110/681 | 0.612824 | 9 | 5 |

固定准入版本将具体window观察正确归回window457。9个消歧结果中最终组一致7项、未知2项、
明确错误0项。GT支持集合保留100%。相比原融合FP仅减少5.17%，未达到25%门槛。
19观察结果TP9/FP1/FN11，F1=0.6，仍有2项暂缓；该切片没有达到F1门槛。
因此本例根因修正有效，但整体融合仍未验收，保持实验选项。

第一次等权对照同时改变准入造成混杂，原始产物保留，不能把其F1提升归因于排序改进。
新的诊断方向：检查19观察新增暂缓项和298观察残留混合的观察级空间支持，区分宽mask污染、
实际相邻物体和合理部分视图；维持已冻结门槛，优先控制误合并而不全局收紧准入。

产物：
- `outputs/prior_diagnostics/matched13_wrong_resolution_trace.json`：决策时刻的候选及原始成员分数。
- `matched13_wrong_resolution_audit.json`：运行结束后关联GT的独立审计。
- `matched13_wrong_resolution_receipt.json` / `trace_prior_resolution.py`：输入及脚本身份。
- `matched13_owned_balanced_rank.json` / `TEEsav_final_owned_balanced_rank.json`：最终单因素重放。
- `owned_balanced_rank_evaluation.json`：最终评测。
- `*_owned_balanced.json` / `owned_balanced_evaluation.json`：保留的准入+排序混杂对照。

## 2026-11-01：修复暂缓状态版本超前，并定位残留chair混合

### 确定性状态修复

`add_frame`和`resolve_pending`在固定实例状态上评分，却曾在随后提交时由`_defer`读取最新版本。
当同帧另一观察已先更新候选实例，待定项会把尚未评估的新证据标为已检查，漏掉重评。
现改为显式传入评分时的版本快照。撤销使用撤销后的版本作为等待新证据的基线。

新增测试先在旧实现稳定失败(None而非'a')，修复后通过；维护镜像13项关联测试全部通过。
冻结重放保持balanced候选排序与原准入不变，只改变版本记录：

| 输入 | TP/FP/FN | F1 | 最终暂缓 | 消歧最终组一致/错误/未知 |
|---|---|---:|---:|---|
| 298观察，修复前 | 626/110/681 | 0.612824 | 5 | 7/0/2 |
| 298观察，修复后 | 630/110/677 | 0.615535 | 4 | 8/0/2 |
| 19观察，修复后 | 9/1/11 | 0.600000 | 2 | 0/0/0 |

这是漏触发修复，不能解读为身份判断问题已解决。有效支持对象集合仍全部保留，质量门槛仍未通过。
产物：`matched13_owned_scoring_version.json`、`TEEsav_final_owned_scoring_version.json`、
`owned_scoring_version_evaluation.json`，均位于`outputs/prior_diagnostics`。

### 残留错误分解（基于版本修复前的110个FP）

`matched13_remaining_mixed_decomposition.json`将错误对分解到9个混合UID：
chair组40、bench/wall组20、wall/window组16、不同wall组12、cabinet/wall组7，其余共15。
仅45个FP的两端mask均有超过80%同一GT像素支持；宽mask/边界混杂影响明显，但纯净mask间也有误关联。
该80%统计仅作GT离线诊断，不作为运行时过滤条件。

最大chair组`frame0:detection7`的初始mask本身GT416支持仅49.3%，GT415也占有大量像素。
它作为实例种子，随后同一预测UID吸收两把椅子的多帧观察。已有实例的累计支持不能保证种子身份纯净。
第11帧detection3(GT415,87.5%)与detection6(GT416,85.2%)的mask IoU仅0.0392，却归入同一历史UID；
第12帧detection36与detection37的IoU仅0.00834，分别主导GT416/415。
这给出了后续可用的同帧冲突证据，而非仅凭类别或相近位置猜测。

当前帧选择在固定历史members上判断，尚未联合审查本帧多个拟提交观察间的冲突；同时“其他帧有支持”
可能来自已经污染的成员。下一步应先补齐本帧拟提交组的冲突检测并记录观察级依据，
再评估以独立原始观察支持局部拆分/撤销。禁止用已有混合实例的累计点云自证其身份。
需保留同一对象局部视图的对照，避免把所有低IoU都当作不同实例。

同帧掩码交并、包含率证据：`outputs/prior_diagnostics/matched13_chair_coview_overlap.json`。
本轮未改默认建图或在线策略，未扩大到全场景，未启动外部图像模型。

## 2026-11-01：同帧拟提交冲突审计及负例验证

新增`PriorAssociation._record_coview_conflicts`，在当前帧提交和暂缓消歧提交前检查
同一预测UID的同帧观察；暂缓消歧也包含该帧已提交成员。记录原始观察ID、mask IoU、
双向包含率与提交阶段，仅作为证据，不改变接纳。
维护镜像15项关联测试通过；298观察最终归属与上一轮逐项完全一致，F1仍0.615535。

40对去重冲突中，26对GT相同、12对GT不同、2对GT支持不足无法确定。
同GT的26对包括table15、flower vase6、window3、wall1、cabinet1。
因此低IoU不能直接触发拆分或撤销；这种操作会误伤大量同一实例的局部视图。

进一步对这些冲突的原始过滤点云计算几何（GT仅用于事后分组）：

| 几何量 | 同GT中位数 | 不同GT中位数 | 边界 |
|---|---:|---:|---|
| 最小点间距/m | 0.02276 | 0.01952 | 两组都可为0，不能单独区分 |
| 中心距离/m | 0.16763 | 0.51994 | 同GT最大0.45177、不同GT最小0.14473，有重叠 |
| 第一方向0.1m覆盖 | 0.65152 | 0.33505 | 同GT也有0；方向取决于观察顺序 |
| 第二方向0.1m覆盖 | 0.45833 | 0.13017 | 不同GT也可达1.0 |

这些结果支持继续使用原始观察的多证据判定，否定“低IoU+最小距离”直接当作身份反证。
中心距离和表面覆盖存在组间差异，但样本很小且范围重叠，尚不据此设新生产阈值。
后续局部重建应先验证多视角稳定的分组证据，保留同一对象局部视图作为对照。
当前默认算法与完整地图验收状态保持不变。

产物（outputs/prior_diagnostics）：
- `matched13_owned_conflict_audit.json`：运行时无GT冲突记录，重放20.029秒。
- `matched13_coview_conflict_evaluation.json`：独立GT评测与40对完整记录。
- `matched13_coview_conflict_geometry.json`：原始观察几何及分组统计。
- `inspect_conflict_geometry.py`：冻结几何计算脚本。

## 2026-11-01：原始观察强支持分组与跨位置验证

新增 `scripts/diagnose_prior_support_groups.py`：对冻结预测实例内部所有原始观察对计算双向
0.1m近邻覆盖；两向均>0.85才连边，阈值沿用既有目标融合的强覆盖数值。
图构建不读取GT、不改变归属；连通分量仅作支持证据。记录帧数、相机位置数及源码/输入哈希。
相机位置按0.0001m四舍五入计数，仅用于区分采集位置，不代表独立语义证据保证。

### 13帧局部诊断

779对、246条强边：239条同GT、1条跨GT、6条GT支持不足。
158个分量中156个单GT、1个混合、1个未知；跨帧分量62个，但跨相机位置分量为0。
原始两把椅子的观察分别形成两个跨3帧分量，对应GT416/415；仍有部分GT415观察形成单例。
这13帧来自一个原地扫描位置；RGB文件哈希无完全重复，但80条强边的相机旋转差<1度，
166条>15度。不能把多帧支持等同于跨位置验证。

若将所有分量直接视为实例，TP252/FP2/FN1055，F1=0.322870，明显过度拆分。
该反事实仅离线计算，未应用到地图。

### 最小跨位置检查

追加使用已有捕获的第13—21帧行走观察，共22帧533个观察；没有新增采集、检测或外部模型请求。
协调器重放61.39秒，强支持图6.98秒。此检查用于验证位置变化这一缺失条件，不是完整场景扩测。

3488对、953条强边：931条同GT、1条跨GT、21条GT支持不足。
222个分量中218个单GT、1个混合、3个未知。
跨相机位置分量30个，包含231个观察，独立审计均为单GT；单GT依然只指多数像素GT一致，
不能据此声称mask完全纯净或对象类别正确。
两把椅子的主要组分别跨4个和6个相机位置；同一GT415仍有另一个跨2位置的小分量。

### 局部拆分反事实

按无GT规则：只选择至少存在两个跨位置强支持分量的旧UID，将强分量成员分别赋新身份，
其他观察暂留原身份。共涉及5个UID、131个观察。

| 22帧指标 | 原协调器 | 局部强分量拆分反事实 |
|---|---:|---:|
| TP | 2922 | 1893 |
| FP | 294 | 118 |
| FN | 4283 | 5312 |
| F1 | 0.560791 | 0.410807 |

误合并明显减少，但弱支持观察与强分量之间的正确关联被切断，召回损失超过收益。
因此强分量可作为局部重建的身份参考，不能直接作为最终实例划分。
下一步应验证剩余观察向这些参考组的唯一归属：引用原始成员支持、保留竞争候选，
无法确定者继续待定；必须连同漏关联一起验收，避免只汇报FP下降。
当前未将拆分应用到地图，默认融合及完整地图验收状态保持不变。

产物位于`outputs/prior_diagnostics`：
- `matched13_raw_support_groups.json`、`matched13_raw_support_groups_audit.json`
- `matched_capture_view_independence.json`
- `matched13_support_component_assignment_diagnostic.json`
- `matched22_owned_support_input.json`
- `matched22_raw_support_groups.json`、`matched22_raw_support_groups_audit.json`
- `matched22_local_component_split_diagnostic.json`

## 2026-11-01：弱观察重归属对照与拆分损失根因修正

新增 `scripts/diagnose_prior_group_assignment.py`，固定跨位置强支持组，不让新归属观察更新参考。
只处理存在多个强支持组的5个旧UID。原始成员对pool>=0.25才进入候选，组分数为符合准入成员的
双向等权覆盖最大值；前两组分差>=0.10才接纳，否则暂缓。构造过程不读GT，不修改地图。

131项强组成员之外，实际只有7个弱观察，6项完成唯一归属、1项暂缓。
离线指标：TP1970/FP118/FN5235，F1=0.423975；比直接拆分0.410807有所恢复，
仍明显低于拆分前0.560791。最终待定11项（原10项加1项），质量门槛未通过。

**修正上一轮原因判断：**此前将主要损失归因于弱观察与强组切断，证据不足。
进一步分解剩余952对丢失的正确关联：913对发生在不同强支持组之间，仅39对涉及弱观察。
主要原因是同一真实对象存在多个强支持组，缺少强边不能证明不同身份。
因此继续调弱观察阈值无法解决主要召回损失，暂不将此方案接入地图更新。

### 独立空间证据检查

5个旧UID的跨位置支持组共有12个组对：独立GT审计10对同实例、2对不同实例（两把椅子）。
三维AABB间距：不同椅子的两对均为0，同实例组对可为0至0.017m；边界盒相交无法可靠区分。
中心距离：不同椅子0.345/0.397m，同实例0.074至0.298m。
仅两对不同身份且范围间隔很小，不能据此选择新的通用距离阈值。

下一步应寻找组间不同身份的正面证据（例如跨视角稳定的分割边界与不同表面占据），
并保留同一物体不同局部的对照。强支持组可作为参考证据，不能自动提升为独立实例。
本轮所有变体保持离线，原地图、默认融合和在线导航未改动。

产物（outputs/prior_diagnostics）：
- `matched22_weak_group_assignment.json`：候选分数、支持观察、反事实归属及源码/输入哈希。
- `matched22_weak_group_assignment_evaluation.json`：完整指标及正确关联损失分解。
- `matched22_support_group_extent_pairs.json`：无GT原始三维范围。
- `matched22_support_group_extent_audit.json`：GT事后审计。
- `inspect_support_group_extent.py`：范围计算脚本。
新脚本在维护镜像中实际执行成功；py_compile和git diff --check通过。

## 2026-11-01：部件粒度核对与先验查询实际用途检查

查看frame0/frame11原始RGB并复核GT：检测器标为drawer的frame0:detection16/19/20
位于桌子抽屉区域，GT均归table357，mask支持100%。不同部件存在分割边界是正常现象，
不能将边界直接等价为不同物体。两把椅子仍是不同GT415/416，需要区分整体、部件及邻接物体。
更正前文笔误：frame11:detection3的GT415支持为87.5%；97.2%属于frame1:detection3。

为落实“导航够用”，本轮通过现有 `query_prior_bundle.py` 对已完成的80帧原地图执行
chair/table/window/cabinet/piano/flower vase六项字面概念检索，GT只作事后审计。
这补充功能验收，不替代已冻结的融合门槛。

| 查询 | 首选候选的多数GT证据 | 结果边界 |
|---|---|---|
| chair | 13/13支持观察为chair | 具备正确类别证据 |
| table | 6/8支持观察为table，其余chair | 候选存在混合 |
| window | 9/15支持观察为window，其余wall | 候选存在混合 |
| cabinet | 9项支持观察均为art frame/bookshelf | 字面类别检索缺乏正确证据 |
| piano | 无候选 | 检索缺口 |
| flower vase | 3项支持观察均为decorative plant | 存在语义粒度歧义，暂不判错 |

没有将上述六个查询转换为导航成功率；完整任务语义和在线验证尚未执行。

### baseline巡视视点适配修复

发现 `resolve_habitat_viewpoints` 只接受coverage_scan/continuous_walk，拒绝已有baseline_patrol。
核对collect_hm3d_patrol.py及保存的Habitat配置：RGB/Depth安装偏移均[0,0.88,0]、安装角为0；
记录位姿采用相同OpenCV相机轴和map_z_up转换。
新增明确支持 `(habitat_gt_camera_pose, baseline_patrol)`，该模式原网格哈希从capture_profile读取。
带重建回执时继续校验场景与NavMesh哈希。其他采集类型保持拒绝；未修改运动准入或STOP条件。

维护镜像实际PathFinder验证：针对5个有候选的查询，使用已哈希绑定的wcojb4TFT35-r010-h150
重建NavMesh，从记录的起点查询。chair/table/window/cabinet/flower vase分别有6/6/11/3/3个
可达历史视点，均无拒绝项。24项capture/query-binding回归测试通过。

按现有最短路径选择的视点：chair具备chair证据；table同时含table/chair；window同时含window/wall；
cabinet含art frame/bookshelf；flower vase对应decorative plant。可达不等于目标语义正确。
本轮仅做路径查询，没有实际导航、外部VLM或机器人动作。

后续应针对查询目标检查观察级语义证据与视点选择，避免混合UID的所有观察被当成同一个目标证据；
也需区分对象缺失与类别/部件表示差异。保持当前最终验证和ranking_only权限边界。

产物（outputs/prior_diagnostics）：
- `matched80_*_retrieval.json`：6项实际查询回执。
- `matched80_retrieval_evidence_audit.json`：字面类别证据审计。
- `matched80_*_reachable_views.json`：5项真实PathFinder回执。
- `matched80_retrieval_viewpoint_audit.json`：实际选中视点的独立GT审计。
- `check_prior_retrieval_views.py`：只读路径检查脚本。

## 2026-11-01：查询目标绑定到原始观察，再解析历史视点

修复查询导航把同一融合UID的所有观察都当作目标证据的问题。
`prior_map/query_binding.py::query_evidence_observation_ids` 先复用既有map_id/revision/hash校验，
再从当前bundle读取所选UID的观察，使用原query服务的目标词提取、归一化和文本相关性规则
筛选detector_label匹配的观察。没有新增语义模型或类别表，检测标签匹配仍然只是候选依据。
无匹配证据时显式报错，保留地图中的全部原始观察及冲突证据。

`resolve_habitat_viewpoints(..., observation_ids=...)` 增加观察级范围约束，
拒绝引用所选实例之外的证据；结果记录实际绑定的历史观察ID。
`check_prior_motion.py` 的query-receipt入口先计算该绑定，再解析视点。
直接指定object_uid的诊断入口保持实例级范围。任务版本、运动准入和最终验证规则未改变。

26项capture/query-binding测试通过，包含混合UID按目标分别选择chair/table观察、
地图冲突证据仍保留，以及缺乏匹配证据时显式失败。py_compile及git diff --check通过。

### 真实NavMesh只读对照

使用同一80帧地图、同一起点、同一重建NavMesh和5个查询，实际PathFinder通过。

| 查询 | 可达视点数：绑定前→后 | 绑定后的选中观察审计 |
|---|---|---|
| chair | 6→6 | 3项chair观察，保持 |
| table | 6→6 | 仅保留frame0:detection0（GT table，65.3%） |
| window | 11→9 | 仅保留frame0:detection1（GT window，91.2%） |
| cabinet | 3→1 | frame56:detection17仍为GT bookshelf（85.1%） |
| flower vase | 3→3 | decorative plant，语义粒度歧义保持 |

table/window不再向下游传入同UID中的chair/wall观察。
cabinet结果说明该修改只能消除证据混用，不能纠正检测器自身误识别；不得把绑定成功当作目标验证。
本轮没有实际运动、VLM调用或融合重建，也没有宣称地图关联FP改善。

产物（outputs/prior_diagnostics）：
`matched80_*_bound_reachable_views.json`、`matched80_bound_retrieval_viewpoint_audit.json`、
`check_bound_prior_views.py`。此前未绑定产物保留用于对照。

## 2026-11-01：当前接近与最终验证的深度有效性统一

审计现有链路：历史查询只决定观察位置；到达后重新检测当前候选，再接近、重新采集，
最终通过FinalInstructionVerifier和TaskProgress reducer决定是否发送STOP。
没有发现历史类别直接写入完成状态的路径。

发现确定性几何口径差异：capture_builder调用metric_depth_validity排除已记录的传感器裁剪端点，
prior_approach与arrival_verification此前仅检查depth<=max_depth，可能将量程端点作为近距离表面。
两处现复用同一个有效性函数，保留原平面最近点距离公式：
\[
d=\min_{p\in P_{valid}}\|p_{xy}-x_{camera,xy}\|_2,
\quad P_{valid}=\operatorname{Backproject}(M\land z_{min}<D<z_{max}).
\]
没有裁剪profile的原始深度仍使用原有正深度规则；不把历史传感器profile强加给新模拟器的原始深度。
有效点为空时明确报告Current mask has no valid depth support，保留失败而非生成虚假近距离。

新增两项测试先复现旧实现失败（8点误当有效、全裁剪未报错），修复后通过。
共38项arrival、verification、capture、depth、query-binding回归通过；py_compile及diff检查通过。

保存的TEEsav chair第51帧准备输入重算：原距离与新距离均0.6777199749643563m，
点数均12038。此保存案例没有裁剪profile，故结果保持符合预期。
新产物 `outputs/prior_diagnostics/TEEsav_chair_depth_policy_prepare/` 标明
verification_performed=false、task_completed=false：仅重算输入，不是再次模型验证或导航成功。

这项修复统一了测量事实，未解决错误类别或mask内部跨实例混杂。
后续当前候选的多观察选择、类别错配和完整导航闭环仍需单独验收。

## 2026-11-01：当前候选显式绑定原始观察

当前接近入口此前假定每个候选UID只有一个观察，多观察实例会在解包时失败。
新增current_concept_observation，复用现有目标文本相关性，再以检测置信度及观察ID排序，
返回candidate_uid与observation_id。此选择仍是候选证据选择，不是类别验证。
接近提案和最终验证接收显式观察ID；最终新帧独立选择，沿用当前运动回执绑定。
验证中的对象中心由所选mask的有效深度点计算，证据记录观察ID、mask及深度SHA256，
从而让图像、距离与中心引用同一条原始观察。

41项query-binding、arrival、verification、capture及depth测试通过。
使用保存的TEEsav第51帧及真实NavMesh完成只读接近提案检查：

| 候选 | 绑定观察 | 提案数 | 选中提案路径距离 | 预测可见率 |
|---|---|---:|---:|---:|
| chair | frame51:detection0 | 2 | 0.365m | 0.241 |
| table | frame51:detection1 | 4 | 0.862m | 0.065 |

两者均生成满足现有停止距离准入的提案；table的预测可见率较低，不能将几何准入视为取证质量验收。
产物为outputs/prior_diagnostics/TEEsav_{chair,table}_bound_current_approach.json。
本轮没有运动或模型调用。检测误识别、mask内部混杂及融合质量门槛仍未解决，默认融合保持原配置。


## 2026-11-01：必要交付2——接近评分使用目标俯仰姿态

对保存的第51帧chair/table观察固定点云、内参、位置，仅改变俯仰，水平投影准确复现
原visible_ratio 0.240738/0.064957；对准目标中心后均为1.0。此前低值由评分姿态与执行取证
姿态差异造成，不能据此断言实际取证失败。

`navigation_core/view_geometry.py::rotation_towards`增加默认0的pitch，
`planning/viewpoint_policy.py`增加默认False的align_pitch；仅先验接近入口启用。
公式为theta=atan2(z_target-z_camera, ||target_xy-camera_xy||)，R=R_yaw R_x(theta)。
沿用mapper反射坐标轴，变换行列式为-1；原在线水平评分默认行为保持。

维护镜像中实际NavMesh检查完成：chair候选2→2，table候选4→8，二者最终位置保持；
路径距离分别0.364820m和0.862403m。51项相关回归通过。
产物：outputs/prior_diagnostics/TEEsav_{chair,table}_pitched_current_approach.json。

本轮无导航和模型调用。visible_ratio是已知表面点落入图像范围的比例，未检查新视角遮挡；
评分采用连续目标姿态，执行使用10度离散俯仰、到达误差与实际相机位姿，仍需当前图像验证。
融合质量、固定任务集、手机与实物交付仍未完成。

## 2026-11-01：必要交付2——table接近实际运动与深度取证

首次运动入口在启动simulator前报错：指定当前接近提案仍先调用历史视点解析，
当前单帧motion-arrival包不属于历史采集模式。修复check_prior_motion.py：
当前提案与历史视点分别选择目标，共享场景/网格哈希校验及建网机器人尺寸。
不扩大历史采集模式白名单，不伪装当前观察来源。

修复后维护镜像实际执行完成，产物outputs/prior_motion/TEEsav-table-pitched-approach001。
22动作，0.750000m，零碰撞，reached，目标位置误差0.115613m；
偏航误差3.364351度，俯仰误差-4.323074度，均在离散10度动作的半步范围内。
最终图像rgb/000022.png中可见扶手椅旁边桌；本轮没有检测器或外部模型判断。

将原frame51:detection1的1755个有效深度点投到实际第22帧：全部在画面内，
全部具有有效当前深度，99.886%深度差不超过0.1m；绝对深度误差中位数0.001676m，
被前景遮挡超过0.1m的比例为0。详细记录surface_reprojection.json。
此为预测表面的空间取证验证，类别和指定实例身份仍需当前语义验证。

这次只执行当前接近子过程，不计入完整新指令成功率；不替代固定任务集验收。

## 2026-11-01：必要交付2——table当前检测与验证输入

对TEEsav-table-pitched-approach001最终第22帧复用维护镜像、原SAM/GroundingDINO权重、
原100类词表生成arrival_capture和arrival_bundle。当前table观察为frame22:detection1，
实例d8da6099-bce7-4b94-9cf3-f9e3e65f1876:frame22:detection1。
最终验证输入采用本帧重新检测的mask，3008有效点，平面最近距离0.716948906m。
与上一轮源表面重投影不同，此处是实际新检测证据。

check_prior_arrival.py补齐已有verify_arrival的--observation-id参数，命令行prepare-only实测通过：
一条明确观察，verification_performed=false，completed=false，task_stop_sent=false。
输入及任务分别保存在verification_prepare/、cli_prepare/、task_query.json。

外部豆包调用在执行前被自动审批拒绝，理由是此前图像授权仅覆盖chair；已向用户请求
本次table第22帧/框/派生证据的授权。尚无本次最终语义判断，未发出模型请求。
后续批准后可使用现有check_prior_arrival.py入口，读取task_query.json与显式候选/观察ID。
该入口验证保存证据，不发送STOP；完整任务结果必须使用同会话运动入口验收。

真实手机交付：在当前Code工作空间未找到odometry.csv、camera_matrix.csv或.r3d样本，
已请求真实采集目录及机器人连接/启动配置。不能据此推断其他磁盘不存在数据。

## 2026-11-01：必要交付1/3——统一地图到baseline frontier的坐标绑定

正式benchmark已有PriorMapSimulationRuntime及房间/frontier软排序接入；
此前单例check_prior_motion的直接历史视点导航是另一入口，不能直接充当三组对照。
核对发现统一采集map_z_up=(x_H,-z_H,y_H)与旧查询Habitat水平面(x_H,z_H)不一致。
默认identity未标明新运行平面，可能把搜索偏置引到目标镜像侧。

新增回归首先复现失败：目标(10,-10,1)对应Habitat(10,0,10)，旧实现选择(10,0,-10)。
simulation默认对map_z_up声明同名运行平面，query对明确Habitat/mapper局部来源使用(x,-z)。
已转换的map_z_up位置和实物原生快照继续使用xy；旧Habitat地图保持原投影。
没有新增评分权重或运动控制路径。

15项query/simulation/active-planner/real-robot相关测试通过，包含非零episode原点转换及
原生统一地图点不重复翻转。此为固定任务集对照的接入修复，尚未产生新的导航成绩。
对象加证据分支需要与对象先验共享导航执行器后再进行公平对照；现有直接视点诊断单列。

## 2026-11-01：必要交付6——统一地图包进入共享运行时

PriorMapLoader现接受完整地图包目录（auto或bundle），调用既有load_bundle验证坐标、身份和证据引用，
返回相同PriorMapData。仿真与实物配置入口共享此能力，核心JSON路径继续支持。
地图包引用检查不意味着历史图像已进入高层模型；对象加证据分支的消费链仍待接入验收。

实际80帧地图通过两种运行时配置工厂加载，200对象、map_z_up，核心数据完全一致。
期间发现实物工厂传入high_level_interval但构造器缺参数；补齐参数与成员，复用既有调度规则。
22项capture、simulation、real-robot相关测试通过。产物outputs/prior_diagnostics/bundle_runtime_load.json。
这是本地运行时初始化验证，不是实物执行验收。

用户已明确实施顺序：先仿真，手机扫描及实物接入最后进行；当前无需等待实物输入。

### table保存到达证据的真实豆包验证完成

用户明确授权本次table第22帧/框/派生证据后，调用doubao-seed-2-0-pro-260215，
使用既有关闭thinking的请求路径。返回accept、confidence=0.90、semantic_satisfied=true、
view_sufficient_for_stop=true，认为框内边桌满足Find a table。
当前mask平面距离0.716948906m通过1m约束，reducer completed=true。
receipt位于outputs/prior_motion/TEEsav-table-pitched-approach001/final_verification/receipt.json。

该验证在原simulator会话结束后运行，task_stop_sent=false。它证明这组当前证据和任务判断链路，
不构成新的同会话完整导航成功率样本。此前本组外发授权阻塞已解除。

## 2026-11-01：必要交付3——正式运行时消费历史对象证据

仿真benchmark新增--enable_prior_object_evidence，要求已启用--enable_prior_map_vlm并指定完整地图包。
PriorMapSimulationRuntime在既有高层选择时，取排名首位对象中与当前目标词匹配的原始观察，
按文本相关性、检测置信度、观察ID确定一条参考，生成带框历史图像。
无匹配观察时本次不附加历史图像；暂缓观察不属于已归属实例，不进入参考。

高层选择器仍接收同一BEV、同一候选集合，额外图像被标记为历史检测假设。
map_id/revision、core hash、对象UID、观察ID、mask hash与图像hash进入请求记录和缓存键。
它只参与搜索排序，不写入当前实例身份或任务完成状态。
模型选择、局部导航、当前感知和最终验证继续使用既有执行路径。

30项query-binding、多模态请求和仿真集成测试通过。
实际80帧地图在正式运行时构造请求，mock模型收到2张图（动态BEV＋历史chair框图），
绑定frame11:detection1、对象matched-baseline-concepts100:frame0:detection4。
产物outputs/prior_diagnostics/historical_evidence_runtime/request_wiring.json。
此次仅本地请求构造，无外部图像发送、运动或导航成绩。

对照入口：无先验保持--enable_prior_map关闭；对象先验启用--enable_prior_map与
--enable_prior_map_vlm并指向统一包；对象加证据在相同配置上增加--enable_prior_object_evidence。
后两组保持相同高层选择器，以隔离附加历史原始证据的影响。
固定任务、起点、预算和实际评测仍待运行；不能把mock请求结果视为模型或导航能力验收。

## 2026-11-01：必要交付3——首批三组任务与数据身份冻结

outputs/prior_evaluation/wcojb-three-arm-001/manifest.json冻结三组命令、地图core哈希、
数据哈希及起点/朝向。源为wcojb4TFT35 ObjectNav val，选取原索引0/2/12：
TV、sofa、chair各一例，均为当前先验所在低层起点；每组500步，baseline_guided，
模型doubao-seed-2-0-pro-260215，共9次运行。该批仅为普通目标先导验收，
相似实例、属性消歧、历史对象当前不可见仍保留后续验收范围。

维护镜像实际Habitat加载已通过。加载器from_json显式将episode_id重编号为str(i)，
源ID为2/2/4，运行ID为0/1/2；类别、场景、位置和旋转逐项一致，完整身份互异。
源/运行身份同时保存在manifest和dataset_load_receipt.json，加载器方法源码一并留存。
不得仅按原episode_id判定重复或把运行重编号误当换样本。

本批尚未启动导航。已请求针对9次运行当前图像、BEV及历史框图的豆包外发授权；
已有明确授权覆盖指定chair/table图像，未扩展到本批全部图像。

## 2026-11-01：必要交付3——高层frontier偏好进入当前房间策略

审核baseline_guided调用链确认：select_observed_candidate刷新prior runtime；
此前find_closest_nodes只消费geometry frontier_biases，不读取高层选择器返回的frontier。
因此新增历史图像可能影响模型输出，却无法影响当前房间内选点，仅房间切换路径部分消费。

在既有软排序中加入高层明确选择的frontier，使用与原房间偏好一致的单位分数：
`d_adjusted = d_nav - bias_m * max(geometry_score, selected_frontier_score)`。
现有导航距离、候选节点、偏置权重、可达性和运动所有者保持原实现。
选择不存在于当前候选中的UID不产生分数；prior adapter关闭时保持baseline选择。

新增测试先复现高层选择被忽略，修复后通过；25项planner/query/simulation/multimodal测试通过。
三组正式运行前将当前Python/YAML/shell源文件哈希记录在
outputs/prior_evaluation/wcojb-three-arm-001/source_hashes.json。
没有启动导航或模型调用；本批外发授权仍待用户回复。

## 2026-11-01：必要交付3——实际reset核验与完整任务身份选择

实际Habitat环境发现NUM_EPISODE_SAMPLE会改变迭代器顺序，即使SHUFFLE=False；
首个索引变为sofa，与冻结顺序TV/sofa/chair不同。原select_episode仅验证reset前后
一致，不能证明跨组选择了同一预定任务。

select_episode增加可选expected_identity，按完整身份匹配；benchmark新增--episode_manifest，
读取runtime_episodes并传入。缺失或重复匹配显式报错，旧索引入口保持原行为。
三项episode选择回归通过；三组命令已加入同一清单路径，并更新源码哈希。

维护镜像真实reset完成：TV运行ID0、sofa ID1、chair ID2，身份与冻结清单逐项相同，
起点均可导航；初始distance_to_goal分别11.537122m、3.310102m、4.000765m。
回执outputs/prior_evaluation/wcojb-three-arm-001/environment_receipt.json。
本轮仅reset和传感器初始化，未导航、未调用模型、未发送图像。

## 2026-11-01：三组对照授权与正式启动

用户已明确允许本批9次运行的当前图像、BEV和历史对象框图用于豆包。
新增scripts/run_prior_comparison.py复用原Docker benchmark入口，按manifest运行单组，
启动前后检查源码/数据/地图哈希；结果验证每例完整身份、metrics行数并保留进程退出码和耗时。
旧输出目录明确拒绝复用，失败记录保留，不自动重跑baseline。

当前源码哈希已冻结，无先验组于本次启动，输出
logs/outputs/prior_evaluation/wcojb-three-arm-001/none，
日志/进程回执outputs/prior_evaluation/wcojb-three-arm-001/execution-none。
已实际进入场景和感知初始化，尚未产生完成指标。先验两组尚未启动。
本条只记录启动事实，最终结果以execution receipt及metrics为准。

## 2026-11-01：首例TV成功与多episode编码错误修复

无先验首例TV已完成：success=1，SPL=0.5817417993，distance_to_goal=0.1508264691m。
进入sofa后，AgentRuntime重新读取中文Skill时默认编码为ASCII，触发UnicodeDecodeError。
该批进程退出码1，原记录保留在wcojb-three-arm-001/execution-none；不能将该组记为三例完成。

根因修复为Skill markdown显式read_text(encoding='utf-8')，不修改Skill内容或导航策略。
10项Agent测试通过，新增测试覆盖LC_CTYPE=C下的构造。
运行器支持清单已有--start_episode范围，验证对应身份和该范围metrics数量。

wcojb-three-arm-002保存续跑清单及新源码哈希，引用001中已完成TV指标，
无先验组从索引1续跑sofa/chair；两组先验仍从索引0运行。未重跑TV。
原输入数据、地图、模型和预算保持；仅编码与运行器范围处理改变，分析时保留版本差异。
本条记录续跑启动事实，结果以002/execution-none及其metrics为准。

### 无先验sofa正式失败记录（同批继续）

sofa在第34步内部verifier返回accept/task_complete，但官方success=0、SPL=0，
distance_to_goal=4.581057m。内部预测对象距离0.541902m，二者测量对象不同。
豆包confidence=0.95，描述框内为木框软垫座椅；人工查看check_again/rgb_34.jpg，
框内木质家具外观更像钢琴，附近对象记录包含piano。具体GT实例仍需独立标注核查，
当前确认内部语义接受与官方目标成功不一致。

该失败保留，不中途修改验证器。chair继续同配置运行，尚未完成。

## 2026-11-01：冻结对照的运行异常与加载成本

无先验续跑002的sofa已写入正式失败指标；chair在候选框分类时发生模型协议异常，
进程退出1。原始记录为none/episode-2/lvlm_calls/0017_bbox_object_in_box.json：
输出先出现损坏的JSON键引号，随后连续空白，finish_reason=length，输出4096token，
reasoning_tokens=0，耗时75.14秒。当前证据支持输出退化，不能归因于thinking开启，
也不能将本例记为正常导航失败。成功TV和已完成sofa继续保留。

对象先验组已按002冻结清单启动；运行期间不修改源码和配置。对象加证据组待运行。
维护入口（在STRIVE根目录执行；key-file为本机凭证路径，内容不写入回执）：

```bash
python scripts/run_prior_comparison.py outputs/prior_evaluation/wcojb-three-arm-002/manifest.json --arm objects --check-only
python scripts/run_prior_comparison.py outputs/prior_evaluation/wcojb-three-arm-002/manifest.json --arm objects --key-file /path/to/local/key.txt
python scripts/run_prior_comparison.py outputs/prior_evaluation/wcojb-three-arm-002/manifest.json --arm objects_evidence --key-file /path/to/local/key.txt
```

已有输出的组拒绝重复启动；新尝试须保留原始失败记录并使用新的清单目录。
清单描述固定输入，执行状态以execution-<arm>/receipt.json和实际进程为准。

同一冻结地图的加载成本记录在002/load_resource_receipt.json：
80帧、200对象、2003观察；地图包124365097字节（118.60MiB），核心JSON94906字节。
宿主UrbanNav环境单次完整引用校验加载0.333秒，含导入的进程峰值RSS约62.13MiB。
这是已有文件缓存下、不解码图像的加载测量，不能代表冷启动、建图或导航峰值。
原build_receipt记录建图337.33秒；原采集记录elapsed_seconds=111.14秒。
建图峰值资源与完整在线成本尚缺实际记录，保持未验收。

## 2026-11-01：模型输出诊断、动作域修复与003冻结续跑

002对象先验组在首例TV的首次高层选择中断；两次响应均为损坏JSON，未产生指标。
同图协议诊断保存在002/protocol_check，保持模型、图像、候选及thinking disabled：
显式schema的json_schema首次可解析，json_object仍损坏。随后原始与显式schema
两种请求在TV、chair各一次均可解析。因此错误具有波动性，当前证据不支持将
增加提示词或更换json_object认定为根治；主路径保留原json_schema协议。
官方文档将结构化输出标记为beta：
https://console.volcengine.com/ark/region:cn-beijing/docs/structured-output-beta

诊断中还发现可解析的decision="explore"：schema接受任意字符串，规划器只消费select。
修复为ParsedHighLevelSelection明确select/uncertain及room/frontier/空类型，
提示词声明对应含义，prompt版本更新v2。未知动词进入模型协议错误边界。
另外，候选分类实际发生的LengthFinishReasonError现在与ValidationError共享一次
重新生成预算。原始输出和错误均落盘，第二次失败继续传播；没有扩大token预算、
补写JSON或返回默认识别标签。该变更恢复协议机会，不能保证模型服务始终成功。
25项模型请求记录、先验多模态和规划集成测试通过。

003清单保留001TV与002sofa，none仅从index2重新执行未完成chair；先验两组各三例
尚需执行。440份源码哈希重新冻结，数据及地图保持不变，代码变化写入manifest。
此次先启动none chair。运行状态以003/execution-none/receipt.json及实际进程为准。

## 2026-11-01：无先验三例完成与正式入口边界

003无先验chair正常退出0，74步、路径7.66318m、success=1、SPL=0.509553，
官方distance_to_goal=0.036517m；内部当前对象距离0.331215m。episode驱动229.57秒，
含启动的执行回执244.92秒。运行器核对完整任务身份及冻结源码/数据/地图通过。

003/none_aggregate.json逐例校验身份并引用原始CSV及哈希：
TV(001)成功/SPL0.581742/189步，sofa(002)失败/SPL0/35步，chair(003)成功/SPL0.509553/74步。
三例SR=2/3，平均SPL=0.363765。保留三次版本差异，不重跑已完成baseline；这不是全量HM3D指标。
003对象先验组三例已启动，尚未完成；对象加证据组待执行。

入口审计确认：prior_map/arrival_verification.py和backends/habitat/prior_approach.py
的当前mask/深度绑定用于独立到达入口；正式ObjectNav guided入口仍使用原验证链。
当前三组对照检验先验搜索排序，不能证明新的当前观察几何已接入正式导航入口。
必要交付2已补记这个集成缺口，保持未完成；复杂指令切片同样尚未验收。

## 2026-11-01：003对象先验中断、协议改进004与sofa空间重放

003对象先验TV首次高层选择通过一次重生成恢复；推进到66步后，又连续两次损坏JSON，
进程退出1，未产生完整episode指标。失败回执和原文保留，不能计算该组SR/SPL。

003/decoder_check固定已授权TV和chair输入、同一schema提示，对比服务端json_schema
与text输出后本地校验，共四次均通过。因此尚不能把故障归因于服务端约束解码。
此前无schema提示的高层选择多次失败，而显式schema复验可解析；现复用已有
inject_json_schema为豆包补充同一字段合同，保留API json_schema、本地Pydantic校验、
原始输入/输出及一次重生成边界。25项相关测试通过；完整运行稳定性尚待验证。
004重新冻结源码，沿用数据、地图、模型及预算，保留已完成无先验三例；先启动
对象先验组三例。此次变更属于协议改进候选，不能据局部复验宣称问题彻底修复。

正式链审计：check_again的框来自累计点云投影，_estimate_distance_to_object_final
取整个累计点云最近距离；当前可见区域与测距支持集合尚未统一。
但对sofa失败的实际npz重放表明，最小距离0.541902m的点位于当前图像(311,430)，
点深度0.541785m、传感器深度0.550884m，残差-0.009099m。
回执003/sofa_geometry_audit.json。这排除了该最近点完全不可见这一具体解释；
语义误接受仍待处理，不能声称统一mask/深度就会解决此例。

## 2026-11-01：模型BEV的坐标与可读性根因修复

004实际PNG检查发现文字全部缺失：原_RasterCanvas仅绘制几何，SVG标签未进入模型图像。
同时markers显示robot_position_xy=[0,0]，frontier已是绝对先验坐标。memory.update_from_mapper
直接保存current_position局部坐标，而canonical alignment将其解释为map_z_up。
TV起点的正确平面坐标为[-1.88458,11.11461]，此缺口直接影响模型空间推理。

主动停止004对象先验容器，退出137；原始回执保留，cancellation_reason.json说明原因，
这次记录不计入导航失败或有效对照。无先验三例保持不变。

修复：canonical mapper观察采用p_map=diag(1,-1,1)(p_local+p_initial)，声明map_z_up；
机器人及历史轨迹使用同一坐标。PNG复用已有Pillow/DejaVuSans字体，补齐frontier UID、
机器人、目标T编号及对应完整UID图例。维护镜像依赖现场检查通过，无新增下载。
31项memory/query/multimodal/planner测试通过；真实保存输入重新渲染并视觉检查，
位置与标签可读。重渲染证据为004/bev_corrected/corrected.png和markers。

005冻结新源码并启动对象先验组三例；对象加证据组尚待执行。
仍需完整运行确认先验效果；当前观察几何接入正式验证链及复杂任务切片继续未完成。

## 2026-11-01：005输入生效、批次漏答与006协议边界

005正式BEV确认机器人坐标[-1.884580,11.114610]，模型两次合法选择分别引用T3目标，
证明修复进入真实输入。随后在66步后的概念匹配中断：首次JSON损坏，重生成可解析，
但合并了两项字段，漏掉候选UID，领域检查抛ValueError。该例尚无完整指标。

005/protocol_contract固定批次、选择器、候选框三份实际请求，对比相同schema提示下
json_schema与text输出。text三项均通过本地schema校验，批次UID集合完整；json_schema
在选择器失败，其余两项通过。这是小样本协议证据，不足以断言服务端根因或永久稳定。

006豆包路径采用create(response_format=text)并显式注入已有schema，通过SDK同一个
parse_chat_completion在本地严格解析；截断、schema错误与原文记录保留，一次重生成
预算保持。其他provider维持原路径。结果trace记录structured_output=text_with_local_schema。
28项相关测试通过，维护镜像OpenAI 1.102.0本地解析验证通过。006再次冻结源码、
数据与地图并启动对象先验三例；对象加证据仍待运行。

保留已完成baseline而不重跑，其provider协议与新先验组不同，manifest明确记录
comparison_caveat；未来结果可作为工程对照，不能宣称分离出了先验地图的纯因果增益。

## 2026-11-01：对象先验组三例完成及TV类别冲突证据

006/execution-objects/receipt.json正常完成三例，完整身份及冻结哈希检查通过：
TV success1/SPL0.446012/280步/25.67453m；sofa success0/SPL0/35步；
chair success1/SPL0.509553/74步。SR=2/3，平均SPL=0.318522；
保留baseline平均SPL=0.363765。chair/sofa核心指标相同，退化主要来自TV。
TV首次明显路线分叉在34步，baseline选节点2，先验选节点3；调用数12→24。
协议版本差异继续保留，不能宣称单因素因果结论。

TV误导候选matched-baseline-concepts100:frame7:detection3有四条观察：
frame7 monitor0.48503/mirror0.32236；frame9 monitor0.50739/mirror0.32144。
同帧框区域高度重叠。原图右上为黑框反光墙面物体，外观支持mirror；
具体semantic GT类别尚未加载，但官方唯一TV目标位置已确认在地图另一端。
map_review图的1号对应模型BEV的T3。原始审计006/tv_candidate_semantic_audit.json。
融合内部num_list/conf_list保留多类信息，capture_builder导出只保留胜出label；
检索按该label产生概念匹配，冲突没有传给高层模型。这是当前可定位的信息损失。
不要将此例直接解释为两个空间实例误合并，也不要将几何绑定当作语义修复。

006对象加证据组三例已启动。首个真实请求仍选择靠近T3的frontier1，理由称其为
historical monitor，说明首轮历史图像尚未纠正类别判断；完整任务结果仍待运行。


## 2026-11-01：三组对照完成与共享地图设计更新

9个有效运行已完成，三组均2/3成功；平均SPL：无先验0.3638、对象0.3185、对象加历史图像0.2965。完整任务指标与源文件哈希见[最终汇总](../outputs/prior_evaluation/wcojb-three-arm-006/comparison_summary.json)。baseline保留001/002/003已完成结果，协议版本变化限制因果解释；两个先验组共享006冻结实现。TV额外绕行，sofa三组同样误接受，chair核心结果一致。旧失败尝试不计作独立任务。

现有共享图含200对象、0房间节点、0拓扑边。因此该实验覆盖对象先验偏置，尚未覆盖房间布局与共享地图维护方案。新版[设计第8—9节](prior_map_mode.md#8-共享地图区域决策与导航闭环2026-11-01更新)明确共享图文状态、区域决策、定位轨迹、人工初始化/修正、局部数学模型及参数校准边界。下一步必要工作是上述机制接入和当前验证一致性；实物后置。本轮仅更新设计及结果汇总，没有新增导航实验、修改运行源码或恢复已暂停的实施目标。


## 2026-11-01：共享地图首轮实现与单区域验收

- 维护入口：`docs/prior_map_todo_checklist.md`当前交付清单；历史阶段折叠保留。
- 核心改动：simulation共享快照/区域候选，visualizer朝向与完整已观察轨迹，exploration_policy采用区域子目标并复用baseline局部距离。模型只处理多个可执行区域，单区域确定性执行。
- 必要回归46项通过；新增2个契约用例放在已有测试文件，无新临时测试脚本。
- `shared-map-001`冻结TV实际完成：success=1，SPL=0.5817417993，189步，19.717061m，模型12次，367.5秒。已校验episode身份和冻结源码；与保留baseline的该例核心指标一致。
- 该例全程一个区域，仅证明单区域与导航闭环；多区域选择及先验区域布局绑定仍需继续。
- 产物审计发现最终地图停在最后探索查询（158步），已增加finish_episode终态发布，清除活动选择，补充契约检查4项通过。此后源码与001冻结版本有该终态输出差异，不能把001作为此补充的端到端证明。
- 手机采集、自动重定位、实物运行继续后置；整体六项交付保持未完成。


## 2026-11-01：先验区域到在线frontier的绑定

已将共享地图区域候选从仅在线room扩展到先验多边形：逐个可达frontier查询唯一包含区域，重叠保持在线归属；同一先验区域可以跨多个在线片段，执行只消费该快照绑定的节点。复用原NavMesh距离和局部探索排序。必要回归47项通过，未新增实验运行。

当前用于导航的预测地图仍无区域几何。检查发现维护中的HM3D布局构建器使用GT语义区域与NavMesh；后续需要几何-only入口以产生明确标注unknown的导航区域，而非引入GT房间类型。还需补齐实际多区域选择与终态发布的端到端证据。


## 2026-11-01：几何区域生成与轮廓修复

现有HM3D布局脚本已增加geometry-only入口，仅查询NavMesh；在指定楼层以侵蚀种子和自由空间扩展形成导航区域，全部保留unknown类型。复用原区域/布局格式，增加canonical prior_map.json输出和导航--prior_map_layout组合加载。当前场景生成9个几何区域，scene_id与200对象包一致，标准frame_id=map_z_up。

实际生成图暴露旧_boundary_from_cells使用浮点端点拼边的问题：0.1米栅格共享端点精度不一致，轮廓断开后输出包围矩形。现改用整数半格端点，闭环后统一转米制坐标；保留凹形轮廓，失败明确抛错。必要回归61项通过。

`shared-map-regions-002`已冻结TV任务、对象包、区域布局哈希和源码并启动实际导航；仍需核验多区域LVLM选择和终态轨迹发布。此次几何布局使用仿真NavMesh先验，不能当作纯RGB-D重建或实物验收。


## 2026-11-01：多区域端到端结果与观察语义修复

002冻结运行已完成并校验身份/源码/地图：TV成功，280步、25.674530m、SPL 0.446012，模型21次（区域选择6次），整组536.8秒。第12/34步出现3/2个区域候选，结果实际驱动节点导航；终态280步地图保存280点轨迹，活动选择清空。效率仍低于保留baseline，不能据此宣称收益。

原因审计：旧对象无parent_room_uid，区域候选objects为空；直接对象中心包含仅覆盖13/200，因导航区域表示自由地面而非家具所属房间。现按历史相机位置生成observed-from-region摘要，以唯一(frame,label)计数并保留示例观察ID及索引哈希，不改对象归属。真实摘要覆盖区域1/5/4，各23/53/4帧、38/41/18类；类别冲突保留。固定先验区域ID在frontier更新后保持决策，耗尽再重评。选区提示版本升级为2026-11-01，明确摘要为预测观察来源。

修改后的必要检查60项通过；003冻结同一TV和观察索引哈希后启动，尚待结果。整体任务仍欠正式当前证据一致性、完整代表性验收及后置实物验证。


## 2026-11-01：003观察语义驱动选区的单例结果

`shared-map-semantics-003/execution-objects/receipt.json`正常完成：TV success=1，**149步、16.749999m、SPL=0.684163**，模型13次（选区2次），280.0秒。终态149步地图含149点轨迹，selection清空。

选区依据已由距离转为观察语义：第12步3候选，引用区域1“monitor 8帧”；第71步2候选，引用区域5“monitor 23帧、tv 6帧”；第52、115步为持久区域沿用，未新增模型调用。002同任务依次按距离选区域0/6/6/6/6/5/5/0。单例结果，不作为性能结论；区域1的monitor是否即006审计中的mirror候选尚未核验。产物位于`logs/outputs/prior_evaluation/shared-map-semantics-003/objects/episode-0/prior_map/`。


## 2026-11-01：正式入口取证与停止距离一致性

审计`baseline_guided`调用链：视点评分→motion到达→`rotate_toward_object_for_recheck`→`check_again`→`final_instruction_check`→reducer。图像、bbox、实例与observation_identity原已同帧绑定；不一致点为STOP距离取实例累计点云平面最近点，可能来自当前帧不可见表面。现由同一次投影给出可见支撑集合V（光轴前方、图内、`z_cam-D(u,v)<0.2`）及`d=min_V||p_xy-c_xy||`，`check_again`与最终取证包不再用全点云覆写；投影回执增加`visible_point_index`，图像宽高取自当前深度，替代640×480常量。

评分/执行差异：正式入口未设`arrival_radius_m`，全部视点由`ShortestPathFollower(goal_radius=0.15)`判定到达（旧`step_mod`路径仍以0.5m触发复核，非正式入口）；该判据是两端吸附到NavMesh后的测地距离，XY偏差可超过0.15m（见004记录）；朝向经30°离散转向，相对评分假设的正对目标残留≤15°。`verified`决策日志新增pose_gap，记录observation_id、评分视点、执行位置、位置偏差和光轴残余偏角。

相关回归79项通过；`test_associated_instance_publishes_corresponding_new_image_and_box`两项失败只执行`mapping/object_fusion.py`代码（AST只提取`associate_object_entities`，缺少共享`score_object_candidates`），与本次修改无关，未改动。正式STOP距离变化影响全部臂，冻结baseline结果不可直接对比；sofa失败的最近点原本可见，其语义误接受仍待单独处理。下一步需一次仿真复核以查看实际pose_gap与距离来源。


## 2026-11-01：004五例 none/objects 对照（取证一致性复核）

`shared-map-semantics-004/execution-{none,objects}/receipt.json`均正常完成，wcojb4TFT35单场景5个任务，500步上限，维护镜像`strive-hm3d:local`，正式入口`baseline_guided`。

| 臂 | SR | 平均SPL | 平均步数 | 平均行程 | 模型调用 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| none | 0.6 | 0.306 | 191.2 | 19.22m | 97 | 1914s |
| objects | 0.8 | 0.327 | 274.4 | 32.61m | 168 | 2582s |

| 任务 | none | objects | 分析 |
| --- | --- | --- | --- |
| TV | 189步 SPL 0.582 | 149步 SPL 0.705 | none与shared-map-001逐值一致；STOP距离改为可见支撑未改变该例 |
| sofa(ep1) | 第34步误接受 | 152步成功 | 两臂第34步位姿与视图相同，verifier判定不同；差异来自判定波动，非先验 |
| toilet | 500步超时 | 501步超时 | 先验地图无toilet；objects记录501步，超过上限1步，待查 |
| chair | 74步 SPL 0.510 | 469步 SPL 0.068 | 至第36步相同；objects进入单选项recovery，verifier持续need_better_view，区域策略将机器人带离被暂缓候选约340步，第468步接受同一把椅子 |
| sofa(ep4) | 158步 | 101步 | 历史语义选区缩短搜索 |

取证复核：全部接受的`accepted_distance_source=check_again`；pose_gap朝向残余≤14.9°。位置偏差>0.15m共7次（agent_recovery 0.263–0.281m、baseline_view_feedback 0.317m），对应`motion_terminal`均为`reached`：到达判据是NavMesh吸附后的0.15m测地距离，XY偏差包含视点到NavMesh的吸附位移（推断，尚未逐点用`snap_point`测量）。

结论范围：5例单场景试点，不作为性能结论。暴露的两项问题——先验区域策略与被暂缓候选的竞争、verifier对同一视图的判定波动——记录待处理；按2026-11-01优先级调整，固定任务集验收暂停，先推进手机建图（见checklist“下一阶段”）。

## 2026-11-02：005–007 停止几何根因修复与五例复测

**005**（继承004任务集与地图）：check_again按真实相机高度评分；接近视点取停止可行的最佳投影视点；recovery与search review恢复baseline frontier选项并附加先验区域分支。none臂完成：SR=0.6、平均SPL=0.371、182.0步、18.49m、模型86次、1750s；chair 74→30步，toilet仍501步失败，sofa(ep1)成功而ep4误接受。objects臂第1例崩溃：无词表框过阈值时`C_flag`长度为0，`cv_utils/image_perceiver.py:54`掩码索引越界；修复为保留目标掩码后冻结006，006未执行即由007取代。

**toilet根因：** 视点规划以实例全部点判停止可行，最终verifier以当前帧可见支撑V判距离。马桶低矮，1m内平视时V只含马桶盖，测得约1.05m>1.0m被否决；规划认为可行的视点逐一耗尽后转入探索约290步。

**修复（007）：**
- 停止可行性统一：`planning/viewpoint_policy.py::score_object_view`用与verifier相同的光轴前方、图内点集计算 $d=\min_{p\in V}\|p_{xy}-c_{xy}\|$，停止可行当且仅当 $d+\epsilon_{arrive}\le d_{stop}$；规划阶段无当前深度，不做遮挡检测。
- 俯仰维度：`navigation_core/view_geometry.py`中 $R=R_{yaw}R_x(\phi)$，当前俯仰 $\phi=\arcsin(-R_{22})$；视点和到达后复核在 $\{0,-\theta_{tilt}\}$ 中按（停止可行，投影质量）取最优，`tilt_camera`以 $n=\mathrm{round}((\phi-\phi_0)/\theta_{tilt})$ 执行LOOK_UP/LOOK_DOWN；验证未完成即回平视，避免俯视帧进入后续frontier决策。
- 007冻结包含006之后已提交的手持采集改动（e960fe7、fa579f5、2359453），见manifest `concurrent_changes`。

| 臂 | SR | 平均SPL | 平均步数 | 平均行程 | 模型调用 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 005 none | 0.6 | 0.371 | 182.0 | 18.49m | 86 | 1750s |
| 007 none | 0.8 | 0.512 | 108.2 | 10.72m | 74 | 1278s |
| 007 objects | 0.8 | 0.574 | 89.2 | 9.07m | 74 | 1154s |

| 任务 | 007 none | 007 objects | 分析 |
| --- | --- | --- | --- |
| TV | 189步 SPL 0.582 | 152步 SPL 0.694 | none与001/004/005逐值一致；先验缩短37步 |
| sofa(ep1) | 35步误接受 | 35步误接受 | 两臂与004 none逐值一致：接受0.54m处实例，距目标4.58m；可复现的verifier语义误接受 |
| toilet | 126步 SPL 0.679 | 126步 SPL 0.679 | 接受时pitch=−30°、可见支撑0.86m，首次复核即通过；objects先验首选table，成功来自停止几何修复 |
| chair | 31步 SPL 0.816 | 31步 SPL 0.816 | 004 objects的recovery绕行未再出现 |
| sofa(ep4) | 160步 SPL 0.482 | 102步 SPL 0.683 | 历史语义选区缩短搜索 |

结论范围：5例单场景试点，不作为性能结论；none臂SR提升仅toilet一例可归因于本次修复，ep1/ep4跨批次翻转属verifier判定波动。

**俯仰能力改为平台声明（007运行后）：** 原`_tilt_step`以固定动作号5读取Habitat配置。现`navigation/goal_approach_controller.py::tilt_actions`按名字从`sim_config.agents[0].action_space`取look_up/look_down及其步长；`recheck_pitches`在声明look_down时返回 $(0,-\theta_{tilt})$，否则只返回当前安装俯仰 $\phi_0=\arcsin(-R_{22})$；`tilt_camera`按目标方向选动作，未声明俯仰时不执行动作。Habitat v0动作空间即可模拟固定相机轮式平台，用于“仅统一停止距离、无低头”的对照。实物SysNav回路（`real_robot/sysnav_runtime.py`，运动契约含`LOOK_AT_UNSUPPORTED`）不调用该路径，其视点与停止距离一致性留待部署阶段核对。`tests/test_prior_view_pitch.py`增加固定俯仰用例；维护镜像中`site-packages/tests`遮蔽项目tests包，需在`tests/`目录内以`PYTHONPATH=/workspace/STRIVE python -m unittest test_prior_view_pitch`运行（3项通过）。该文件修改后007冻结哈希不再匹配，007结果以已完成回执为准。

### sofa(ep1) 可复现误接受的离线分析（007 none，仅读已有日志）

- **物体实为立式钢琴。** 早期正面视图（`lvlm_calls/0001`、`0004`）逐框分类均为piano，实例`object:9877…`标签piano，中心(−0.016, 0.70)，尺寸2.10×0.65×0.92m。第30–34步从侧斜方向接近，网格纹理严重扭曲，检测器在同一钢琴上给出sofa（conf 0.80），生成新实例`object:20d9d8…`，中心(−0.099, 0.70)，尺寸1.36×0.65×0.73m，包围盒完全落在钢琴包围盒内。
- **融合路径（推断）：** `mapping/object_fusion.py::associate_object_entities`对目标类有专门分支（150–179行）：与已有实例重叠≥0.25但三项重叠未全>0.85时，新建独立目标实例，不并入已有实例的类别投票。两包围盒包含关系说明表面重叠，<0.25的普通新建路径可能性低；未从保存产物重算重叠分数。
- **语义链只看到同一失真视角：** 新实例的逐框分类（`0012`）、concept match（`0013`）和最终verifier（`0014`，单张0.54m近景，confidence 0.92）都基于第30–34步的扭曲视图，判为“木质带软垫长座椅”。verifier取证包`nearby_objects`已记录0.083m处的piano，但未进入模型请求；verifier按设计只依据当前图像判定类别。
- **结论：** 不是停止几何或先验问题。同一物理表面已有清晰正面视图确认的非目标身份，目标类分支另起无历史的新实例，后续所有语义检查都失去这条反证。005 none与004 objects同例成功，说明模型对该失真视图的判断有波动，不能当作稳定修复依据。
- **候选修复（待定，未实现）：** ①最终verifier附带与候选表面重叠（沿用`score_object_candidates`≥0.25判据）的其它实例的代表观察，作为同一表面的历史图像，由模型独立识别；不改融合和baseline召回。②目标类检测并入重叠实例、以类别投票表达冲突；改动STRIVE原有召回策略，影响全部任务。

### 2026-11-02：008 修法①验证与仅水平视角（Habitat v0）消融

**实现（修法①）：** `objnav_agent_with_process_obs.py::_overlapping_instance_observations`对其它存活实例Q按候选侧覆盖率 $c(P,Q)=|\{p\in P: d(p,Q)<0.1\}|/|P|\ge0.25$（`score_object_candidates`的`overlap_scores1`）筛选，写出Q的`rgb`/`bbox`（最新入图观察）；`check_again`与`_build_final_verifier_evidence`取证包增加`overlapping_instance_observations`。`instruction_adapter/verifier.py`在历史视图之后发送图像、bbox和覆盖率，不发Q的地图标签；`FINAL_VERIFIER_PROMPT`说明该图可能是同一物理物体的另一视角，需独立识别。测试：`tests/test_navigation_root_causes.py::test_final_evidence_views_instances_sharing_candidate_surface`、`tests/test_verification_images.py::test_overlapping_instance_view_is_sent_without_its_map_label`。

**冻结：** `shared-map-semantics-008`（v1，含LOOK_UP/LOOK_DOWN）与`shared-map-semantics-008-level`（同一代码、任务、地图；运行目录内`habitat_lab/configs/tasks/objectnav_hm3d.yaml`改为`ACTION_SPACE_CONFIG: "v0"`、4个动作，经`HABITAT_LAB_PATH`选用并计入冻结哈希）。四臂回执均为complete，任务身份与冻结哈希复核通过。

| 臂 | SR | 平均SPL | 平均步数 | 平均路径 | 模型调用 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| 007 none | 0.8 | 0.512 | 108.2 | 10.72m | 74 | 1278s |
| 007 objects | 0.8 | 0.574 | 89.2 | 9.07m | 74 | 1154s |
| 008 none（v1） | 1.0 | 0.603 | 117.8 | 11.87m | 83 | 1393s |
| 008 objects（v1） | 1.0 | 0.593 | 120.4 | 12.67m | 101 | 1456s |
| 008-level none（v0） | 1.0 | 0.601 | 117.4 | 11.95m | 85 | 1327s |
| 008-level objects（v0） | 0.8 | 0.546 | 96.0 | 9.82m | 80 | 1178s |

| 任务 | 008 none | 008 objects | 008-level none | 008-level objects |
| --- | --- | --- | --- | --- |
| tv(ep0) | 189步 SPL 0.582 | 182步 0.581，verifier 5次 | 189步 0.582 | 182步 0.581，verifier 5次 |
| sofa(ep1) | 第34步拒绝钢琴，83步成功 0.457 | 第34步拒绝钢琴，再拒两次扶手椅，161步成功 0.204 | 第34步拒绝钢琴，82步成功 0.457 | **第34步接受钢琴**，35步失败 |
| toilet(ep2) | 126步 0.679 | 126步 0.679 | 127步 0.679 | 127步 0.679 |
| chair(ep3) | 31步 0.816 | 31步 0.816 | 30步 0.816 | 30步 0.816 |
| sofa(ep4) | 160步 0.482 | 102步 0.683 | 159步 0.471 | 106步 0.655 |

**sofa(ep1)：修法①触发，但未提供独立视角。**
- 六臂第34步当前图`check_again/rgb_34.jpg`逐字节相同；四个008臂的重叠实例图`overlap_34_*.jpg`也逐字节相同，c=1.0。
- 该图就是第34步同一位姿的全景帧：Q在第34步刚被同一观察关联，`ObjectNode.rgb`作为“最新入图观察”已被覆盖为当前视图，早期确认piano的正面视图不在其中。
- 相同输入下模型3次拒绝（理由引用“overlapping observation confirms the same wooden piano”）、1次接受（理由称重叠图“confirms the same instance”为sofa）。007两臂未附该图，均接受。
- 结论：008 ep1的改善不能归因于新的视觉信息，可能来自额外的同视角重复或提示词措辞，属单样本波动范围。根因（失真近景之外缺少该表面的独立身份证据）未解决。修法①的代表观察应取不同于当前观察的视角（例如Q标签形成时的观察），需另行决定。

**v0消融：本组任务未测到俯仰的贡献。**
- none臂v0与v1逐例基本一致（SR 1.0/1.0，SPL 0.601/0.603）。objects臂差异仅来自上述ep1同输入的判定波动，与动作空间无关。
- toilet在v0下同样成功：第121、125步verifier返回need_better_view（可见支撑距离1.49m、1.03m），第126步在0.861m接受，与v1接受距离相同，只多1步和2次verifier调用。可见005 toilet停滞由“规划与验证统一可见支撑”修复，而非LOOK_DOWN。
- 对固定相机轮式平台：本5例上无俯仰不造成成功率损失。场景内无矮小目标，不能推广到需要俯视的物体。
- tv(ep0) objects两组verifier均为5次（007为2次），步数182，属先验路径上的额外复查，未逐次分析。

**5例单场景试点，不作为性能结论。**

## 2026-11-01：A1 深度可通行栅格替代NavMesh（Habitat采集上复核）

实现：`prior_map/hm3d_layout.py`的侵蚀种子+多源BFS分区抽为`build_free_cell_region_layout`，NavMesh与深度共用；NavMesh重建（`outputs/prior_layouts/wcojb-navigation-regions-a1`）与004使用的冻结布局在元数据、多边形、面积、质心、uid顺序上逐项一致。深度栅格`prior_map/capture_builder.py::capture_free_cells`：

- 地面高度z_f：相机下方点高度按0.05m分箱，取众数箱中位数。
- F(c)=1[∃p∈c: |p_z−z_f|<0.2] ∧ 1[∄p∈c: z_f+0.2≤p_z<z_f+h]；F_robot=F⊖Disk(r)；未观测格不属于F。
- 可达性：只保留含相机轨迹投影格的4连通分量。手持采集者走过的地面可达；透过窗口观测到的孤立地面由此排除。

产物：`outputs/prior_layouts/wcojb-rgbd-patrol002/`（`scripts/build_capture_layout.py`，采集`outputs/prior_patrol/matched-baseline-002/capture`，80帧，stride 4，max_depth 6m），`navmesh_overlay.png`中蓝=两者、红=仅深度、橙=仅NavMesh、黑=相机位置。

| 参数 | precision | 已观测区recall | 已观测区IoU |
| --- | --- | --- | --- |
| h=0.88, r=0.18（004运行时机器人尺寸） | 0.872 | 0.848 | 0.754 |
| h=1.5, r=0.10（场景自带NavMesh烘焙参数，`nav_mesh_settings`实测） | 0.867 | 0.889 | 0.782 |
| h=1.5, r=0（不侵蚀） | 0.718 | 1.000 | 0.718 |

对照对象为场景自带NavMesh（区域布局所用，r=0.10、h=1.5）；004运行时NavMesh按机器人尺寸r=0.18、h=0.88重建（`outputs/prior_patrol/matched-baseline-002/runtime_navmesh_receipt.json`），A2按运行时NavMesh对照。

- 地面：估计0.0124m，NavMesh地面0.0164m，误差4mm。
- 仅深度格（r=0.10时341格）：约80%与NavMesh相邻1格，属于格心采样与`round`离散化的沿墙描边；成块的一处（约58格，地图(1.93, 1.77)）位于玻璃推拉门外的室外平台（帧`rgb/000061.png`可见）。深度穿透玻璃，平台经门洞与室内地面连通，轨迹可达性无法排除，留给C6人工核对；手机LiDAR同样穿透玻璃。
- 轨迹可达性移除了不连通的碎块：r=0.18时去掉9个分量共86格；早先12个区域中有8个≤0.1m²的碎块区域，推断来自这类分量（未逐个对应）。
- 区域划分：r=0.10、clearance 0.35时深度布局为2个区域（3.67、22.26m²）；NavMesh布局覆盖整层（含采集未到的北侧），共9个。clearance取0.35/0.40/0.45/0.50/0.55时深度区域数为2/4/3/2/2，非单调：侵蚀种子在窄通道处对1格边界差异敏感。区域边界作为几何划分使用，在手机数据上由C6人工核对，本轮不调参。
- 未解决：多楼层采集上全局众数会选到其他楼层（`wcojb4TFT35-all-islands-coverage`估计到−2.39m），手机扫描按单楼层采集（B2）。

## 2026-11-01：A2 手持相机位姿→底盘候选点（Habitat采集上复核）

实现：
- `camera_to_base_pose`从`backends/habitat/prior_viewpoints.py`移入`prior_map/observations.py`，参数为相机离基座高度h：固定安装取安装高度，手持取逐帧h=z_cam−z_f。
- 按物体/观察筛选证据抽为`prior_map/bundle.py::select_evidence_ids`，Habitat与手持两个解析器共用。
- `prior_map/capture_builder.py::resolve_capture_viewpoints`：p_base=(x_cam, y_cam, z_f)，朝向取光轴水平投影；round(p_base,xy/s)∈F_reach才接受，F_reach即A1的轨迹连通、按机器人半径腐蚀的可通行格。bundle的`build_receipt.capture_manifest_sha256`须等于采集manifest哈希，不一致直接报错。候选记录`reachability_status=trajectory_connected_free_space`，`motion_admission`要求运行时核对机器人尺寸与当前路径；本函数不发运动指令、不改任务进度。
- CLI：`scripts/resolve_capture_viewpoints.py`。契约测试：`tests/test_floorplan_layout.py::test_capture_viewpoints_drop_handheld_poses_onto_reachable_floor`（走过的地面接受；箱体上方、未走到的孤立地面拒绝；采集哈希不一致报错）。

复核：bundle `matched-baseline-concepts100`（采集`matched-baseline-002`，80个observation_pose），栅格按004运行时机器人尺寸r=0.18、h=0.88、0.1m格；对照`runtime.navmesh`上`is_navigable(max_y_delta=0.05)`。产物`outputs/prior_layouts/wcojb-rgbd-patrol002/viewpoint_candidates.json`。

| 栅格r | 两者接受 | 仅栅格接受 | 仅NavMesh接受 | 一致率 |
| --- | --- | --- | --- | --- |
| 0.18 | 76 | 0 | 4 | 0.950 |
| 0.10 | 76 | 0 | 4 | 0.950 |
| 0.25 | 58 | 0 | 22 | 0.725 |

- 仅NavMesh接受的4个是同一位置(0.53, 3.58)的4个朝向：NavMesh障碍距离0.59m；该格在未腐蚀栅格中可通行，但5×5邻域有7格从未被观测（水平相机看不到脚下近处地面），腐蚀按“未观测不可通行”保守拒绝。原因是观测覆盖，判定规则保持不变；手机扫描按B2要求覆盖地面。
- 没有仅栅格接受的视点：栅格没有把NavMesh不可通行处判为可通行。
- 该采集的视点全部位于NavMesh上，拒绝方向的真实数据一致性未测，只由契约测试覆盖。
- 基座高度：栅格候选z=z_f=0.0124；Habitat的z_cam−0.88在部分帧为0.195，而这些帧相机周围0.25m内的深度点都在z_f附近、无抬高表面。推断差异来自NavMesh多边形高度近似（未逐点测量）。

## 2026-11-02：C 阶段——首份手机扫描 d655a5ce86

输入：iPhone 17 Pro + Stray Scanner（App版本未记录），实验室单间约9×10m，57.5s、3447帧，竖握，轨迹9.0m只走中央通道。原始数据`Code/phone_scans/d655a5ce86/`（仓库外）。无卷尺真值（B3未提供）：绝对尺度与物体位置误差均未验收。手机地图尚未在机器人上使用。

### C1 导入

- 首次导入报“Video ended before odometry frame 3446”。根因在上游`DatasetEncoder.swift`：pts取自增前的ARFrame计数，首个保存帧pts=−1/60早于`startSession(.zero)`，被edit list裁掉；默认解码少1帧且RGB整体错位+1帧。改为`OPENCV_FFMPEG_CAPTURE_OPTIONS=ignore_editlist;1`按packet顺序解码：3447帧全部解出，25帧RGB/深度边缘相关峰值均在偏移0。pts间隙来自编码积压丢帧，不影响顺序对应。
- 竖握：roll −95.6°至−83.5°，导入按roll众数整体顺时针转k=1次（RGB/深度/confidence与K、R同步），世界点集不变（5帧稠密核对最大差1.8e−15m）。manifest记`upright_quarter_turns_cw`。
- 无畸变LUT；逐帧fx在1337–1360间变化（对焦）。采集`outputs/prior_capture/phone-d655a5ce86-every10`，every=10，345帧，256×192。

### C2 几何（无卷尺）

- 重力：地面法向与z轴夹角0.41°，平面RMS 1.08cm，相机离地1.41m。
- 跨视角（stride 4，21677对）：短间隔conf2残差中位0.5–1.0cm、5cm内94–98%；conf1中位1.7–2.8cm；conf0中位8–16cm、5cm内仅25–36%。像素占比conf0/1/2=16%/24%/60%。
- 相对尺度：VIO平移乘s≈1.02–1.03时3cm一致率由0.795升到0.89–0.90，与fx存在脊线（fx×0.98+s=1.03同为0.901）。即LiDAR深度与VIO平移约差2–3%，哪一方偏无真值无法判定。
- 回访漂移：7对回访中只有2对高重叠（间隔约35s）ICP收敛，平移4.4–7.3cm、旋转0.5–1.6°；其余因重叠低发散，不计入。

### C3 检测输入

23帧、100类、同一检测器：

| 输入 | 检出/帧 | 物体检出 | 类别数 | 分数中位 | 耗时 | 峰值显存 |
| --- | --- | --- | --- | --- | --- | --- |
| 侧躺 256×192 | 13.2 | 278 | 33 | 0.256 | 49.7s | 6.9GB |
| 竖正 256×192 | 16.0 | 346 | 36 | 0.271 | 49.5s | 6.9GB |
| 竖正 1920×1440 | 17.4 | 382 | 36 | 0.285 | 52.5s | 6.9GB |

方向比分辨率重要（侧躺时键盘判为tv、ironing board 17次）。导入器竖正、保持深度分辨率。

### C4/C5 深度置信度与建图

`metric_depth_validity`增加显式`min_confidence`（缺confidence直接报错），建图`--min-depth-confidence`、布局/视点`--min_depth_confidence`，receipt记`confidence_policy`。三臂同一采集、100类词表、融合参数不变；指标不依赖语义真值：同标签0.5m内成对对象、按证据重建对象点云（5cm体素）后邻域<2点的孤立点占比、2–98%分位跨度。

| 深度 | 对象 | 观察 | 同标签近邻对象 | 孤立点中位/p90 | 跨度中位/p90 | 跨度>3m | 屏幕类(tv/monitor/computer) y<−3 | 全部深度臂≥10观察对象无对应 | 建图耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 全部 | 308 | 5341 | 178 | 2.1%/10.2% | 1.38/3.17m | 36 | 25/79 | — | 1275s |
| ≥1 | 202 | 4409 | 92 | 0.56%/8.8% | 0.97/2.27m | 14 | 3/28 | 7 | 1062s |
| =2 | 140 | 3161 | 63 | 0.28%/6.0% | 0.84/1.87m | 9 | 0/16 | 21 | 904s |

- 全部深度：深色屏幕边缘的conf0飞点使tv对象沿视线拖到(0.2, −6)一线（`phone-d655a5ce86-objects-concepts100.png`）。
- =2：观察减少41%，按类别丢失率shelf 86%、cabinet 78%、speaker 66%、tv 63%、monitor 56%；丢失的21个对象中含(3.57, 1.96)的monitor、(−0.54, −2.02)的chair等真实物体，也含南墙外的疑似重影。
- 选≥1作为C5地图：`outputs/prior_bundles/phone-d655a5ce86-concepts100-conf1`。4409观察证据文件齐全，位姿与采集一致，frame_id即扫描原始帧号。
- 遗留：同标签近邻仍有92个对象，属于关联碎片化，深度过滤无法解决，本轮不改融合参数；显示器多被标为tv。

### C6 区域布局

r=0.10、h=1.5、0.1m格，`outputs/prior_diagnostics/phone-d655a5ce86-layout-conf1/`（`outputs/prior_layouts`为root所有，改写到用户目录）。

| 深度 | 地面格 | 阻挡格 | 可达格 | 可达格中的多视角持续障碍格 |
| --- | --- | --- | --- | --- |
| 全部 | 2304 | 6007 | 342 | 0 |
| ≥1 | 1808 | 2849 | 503 | 0 |
| =2 | 1467 | 1771 | 570 | 5 |

- 全部深度时被堵的格：其中地面点90%为conf2，阻挡点91%为conf0、距离1.4–2.1m，属近距飞点；多视角持续障碍格定义为障碍带点出现在≥5帧且占可见帧≥50%。=2会放开5个这类格，≥1不放开任何一个，故布局与建图同用≥1（`reach_confidence.png`）。
- 区域：1个，clearance 0.25–0.55均为1，与单间一致；三处分支对应桌间过道；x≈1.2处窄颈为采集者走过。南侧y<−4的窗、柜不与轨迹连通，不在可达区，是否隔玻璃未核实。

### C7 导出与查询

- compact（每对象3视角）：202对象、697观察、78MB，2.5s。
- 新进程`query_prior_bundle.py`：sofa 5、refrigerator 1、printer 3、monitor 6个命中，含重载0.3s。
- 证据人工核对（`phone-d655a5ce86-query/evidence_grid.png`）：refrigerator两张证据均正确；sofa首个对象一张为卡座侧板、一张为红色坐垫；printer首个对象一张为图像边缘红色面板、一张为打印机；monitor首个命中是墙上红色标牌。说明碎片化之外还存在跨物体混合与标签错误，查询结果只能作为待现场核验的假设。
- A2：refrigerator 2个候选落在conf≥1可达地面，距物体1.11/1.96m，朝向误差15.0°/29.2°；1个因底座不在可通行区被拒。位置误差无真值。

### C8 成本

| 项 | 值 |
| --- | --- |
| 采集 | 57.5s手持 |
| 建图（≥1，345帧） | 1062s，3.1s/帧 |
| GPU峰值 | 9.4GB（整卡读数，含其他进程约1.4GB） |
| 布局 | 0.8s CPU |
| 地图包 | 完整113MB，compact 78MB |
| 重载+查询 | 0.3s |

图：`outputs/prior_diagnostics/phone-d655a5ce86-bev.png`（点云与轨迹）、`phone-d655a5ce86-objects-concepts100{,-conf1,-conf2}.png`（对象+区域+轨迹）、`phone-d655a5ce86-layout-conf1/prior_map_bev.png`。

### 测试

新增`test_confidence_threshold_is_explicit_metric_support`、`test_low_confidence_depth_neither_blocks_nor_invents_floor`、`test_portrait_scan_is_turned_upright_without_moving_points`；主机489通过，2个失败为既有未提交`mapping/object_fusion.py`所致。

### 边界

无卷尺，绝对尺度、物体位置误差未验收；LiDAR/VIO 2–3%不一致未定位来源；关联碎片化与标签错误未处理；手机地图未在机器人上使用（D阶段）。
