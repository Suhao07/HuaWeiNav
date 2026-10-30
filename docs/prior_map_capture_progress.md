# HM3D 先验采集首轮实现记录

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
