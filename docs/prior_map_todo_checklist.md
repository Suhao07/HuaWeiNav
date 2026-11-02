# VLN 先验地图模式待做 Checklist

本文档跟踪先验地图模式的实现工作。先验地图只能提供 context、ranking 和 evidence
summary，不能直接发布导航目标或绕过 final verifier。

## 当前交付清单：仿真共享地图（2026-11-01）

**本节是当前进度入口；后文为旧阶段记录，不代表新版已验收。** 最大复用现有`prior_map`、baseline策略与导航器；实物接入后置。

| 状态 | 必要交付 | 验收依据 |
| --- | --- | --- |
| 已完成 | 核对运行入口、坐标和区域候选，保留已有未提交改动 | 真实Room_node使用room_id、视点使用idx；复用现有NavMesh距离与baseline |
| 单区域仿真已验证 | 共享地图快照：地图/对齐版本、区域候选、位姿和轨迹统一输出 | 图文ID和版本一致、原始地图保持不变 |
| 运行中可视化已验证；终态发布已接入 | 定位轨迹可视化：实际位姿、朝向、轨迹及来源 | 仿真生成可查看SVG/PNG与同版本JSON |
| 单区域、多区域单例已验证 | 区域决策：只向模型提供可执行区域，按事件选择，复用区域内baseline | 选区实际影响导航，单候选不调用模型 |
| 5例仿真已复核（004、007） | 取证与任务闭环一致：接近、图像、实例、距离和完成判断绑定同一份当前观察 | 调用链审计结论；评分姿态与执行姿态差异有记录；正式STOP距离取当前帧可见支撑，视点规划与之同一支撑集合，`verified`日志含pose_gap（含pitch_deg） |
| 004–007已记录；固定任务集验收暂停 | 验证与交付记录 | 复用现有必要回归测试，运行维护中的仿真入口，记录结果与局限 |

## 下一阶段：手机直接构建先验地图（2026-11-01起）

**顺序调整：** 先做手机数据生成同格式地图（设计交付项4），实物部署（交付项5）放最后；固定任务集仿真验收（交付项3）暂停。现有区域布局依赖Habitat NavMesh，手机数据没有NavMesh；导入器把RGB缩到深度分辨率256×192，C3实测竖正后高分辨率只多9%检出，保持该分辨率；首份真实扫描已走完C1–C9（见下）。

**A. 无需手机数据**

| 状态 | # | 工作 | 复用 | 验收 |
| --- | --- | --- | --- | --- |
| 完成（区域划分待C6核对） | A1 | 深度点云可通行栅格替代NavMesh：估计地面高度z_f，F(c)=1[c内有地面点 ∧ c内无z∈[z_f+0.2, z_f+h_robot)的点]，F_robot=F⊖Disk(r)，只保留含相机轨迹格的4连通分量 | 分区抽为`build_free_cell_region_layout`共用；深度栅格`capture_free_cells`；CLI `scripts/build_capture_layout.py`；NavMesh重建与004冻结布局逐项一致 | patrol002（对照场景自带NavMesh，参数r=0.10、h=1.5）：地面估计误差4mm；可通行格precision 0.867、已观测区recall 0.889、IoU 0.782；仅深度格多为沿墙1格离散化，另有透过玻璃门观测到的室外平台；区域数2（NavMesh布局覆盖整层为9），侵蚀种子划分随clearance非单调（0.35–0.55→2/4/3/2/2） |
| 完成 | A2 | 手持相机位姿→底盘候选点p_base=(x_cam, y_cam, z_f)，要求落在A1的F_reach（轨迹连通、按r_robot腐蚀） | `camera_to_base_pose`移入`prior_map/observations.py`共用；证据筛选抽为`select_evidence_ids`；`resolve_capture_viewpoints`与CLI `scripts/resolve_capture_viewpoints.py`；bundle须绑定同一采集manifest哈希 | patrol002的80个视点对照004运行时NavMesh（r=0.18、h=0.88）：76个两者都接受，0个仅栅格接受，4个仅NavMesh接受（同一位置4个朝向，邻域7/25格未被观测，腐蚀保守拒绝）；该数据全部视点在NavMesh上，拒绝方向只由契约测试覆盖；相对当前起点的路径可达性留给运行时 |
| 完成 | A3 | 区域摘要隐藏wall/floor/ceiling/window等结构标签，原始地图保持不变 | `historical_region_observations`；HM3D两个构建器原各有一份相同集合，合并为`prior_map/contracts.py::STRUCTURAL_LABELS`，并补入词表中的场所类staircase/hallway/entryway | 摘要只统计非结构类别，frame_ids仍计全部观察帧；concepts100真实地图3个区域（23/53/4帧）摘要不再含window/floor/door/ceiling/wall，observations.jsonl与prior_map.json未改；契约`test_region_observations_keep_conflicts_and_deduplicate_frames`；HM3D构建器对新增3个场所标签同样过滤 |

**B. 用户提供**

| 状态 | # | 内容 |
| --- | --- | --- |
| 已提供 | B1 | iPhone 17 Pro + Stray Scanner；App版本未记录（输出格式与核对提交`ec3e1dc`一致：逐帧内参、无畸变LUT） |
| 已提供 | B2 | 单楼层扫描 d655a5ce86：实验室约9×10m单间，57.5s、3447帧，轨迹9.0m只走中央通道，竖握、俯仰中位−27° |
| 未提供 | B3 | 卷尺真值：用户不提供（测量不便）。C2绝对尺度、C7位置误差因此无法验收 |

**C. 数据到手后**

| 状态 | # | 工作 | 复用 | 验收 |
| --- | --- | --- | --- | --- |
| 完成 | C1 | 导入：检查畸变LUT，选择抽帧间隔 | `import_stray_capture --every` | 无LUT；every=10→345帧。修复上游视频首帧pts=−1/60被edit list裁掉导致的RGB错位一帧（改为忽略edit list按packet顺序解码，3447帧全部解出，25帧边缘相关峰值均在偏移0）；竖握扫描按roll众数整体顺时针转k=1次，世界点集不变 |
| 部分完成（无绝对尺度） | C2 | 几何验收：多帧投影重合、尺度、重力方向、漂移 | `diagnose_prior_capture.py` | 地面法向夹角0.41°（<2°通过），平面RMS 1.08cm；短间隔跨视角conf2残差中位0.5–1.0cm；卷尺缺失→绝对尺度未验收，相对检验显示LiDAR深度与VIO平移约差2–3%（最优平移比例1.02–1.03，无法判定哪一方偏）；回访漂移仅2对高重叠ICP收敛：平移4.4–7.3cm、旋转0.5–1.6° |
| 完成 | C3 | 检测分辨率对照：同批帧用256×192与更高分辨率RGB检测 | 现有检测器 | 23帧100类：侧躺低分辨率13.2个/帧、竖正低分辨率16.0、竖正高分辨率17.4（+9%，峰值显存相同6.9GB）；方向影响大于分辨率（侧躺把键盘判为tv）。决定：导入器竖正，保持深度分辨率 |
| 完成 | C4 | 深度有效性：仅confidence=2与保留全部深度对融合的影响 | `metric_depth_validity`增加显式置信度阈值；`--min-depth-confidence` | 全部/≥1/=2三臂：实例308/202/140，孤立点占比中位2.1%/0.56%/0.28%，跨度中位1.38/0.97/0.84m；全部深度沿视线拖出一串tv重影（屏幕类y<−3：25/3/0）；=2丢掉深色/反光物体（shelf、cabinet、tv观察丢失56–86%，21个≥10观察的对象无对应）。选≥1 |
| 完成 | C5 | 建图 | `build_prior_capture`，100类词表，融合参数不变 | `outputs/prior_bundles/phone-d655a5ce86-concepts100-conf1`：202对象、4409观察，证据文件全在、位姿与采集一致、frame_id即扫描原始帧号。遗留：同标签0.5m内成对的对象92个；2026-11-02分解为部件/整体、同类相邻物体与少量重复实例，新增建图后重复表面合并（双向覆盖>0.85，冻结重放手机202→197对象、HM3D F1 0.424→0.456），重建为`-conf1-dedup`（197对象，近邻92→87），其余需几何以外证据；显示器多被标为tv |
| 完成 | C6 | 区域布局 | A1，`capture_free_cells`与`resolve_capture_viewpoints`增加同一置信度阈值 | conf≥1（r=0.10、h=1.5）：可通行503格（全部深度342格，conf0近距飞点堵死桌间通道；=2为570格但放开5个多视角持续障碍格）；1个区域，clearance 0.25–0.55均为1；三处分支对应桌间过道；南侧y<−4有窗、柜（conf2深度支撑），其地面不与轨迹连通、不在可达区；是否隔着玻璃未核实；窄通道（x≈1.2）为采集者实际走过 |
| 部分完成（无B3） | C7 | 导出、换进程重载、查询 | compact bundle与loader | compact 78MB/697观察；新进程查询sofa/refrigerator/printer/monitor均有命中（含重载0.3s）。证据人工核对：refrigerator正确；sofa、printer首个对象混入相邻面板；monitor首个命中是墙上红色标牌（真实显示器标为tv）。refrigerator的2个A2候选在可通行地面、距1.1/2.0m、朝向误差15°/29°；位置误差无真值 |
| 完成 | C8 | 可视化与成本 | 现有visualizer | 采集57.5s；建图345帧1062s（3.1s/帧）、GPU峰值9.4GB（含其他进程约1.4GB）；布局0.8s；完整包113MB、compact 78MB；重载+查询0.3s。图：`outputs/prior_diagnostics/phone-d655a5ce86-*` |
| 完成 | C9 | 更新清单与进度文档 | — | 见`prior_map_capture_progress.md` 2026-11-02节；手机地图尚未在机器人上使用，绝对尺度与物体位置误差未经真值验证 |

**D. 后置：实物部署** —— 人工初始化T^M_O=T^M_B,human·(T^O_B)⁻¹；ROS快照驱动先验查询；现有运动所有者执行，现场取证核验后STOP。

**实现要求：** 最小必要改动；明确不确定性；错误直接暴露；共享已有算法；核心数学关系写入注释。不增加临时测试脚本、不重复运行历史baseline实验。

### 本次验证记录

- 必要回归：现有simulation、multimodal、guided navigation、active planner、memory共47项通过；在原测试文件维护3个必要行为契约测试（快照/选区持久性、区域执行、多边形唯一归属）。
- 实际运行：`outputs/prior_evaluation/shared-map-001`，冻结TV一例、500步上限、维护镜像`strive-hm3d:local`、豆包thinking关闭。已完成：success=1、SPL=0.5817418、189步、19.7171m、模型12次，含启动和产物写入367.5秒。核心指标与保留baseline一致；该例全程只有一个可执行区域。
- 运行产物审计发现最终目标接近阶段没有新的高层查询，地图停留在第158步。已补上episode终态发布，保存完整轨迹并清除活动选择，最终发布契约4项通过；该补充尚待实际入口验证。
- **先验区域绑定已实现：** 可达frontier按唯一多边形包含关系绑定先验区域；重叠/未知保留在线区域。同一先验区域可跨在线片段，执行按冻结节点集合进入原距离策略。契约覆盖单在线区域被两个先验多边形划分，以及重叠区域保持未知。
- **几何区域布局已生成：** 复用HM3D布局入口新增geometry-only模式，当前楼层生成9个unknown导航区域及相邻关系。明确使用NavMesh几何先验，未使用GT房间类型；它不是从手机观察重建的结果。
- **轮廓根因修复：** 现有浮点端点拼接会断开栅格轮廓并退为矩形。改用半格整数端点，米制转换只在导出时进行；凹形面积、通道覆盖与坐标契约通过。
- **002实际验证完成：** TV成功，280步、25.6745m、SPL=0.4460，模型21次（选区6次），536.8秒。实际出现多区域模型选择；终态JSON含280个轨迹点，selection已清空。多区域链路有效，但效率低于baseline。
- **002中间审计：** 实际LVLM已接收3区域/2区域候选并驱动导航；对象摘要为空，因旧对象缺少parent_room_uid。简单对象中心包含仅覆盖13/200个对象，导航区域的自由空间不能替代对象房间边界。已使用历史相机位置生成“从该区域观察到”的预测语义摘要：同帧同类别计一次，保留冲突类别和原始观察引用；固定先验区域ID在frontier更新后保持选择。修改后相关回归60项通过；真实摘要覆盖3区域、23/53/4帧。`shared-map-semantics-003`已冻结并运行，同一TV任务验证语义输入和调用次数。
- **003实际验证完成：** TV成功，**149步、16.75m、SPL=0.6842**，模型13次（选区2次），280.0秒；终态JSON含149个轨迹点，selection已清空。选区依据由距离变为历史观察语义：第12步按“monitor 8帧”选区域1，第71步按“monitor 23帧、tv 6帧”切到区域5；第52、115步沿用持久选择、未再调用模型。对照002（仅按距离先进入区域6）步数减少47%。**仅为单例结果，不作为性能结论**；区域1的monitor是否为同类干扰物尚未核验。产物位于`logs/outputs/prior_evaluation/shared-map-semantics-003/objects/episode-0/prior_map/`（容器内save_dir相对`logs/`解析），`manifest.json`的status字段为准备阶段快照，以`execution-objects/receipt.json`为准。
- **取证一致性审计（正式入口`baseline_guided`）：** 图像、bbox与实例原已绑定同一observation_identity；STOP距离原取实例累计点云最近点，可落在当前帧不可见的表面。现由`_project_object_bbox_on_current_view`同时输出bbox与当前可见支撑距离（`distance_source=current_view_visible_support`），`check_again`与`_build_final_verifier_evidence`不再覆写；投影回执增加`visible_point_index`，图像尺寸取自当前深度。评分/执行差异：所有视点到达均由planner判定（正式入口未设`arrival_radius_m`），判据为NavMesh吸附后0.15m测地距离，而非XY距离，004实测XY偏差可达0.32m；朝向经30°离散转向残留≤15°，评分假设正对。`verified`日志新增pose_gap（评分视点、执行位置、位置偏差、光轴残余偏角）。该修改改变全部臂的正式STOP距离，冻结baseline不可直接对比；sofa失败的最近点原本可见，此修复不解决其语义误接受。
- **004五例对照（`shared-map-semantics-004`，wcojb4TFT35单场景）：** none臂 SR=0.6、平均SPL=0.306、平均191.2步、模型97次；objects臂 SR=0.8、平均SPL=0.327、平均274.4步、模型168次。TV 189→149步（none臂与shared-map-001逐值一致）；sofa(ep4) 158→101步；sofa(ep1) none误接受、objects成功，两臂第34步位姿与视图相同，差异来自verifier判定波动而非先验；toilet两臂失败，先验地图无toilet；chair 74→469步，第36步后objects进入单选项recovery，verifier持续need_better_view，区域策略将机器人带离被暂缓候选约340步。objects臂ep2记录501步，超过500上限1步，待查。全部接受距离来源为check_again当前帧可见支撑；pose_gap朝向残余≤14.9°，位置偏差>0.15m的7次均为`reached`。**5例单场景试点，不作为性能结论。**
- **005（`shared-map-semantics-005`）：** check_again按真实相机高度评分、接近视点取停止可行的最佳投影、recovery恢复baseline frontier选项。none臂 SR=0.6、平均SPL=0.371、平均182.0步、模型86次；chair 74→30步，toilet仍失败（501步）。objects臂第1例崩溃：检测器无词表框过阈值时`C_flag`为空，掩码索引越界（`cv_utils/image_perceiver.py:54`），已修复并冻结006；006未执行即被007取代。
- **toilet停滞根因与修复（007）：** 视点规划用实例全部点判停止可行，verifier用当前帧可见支撑；1m内平视只看到马桶盖，verifier距离约1.05m，视点耗尽后探索约290步。现两者共用同一可见支撑（`planning/viewpoint_policy.py::score_object_view`，规划不含遮挡检测），视点与复核在{0, −TILT_ANGLE}中选可执行俯仰，取证前LOOK_DOWN、未完成验证后LOOK_UP回平视。
- **007五例对照（`shared-map-semantics-007`，同004任务集与地图）：** none臂 SR=0.8、平均SPL=0.512、平均108.2步、模型74次；objects臂 SR=0.8、平均SPL=0.574、平均89.2步、模型74次。toilet两臂均126步成功（SPL 0.679），接受时pitch=−30°、可见支撑距离0.86m；objects先验首选对象为table，toilet成功来自停止几何修复而非先验。先验缩短路径：TV 189→152步、sofa(ep4) 160→102步；chair两臂均31步。sofa(ep1)两臂与004 none逐值一致（35步，接受0.54m处非目标实例，距目标4.58m），为可复现的verifier语义误接受：钢琴侧斜失真视图被检测为sofa，目标类融合分支另起无历史实例，正面视图的piano身份未进入后续语义检查（分析与候选修复见进度文档，待定）。**俯仰已改为平台声明：** `tilt_actions`按名字读取动作空间中的look_up/look_down，未声明时（Habitat v0、固定相机轮式平台）复核只用安装俯仰、`tilt_camera`不执行动作；实物SysNav回路不经过该路径，其停止距离一致性留待部署（D）核对。**5例单场景试点，不作为性能结论。**
- **008：修法①验证与v0消融（`shared-map-semantics-008`、`-008-level`）：** 最终verifier附带与候选共享表面（c≥0.25）的其它实例最新观察图，不发其标签。v1 none/objects SR=1.0/1.0、SPL 0.603/0.593；v0 none/objects SR=1.0/0.8、SPL 0.601/0.546。sofa(ep1)四臂第34步的当前图与重叠实例图逐字节相同（重叠图即当前帧，实例rgb已被同一观察覆盖），模型3拒1接，改善不能归因于新视觉证据。**待定：** 重叠实例代表观察改取非当前观察的视角。v0下toilet仍成功（0.861m，多2次need_better_view），本组任务未测到俯仰贡献，固定相机轮式平台无成功率损失（无矮小目标，不外推）。**5例单场景试点。**
- **已知问题（2026-11-02挂起）：** sofa(ep1)钢琴侧斜失真近景被判为sofa，同一输入下verifier结果不稳定（008四臂3拒1接）。修法①保留在代码中，但其代表观察会被当前观察覆盖，未提供独立视角；改为取实例标签形成时的观察留待仿真验收恢复后处理。不阻塞手机建图。
- 明确范围：区域来自现有在线划分，房间类型缺失时保持unknown；连续轨迹使用Habitat真实位姿，初始对齐复用现有alignment配置。本轮不实现自动重定位或实物入口。

<details>
<summary>历史阶段记录（不作为当前执行清单）</summary>

## 0. 当前审计结论

- [x] 当前 1-11 阶段完成的是 prior-map contract、loader、memory、query、artifact 和闭环 smoke。
- [x] 当前 HM3D `*.semantic.txt` builder 只能生成 scene semantic inventory：包含 object label、instance id、region id，但不包含 object position、room boundary、room centroid 或 room-room topology。
- [x] 当前 `wcojb4TFT35` A/B 完整测试验证了 prior-on 不破坏主线：两边均 `success=1.0`、`Episode Steps=283`、`SPL=0.2402441547`。
- [x] 当前 `wcojb4TFT35` A/B 同时证明 `.semantic.txt` prior 没有改变路径：`travel distance`、`accepted_candidate_uid`、`final_stop_accept_step` 完全一致。
- [x] 当前 `.semantic.txt` prior 的 `alignment_type=unavailable`、`enabled_for_ranking=false`，因此 `frontier_biases=[]`，不能作为可导航先验地图。
- [x] 下一阶段目标改为 HM3D ground-truth geometric prior：用 Habitat semantic scene + navmesh/topdown 生成 FloorPlan-VLN 风格的 room/object/topology prior。

## 1. Contract Layer

- [x] 新增 `prior_map/contracts.py`。
- [x] 定义 `PriorMapData`，包含 scene id、rooms、objects、topology、source format、frame id 和 metadata。
- [x] 定义 `PriorRoom`，包含 uid、label、boundary、centroid、neighbors、confidence、source。
- [x] 定义 `PriorObject`，包含 uid、label、position、parent room、exactness、confidence、aliases。
- [x] 定义 `PriorTopologyEdge`，表达 room-room、room-object 或区域连接关系。
- [x] 定义 `PriorObservationRecord`，记录 runtime pose、observed objects、room hypothesis 和 source。
- [x] 定义 `SearchPriorResult`、`RoomPrior`、`ObjectPrior`、`FrontierPrior`。
- [x] 保证 contract 层不 import ROS、Habitat、OpenCV、detector 或 live LLM client。
- [x] 增加 JSON roundtrip 单元测试。

## 2. Loaders

- [x] 新增 `prior_map/loaders.py`。
- [x] 实现 `PriorMapLoader.load(path, source_format="auto")`。
- [x] 支持标准 floorplan JSON。
- [x] 支持 FloorPlan-VLN JSON。
- [x] 支持 OSM/XML 简化格式。
- [x] 支持 HM3D top-down / generated prior map JSON。
- [x] 支持 HM3D scene `*.semantic.txt` 到 canonical `PriorMapData` 的语义清单构建；该来源不是可导航几何先验。
- [x] 为 VLM reconstruction 输出预留 loader。
- [x] 增加最小 fixture 测试，覆盖 room、object、topology 和 metadata。
- [x] 明确 loader 只解析文件，不进行 runtime 坐标对齐和导航决策。

## 3. Alignment

- [x] 新增 `prior_map/alignment.py`。
- [x] 定义 `PriorMapAlignment`，包含 `prior_to_runtime()`、`runtime_to_prior()` 和 `confidence()`。
- [x] 实现 identity alignment。
- [x] 实现 affine 2D alignment：scale、rotation、translation。
- [x] 支持保存和加载 `alignment.json`。
- [x] 增加 inverse consistency 测试。
- [x] 在 diagnostics 中记录 frame id、source points、target points、误差统计。
- [x] 明确坐标对齐失败时 prior map 降级为 prompt context，不参与 frontier/object 排序。

## 4. Memory

- [x] 新增 `prior_map/memory.py`。
- [x] 实现 `PriorMapMemory(base_map, alignment)`。
- [x] 实现 `update_from_mapper(mapper, step)`，用于仿真模式。
- [x] 实现 `update_from_snapshot(snapshot)`，用于实物模式。
- [x] 实现 `mark_room_visited(room_uid, step)`。
- [x] 实现 `mark_object_verified(prior_uid, runtime_uid, step)`。
- [x] 实现 `mark_prior_rejected(prior_uid, reason, step)`。
- [x] 实现 `current_map()`，返回合并 runtime state 后的只读地图视图。
- [x] 增加 visit count、verified/rejected、confidence moving average 测试。
- [x] 确保 memory 不修改原始 `PriorMapData`。

## 5. Query Service

- [x] 新增 `prior_map/query.py`。
- [x] 实现 `PriorMapQueryService.query(plan, runtime_context, memory)`。
- [x] 根据 `InstructionPlan.target_detector_prompts` 和 concept aliases 计算 object relevance。
- [x] 根据 `InstructionPlan.search_priors.room_hints` 计算 room relevance。
- [x] 根据 support objects 和 affordances 生成 support region ranking。
- [x] 根据 visited/exhausted/rejected state 调整 soft score。
- [x] 根据 live observed objects 提高 matching prior 的 confidence。
- [x] 输出 `SearchPriorResult`，包含 score components 和 reason。
- [x] 增加测试：prior 只能改变排序，不能生成 motion goal。

## 6. Policy Adapter

- [x] 新增 `prior_map/policy_adapter.py` 或并入 `prior_map/query.py` 的 adapter 层。
- [x] 实现 `rank_rooms(rooms, prior_result)`。
- [x] 实现 `rank_frontiers(frontiers, prior_result)`。
- [x] 实现 `annotate_target_candidates(candidates, prior_result)`。
- [x] 提供可接入 `planning/room_policy.py` 或 `planning/exploration_policy.py` 排序路径的稳定 API。
- [x] 提供可接入 `planning/target_selection_policy.py` 的 debug annotation，不改变 final verifier 逻辑。
- [x] 增加 A/B 测试：关闭 prior map 时保持现有行为。

## 7. Prompt Context And SoM

- [x] 新增 `prior_map/prompt_context.py`。
- [x] 实现 prior map natural-language summary。
- [x] 实现 OSM-like compact XML summary。
- [x] 实现 `SearchPriorResult` 的 prompt-friendly summary。
- [x] 新增或迁移 SoM visualizer 到 `prior_map/visualizer.py`。
- [x] 输出 global view、room view、markers 和 legend。
- [x] 限制 prompt context 长度，避免把完整地图无节制塞给 LLM。
- [x] 增加测试：marker id 稳定、room/object labels 可追踪。

## 8. Simulation Integration

- [x] 增加 CLI 参数：`--enable_prior_map`、`--prior_map_path`、`--prior_map_source`、`--prior_map_alignment`。
- [x] 在 episode/run 初始化时加载 prior map。
- [x] 在每个 planning cycle 后调用 `PriorMapMemory.update_from_mapper()`。
- [x] 在 room/frontier/object selection 前调用 `PriorMapQueryService.query()`。
- [x] 将 `SearchPriorResult` 作为排序 context 传给 planning policy。
- [x] 保存 `prior_map/query_*.json` 和 `search_prior_result_*.json`。
- [x] 新增 Habitat/ObjectNav dataset 到 canonical `PriorMapData` 的生成脚本。
- [x] 新增 HM3D semantic txt 到 canonical `PriorMapData` 的生成脚本，避免 ObjectNav goal prior 泄漏目标位置；该脚本只用于 semantic inventory smoke。
- [x] 做 1 episode offline smoke，确认无 live API 依赖。

## 9. Real-Robot Integration

- [x] 在实物高层 node 中增加 prior map 参数。
- [x] 用 `SemanticMapSnapshot` 调用 `PriorMapMemory.update_from_snapshot()`。
- [x] 将 `SearchPriorResult` 作为 ranking/context evidence 传给 `SemanticMapSnapshotPolicyContext` / 现有 planning policy adapter。
- [x] prior map 和 live map 冲突时，live evidence 优先，并写入 diagnostics。
- [x] 确认 prior map 不直接生成 `/way_point`。
- [x] 增加 fake snapshot replay 测试。

## 10. Artifacts And Evaluation

- [x] 保存 `prior_map/base_map.json`。
- [x] 保存 `prior_map/alignment.json`。
- [x] 保存 `prior_map/runtime_state_*.json`。
- [x] 保存 `prior_map/query_*.json`。
- [x] 保存 SoM PNG 和 marker metadata。
- [x] 在 metrics/debug 中记录 prior map enabled、top room/object prior、alignment confidence。
- [x] 增加 failure mode 统计：wrong prior、alignment mismatch、live conflict、prior exhausted。

## 11. Acceptance

- [x] Contract/loaders/alignment/memory/query 单元测试通过。
- [x] 仿真 smoke：启用 prior map 后可以完成至少一个 episode，不改变 stop authority。
- [x] 仿真 A/B：关闭 prior map 后行为与当前主线一致。
- [x] 真实 HM3D semantic inventory A/B：同一 episode 下 prior-on 生成 `SearchPriorResult`，prior-off 不生成 prior runtime artifact，但 prior-on 没有改变路径。
- [x] 实物 replay smoke：fake `SemanticMapSnapshot` 能生成 prior-aware `NavigationIntent`。
- [x] 文档同步：`docs/prior_map_mode.md` 与本 checklist 保持一致。

## 12. HM3D Ground-Truth Geometric Prior Builder

- [x] 新增 `prior_map/hm3d_groundtruth.py`。
- [x] 新增 `scripts/build_hm3d_groundtruth_prior_map.py`。
- [x] 定义 `HM3DGroundTruthPriorMapBuilder`，只负责从仿真 ground truth 生成 `PriorMapData`，不依赖 runtime mapper 的已探索点云。
- [x] 定义 `HM3DGroundTruthBuildConfig`：topdown resolution、floor height tolerance、min room area、mask dilation radius、是否拆分 disconnected components、是否保留 structural labels。
- [x] 支持从已有 Habitat simulator 构建：`build_from_sim(sim, scene_id, config)`。
- [x] 支持从 scene directory 构建：读取 `*.basis.glb`、`*.semantic.glb`、`*.basis.navmesh`，初始化 Habitat-Sim 后调用 `build_from_sim()`。
- [x] 从 `sim.semantic_scene.objects` 提取 object uid、label/category、AABB center、AABB size、semantic object id、parent region id。
- [x] 生成 `PriorObject.position_xyz`，并在 metadata 中记录 AABB、semantic id、region id、navmesh snap point、distance-to-navmesh。
- [x] 从 `sim.semantic_scene.regions` 提取 region uid、label、AABB、floor/level id。
- [x] 从 `sim.pathfinder` 或 topdown map 采样可通行 mask，坐标保持 Habitat world frame：2D 平面使用 `(x, z)`，3D 使用 `(x, y, z)`。
- [x] 用 region AABB 与 navmesh traversable mask 相交生成 room mask。
- [x] 对同一 semantic region 内的 disconnected navigable components 做拆分或记录，避免把不可连通区域当成同一个 room。
- [x] 从 room mask 生成 `PriorRoom.boundary_xy`、`centroid_xy`、area、nav sample count、floor id。
- [x] 实现 room-object containment：优先 semantic parent region，其次 object center 是否落入 room polygon/mask，最后使用 nav-snap 最近 room。
- [x] 实现 room-room topology：基于 mask dilation/contact points 和 pathfinder geodesic distance 生成 `room-room connected` edges。
- [x] 实现 room-object topology：为每个 prior object 生成 `room-object contains` edge。
- [x] 写出 canonical `PriorMapData` JSON，`source_format="hm3d_groundtruth_semantic_scene"`，`frame_id="habitat_world"`。
- [x] 写出 `alignment.json`：仿真默认 `identity`、`base_confidence=1.0`、`enabled_for_ranking=true`。
- [x] 明确 builder 不读取 ObjectNav goal positions，不从 episode target 泄漏答案。
- [x] 增加 fake semantic scene + fake navmesh 单元测试：object position、room boundary、centroid、neighbors、topology 全部生成。
- [x] 增加真实 scene smoke：`wcojb4TFT35` 导出的 TV `position_xyz != None`，至少一个 room 有非空 `boundary_xy`，至少一个 `room-room` edge。
  - 容器内 smoke 已导出 `logs/prior_maps/wcojb4TFT35_groundtruth_prior_map.json`：`object_count=779`、`room_count=90`、`topology_edge_count=1039`。
  - TV `prior_object:wcojb4TFT35:tv_349` 使用 `semantic_glb_texture_bounds` 几何来源，`position_xyz=[-0.18259349465370178, -0.9289894700050354, 1.8373835682868958]`。

## 13. Geometry-Aware Query And Ranking

- [x] 扩展 `PriorMapQueryService`，在 prior object 有 `position_xyz` 且 alignment 可用时，按目标对象位置和 parent room 生成几何 search prior。
- [x] 将 target object prior 的 parent room 排名提前到 room ranking，而不是只给所有 unvisited room 相同 `0.2` 分。
- [x] 为 runtime frontier 建立稳定 uid 和 world position，供 `PriorMapQueryService._rank_frontiers()` 消费。
- [x] 实现 runtime frontier/node 坐标到 `habitat_world` 的转换，避免 mapper local frame 和 prior frame 混用。
- [x] 基于 geodesic distance 或 euclidean fallback，计算 frontier 到 top prior room/object 的 score delta。
- [x] `frontier_biases` 必须包含 score components：target object relevance、target room relevance、distance score、alignment confidence、visited/exhausted penalty。
- [x] 当 alignment 不可用或 prior object 无坐标时，显式降级为 prompt context，并在 diagnostics 写明 `geometry_disabled_reason`。
- [x] 增加测试：有几何 prior 时 `frontier_biases` 非空；无几何 prior 时保持现有行为。
- [x] 增加测试：prior 只能改变 frontier/room 排序，不能生成 motion goal、不能绕过 final verifier。

## 14. Active Planner Integration

- [x] 在 active frontier/room selection 前刷新 geometry-aware prior query。
- [x] 将 `frontier_biases` 接入实际使用的 frontier/waypoint selection 路径，而不是只写 artifact。
- [x] 在 `planning/room_policy.py` 中保留 deterministic nearest-frontier baseline，并记录 prior-adjusted distance。
- [x] 在 active non-relocate planning 路径中记录 prior-on 选中的 frontier uid、raw distance、prior score、adjusted score。
- [x] 增加 debug artifact：`prior_map/chosen_frontier_*.json`，包含候选排序前后对比。
- [x] 增加 A/B guard：关闭 prior map 时排序结果和当前主线一致。
- [x] 确认 prior map 不直接设置 `found_goal`，目标发现仍来自 live perception + existing target selection。
- [x] 确认 final stop authority 仍由 physical stop contract + final verifier 决定。

## 15. Ground-Truth Prior Acceptance

- [x] 构建真实 `wcojb4TFT35` HM3D ground-truth prior map。
- [x] 检查 `base_map.json`：TV `position_xyz` 非空、parent room 非空、room boundary/centroid 非空、room-room topology 非空。
- [x] 检查 `alignment.json`：`identity`、`confidence=1.0`、`enabled_for_ranking=true`。
- [ ] 运行同一 episode prior-on/off A/B，确认 prior-on 产生 `frontier_biases`。
- [ ] 运行同一 episode prior-on/off A/B，确认 chosen frontier 或 waypoint 至少在一次 planning cycle 中不同。
- [ ] 记录 first target seen step、final accept step、travel distance、SPL、LLM call counts。
- [ ] 若单 episode 未改善，先确认 prior signal 是否生效；不要只用 success 判断 prior 是否有效。
- [ ] 跑多 episode 统计：至少包含 mean/median first target seen step、success、SPL、travel distance。
- [ ] 增加 failure mode：ground-truth prior unavailable、geometry missing、alignment disabled、frontier bias empty、prior did not affect action。

## 16. 文档同步

- [ ] 更新 `docs/prior_map_mode.md`，区分 semantic inventory prior、ObjectNav goal oracle prior、ground-truth geometric prior。
- [ ] 补充 HM3D ground-truth prior 构建命令和输入文件要求。
- [ ] 补充 A/B 评测标准：必须检查 `frontier_biases`、chosen frontier 差异和 first target seen step。
- [ ] 补充安全边界：prior map 只做 ranking/context，不能生成 STOP 或绕过 final verifier。

</details>
