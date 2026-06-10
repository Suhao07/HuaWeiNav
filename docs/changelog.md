# Instruction Adapter Changelog

## 2026-06-09

### Added

- 新增统一 Concept grounding:
  - `ConceptQuery`
  - `RuntimeConceptMatcher`
  - `ConceptMatchRecord`
  - target 与 relation anchor 共用同一 prompt-first 概念匹配接口
  - `Constraint.object` 现在会生成 `object_concept`，不再只依赖精确字符串
- 新增 anchor-first relation search 支持：
  - LLM 决定 `execution.mode=anchor_first_relation_search`
  - anchor 只作为导航参考物，不参与最终 stop
  - `AnchorSearchLedger` 记录已搜索失败的 anchor 实例
  - anchor 失败只屏蔽该实例，不屏蔽整个类别/概念
- 新增 `instruction_adapter` 独立模块：
  - `InstructionPlan` / `TargetQuery` / `Constraint` canonical schema
  - `StriveInstructionSpec`
  - `StriveInstructionParser`
  - CogNav `episode.info` metadata compiler
  - CogNav 风格 LLM prompt parser
  - detector vocabulary grounding layer
  - structured prompt context renderer
- 新增动态语义边基础模块：
  - `DynamicSemanticEdgeVerifier`
  - `SemanticEdgeCache`
  - object-object relation geometry prefilter + VLM callback interface
- 新增原始指令终止验证模块：
  - `FinalInstructionVerifier`
  - `CandidateInstance`
  - `VerificationLedger`
  - `VerificationResult`
  - 实例级 hard rejection，避免错误候选反复触发停止
- 新增 Phase 2 运行时约束执行模块：
  - `InstructionExecutionState`
  - `TargetProgress`
  - `ConstraintStatus`
  - `ConstraintEvaluator`
  - 支持 count / any-success / sequence 的任务完成状态
- 新增 Phase 3 动态关系验证运行模块：
  - `InstructionSpatialGraph`
  - `ViewNode`
  - `ObjectNodeRecord`
  - `DynamicRelationService`
  - `VLMRelationVerifier`
  - object-view 共视索引、关系几何预筛、CogNav VLM 按需验证
- 新增 benchmark 参数：
  - `--custom_instruction`
  - `--enable_instruction_adapter`
  - `--instruction_adapter_backend`
  - `--instruction_adapter_strict_classes`
- 每个 episode 保存解析结果：
  - `logs/<save_dir>/episode-*/instruction_adapter/plan.json`
  - `logs/<save_dir>/episode-*/instruction_adapter/spec.json`
- final verifier 保存运行证据：
  - `logs/<save_dir>/episode-*/final_verifier/evidence_<step>.json`
  - `logs/<save_dir>/episode-*/final_verifier/result_<step>.json`
  - `logs/<save_dir>/episode-*/final_verifier/current_bbox_<step>.jpg`
  - `logs/<save_dir>/episode-*/final_verifier/object_crop_<step>.jpg`
- 新增 LLM/LVLM 原始返回追踪：
  - `logs/<save_dir>/episode-*/lvlm_calls/0001_<kind>.json`
  - 保存 `kind`、元数据和 `raw_response`
  - 不保存 prompt 图像 base64，避免日志膨胀
- 新增 `ask_gpt_object_in_box()` bbox crop 缓存：
  - `logs/<save_dir>/episode-*/detection/object_box_cache.json`
  - 同一 crop hash、bbox 和 detector class 列表不重复调用 LVLM
- instruction adapter 保存运行状态：
  - `logs/<save_dir>/episode-*/instruction_adapter/runtime_state_<step>.json`
  - 其中包含 `concept_matches`、`concept_matcher_stats`、`lvlm_call_counts`、`anchor_search_ledger`、`relation_pair_ledger`、`semantic_edges`
- `metrics.csv` 新增 instruction-level 字段：
  - `instruction_success`
  - `instruction_decision`
  - `instruction_accept_step`
  - `accepted_candidate_uid`
  - `accepted_relation_edge`
  - `lvlm_call_count_by_type`
  - `lvml_call_count_by_type`
- 新增中文设计文档：
  - `docs/instruction_adapter.md`

### Changed

- 指令解析从本地硬编码 ontology 改为 prompt-first 分层：
  - 优先复用 CogNav episode metadata。
  - metadata 不足时调用 LLM 输出结构化 plan。
  - `tv_monitor -> tv` 等类别差异移到 grounding 层处理。
- adapter 开启时，STRIVE 不再用 `ask_gpt_similar_objects()` 扩展目标同义词。
  - 目标检测词来自 `StriveInstructionSpec.target_detector_prompts`。
  - 新 grounding 层仅在精确匹配不足时可复用 legacy LLM similarity fallback。
- `mapper.object_found_no_gpt()` 从裸 `tag == target` 改为 normalized target alias 匹配。
- `mapper.object_found_no_gpt()` 会跳过已被 final verifier hard-rejected 的同一对象实例，但不会屏蔽同类别其它对象。
- `objnav_agent_with_process_obs.py` 在几何 `final_check()` 通过后新增原始指令满足度验证；只有 `decision=accept` 才允许最终 stop。
- final verifier 改为可插拔 `auto` 模式：普通 benchmark 没有 `InstructionPlan/InstructionSpec` 时直接旁路，不改变 STRIVE 原始停止逻辑。
- mapper 中增加 `instruction_plan` / `instruction_spec` / `target_aliases` 字段，但不改变 STRIVE 的 room/viewpoint/path planner。
- 启用 instruction adapter 时，`mapper.object_found_no_gpt()` 在 sequence 模式只暴露 active target，防止后续子目标提前触发 stop。
- `objnav_agent_with_process_obs.py` 在 final verifier 前调用 `ConstraintEvaluator`：
  - attribute/room 作为原始指令 verifier 的显式证据；
  - relation 走 geometry prefilter + VLM dynamic semantic edge；
  - final accept 后由 execution state 判断 count/sequence 是否真正完成。
- `mapper.object_found_no_gpt()` 在 instruction mode 中改为 runtime concept matching：
  - terminal target 由 `RuntimeConceptMatcher` 判断；
  - anchor-first 模式下，没有 terminal candidate 时可导航到未搜索 anchor；
  - anchor reference 到达后写入 `AnchorSearchLedger` 并继续搜索，不能作为成功。
- relation constraint anchor 匹配改为 concept-instance match：
  - 优先使用 `constraints[*].object_concept`；
  - runtime mapper label 不需要和 instruction 文本完全相同。
- 指令模式下 `check_again()` 改为 evidence-only：
  - 普通 benchmark 仍保留原 `check_again_object_in_bbox()` 类别复核；
  - instruction mode 只采集 bbox 图、几何和附近对象证据；
  - `FinalInstructionVerifier` 统一判断 candidate、relation、instruction satisfaction 和 view sufficiency。
- `FinalInstructionVerifier` 输出增加视角反馈字段：
  - `semantic_satisfied`
  - `view_sufficient_for_stop`
  - `view_feedback`
  - `preferred_view_goal`
  - `view_objective`
  - 只有语义和视角质量同时满足才允许 stop。
- 新增 `ViewControlState`：
  - `need_better_view` 后不再只做一次 retry；
  - 保存 baseline view quality、多个 viewpoint proposals、attempt history；
  - final verifier 会看到 `view_control` 历史；
  - 如果 evidence 没有相对 baseline 足够改善且仍有 proposal，控制层继续
    `need_better_view`，避免过早 stop。
  - 首次 `accept` 前也会执行 initial-accept deferral：若存在明显更好的通用几何
    proposal，则先采集更好 evidence，再允许最终停止。
  - better-view 子目标会 pin 已验证的 `DynamicSemanticEdge`，避免靠近后
    mapper 实例 uid 漂移导致重新验证其它 pair。
- relation constraint 在 view-control active 时优先复用 `pinned_relation_context`。
- `check_again` 强视觉证据下，关系几何预筛失败可降级为 VLM override，并绕过旧的
  geometry failure cache。
- `FinalInstructionVerifier` 新增通用 final view guard：
  - 只使用 `view_quality_facts` 中的投影、中心偏移、边界余量和 bbox 面积；
  - 语义正确但停止视角过差时强制转为 `need_better_view`；
  - 不写目标/anchor 类别规则，阈值可通过 `STRIVE_FINAL_VIEW_*` 环境变量配置。
- `whether_to_check_again()` 从旧的早停阈值改为通用候选视角评分：
  - visibility；
  - centerability；
  - border margin；
  - projected area；
  - distance suitability。
  该评分只负责生成更清晰证据，最终语义仍由 final verifier 判断。
- 新增 `RelationPairLedger`：
  - 关系失败只拒绝 `(terminal_uid, relation, anchor_uid)`；
  - 关系成功写入 `SemanticEdge`，后续按动态语义边复用。
- `RuntimeConceptMatcher` 新增 `match_many()` 批量接口：
  - exact/cache 先行；
  - 非精确候选合并成一次 prompt-first grounding；
  - 小 crop 会自动 resize，避免 VLM provider 因尺寸过小拒绝请求。
- `RuntimeConceptMatcher` 记录真实调用统计：
  - `batch_llm_calls` 是实际批量请求次数；
  - `batch_items_requested` 是批量请求覆盖的候选 object 数量；
  - `cache_hits` 和 `exact_matches` 用于区分“模型没被调用”和“模型被调用但批量处理”。
- CogNav / Gemini / OpenAI compatible `parse()` 统一经过 trace wrapper，未显式标注的调用会退化为 `parse`，STRIVE 关键调用已补充语义化 `trace_label`。
- `semantic_edges` 几何预筛改为使用 STRIVE 点云 z 轴作为垂直方向，并优先使用点云边界判断 `on / inside / under`。
- CogNav episode metadata compiler 增强为 best-effort 读取 `complex_constraints` 中的 relations / attributes / room constraints。

### Boundary

- `support_objects` 只进入 room/search prompt context。
- `support_objects` 不进入目标检测停止链。
- parser 不保存“目标常见房间/常见支持物”硬编码表。
- 名称泛化不写 `shelf -> bookshelf`、`book -> books` 这类代码规则；所有非精确匹配交给 ConceptQuery grounding 与 RuntimeConceptMatcher，并记录到日志。
- 动态语义边只提供 verifier/cache 接口，具体共视角提取和 VLM 调用由运行时接入。
- final verifier 不写“红色椅子/TV/客厅”等目标常识规则，只把原始指令、plan、候选实例和证据包交给 VLM 判断。
- 关系约束失败默认拒绝对象对/关系边，不拒绝 detector 类别；属性失败才拒绝具体对象实例。

### Fixed

- 修复 execution mode 判断使用 `normalize_term()` 导致下划线模式失效的问题。
  - `anchor_first_relation_search` 现在会正确进入 anchor-first 分支。
  - `all_targets_success` 等计数/多目标模式也统一使用 `compact_key()` 比较。
