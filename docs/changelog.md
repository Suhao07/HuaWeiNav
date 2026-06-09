# Instruction Adapter Changelog

## 2026-06-09

### Added

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
- instruction adapter 保存运行状态：
  - `logs/<save_dir>/episode-*/instruction_adapter/runtime_state_<step>.json`
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
- `semantic_edges` 几何预筛改为使用 STRIVE 点云 z 轴作为垂直方向，并优先使用点云边界判断 `on / inside / under`。
- CogNav episode metadata compiler 增强为 best-effort 读取 `complex_constraints` 中的 relations / attributes / room constraints。

### Boundary

- `support_objects` 只进入 room/search prompt context。
- `support_objects` 不进入目标检测停止链。
- parser 不保存“目标常见房间/常见支持物”硬编码表。
- 动态语义边只提供 verifier/cache 接口，具体共视角提取和 VLM 调用由运行时接入。
- final verifier 不写“红色椅子/TV/客厅”等目标常识规则，只把原始指令、plan、候选实例和证据包交给 VLM 判断。
- 关系约束失败默认拒绝对象对/关系边，不拒绝 detector 类别；属性失败才拒绝具体对象实例。
