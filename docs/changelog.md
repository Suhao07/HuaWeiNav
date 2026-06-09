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
- 新增 benchmark 参数：
  - `--custom_instruction`
  - `--enable_instruction_adapter`
  - `--instruction_adapter_backend`
  - `--instruction_adapter_strict_classes`
- 每个 episode 保存解析结果：
  - `logs/<save_dir>/episode-*/instruction_adapter/plan.json`
  - `logs/<save_dir>/episode-*/instruction_adapter/spec.json`
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
- mapper 中增加 `instruction_plan` / `instruction_spec` / `target_aliases` 字段，但不改变 STRIVE 的 room/viewpoint/path planner。

### Boundary

- `support_objects` 只进入 room/search prompt context。
- `support_objects` 不进入目标检测停止链。
- parser 不保存“目标常见房间/常见支持物”硬编码表。
- 动态语义边只提供 verifier/cache 接口，具体共视角提取和 VLM 调用由运行时接入。
