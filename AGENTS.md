# Codex Instructions

These instructions apply to code and documentation under `Agent_part/RouteServer`.

## Code Documentation

When adding or modifying Python code:

- Every public class must have a docstring.
- Every public function or method must have a docstring.
- Docstrings should include:
  - A short purpose summary.
  - `Args:` for parameters.
  - `Returns:` for return values.
  - `Raises:` when the function may raise meaningful exceptions.
- Dataclass docstrings should document constructor fields under `Args:`.
- Private helpers should have concise docstrings when their behavior is not obvious or when they implement benchmark semantics.
- Core algorithms, metrics, ranking rules, and geometry calculations must include mathematical definitions in the docstring when applicable.
- Use LaTeX-style formulas for math, for example:

```text
\[
SPL = RS \cdot \frac{L(P^*)}{\max(L(P^*), L(\hat P))}
\]
```

## Benchmark And Evaluation Code

For dataset, benchmark, and evaluator code:

- Keep schema, IO, metrics, runner, exporter, and CLI concerns separated.
- Evaluators must be offline and deterministic.
- Evaluation code must not call live map APIs, LLMs, or external mutable services.
- Use frozen cache or replay data for reproducibility.
- Match predictions to ground truth by stable ids first, then use coordinate fallback only as an explicit tolerance rule.
- Treat tool-call traces as diagnostics unless a benchmark explicitly defines tool-call quality as a metric.
- Keep the main benchmark metric set small; add extra quantities as debug or slice analysis unless explicitly promoted.

## Architecture And Iteration Management

When implementing core code:

- Manage iterations explicitly. Prefer extending stable interfaces over duplicating near-identical versioned implementations.
- Start from the most general and minimal base abstraction, then implement specialized variants through inheritance, adapters, or small composition layers.
- Keep provider, backend, runner, evaluator, builder, and visualization responsibilities decoupled.
- Do not mix multiple feature versions, provider-specific branches, and shared utility functions into one large script.
- Extract reusable utilities into library modules when two or more scripts need the same behavior.
- CLI scripts should be thin orchestration layers: parse arguments, call library code, and write artifacts.
- New versions should make their behavioral differences explicit through class names, config fields, or strategy objects, not hidden conditionals scattered through unrelated code.
- 禁止补丁式修改：核心功能必须从原理性建模、稳定接口和可复用抽象出发实现，不为单个样例、短期现象或局部失败堆叠特例分支。

## UrbanNav Benchmark Rules

For `instruction -> destination -> route` benchmark work:

- GT cases must preserve `acceptable_destinations`, `ranked_destinations`, `optimality_scope`, and `cache_keys` when available.
- If a prediction matches a non-rank-1 acceptable destination, route metrics must use that matched destination's reference route and reference distance.
- Do not compare a route to rank-1 GT when the predicted destination is another acceptable target.
- `destination_success`, `hit@5`, `route_success`, `median_route_ratio`, and `spl` are the main single-target metrics.
- Coordinate radius arguments are fallback matching tolerances, not metric names.

## Testing

For new benchmark/evaluator behavior:

- Add focused unit tests for schema parsing, IO roundtrip, destination matching, route success, SPL, and edge cases.
- Include tests for non-rank-1 acceptable destination matching when route metrics depend on the matched destination.
- Run targeted tests after implementation.

## Editing Discipline

- Keep changes scoped to the requested task.
- Do not modify unrelated files.
- Prefer existing project patterns over introducing new abstractions.
- Avoid broad refactors unless they are necessary for the requested work.
- If a script grows to contain multiple versions or unrelated tools, split it before adding more functionality.
