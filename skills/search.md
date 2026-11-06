# Object and Relation Search

Read the output of `observe`, then call `query_scene` for the current target in `TaskSpec`. `target_id` identifies a task concept; `candidate_uid` identifies an observed instance. Select the target or anchor according to the task relation, and use `query_viewpoints` to obtain observation viewpoints for that instance.

Use the returned IDs, current `revision`, and `control_epoch` in motion requests. Assign a new `request_id` to each new motion; reuse the original `request_id` when retrying that request.

`TaskSpec` determines task order, quantity, and exhaustive-search requirements. When no candidates are available, obtain further observations through the exploration capabilities exposed by the backend. Report missing capabilities and keep the task incomplete.

In `baseline_guided` mode, the shared room policy handles routine exploration. The model handles only semantic choices or recovery events explicitly presented by the runtime. Continue exploring the current room while it has available frontiers; choose the next room after the current region is exhausted. The navigation service supplies path costs and reachability.
