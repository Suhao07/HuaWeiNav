# Object and Relation Search

Read the output of `observe`, then call `query_scene` for the current target in `TaskSpec`. `target_id` identifies a task concept; `candidate_uid` identifies an observed instance. Select the target or anchor according to the task relation, and use `query_viewpoints` to obtain observation viewpoints for that instance.

Use the returned IDs, current `revision`, and `control_epoch` in motion requests. Assign a new `request_id` to each new motion; reuse the original `request_id` when retrying that request.

`TaskSpec` determines task order, quantity, and exhaustive-search requirements. When no candidates are available, obtain further observations through the exploration capabilities exposed by the backend. Report missing capabilities and keep the task incomplete.

The Agent selects every navigation action. The backend supplies room membership, observed room objects, path costs, reachability and coverage recommendations; these do not select or execute a branch for you. Prefer useful frontiers in the current room, then inspect other rooms when local evidence or coverage is exhausted. If no target is eligible, search near a relevant anchor and use `scan_scene` when available to inspect its surroundings.
