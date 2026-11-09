# Recovery from Blocked Motion

Use a terminal motion receipt directly when the runtime has already supplied one. Otherwise, call `motion_status` to obtain execution feedback.

Preserve current control ownership while motion is running. To switch targets, call `cancel_motion`, then query the motion status to confirm its terminal state.

`unknown` means the execution outcome still requires confirmation. Preserve the original `request_id` and reconcile the request with the execution backend. Releasing `HOLD` is the responsibility of the runtime's operational interface.

Once motion is confirmed blocked, query other available viewpoints. Viewpoint queries do not consume attempts; terminal execution receipts do. If an instance has no untried executable views, inspect other candidates or choose `scan_scene` when available. A pending verified pair is a preference, not an exclusive candidate lock. Deferred evidence is not a semantic rejection. Preserve original execution errors in the tool logs.
