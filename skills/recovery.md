# Recovery from Blocked Motion

Use a terminal motion receipt directly when the runtime has already supplied one. Otherwise, call `motion_status` to obtain execution feedback.

Preserve current control ownership while motion is running. To switch targets, call `cancel_motion`, then query the motion status to confirm its terminal state.

`unknown` means the execution outcome still requires confirmation. Preserve the original `request_id` and reconcile the request with the execution backend. Releasing `HOLD` is the responsibility of the runtime's operational interface.

Once motion is confirmed blocked, query other available viewpoints. Preserve original execution errors in the tool logs.
