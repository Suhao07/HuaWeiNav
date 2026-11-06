# Completion Checks

After `verify_target` returns acceptable evidence, read the task progress. For ordered tasks, continue with the current unfinished target. For quantity requirements, count distinct instance UIDs.

When the task is complete, call `propose_finish`, which checks task completion and motion ownership. Report completion only when it returns `accepted=true`. Cite the actual verification and execution receipts in the completion report. A model judgment alone is insufficient evidence of success.
