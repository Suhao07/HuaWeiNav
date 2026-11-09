# Completion Checks

After `verify_target` returns acceptable evidence, read the task progress. For ordered tasks, continue with the current unfinished target. For quantity requirements, count distinct instance UIDs.

Before finishing, reread the complete original instruction and compare every destination, attribute, relation, reference-object property, prohibition, quantity, order and functional requirement with actual evidence. The structured plan cannot waive a clause by omitting it or calling it a search hint. A contradiction is not a weaker version of the requested property; missing evidence remains unknown. Correct an incomplete interpretation and acquire the missing evidence rather than repeating an unsupported acceptance.

When the full-instruction review and task progress are complete, call `propose_finish`, which checks current full-instruction verification, task completion and motion ownership. A previously accepted candidate does not authorize finishing after another motion, observation or interpretation revision; verify the current stop again. Report completion only when it returns `accepted=true`. Cite the actual verification and execution receipts in the completion report. A model judgment alone is insufficient evidence of success.
