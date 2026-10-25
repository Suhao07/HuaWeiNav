# 完成检查
verify_target 返回可接受证据后读取任务进度。顺序任务继续当前未完成目标；数量任务使用不同实例 UID。任务完成时调用 propose_finish，只有 accepted=true 才报告完成。propose_finish 读取完成状态和运动占有状态。完成报告应引用实际验证与执行回执；模型判断本身不构成成功证据。
