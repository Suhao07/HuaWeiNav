# 主动观测
到达后调用 verify_target，由后端提供满足到达屏障的观测。need_better_view 保留已匹配的对象对，查询同一实例的其他视点再取证。uncertain 表示证据不足；reject_candidate 表示实例反证；reject_relation 仅表示当前对象对不满足关系。使用这些不同结果选择下一次查询，实例计数由领域 reducer 更新。
