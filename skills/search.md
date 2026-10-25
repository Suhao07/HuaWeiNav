# 对象与关系搜索
读取 observe，按 TaskSpec 的当前目标查询 query_scene。target_id 表示任务概念，candidate_uid 表示观测实例。按任务关系选择目标或 anchor，使用 query_viewpoints 获取该实例的观测视点。运动请求使用返回的 ID、当前 revision 和 control_epoch，每次新的运动使用新的 request_id，重试复用原 request_id。任务顺序、数量与穷尽要求由 TaskSpec 决定。没有候选时依据后端提供的探索能力更新观测；能力缺失应报告，保留任务未完成状态。
