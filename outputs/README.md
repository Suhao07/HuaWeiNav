# Generated Outputs

本目录只保存本地运行生成的产物，不保存项目源代码或正式设计文档。除本说明文件外，
内容默认被 Git 忽略。

常见子目录：

```text
outputs/docs/figures/     文档图表 PNG/SVG 渲染结果
outputs/lvlm/             LVLM 服务预检、schema smoke 和部署验收回执
outputs/benchmark/        可选的 benchmark 汇总结果
```

实验过程日志、视频和逐 episode 产物使用 `logs/`；模型权重、数据集和 ROS build/
install/log 目录不应复制到这里。正式文档只记录生成命令、输入版本和输出 schema，
不把某次运行的结果伪装成项目文档。
