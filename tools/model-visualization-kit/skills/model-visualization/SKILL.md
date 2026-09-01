# Model Visualization Skill

当用户提到模型结构、算子、Tensor shape、dtype、layout、性能瓶颈或算子开发时，优先使用 `model_visualization_kit`。

## 调用策略

1. 先调用 `search_visual_assets`，确认适合当前任务的资产。
2. 如果用户提供模型 JSON 或仓库中的模型描述，调用 `render_model_graph`。
3. 如果用户提供 Chrome Trace/DFX `traceEvents` JSON 并要求泳道图，调用 `render_swimlane`。
4. 如果用户提供 Python/PyTorch 模型源码并要求整网结构，调用 `render_python_model_graph`；它只做静态解析，不执行源码。
5. 如果用户聚焦某个算子，调用 `inspect_operator`，并明确区分事实、静态推断和待实测结论。
6. 用户要求开始开发算子时，调用 `generate_operator_scaffold`，先返回草稿和验证清单，再建议用户审查后接入真实实现。
7. 返回生成的 HTML/JSON 产物路径，并用简短摘要说明最重要的结构、风险和下一步。

## 可信度边界

- 模型 JSON 只能证明输入描述中的结构，不等同于真实后端编译或设备执行结果。
- 不要把静态 shape/dtype 推断描述成真实性能结论。
- 修改源码、运行高开销 profiling 或写入用户项目之前，应获得明确授权。
