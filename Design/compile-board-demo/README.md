# 昇腾计算可解释/可观测体系演示

第一页为 `intro.html`（项目背景与用户问题：左边是 PyPTO 的 Tensor → Tile → 芯片三层，右边是三个用户断点，各自链到对应的一幕；内容取自 `Product_Planning/` 与 `Insight/` 中引用的 Issue 和 `repo/pto/README.zh-CN.md`），点「进入演示」到首页；首页左上角可回到背景页。

首页为 `index.html`，提供算子开发、可视化编译和性能调优三个入口，性能调优直接进入第三幕 Device Perf Lane（`result.html`）。算子开发进入 `workbench.html`，点击「编译并上板」后经过模拟进度，进入设备性能分析。

编译与上板均为模拟。算子开发页面使用 CSA 编译产物，性能分析使用同一运行的预置 trace；页面交互不会触发重新编译，也不会改变结果数据。页面复用相邻 Demo 的资源，需保留其相对目录结构；可直接打开或通过本地静态服务器访问。
