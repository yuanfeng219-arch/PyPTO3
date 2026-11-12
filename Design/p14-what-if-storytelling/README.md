# P14 · 统一计算模型驱动的三视角联动

打开 [index.html](index.html) 查看独立工作台。默认上下文为 Task 48 / Batch 6 / KV Head 0 / Block 00。可以编辑顶部的 Sequence Length，也可以直接切换原配置和对比配置。合法的新输入会记为对比配置。Reset 恢复 3072 和 1153 示例，并重置三个视图的局部状态。

## 原组件与共享 Scenario

- **Graph** 直接挂载 `PtoVisualization2GraphIntra` 与原 Graph 渲染器。节点、连线、展开、执行次数及层叠继续由原组件生成。Graph 原本就通过 `scenario.task('graph', id)` 接收投影。本次为工作台增加可选的同 Task 焦点与镜头保留，并用输入范围上限的模型投影预留节点层叠空间。当前输入仍决定显示的逻辑执行次数与层数。
- **Assignment** 直接挂载 `PtoVisualization2Anatomy` 的 Q Tile、KV Cache、Stack 和 Page。原生工具栏通过挂载参数移到工作台顶部。数值输入、滑杆、步进、校验状态与 Reset 继续走原监听器。
- **Pipeline** 直接挂载 `PtoVisualization2PipelineIntra`。原生阶段、Logical Tick、READY、数据依赖、Slot 复用和详情保留。工作台增加紧凑间距与同 Task 焦点保留。Graph 与 Pipeline 的 READY 入口仍可打开对应对象。

一次有效输入先经 `PtoComputeScenario.edit()` 校验，再由 `project()` 从同一份 Anatomy 数据生成三路投影。工作台在同一次更新中把当前投影交给三个控制器。存在于新配置中的对象可以保留焦点。超出新输入范围的 Page、Stack、Tick 与复用路径会清除对应选中态。Logical Tick 表达源码循环轮次，硬件执行耗时需要设备采集。

## 投屏适配与改动范围

左侧纵向展示 Graph，右侧以双列 Stack 和原 Q Tile 展示 Assignment，下方放置 Pipeline。工作台共用 PTO `ide-frame` 与设计令牌，直接嵌入组件。三个视图保留各自的等比镜头、缩放与详情入口。Graph 的紧凑标签沿用原控制器的视口标签，Pipeline 调整刻度字号和阶段间距。

`index.html` 提供工作台入口，`story.css` 负责布局与字号，`story.js` 负责统一参数和视图适配。四个原生控制器文件 `visualization2-anatomy.js`、`visualization2-graph.js`、`visualization2-graph-intra.js`、`visualization2-pipeline-intra.js` 增加工作台所需的挂载选项。演讲端可通过 `window.ptoWhatIfStory` 恢复步数与输入上下文。交付入口为独立预览。

## Chrome 验证

| Sequence Length | Graph | Assignment | Pipeline |
| --- | --- | --- | --- |
| 3072 | QK ×24，Online Softmax ×6，Output Normalize ×1 | 24 Pages / 6 Stacks，P23 有效 128 tokens | 8 Ticks，18 个 READY 对象，9 条 Slot 复用关系 |
| 1153 | QK ×10，Online Softmax ×3，Output Normalize ×1 | 10 Pages / 3 Stacks，P9 有效 1 token | 5 Ticks，9 个 READY 对象，0 条 Slot 复用关系 |

验证了真实键盘输入、滑杆、步进、边界校验、三轮来回切换、Reset、Graph 与 Pipeline 的 READY 双向跳转和 Pipeline 源码查看。两配置的 17 个 Graph 节点坐标及 22 条路径一致。有效焦点保留，退出范围的焦点清除。Chrome 未报告未捕获异常。

在 1920×1080、1600×900、1280×720 和 1440×900 检查了基准与对比视图。基准的六组 Stack 完整可见，图例和时间轴分开。720p 总览中的细粒度标签可用对应视图的局部缩放查看。输入上限出现的第七组 Stack 可在 Assignment 内滚动查看。
