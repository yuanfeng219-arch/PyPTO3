# P06 · 循环执行规模直观可见

打开 [独立演示](index.html) 即可查看 Graph 的 QK Matmul 特写和对应 Task 的 Assignment Block 内视图。默认从短序列开始轮播，每个 Task 停留约 4.6 秒。点击任一 Task、上一个、下一个或重置后会暂停轮播。点击「继续轮播」从当前 Task 继续。重置回 Task 24。

| Task | Batch | KV Head | Sequence Length | Pages | Stacks |
| --- | --- | --- | --- | --- | --- | --- |
| 24 | 3 | 0 | 128 | 1 | 1 |
| 65 | 8 | 1 | 769 | 7 | 2 |
| 18 | 2 | 2 | 2048 | 16 | 4 |
| 43 | 5 | 3 | 3584 | 28 | 7 |

四组 Task 覆盖单 Page、尾页、完整中等分块和示例中的最大规模。Assignment 使用原 Block 内渲染器绘制 KV Cache 与 Q Tile。页数、Stack 数和 QK 次数来自同一个 Task 模型。

Graph 沿用 P06 的 Task 48 原生节点坐标与 SVG 走线。以 Task Inputs、QK Matmul 至 PV 的节点、次数标签和最深的 QK 叠层测量裁切边界，视窗在 Partial Context 前结束。所有 Task 共用这套镜头，切换时只改变模块轮廓与 QK 节点叠层的可见数量。次数表示源码推导的逻辑重复，QK 按每 AIC、每 Task 计数。
