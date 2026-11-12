# P06 · 循环执行规模直观可见

打开 [独立演示](index.html) 即可查看 Task 48 的完整视图。左侧总结 Attention 计算模块 ×6 和 QK Matmul ×24，中间展示 Assignment 的 Block 00 内数据，右侧展示同一 Task 的 Graph。

Assignment 复用 Compute Explorer 的 Block 内渲染器，取其 KV Cache 与 Q Tile 图形。KV Cache 显示 6 个 Stacks、24 个逻辑 Pages。Graph 沿用原生节点布局和 SVG 走线，通过轮廓叠层突出 Stack 级重复与 Page 级重复。

模块次数按有效 Stack 计算。QK 次数按每 AIC、每 Task 计算。这些次数来自源码推导的逻辑计算规模。
