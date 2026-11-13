# P06 · 循环执行规模直观可见

打开 [独立演示](index.html) 或 Compute Explorer HTML Deck 的 P06。切换 Stack Count 可以同步查看 KV Cache 的有效 Page、Attention 模块叠层和 QK Matmul 叠层。

| Stack Count | 有效 token 数 | 有效 Page 数 | 末尾 Stack 的有效 Page |
| --- | --- | --- | --- |
| 1（默认） | 128 | 1 | 1 个完整 Page |
| 2 | 769 | 7 | 2 个完整 Page，1 个仅有 1 个有效 token 的 Page |
| 3 | 1536 | 12 | 4 个完整 Page |
| 4 | 2048 | 16 | 4 个完整 Page |

沿用原 Demo 的分块和执行模型。每个 Page 容纳 128 个 token，每个 Stack 容纳 4 个 Page。每个 Stack 对应一轮 Attention 模块，每个有效逻辑 Page 对应一次 QK Matmul（每 AIC、每 Task）。尾页有有效 token 时计为一次，未用的 Page 槽位不计入。右侧在同一组 Stack 和 Page 元素上更新有效容量，左侧复用 Task 48 的原生节点坐标、依赖和 SVG 走线。
