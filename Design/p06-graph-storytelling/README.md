# P06 · 循环执行规模直观可见

打开 [独立演示](index.html) 或 Compute Explorer HTML Deck 的 P06。选择逻辑 Block 01 认领的 Task，可以同步查看对应的 KV Cache 分块、Attention 模块叠层和 QK Matmul 叠层。

用「冲印店 / 算子」切换同一张工单的两种表达。冲印店的处理批次对应 Stack，标准照片包对应 Page，照片数量对应有效 token 数。左图用外框叠层表示筛选去重、修图、打印共重复多少批，用筛选节点的叠层表示照片包数量。装帧位于批次处理区之外。

| Task | 默认场景的有效 token 数 | Stack 数 | 有效 Page 数 |
| --- | ---: | ---: | ---: |
| 1（默认选中） | 1025 | 3 | 9 |
| 25 | 128 | 1 | 1 |
| 49 | 3072 | 6 | 24 |
| 73 | 1024 | 2 | 8 |
| 97 | 2560 | 5 | 20 |
| 121 | 3328 | 7 | 26 |

每个 Page 容纳 128 个 token，每个 Stack 容纳 4 个 Page。每个 Stack 对应一轮 Attention 模块，每个有效逻辑 Page 对应一次 QK Matmul（每 AIC、每 Task）。尾页有有效 token 时计为一次，未用的 Page 槽位不计入。切换 Task 时，左侧 Graph 保持节点和依赖布局，更新重复次数。
