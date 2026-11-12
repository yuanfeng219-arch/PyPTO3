# P06 · 循环执行规模直观可见

在 Chrome 中打开 [独立演示](index.html)。底部可逐步前进、后退或重置，也可直接选关键帧。左右方向键逐步切换。「完整慢放」从 Stack 0 连续播放到完整 Graph，播放时可暂停和继续。选择「展开 Graph」可单独重放该段。

| 场景 | 逐步看到什么 |
| --- | --- |
| 从 1 个 Stack 开始 | 先亮起 Stack 0，再依次展开 P0–P3。每个 Page 触发一次 PV 矩阵乘贡献，部分结果沿链传给下一个 Page，最后交给 Context Accumulate |
| 扩展到 2 个 Stack | Stack 1 从 P4–P7 形成自己的 Partial Context。前一 Stack 的 Context 经重标定后，与当前 Partial Context 合并 |
| 从 Stack 到 Graph | PV 节点从局部计算链移到 Graph 的位置。Stack 0 和 Stack 1 先进入同一条分块带，其余 Stack 逐个加入，随后补齐上游和跨 Stack 状态 |
| 完整回看 | Task 48 的 6 个 Stacks、24 个 Pages 对应 6 次 Matmul 和 18 次 Matmul Accumulate。完整 Task Graph 保留 PV、Partial Context、Context Accumulate 和输出依赖 |

Task 48 的 Batch、KV Head、有效长度和分块使用 Compute Explorer 模型。逐 Page 操作读取 `PtoVisualization2GraphIntra.pageInstances`，整体使用 `PtoStudioComputationGraph.renderScene`、`applyExecution` 及已有语义依赖。舞台动画按讲述顺序揭示逻辑依赖。设备调度、READY 同步时序、实际指令和硬件耗时需由独立运行证据验证。
