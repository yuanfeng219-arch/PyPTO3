# P06 · 循环执行规模直观可见

打开 [独立演示](index.html) 或 Compute Explorer HTML Deck 的 P06。选择逻辑 Block 01 认领的 Task，可以同步查看对应的 KV Cache 分块、Attention 模块叠层和 QK Matmul 叠层。

用「冲印店 / 算子」切换同一张工单的两种表达。冲印店的处理批次对应 Stack，标准照片包对应 Page，照片数量对应有效 token 数。画布用外框叠层表示筛选去重、修图、打印共重复多少批，用筛选节点的叠层表示照片包数量。装帧位于批次处理区之外。

| Task | 默认场景的有效 token 数 | Stack 数 | 有效 Page 数 |
| --- | ---: | ---: | ---: |
| 1（默认选中） | 1025 | 3 | 9 |
| 25 | 128 | 1 | 1 |
| 49 | 3072 | 6 | 24 |
| 73 | 1024 | 2 | 8 |
| 97 | 2560 | 5 | 20 |
| 121 | 3328 | 7 | 26 |

每个 Page 容纳 128 个 token，每个 Stack 容纳 4 个 Page。每个 Stack 对应一轮 Attention 模块，每个有效逻辑 Page 对应一次 QK Matmul（每 AIC、每 Task）。尾页有有效 token 时计为一次，未用的 Page 槽位不计入。切换 Task 时，左侧 Graph 保持节点和依赖布局，更新重复次数。


## SVG 版卡片

以用户提供的 Frame 427321240.svg 为视觉依据，保留原始 593px 宽度和内部排版，上下横线沿用 SVG；背景、边框和圆角使用设计系统的 --card-bg、--card-border、--card-radius，高度随 Stack 行数变化。Block 01 / Tasks 与横向 tabs 按 SVG 定位；选中 tab 和计算重复次数使用 SVG 的 #B1EF6B，三组主数值使用 32px 粗体，计算次数为 22px，标签为 12px。SVG 中的字形已经转为路径，页面使用可交互文本复现其层级。

Stack 使用三列连续灰色分段，每段对应 128 token 容量。Task 1 真实值为 1025、3 Stacks、9 Pages，显示三组，尾页斜线底纹严格复用 SVG 的表达，页面只显示 Stack 编号、P 标签和分段，容量数字及额外引线均移除。P8 有效比例仍为 1 / 128，详细容量写入无障碍标签。Task 121 有七组，使用三行、46px 行距，保持全部内容在卡片内。SVG 示例的 1230、重复 Stack 0 标签和示意数量由真实数据替换。

Deck 与独立页面共用 presentation.js 和 presentation.css。Task 切换、模式切换、叠层联动、翻页和减少动画设置沿用原逻辑。


面板通过本地 fonts/InterVariable.woff2 加载 Inter 可变字体，附官方 OFL 许可证；中文使用系统字体回退。「冲印店 / 算子」已移至页眉，位置、分段按钮和选中态与 P05 一致。Deck 与独立页面均复用该控件。

卡片位置上移 28px。Stack 区进入文档流，由实际可见行撑开卡片高度；数字、标签、字体、分段和底纹保持原有设置。
