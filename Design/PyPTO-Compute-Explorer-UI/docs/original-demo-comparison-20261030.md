# 原版 Demo 对照审计 · 2026-10-30

## 基线与备份

- 远端：git@github.com:yinyucheng0601/PyPTO3.git，main。
- 现场通过 git ls-remote 核实提交：3524a5e3c85c845b1c84ed936bec51457281422a。
- 本地已有同一提交对象，使用 git archive 导出完整仓库快照，不改工作树、不切分支、不覆盖当前文件。
- 备份：/Users/yin/PyPTO3-main/.codex_work/original-demo-3524a5e3/，目录被现有 .gitignore 忽略。
- 原版入口：http://localhost:8874/Design/PyPTO%20Compute%20Explorer/index.html，点击 Compute Explorer。
- 当前入口：http://localhost:8873/Design/PyPTO%20Compute%20Explorer/explorer.html。
- 原版证明：BACKUP-PROVENANCE.json；目标 Demo 全部 Git 文件逐字节核对提交对象。

## 结论

不能认定“功能和交互保持不变”。底层计算模型与映射保留，但 UI 工作已改动浏览手势、相机范围、层级入口和视觉编码。当前密集图元和字体覆盖是新增问题。

| 项目 | 原版 | 当前版 | 判断与证据 |
| --- | --- | --- | --- |
| Task / Block 映射 | 16 Batch × 8 KV = 128 Tasks；T % 24 分配 | 同一映射 | 模型文件未变，布局测试通过 |
| Task 内部编码 | Generation 显示 5 个 Q Head 圆点；Assignment 淡出 Q、显示按 512 tokens 分段的序列容量 | 两阶段同时显示 Q、Stack/Page、图标及 Task 标识 | 已改变信息编码；浏览器在 Assignment 端点实测原版 Q opacity=0，当前=1，序列两版=1 |
| Generation 的 Batch 序列 | 序列先位于 Batch 元数据位置，再随变形进入 Task | 提前放入 KV0 Task | 原动画解释路径被改变，源码差异确认 |
| 方格比例 | 原版序列为容量分段，并非方形对象网格 | 新增 Q/Stack/Page 对象却允许非正方形尺寸 | 当前 DOM：Q 5.7×6；Stack 9.8×7；Page 随之为 4.6×3.2。不是浏览器偶然拉伸，是生成几何错误 |
| 放大文字 | 无新增 Q/S 文字 | ≥225% 显示 Q/S 文字 | Q 的 SVG font-size=4.6，但 computed font-size=12px；通用 SVG text CSS 覆盖属性。S 标签同样受影响，导致放大重叠 |
| Assignment 相机 | 缩放 0.8–3 | 0.08–8，新增细节阈值 | 范围变化；滚轮跨 2.8 进入 Task 的逻辑仍保留 |
| Anatomy 浏览 | 相邻 Task 卡片轮播；卡内纵向滚动 Stacks；缩放 0.6–2；低缩放继续缩小退出 | 隐藏相邻卡片、全宽单卡；滚轮缩放；二维平移；0.05–8；移除低缩放退出 | 明确交互变更，超出换肤；按钮分页仍存在 |
| Pipeline 总览 | 滚轮纵向浏览 Block，横向滚动/拖动，缩放主要影响横向 | 滚轮缩放、二维自由平移、统一缩放 | 浏览器实测 overflow 原版 hidden auto，当前 hidden；事件处理源码也改变，且当前 aria-label 仍写“无限画布”，违背用户要求 |
| Pipeline 内部 | Logical Tick 原生滚动浏览，最大缩放 3 | transform 平移和指针锚定缩放，最大 8 | 交互改变；对象/依赖模型仍保留 |
| Graph | 原层级、展开/收起、实例和源码检查 | 同一语义模型；新皮肤、Fit 范围和标签宽度；范围按钮隐藏 | 两版实际打开并展开 Q Norm → RoPE，均出现215–251行源码、输入/输出依赖；相机/导航入口不是完全相同 |
| 控件与样本 | 原控件及说明 | Liquid 底栏、独立缩放、额外默认展开图元释义面板 | 前两项是用户要求；新增面板及高密度对象表达增加遮挡和认知负担 |

## 已执行验证及边界

1. 两版浏览器打开，进入 Compute Explorer，切换 Assignment、Pipeline、Graph。
2. 两版双击 Task 可进入 Anatomy，切至 Pipeline 保留 Task 上下文，返回 Block 间入口可用。原版双击过程中命中了动画中的 Task 8，当前为 Task 0；此项只证明路径可达，不作为同对象数值对照。
3. Graph 两版展开同名 Q Norm → RoPE，源码位置与依赖信息一致。
4. 已核对 Assignment 最终态 Q/序列 opacity、当前图元宽高及 computed font-size。
5. 当前六个临时测试文件共28项通过：分配布局、Phase0变形时钟、Pipeline模型、READY导航、Graph状态、Anatomy分页输入。它们不覆盖CSS、实际卡片浏览手势及高缩放可读性，不能据此宣称零回归。
6. paged-attention-model.js、paged-attention-semantic-graph.js、paged-attention-stack-dependency-model.js、visualization2-task-encoding.js、csa-model.js、csa-semantic-graph.js 与基线字节一致。
7. 截图：/private/tmp/compute-explorer-remote-original.png；/private/tmp/compute-explorer-current-audit.png。
8. 未完成全部源代码跳转、全部层级、全部缩放和所有播放组合的浏览器穷举；这些不能宣称通过。

## 后续修复应遵循的边界

- 以原版控制器的滚动、分页、退出和相机契约为基线，撤除未要求的无限画布行为。
- 保留用户明确要求的独立 Liquid 控件组、系统语义颜色、提取的三个 Tab 主体及无意义上下按钮移除。
- Task → Block 只改变组织关系；保留 Task 身份，不把所有内部对象信息挤进一张107×22卡片。
- Q / Stack / Page 分清对象层级；对象网格采用相同边长，标签只在足够空间显示；不能用容量填充误充对象外形。
- 修复需覆盖100%、225%、244%及切换中间态，检查实际渲染，而非只测试生成字符串或底层数学。

本轮只增加独立原版备份和审计记录，没有回退或继续改写现有 Demo。

## 修复落实 · 2026-10-30

- Anatomy、Pipeline总览恢复原版控制器；Pipeline内部恢复原版相机/滚动实现，仅保留系统OP图元。恢复纵向滚动、Task轮播、缩小退出；撤除无限画布与额外缩放范围。
- Assignment恢复0.8–3缩放范围、Q淡出/序列淡入以及KV0序列从Batch元数据进入Task的路径。
- Q与Stack外形、Page容量框统一等边；部分有效tokens只改变格内填充，不改变对象边界。
- 缩略图不再显示Q/S小字；详细样本按需打开，默认不遮挡。图元文字用显式CSS字号，避免全局SVG规则覆盖。
- 保留独立Liquid操作/缩放组件、系统数据语义颜色、Task/OP图元与独立入口。
- 浏览器核对100%和244%分组态，Task分页、低缩放退出、Pipeline原生纵向overflow、Graph展开/源码信息；控制台无warn/error。28项既有模型/状态检查通过。没有宣称所有层级和播放组合均已穷举。
- 修复截图：/private/tmp/compute-explorer-fixed-244.png。


## 表达质量分支收敛修复 · 2026-10-30

- 保留原版固定 Q Tile、Task 轮播、Task/Block 映射、阶段关系和浏览控制器；没有引入对象语义探索版布局。
- Stack 缩略格恢复有效容量比例：Task 0 的 1025 tokens 显示 512、512、1，方形容量边界保持不变。
- Anatomy 和跨 Tab 转场快照改为读取可见 Stack 外框/填充，避免读取隐藏条带；删除 Task 编号后的坐标读取提供现有图元回退，防止空节点异常。
- 尾页不足半页时用浅色标签；完整 Page 保留黑字，数据颜色不变。
- Graph 低缩放标签取消省略号并允许换行，不改变节点坐标与图结构。整张长图 Fit 后仍小，保留为已知限制。
- Assignment 主体避开顶部说明和底部控制区；撤下额外新增的图元释义面板，收敛少量已失效的 Task 状态覆盖规则。
- 既有28项模型/状态测试通过；浏览器核验 Task 0 尾块、详情进入、返回、Assignment→Pipeline→Graph 和 Graph Fit，无控制台 error。未穷举全部手势及转场逐帧组合。
