# Compute Explorer 视觉优化总结与 CSA 迁移 Spec

版本：v1.0 · 2026-10-31
状态：迁移实施规格，尚未实施 CSA 专项改造。
适用目录：`Design/PyPTO Compute Explorer UI/`。原版 `Design/PyPTO Compute Explorer/` 为对照，不修改。

## 1. 目标与边界

将 Paged Attention 已确认的视觉优化迁移到 Decode CSA：统一外壳、控件、图元、层级、提示和详情组织，同时保留 CSA 的领域模型、对象身份、参数、分支、源码关联及跨视图上下文。不是把 Paged Attention 换名，也不是重写已有算子逻辑。

对照基线为原版 Demo；当前实现以本目录实际源码为准。旧版 `ai-infra-ui-migration.md` 是迭代记录，包含已被后续决定替代的颜色、面板和布局方案，不应直接当作最终规格。

本轮不修改算法、源码快照、synthetic 输入生成、队列排序或实际调度；不增加未经设备数据支持的时间、吞吐和瓶颈结论。

## 2. 从原版到当前：已落地变更

| 范围 | 原版/早期表现 | 当前视觉优化版与迁移要求 |
|---|---|---|
| 版本隔离 | 优化曾与原版共享资产，容易覆盖原样式 | 独立 UI 目录及独立入口；发布仅带当前目录，保留原版 |
| 页面外壳 | IDE 入口与可视化耦合 | `index.html` 保留源码/IDE；`explorer.html` 为独立可视化入口，仍复用控制器 |
| 标题与导航 | 标题、阶段说明、范围控件占据多层区域 | 左上英文产品名 PyPTO Compute Explorer；三主视图居中；右侧文件名、设置、Info |
| 控件 | 控件散落、底部贯穿背景、材质重复嵌套 | 左下操作组、右下缩放组，各自独立胶囊；透明定位层，不套整条底板 |
| 材质 | 不透明背景遮盖玻璃效果 | 复用本地 Liquid Glass control/panel；避免嵌套滤镜和多余背板 |
| 语言 | UI 中英文混杂 | 操作 UI 中文；Block、Task、Rank 等领域术语保留英文；Graph 节点和阶段名称英文 |
| Info | 独立大浮窗、说明散落底部 | 右上面板承载用法、图例解释和来源边界；画布底部保留操作控件 |
| 提示 | 原生 title 与自定义 tip 重复，重复按钮名称 | 每个目标仅一个提示；对象提示包含身份/范围/分支，按钮提示说明动作 |
| 详情 | 详情挤压画布，或与右上按钮重叠 | 覆盖式详情，不重排主图；顶部避开全局文件名、设置和 Info，当前约 64px |
| 任务卡片 | 容器边框重、标题中文泛称 | Block 容器 #000000 fill、无边框；标题如 Block 02 · 6 Task；不以边框表现选中 |
| 数据图元 | 条状容量、普通卡片，身份不清 | Q Head / Stack / Page 使用对象图元；填充对应有效范围，不能把纹理当测量量 |
| Assignment | 分组轴距卡片远、视觉偏心 | 分组轴靠近内容，按实际可见内容居中；保留对象 ID 与转场身份 |
| Pipeline 高度 | 行高过大，需要纵向滚动 | 压缩行高使默认总览尽量全显；通过 Fit/缩放查看，不以加滚动代替行高控制 |
| Pipeline 网格 | 准备阶段色块拉成长条 | 几何格子为正方形；先定义一格对应什么，再绘制数量/填充 |
| Pipeline 性能 | hover 反复重置 Canvas 和布局热区，观察器全量扫描 | RAF 合并 hover；尺寸变动才重建 Canvas；复用热区与样式；局部处理 DOM 变化 |
| Graph OP | 展开后变灰、子节点风格漂移、固定宽度导致换行 | 同一套 OP 图元覆盖展开前后；主名称单行、宽度适配文本；小字在节点外下方 |
| Graph Tensor | 条形图元文字溢出、重复黑底标签 | 有元信息：四行网格、两排文字；无元信息：两行网格、一排名称；去掉重复标签 |
| Graph 图层 | 数据节点单独开关，与 layer 割裂 | 数据节点纳入图层；默认开启全部图层；选中态深灰 #383838 + 浅色文字/勾选 |
| Graph 布局 | 数据关掉后留大洞、底部 Fit 被遮挡 | 按可见节点重算布局/依赖连接并 Fit；保留底栏安全区；收紧间距，避免过密或大空洞 |
| 同步说明 | 长串同步事件文字压在图上 | 保留依赖/同步关系；长串解释移入 Info 或当前对象详情 |
| 讲解模式 | 默认重复标题；示意图小且无帮助；侧栏挤压画布 | 默认关闭；开启仅叠加紧凑说明，画布几何及相机不变；示意图及开关已删除 |

颜色是本 Demo 的宿主约定：Q/KV 数据蓝色 `#80B7F7`，Task 调度图标绿色 `#9EF04B`，归约角色按实际语义使用角色色。不能由“蓝色”推断物理内存；CSA 权重、通信、状态不能全部套成蓝色激活。

## 3. 当前代码复用面与接入缺口

| 文件/模块 | 当前职责 | CSA 迁移方式 |
|---|---|---|
| `styles/ai-infra-ui.css` | 视觉适配、停靠控件、Graph 图元、面板 | 提取/复用现有规则，检查 `.viz2` 限定；不要直接全局替换选择器 |
| `js/ai-infra-ui.js` | UI 中文化、单提示、材质挂载、图元适配、Info | 已识别 `.csa-explorer`；逐个核对 CSA 动态节点、事件及可访问名称 |
| `assets/ai-infra/` | 本地 VisualSystem、OP 图标、Tensor 图元、Liquid Glass 等 | 复用同一资产与版本；不再复制一套近似图标 |
| `js/task-object-glyphs.js` | Q Head / Stack / Page 图元 | 仅在 CSA 确实存在同义对象时使用；不能拿 Stack 代表 Work Item |
| `js/compute-graph-scene.js`、`semantic-computation-*.js` | 共享语义图、可见性、布局、绘制 | 沿现有入口接入 CSA；确认紧凑尺寸和数据隐藏的布局选项实际传入 |
| `js/csa-explorer.js` | CSA 主视图、Rank、focus、每视图 camera、Graph 展开及导航 | 保持状态结构和 object/graph ID 映射；迁移 chrome，不重建另一套状态 |
| `js/csa-model.js`、`csa-semantic-graph.js` | CSA 对象解释、来源与语义层级 | 保留；图元绑定依据来自这里，不能由标签猜类型 |
| `js/csa-qk-work.js` | Work Item 生成、valid-first 排序、Lane 分配与画布 | 保留纯数据函数；改外观、对象编码、控件位置与热点处理 |
| `js/csa-qk-pipeline.js` | Work Item 内部依赖及分支 | 保留 valid/invalid 分支、选择和源码关联；接入相同 OP 与详情样式 |
| `js/presentation-mode.js` | 当前仅绑定 `.viz2` 的 Paged Attention 讲解 | **尚未适配 CSA**；应接入 CSA state/focus，不沿用 PA 文案/事件假设 |

当前存在部分共享样式不等于 CSA 已完成迁移。已有早期 CSA 回归记录，也不等于本规格下所有新样式已经验收。

## 4. CSA 对象定义：先确认语义，再迁移图形

| 对象 | 当前实现中的定义/来源 | 显示约束 |
|---|---|---|
| Rank | `csa-model.js` 的 rank 对象；绑定 device，含本地 query、KV 副本和权重分工 | 保留 Rank 切换与数据归属，不改称 Block |
| Request / Query | 默认 source 模式每 Rank 256 Query，8 token/Request；参数与源码配置关联 | Request 是分组，Query 是计算输入身份，不互换 |
| Sparse Block | WIN=128、IDX_TOPK=512、ATTN_K_TILE=128，对应当前模型 5 个 Sparse Blocks | 与执行 Block 分开，优先显示 SB0…SB4 |
| Work Item | Query × Sparse Block；默认 source 模式 256×5=1280 项 | 保留 item ID、token、sparseBlock、valid、validKvCount；不套 PA 的 Batch×KV Head Task 定义 |
| qk_order | `buildQueue`：valid 项在前、invalid 项在后，两组保持既有顺序 | invalid 不删除；排序位置不能当 item ID |
| SPMD Lane / Block | `NUM_QK_CORES=24`，按队列位置分配；具体见 `assignToBlocks`、`getWorkItemDetails` | 不能照搬 PA 的 item ID % 24；保留 queuePosition → Lane / iteration 规则 |
| valid / invalid | 当前 UI 使用 synthetic idx_topk 和可能的手动 override 推导 | 明确当前输入来源；不要写成设备实测或真实 Indexer 输出 |
| invalid 路径 | 写入中性 m/l/o，不执行该项的 QK/PV 计算 | Graph/Pipeline 均显示此分支，不能用低透明度把分支语义抹掉 |
| Tensor / Cache / Weight / State / Communication | 依据模型 type、来源和算子语义判定 | 采用不同图元/角色；通信不能渲染成普通 OP 后丢失跨 Rank 语义 |

讲解不使用“对象：N 个 X”占位句。应说明当前视图回答的问题、工作单位的定义、映射规则及源码来源。例如：“Work Item 的队列编排与 SPMD 分配”，随后说明 valid-first、invalid 保留及队列位置分配。

## 5. CSA 分页面实施要求

### 5.1 Assignment

1. 保留整体 Rank/Data 分配与 QK/PV 工作分配两个层级；保留 Request、Query、Work Item、Lane 的选择范围。
2. generation → ordering → assignment 保持同一批 item 的连续身份，不拆成无关联的三张图。
3. 分别清楚表达 Query×Sparse Block、qk_order、Lane/iteration；对象数量取当前 deriveScenario，不硬编码展示数字。
4. Lane 容器采用黑底无边框；标题先沿用模型 Lane 命名，必要时说明对应 SPMD Block，不能擅自改所有 Lane 为 Block。
5. valid/invalid 除颜色外需有状态/标签辅助；实际有效量与背景纹理分开。默认看全量，放大查看身份。
6. Rank、Mixed/Uniform、Position、override 等输入保留；参数变更时协调当前选择，不能显示上一次场景的详情。

### 5.2 Pipeline

1. 区分整体 distributed execution 与 QK/PV 内部流水，保留跨 Rank 通信和源码定义的依赖。
2. 行距依可见条目与可用高度计算；总体布局先 Fit，再允许缩放/平移。极窄窗口可降级，不能承诺任意尺寸均无滚动。
3. QK、Softmax、PV、Gather、Neutral Write 用对应类别图标；valid 和 invalid 路径分别可检查。
4. 时间轴若是逻辑 tick/步骤必须按实际模型命名；没有 trace 数据不显示伪 ms 或性能结论。
5. hover 只更新提示/绘制，不重置 canvas、不重建所有热区；点击仍绑定同一 item。

### 5.3 Graph

1. 统一整体与展开子图的 OP/Tensor 图元；复用 `objectToGraph`、`nearestObject` 和 focus 投影规则。
2. OP 胶囊参考当前高度 48 模型单位，宽度按文本与左右图标测量；不能用增高解决英文名称换行。
3. 输出精度/执行域等小字放节点下方；计入节点边界与 Fit，但不生成另一张黑底标签。
4. Tensor 有元信息参考高度 64，无元信息约 35 模型单位；四行/两行网格随元信息切换。纹理为图元样式，不等于元素个数。
5. 数据、形状、精度、存储及执行、状态/控制、执行次数、同步统一由图层控制；只暴露 CSA 有实际数据的能力，缺元信息不能编造。
6. 当前 CSA state 只有 data/control overlay 等已有配置；新增图层须有实际 adapter，实现后再默认开启。
7. 关闭数据节点需要维护依赖可达关系、重排并 Fit，不能只设 display:none 后留下空白；控制边与数据边分开处理。
8. 紧凑间距参考现实现 root gap 64、child gap 56 模型单位，再按 CSA 标签/结构调整；不能把参考值当所有图通用常量。
9. 数据开关、展开、Fit、选择均保留可达的返回入口；单纯开启详情或讲解不得重置相机。

## 6. 外壳、信息与讲解的统一契约

- 顶部产品名不限定单一算子；文件名反映当前源码。三个主视图名为任务分配、流水线、计算图。
- Settings/Info 使用现有标准 SVG，18px 图标、36px 点击区，显式居中；不得手绘近似齿轮。
- 图层菜单向画布内展开，当前参考宽 216px、可视区不足时面板内部滚动；不截断在页面外。
- 图层选中态灰色填充及勾选；不能恢复成整排白底。键盘 focus 可见。
- 详情浮层避开全局按钮；打开/关闭前后 viewport、camera、selection 均不变。
- 提示不重复标题，不同时出现原生 title 与自定义 tooltip；对象提示提供当前对象信息，长推导放详情。
- 讲解默认关闭。关闭时没有额外场景标题/说明；开启时只显示一套紧凑文案。
- 讲解采用覆盖层，**不预留行高、侧栏宽度，不触发 Fit、不改变 zoom/pan**。当前参考标题 20px、正文 13px、行高 1.45；长来源转入 Info/详情，避免扩张覆盖面积。
- 不再添加语义示意图或示意图开关。CSA 讲解须读取可见 view、scope、focus、Rank 和当前场景，不能从隐藏控件猜当前状态。
- 局部提示与面板状态不得写回领域数据；避免 DOM 观察器自触发或全页面反复扫描。

## 7. 保留的交互契约

| 操作 | 迁移后必须保留 |
|---|---|
| 源码入口 | decode_csa.py → decode_sparse_attn_csa.py 等跨文件导航与行段定位 |
| 切换 Tab | Rank、Scenario、Work Item 上下文连续；沿现有 focus 映射投影至 Graph |
| 展开/返回 | module/stage/operation 层级正确，返回恢复合理范围与上下文 |
| 选择/取消 | 当前对象详情与图上高亮一致；取消恢复，不残留上一条分支 |
| 参数切换 | Mixed/Uniform、Position、Rank 与 override 变化对应重新派生结果 |
| 相机 | 平移、缩放、Fit、各视图相机保留及对象命中一致 |
| 播放 | 播放/暂停/重播及手动阶段切换保留；自动播放策略另行明确 |

已知待修：当前 PA morph player 默认 once；后台加载/visibility 暂停后不自动恢复，pointerover 也会暂停。该逻辑不是本次迁移的理想标准。CSA 有独立播放器，必须独立核验；不能简单复制这个行为，或宣称本轮已修复。

## 8. 推荐实施顺序与验收

1. **基线记录**：记录 CSA 当前对象数、队列、valid/invalid 分布、焦点及源码入口；保留原版截图。先列出哪些 CSS/adapter 已生效。
2. **共享外壳**：标题、导航、两组底栏、标准图标、Info、单提示、详情覆盖；先不改领域数据。
3. **Assignment**：复用卡片与图元；核对全量1280项、队列身份、Lane/iteration、场景参数。
4. **Pipeline**：压缩行高、统一 OP、检查两条分支，再优化 hover 绘制热点。
5. **Graph**：接入共享皮肤、尺寸、英文名、图层与可见布局，验证展开及返回。
6. **讲解**：接入 CSA state 事件/适配接口，按真实上下文生成文案；最后再接，不依靠全局查找文本。
7. **回归与发布**：通过下表后评审差异；得到发布指令后独立提交，不带无关工作区变更。

| 验收项 | 通过条件 |
|---|---|
| 纯模型回归 | 同一输入的 Work Item、qk_order、Lane、iteration、valid 分支与迁移前一致 |
| 代表场景 | Rank 0/1；Mixed/Uniform；Position 7/8192；分别检查 valid 和 invalid 项 |
| 已有基准实例 | 复核 Uniform、Position 7 的 Work Item 2 是否仍为已有 invalid 路径；Lane/iteration 以实际模型计算为准 |
| 身份连续 | 同一 Work Item 在三视图、详情与源码定位可追踪 |
| Graph | 默认/展开/收起、图层全开/关数据、Fit、选中/取消、返回均可用 |
| 布局 | 1280×720 作为当前规范视口；图层及详情不越界，缩放控件不遮主要节点 |
| 讲解 | 开关前后 viewport 矩形、scene transform、focus 一致；关闭无残留说明；无示意图 |
| 可访问性 | 图标有准确名称；Tab/图层状态同步 aria；键盘可到达并关闭面板 |
| 性能 | 连续 hover 不触发 canvas 尺寸重建和全量热区布局；有实际前后记录才声称性能提升 |
| 发布 | 独立 UI 目录差异、原版未动、PR 检查与部署结果分开记录 |

模型回归优先复用已有函数和测试。新测试针对身份/分支/布局契约，不为每条 CSS 编写镜像断言。最后做一次实际浏览器 smoke test；语法检查不能替代 UI 验收。

## 9. 交付清单

- 本目录下 CSA 页面及适配器变更；不修改原版。
- 对象/图元映射表与源码来源；区分真实源码规则、synthetic 输入与 UI 表示。
- Assignment / Pipeline / Graph 的默认、选择、展开、隐藏数据及讲解截图。
- 回归结果、未覆盖范围、性能观测与已知缺陷。
- 新增独立入口时再更新 launch；不为外观迁移另建重复领域模型。

参考实现：`ai-infra-ui.js`、`ai-infra-ui.css`、`presentation-mode.js`、`visualization2-assignment-morph.js`、`visualization2-pipeline.js`、`visualization2-graph.js`、`semantic-computation-layout.js`、`semantic-computation-visibility.js`。CSA 接入以 `csa-explorer.js`、`csa-qk-work.js`、`csa-qk-pipeline.js`、`csa-model.js`、`csa-semantic-graph.js` 为入口。
