# PyPTO Agent Decision Explorer

历史 Softmax Agent Run 的交互审查原型：从需求与设计进入源码、正确性、性能实验和停止证据。仅展示历史材料，不连接 Agent 执行器，不声称在线重跑或硬件控制。

## 预览与完整旅程

当前本机服务：http://127.0.0.1:8765/agent-decision-lineage-softmax.html 。未部署 GitHub Pages。

服务停止时，在本目录执行 `python3 -m http.server 8765 --bind 127.0.0.1`。也可直接打开 HTML；证据数据通过本地 JS 包加载，无 npm 构建依赖。

1. Overview：识别纯 Vector/VF、五阶段完成、8/8 PASS、大用例 Baseline→Final。
2. 选择 Mixed Precision、Single Tile、UB 或 Scheduling，自动定位 Decision Inspector，查看可选对象与高亮代码。
3. 详情上方「下一步」沿审查顺序进入 Kernel、8 个正确性用例、Baseline、性能线索、优化假设、修改、失败与替代、Final 和 Stop。此导航是阅读顺序，不表示新增实验或因果。
4. Experiment Inspector 的 Journey / Comparison / Evidence 共享 Shape；候选与下一步导航同步主 Lineage 选中身份。
5. 「返回 Lineage」保留当前节点并选择适合的 Design/Optimize 范围；Fit to View 恢复当前范围，Focus Selection 聚焦选中对象及关联。Overview/Design/Optimize 或空白重置选择；返回全局恢复全局缩放。

顶部 IDE 标题栏和底部状态栏固定，main 工作区独立纵向滚动，代码区可独立滚动并通过 Tab、PageDown 操作。Inspector 出现后焦点到上下文标题。

## 工程结构

| 文件 | 责任 |
|---|---|
| agent-decision-lineage-softmax.html / page.css | IDE shell、滚动、Lineage 与详情容器及局部视觉 |
| app.js | Lineage 渲染、语义高亮、聚焦、共享详情选择和滚动定位 |
| lineage-data.json / lineage-data.js | 同一事实模型的 JSON 和浏览器包；节点、关系、证据引用与 evidenceKinds |
| lineage-layout.js | 语义展示投影：分组、固定坐标、分支、主线来源；带宽不编码数量 |
| decision-inspector.js / decision-evidence.js | Precision、Tile/Memory、Scheduling；原始设计材料包 |
| experiment-inspector.js / experiment-data.js | Journey、正式对比、PMU 与停止审查；实验数据与前后源码包 |
| review-inspector.js / audit-evidence.js | 连续审查导航、统一证据语义、Kernel 和全部正确性用例 |
| evidence/ | 11 份原始快照，记录 SHA256；包括 Baseline test_softmax.py.bak_stage5 |
| assets/ | 从 V3 同源设计系统复制的 tokens、CSS、IDE Frame / workbench 依赖；原 V3 未改 |
| checks/ | 可重复执行的证据审计与真实浏览器回归脚本 |
| previews/ | 关键状态截图与机器可读审计结果 |

继续使用 `ide-frame` direct embedding 与共享深色 tokens。没有复刻 Cube/Load3D/C→V，也没有使用候选数量收敛模型；详情专题不取代主图谱。

## 模型与证据规则

节点含 id / type / status / stage / reason / relatedNodeIds / evidenceRefs / assertionBasis / evidenceKinds。类型为 Workflow、Decision、Experiment、Evidence、Finding。关系有独立来源位置；阶段归属不绘制成发散连线。

| 证据分类 | 含义 |
|---|---|
| Measured | 历史正确性、Kernel 计时或 Profiler 材料；不代表本机复测 |
| Derived | 源码、资源约束、Roofline 模型或 Tile DAG 推导 |
| Design Decision | 采用的方案或优化假设 |
| Reported Attempt | 报告记录的真实尝试，可能缺少失败版本源码或原始日志 |
| Not Attempted | 未执行；没有性能结果或排名 |
| Missing Evidence | 缺失或采集失败；不等同于 Kernel 执行失败 |

证据种类与候选状态分别表达。同一 Stop 结论可引用多类证据。原始材料/Workflow 节点不强制标成尝试。

### 本轮语义修正

- Journey `8/8 PASS` → `x-verify`，展示全部 8 个实际报告用例；不再跳至 Final 性能节点。
- `f-mask` 为报告中的循环内 mask 线索；`h-mask` 为优化假设；`x-mask` 为实施修改。不是三次新增实验。
- `x-tail` 为 Mask Lifting 的组成部分，关系 `includes`；没有独立性能归因。
- `x-reuse` / `x-fusion` 分别表示 Exp Reuse / Pass Fusion，主图与 Candidate Ledger 身份一致。
- Scheduling 入口直接命名，消除仅使用「行块跨核 striding」的可发现性问题。

### 数值口径与边界

- 正式 Kernel Duration 取 3 samples 的 trimmed mean（去最大/最小，等于中间值）。Baseline / Final 为 3.250→3.428、4.714→4.015、5.926→4.946 μs。
- 耗时变化为 `(Final−Baseline)/Baseline`：+5.48%、−14.83%、−16.54%；speedup 为 Baseline/Final。报告小用例的约 5.2% 未直接复制为同口径变化率。
- Golden E2E / PyPTO target kernel 是跨范围默认目标，不能当作本次优化 speedup。
- 大用例 Profiling 表单次 Task Duration 5.834→4.946 μs，与正式基线中位数 5.926 分开。PMU 百分比不是可相加的执行时间分解；vec_resc_cflt=121.50% 不解释为超100%利用率。
- 8 个正确性用例来自 PERFORMANCE_REPORT §8，matched_ratio、max_abs_error逐项保留。通过不等于穷举整个动态范围。
- compute_bound 是 Roofline 推导；小用例实际时间仍以启动/setup 为主。not_applicable_with_dag 为每核一个工作项的依赖证明。
- Timeline 因设备重映射变量冲突在 preflight 失败，未画成算法失败或伪造时间轴。
- Stage retry 次数、KB Pattern 数、页面节点数均不是实验次数或候选总数。

## 可重复验收

### 证据审计

在本目录执行：

`python3 checks/audit_evidence.py http://127.0.0.1:8765/`

不传 HTTP 参数时只审计本地材料。检查原始文件与快照、SHA256、全部证据文件的 HTTP 内容、JS/JSON 与代码包一致性、正式 samples/中间值、8用例数据、唯一ID、关系与引用。

结果：`previews/evidence-audit.json`，82 项通过。

### 浏览器回归

`checks/e2e.cua.mjs` 导出 `runReviewRegression({tab,browser,width,writeFile,directory})`，在具备文档化 cua 浏览器句柄的 cua_repl 中调用。仅使用公开 DOM/键盘/点击/滚动 API；不注入应用状态或调用隐藏业务方法。分别传入 width=1440、1600，height 固定为900。测试结束应 reset viewport。

本轮两种视口各48项通过。记录：`previews/regression-1440.json`、`regression-1600.json`。覆盖 Overview/Design/Optimize、三个设计专题、Kernel、8正确性用例、全部候选、3 Shape/Samples、PMU/Stop、连续下一步与返回、键盘焦点、内部代码滚动、主区滚动、选择清理、Fit/Focus、横向溢出和Console。

截图：`audit-overview-{width}.png`、`audit-design-{width}.png`、`audit-optimize-{width}.png`、`audit-experiment-{width}.png`、`audit-stop-{width}.png`，width为1440或1600。截图反映独立滚动区域，Overview底部分支可继续下滚查看。

## 已知限制与未验证部分

这是为一次 Softmax Run 编排的语义布局，不是任意拓扑自动布局器。Overview文字密度高，聚焦视图有留白；<900px 窄屏保留画布横向滚动，不属于本轮桌面验收范围。

没有逐步精度误差、其他 TILE_ROWS 性能对照、全FP16/全FP32路线本次实测对照、Runtime-if失败源码/编译日志快照、真实Cycle Timeline、逐指令因果或统计显著性依据。也没有完整 Agent 对话和逐次事件时间戳。对这些内容只披露限制，不补造数据。

本轮未运行 NPU 测试，未连接执行器，未发布、commit、push 或合并。只修改本 Demo；事实依据为本地原始材料与快照，线上版本不自动同步。

## 紧凑 Experiment River

Overview / Optimize 使用 `experiment-river.js` 与 `lineage-layout.js` 的展示几何：Baseline→Finding线索→Mask Lifting→Final→Stop；Finding是同一主段的轻量注释，不是新增实验。Runtime if从候选展示分岔处分出、红色封口、不汇入Final。三个机会从Stop附近并列以虚线和淡出端点呈现，不互为下游，不表示已执行。

主带宽32、失败支线10、机会支线3均为叙事层级，绝不编码数量/性能/概率。布局画布由1688×500承载完整全局（此前1540×750）；Optimize内部叙事带保持约350逻辑像素高。设计区原有节点坐标保留。新增junction仅属于展示层，不记录为历史节点。Stop→opportunity为审查关系，有PERFORMANCE_REPORT §7/§11依据；不表示时间执行因果。点击开放机会直接进入对应Not Attempted说明和停止审查。

更新后的两视口回归仍各48项通过。新增截图：`previews/river-overview-1440.png`、`river-optimize-1440.png`。1440下完整Optimize节点与机会端点可见，不需要滚动很长页面；Inspector继续按需自动定位并独立滚动。

## 运行后未探索方向

`06 OPEN OPPORTUNITIES` 是产品呈现章节，不是 Orchestrator 第六阶段。配置位于 `LINEAGE_VIEW.river.postRunOpportunities`，按 PERFORMANCE_REPORT.md §7 的原始原因归纳为“目标已达成”和“需要改设计”。三个候选复用 x-rows / x-reuse / x-fusion，原始事实模型及实验父子关系保持不变。

统一从 Target / Stop 右侧分叉，以中性曲线连接类别区域；候选仍调用原有 Inspector，类别与候选的悬停/聚焦同步强调所属分叉。完整 SVG 继续由实际元素 bounds 计算 viewBox。

## Lineage Dock 精修（2026-10-31）

复用 PTO `ide-frame`（direct embedding）和现有 surface、foreground、primary、success、danger tokens。保留已确认的单主 Pane 和深色主题；未替换 shell、未引入框架或 Liquid Glass。参考 AI_Infra Spatial Systems UI 的开放结构、语义缩放及中性选择态，没有引入时间比例。无新增私有视觉 token 或 preview 审批项。

### 视觉与语义

- 原先的矩形 Stage 底色与下方重复上下文已移除，当前上下文合并到顶行，Dock 高度 108px。
- Explore / Design 是共同主干上的集合节点，不是执行序列；Design picker 只列出本次采用的五项决策。未采用分支根据 parent 定位所属 Stage。
- Optimize 的成功主干、Finding、Tail Handling 与 Runtime if 失败端分开呈现；选择使用白色轮廓，保留红色失败身份。
- Open Opportunities 根据现有 categories 分成两条原因支路；未尝试，不带完成勾选。
- Compact projection 包含 stageId、nodeId、groupId、relationType、status、compactPosition。选中状态读取 app.js 唯一 selected 的只读 getter，不保存第二份选择。关系来自 view.groups / view.river / data.edges；几何是展示尺度，不编码时长或收益。

### 状态与滚动

统一入口为 `selectLineageNode` → `enterInspect(nodeId)`；关闭 Analysis、Escape、展开按钮和上滚手势均进入 `returnToOverview(reason)`。四种状态为 overview、entering-inspect、inspect、returning-overview。

高度交接使用 Web Animations API 300ms，不缩放 SVG；等待动画完成及下一布局帧后，按 Analysis、主 Pane、Dock 的实际矩形定位。程序定位立即完成，不遗留平滑滚动尾部；过渡期禁止手势返回。Reduced Motion 跳过动画。禁用主 Pane scroll anchoring，Analysis 不再拥有主阅读范围的内部滚动。

上滚恢复要求：inspect、详情顶部附近、前一个 scroll 观测也已到顶，且不处于刚到达顶部的短暂窗口；随后累计向上意图达到 78px 才返回。滚轮、触摸下拉、键盘上翻共享这一判定。Code Preview、Picker、输入控件的事件保留给局部组件。反向滚动或离开顶部清空累计。返回时统一隐藏 Analysis、保留节点、恢复图谱横向位置与主 Pane 顶部。只有重新明确选择节点才会再进入 Inspect。

### 截图与快速回归

截图目录：`previews/dock-refinement/`。

| 状态 | 截图 |
|---|---|
| 修改前 | [before.png](previews/dock-refinement/before.png) |
| 新 Design / 默认 Inspect | [design-1440.png](previews/dock-refinement/design-1440.png) |
| UB Buffer | [ub-buffer-1440.png](previews/dock-refinement/ub-buffer-1440.png) |
| Runtime if | [runtime-if-1440.png](previews/dock-refinement/runtime-if-1440.png) |
| Open Opportunities | [opportunities-1440.png](previews/dock-refinement/opportunities-1440.png) |
| Design Picker | [picker-1440.png](previews/dock-refinement/picker-1440.png) |
| Opportunity Picker | [opportunities-picker-1440.png](previews/dock-refinement/opportunities-picker-1440.png) |
| 返回全图 | [returned-overview.png](previews/dock-refinement/returned-overview.png) |
| 1600 / 800px | [1600](previews/dock-refinement/design-1600.png) / [800](previews/dock-refinement/design-800.png) |

按用户要求只做快速 Chrome 回归：默认全图、选中、标题位于 Dock 下、详情切换、中段上滚、到顶不立即返回、小幅上滚不返回、继续上滚恢复、失败身份和选中共存、Picker Escape、页面 Escape、1600/800 无整体横向溢出、Reduced Motion、键盘上滚、Console 无异常均通过。机器记录见 [regression.json](previews/dock-refinement/regression.json)。另有 1440×900 截图检查。

限制：没有真实 Mac 触控板惯性或触屏设备实测；未全面重跑各 Analysis 业务回归。Single Tile、UB Buffer、Scheduling 等业务模块源码未修改；Shared Tile State 沿用现有模块。未 commit、push、合并或部署。

## IDE 工作区与 Agent 对话故事线（2026-11-02）

### 结构与复用

本轮只改 Shell、历史对话与定位入口，未改 Tile / UB / Scheduling / Optimize 专业可视化。现有 main DOM 整体保留，迁入 `ide-frame` 的 direct embedding 工作区；主 Pane 的选择器同步放宽祖先层级。原单 Pane shell 升级为 Activity Rail + 可隐藏 Files + Decision Explorer / Agent Chat 水平分割。未创建 iframe、代码编辑器、Terminal、额外面板或私有 shell。

复用 `ide-frame` 的 Activity Rail SVG、Pane Header / Body、Explorer Toggle、Status Strip 和 `workbench-shell` 的鼠标 / 键盘分割线。使用原 PTO 深色 surface、foreground、primary、success、spacing、radius tokens，没有另建视觉 tokens 或替换 pane 皮肤。双 Pane 和深色主题由本轮需求指定。无缺失系统部件或额外视觉审批项，未创建 preview page；外围容器没有新增渐变、装饰侧线或嵌套卡片。

Files 默认隐藏，Explorer 图标可反复开关；其他宿主命令保持禁用，不伪造源控、搜索或终端行为。桌面初始 Chat 占剩余空间约 33%（1440px 时约 442px），共享分割线最小尺寸为 Explorer 600px / Chat 360px。空间不足时 Chat 默认收为右侧标签，显式展开使用右侧浮层。左右独立滚动，Chat composer 固定在 Chat 内底部。

### 故事与证据

`agent-run-story.js` 分离故事内容与渲染：用户措辞是 SPEC 重构示例，明确标注“非原始逐条对话”。SPEC 机器合同快照来自 `Data/softmax 新研pto pro agent产物数据/SPEC.md`，与现有 `evidence/SPEC.md` 对应；容差引用该文件 §7 官方样例来源，A5 / 950 保持默认假设，Performance Target 为 null。未把设备、容差、速度目标写入用户消息。

Stage 1–5 状态从 `LINEAGE_DATA` Workflow 读取；Correctness 从 `AUDIT_EVIDENCE.correctness` 读取，性能从 `EXPERIMENT_DATA.cases` 读取，未探索候选从 `LINEAGE_VIEW.river.postRunOpportunities` 读取。Stage 06 没有混入执行活动。Improvement 与 Speedup 分别由同 Shape 的 Kernel Duration 计算，保留小 Shape 回退说明，不使用 Golden E2E Ratio 作为优化倍数。

故事顺序：示例用户需求 → 默认展开 SPEC Summary（Contract / Acceptance / Assumptions 与来源）→ 五条紧凑、可展开执行记录 → 完成响应（实现、8/8 PASS、性能）→ 四份真实文件链接 → 两类原因下三个未尝试方向。Composer 只读，附件和发送按钮禁用。

### 导航与状态

单节点 SPEC 使用现有 `selectLineageNode('e1')`；SPEC Summary 在原有 Analysis 容器内呈现。多节点阶段复用 `LineageDock.navigateStage(stageId)` 和原 Node Picker；Overview 定位章节后打开 Picker，Inspect 直接打开对应 Dock Picker，不随意选择某个 Design 决策。外部 Picker 的 Escape、点击外部及选中关闭复用同一逻辑。

`lineage-selection-change` 仅更新 Chat Header 的关联章节，不调整 Chat scrollTop 或插入消息。选择身份仍由 app.js 唯一 selected 管理。Dock 的四态、向上返回、Shared Tile State 和 Analysis 业务渲染保持原有实现。

### 修改文件

新增 `agent-run-story.js`（故事与事实引用）、`agent-chat.js`（对话、文件与 shell 命令）、`agent-chat.css`（Chat 内容布局）、`checks/ide-chat.cjs`（精简浏览器回归）。调整 HTML shell 与脚本入口、app.js 的选择通知和 SPEC 接入、lineage-dock.js 的共享阶段 Picker 入口、page.css 的主 Pane 祖先选择器。README 与 `previews/ide-chat/` 为交付说明和截图。

### 截图与快速检查

- [完整 IDE 1440×900](previews/ide-chat/ide-1440.png)
- [SPEC Summary](previews/ide-chat/spec-summary.png)
- [五阶段执行故事线](previews/ide-chat/execution-story.png)
- [Chat → Design Picker](previews/ide-chat/chat-to-design.png)
- [Analysis 展开且 Chat 保持原位](previews/ide-chat/analysis-chat-stable.png)
- [最终响应](previews/ide-chat/final-response.png)
- [文件产物与未探索方向](previews/ide-chat/artifacts-opportunities.png)
- [拖动分割线后](previews/ide-chat/resized-split.png)
- [1600×900](previews/ide-chat/ide-1600.png)
- [850×900 收起](previews/ide-chat/narrow-collapsed.png) / [Chat 展开](previews/ide-chat/ide-850.png)

精简 Chrome 回归共 22 项通过：默认文件树隐藏、历史 Completed 状态、五阶段、Composer 只读、SPEC Analysis、Design 五项 Picker、Chat 滚动保持、Single Tile、Dock 标题避让、向上返回、Split 拖动、文件树反复开关、真实性能、三个未尝试候选、机会分类、1600/850 无整体横向溢出、窄屏收起 / 展开和 Console 无错误。完整记录见 [regression.json](previews/ide-chat/regression.json)。未运行 NPU 或全面重跑各专业 Analysis 业务测试，未连接 Agent backend。未 commit、push、合并或部署。

## IDE Frame 正式结构对齐（2026-11-02）

本节覆盖前述 Shell 滚动容器说明：当前使用单层 Split，Decision Pane 采用固定的 32px 标准 Header + `#decision-pane-body` 主阅读滚动区；main 本身不滚动。移除了外层 16px Padding、私有 Topbar / Run Header / Status Strip 样式与重复 Split。Chat 历史上下文已归入 Chat Body。共享组件与 Token 未改动。

详见工作区根目录 [IDE Frame 结构与布局对齐](../../../PyPTO%20IDE%20Frame%20结构与布局对齐.md)，本轮截图和几何记录位于 [frame-alignment](previews/frame-alignment/comparison.html)。

## 详情分析 UX 精简（2026-11-03）

Baseline / Final 始终同时显示三个 Shape，删除 Shape 单选状态、Tabs、可点击图表行、Dumbbell SVG 和对应旧 CSS。新对比采用两条水平条形：Baseline 中性灰 / Final 蓝色，两者统一零起点、共用同一最大值尺度；数值直接标注，Δ 耗时回退红色 / 改善绿色，保留正负号。图表实测高 202px，1280×900 首屏完整显示三组数据；600px 无图表横向溢出。

层级：对象标题 → 三组直接比较 / 决策状态 / 结构变化 → 折叠技术证据。移除重复 ANALYSIS 小标题、实验上下文段落、图例段落和单个 Shape 的重复摘要。Mask Lifting 的 Before / After 结构与代码 Diff 相邻；Runtime if 明示失败类型和替代方案，不把最终代码称为失败源码。Stop 使用紧凑验收表；Open Opportunities 先呈现未尝试原因。Design Decision 主区保留 Decision / Why，Constraints / Alternatives 按需展开，知识绑定与源码映射保持可访问。

保留的二级证据：三 Shape 全部 Samples、round_001 / round_002 collection ID、trimmed mean / Δ / speedup 公式、小 Shape 报告百分比与统一分母差异、统计显著性限制、Quick A/B 与 Golden E2E 口径区别、[32,4096] Profiler 完整表及单次 / max-core 口径、PMU 不等于逐指令因果、Runtime if 无成功性能测量与失败源码、未尝试方向无收益结论、停止目标 / DAG / Roofline 推导与缺失 Timeline。

真实数据文件未改；主图、Dock 几何、进入详情 / 返回导航和 Chat 联动未改。用户既有的其他项目删除 / 重命名改动未触碰。

截图： [1280 性能比较](previews/compact-details/comparison-1280.png)、[Measurement details](previews/compact-details/measurement.png)、[600 窄屏](previews/compact-details/comparison-600.png)、[Runtime if](previews/compact-details/x-if.png)、[Stop](previews/compact-details/d-stop.png)、[未尝试方向](previews/compact-details/x-reuse.png)、[Design Decision](previews/compact-details/d-precision.png)。快速浏览器检查未发现 pageerror，结果位于 `previews/compact-details/checks.json`；未全面运行其他业务回归。未 commit、push 或部署。

## Compact 节点直接导航（2026-11-03）

缩略图的阶段容器不再是按钮，也不再绑定 Node Picker。22 个现有节点以原 ID、坐标和 shared selected 状态直接调用 `selectLineageNode`；点击当前节点不重渲染详情。展开模式的阶段选择保留原有用途，Chat 在 Inspect 下定位阶段则滚动到该阶段并聚焦节点，不弹菜单。

每个节点使用透明 SVG Hit Path：按实际 SVG screen CTM 将 24×24 CSS px 投回原坐标，再用与其他节点的距离平分线裁分；密集处命中区适当收窄，互不重叠。连线、文字和视觉图元不截获事件。ResizeObserver 在尺寸或可见性变化后重算命中区，不修改节点几何。

Hover / Focus 共用唯一的顶层 Tooltip（完整领域名称 + 简短状态，无按钮，pointer-events:none），按节点实时屏幕矩形定位并限制在窗口内；滚动时重新定位，节点离开可见区域则关闭。移开鼠标清除 Hover，Selected 保留。Hover 外圈、Pressed 明暗和 Focus 虚线圈分别反馈，失败节点保留红色身份。节点支持 Tab、Enter 和 Space；不可用身份使用 aria-disabled、移出 Tab 顺序并停用 Hit Path。选中节点的 aria-pressed 与原 app 状态同步。

快速 Chrome 检查：1440 / 850 下 Tooltip、一次点击、当前节点不刷新、Enter / Space、移开清理、横向滚动后的定位、快速跨节点切换及不生成 Compact 菜单均通过，未发现 pageerror。窄屏鼠标焦点不自动移动阶段，避免按下到抬起间目标位移；键盘聚焦仍能显示节点名称。截图：[Runtime if Hover](previews/direct-dock/runtime-hover.png)。本轮只修改 lineage-dock.js / lineage-dock.css，原始关系、数据、详情和主图未改；未 commit、push 或部署。

## Scroll-driven Collapsing Lineage（2026-11-03）

本节取代之前点击驱动的 overview / entering-inspect / inspect / returning-overview 与顶部上滚意图阈值说明。

当前 DOM：`decision-pane-body`（唯一主阅读滚动）→ `lineage-scroll-region`（自然高度的 Sticky 区域，包含 Full 图谱、阅读注释和 Compact Renderer）→ `analysis`（普通文档流详情）。Mini 是该 Sticky 区域底部的紧凑投影，不是另一个页面或页面级浮层。完整图谱始终留在 DOM 与文档流中，不再高度归零、互斥切换或隐藏阅读注释。

折叠距离 = Full 区域实际自然高度 − Mini 实际高度。Sticky top 为折叠距离的负值：区域顶部随主 Pane 滚出，最后 108px 自然停靠在 Pane Body 顶部。Full / Mini 在最后一段滚动距离交接透明度；滚动位置决定 mode，仅切换命中与键盘可访问性，不改写 scrollTop。ResizeObserver 重算几何，scroll 监听通过 requestAnimationFrame 合并。无滚轮 / 触摸方向拦截、延时猜测、每次状态切换归零或新增空白占位元素。

点击节点：原入口同步 Selected 与详情 → 下一布局帧计算 Analysis 文档位置减去 Mini 高度 → 原生 smooth scroll。连续点击由最新请求和浏览器原生滚动接管；手动滚动可直接中断。Reduced Motion 使用即时定位，不播放强制动画。短详情采用 Analysis 自身的最小可读视口高度，以便将标题完整定位到 Mini 下；这不是另加 Spacer，长详情继续按真实内容自然延伸。

上滚重新进入图谱区域会自然恢复 Full，详情不收起、Selected 不丢失；“展开图谱”只平滑滚回顶部，同样保留详情。只有“收起详细分析”或 Escape 会在回到顶部后关闭详情，避免途中移除详情造成高度跳动。全图横向位置不再需要复制 / 恢复，原 scroller 一直存在。详情打开期间点击 Full 空白不清除 Selected，防止与下方详情失配。

快速检查：1280×900，UB 详情打开后图谱自然高度 530px、主文档 scrollHeight 1501px；折叠、中途、回到顶部均保持这两个高度。折叠滚动位置 422px，Analysis 顶部与 Mini 底部一致；main.scrollTop=0，只有 Pane Body 滚动。向下阅读、手动上滚、无点击情况下由滚动恢复 / 折叠、Mini 直接切换、返回保留详情、显式关闭、Reduced Motion 均通过，未发现 pageerror。未全面重跑其他业务测试。

截图：[初始](previews/scroll-lineage/overview.png)、[吸顶详情](previews/scroll-lineage/inspect.png)、[滚动中途](previews/scroll-lineage/mid-scroll.png)、[回到完整图谱且保留详情](previews/scroll-lineage/restored-with-detail.png)。记录：[checks.json](previews/scroll-lineage/checks.json)。修改仅涉及 lineage-dock.js / lineage-dock.css / app.js 与说明，原图数据、节点关系、Inspector 内容和 IDE Shell 未变。未 commit、push 或部署。

## UB Memory Planning Explorer（2026-11-03）

Memory · 192 KB 已从独立的 Map / Selected Slot / Rotation / Pipeline 模块合并为同一个规划探索区：静态容量 Summary → Shape / Tile / Iteration 控件 → 等比例 UB Map → 槽上下文与相关数据流 → 默认折叠 Evidence。

容量比例为 48:48:48:48:56（248 KiB 单 Vector Core 地址空间）；77.4% 明确标记为静态分配率。蓝色输入、绿色输出，Active 为较强填充 + PING/PONG 状态文字，Inactive 保留低权重，Selected 单独用白色轮廓。Iteration 只更新节点 class、文字和数据流，不重建 Map，过渡不改变槽位位置 / 宽度。Hover 用轻量原生 Tooltip 显示槽名、半开地址范围和大小；单击选择槽位，Free 可查看未分配属性，不伪造 Mutex。

复用 SoftmaxTileModel 的 Shape / Tile 状态。切换 Shape / Tile 更新 Input、行范围、Valid 与 Allocated；不改变 Iteration 或 Selected Slot。独立的逻辑 Iteration 不代表所选 Tile 实际使用该 Buffer Pair。数据流 GM → Active Input → VF → Active Output → GM 支持 Hover / Focus / 点击关联强调，但不隐式更改槽选择，也不表达 Timeline。

移除了重复的 Buffer Group 按钮、地址轴中间标签、大型 Selected Slot、独立 Rotation / Pipeline 说明和默认 Core Mapping。Evidence 保留全部槽地址、Group / depth、Mutex、字节计算、32B 对齐、set_validshape、.next / auto_mutex 源码入口、DESIGN 和设计绑定。条件化 Core 映射说明仅作为 Illustrative Mapping 的证据边界保留；未展示未经验证的实际映射或硬件重叠时长。

快速 Chrome 检查：1280×900 和 600×900；Iteration 1 高亮 in1 / out1，Selected 不变，各段实测宽度切换前后一致；out1 对应 Mutex 3 与 VA_OUT1 源码；Tile 2 的行范围 6–6、Valid [1,200]、Allocated [3,8192] 正确；Shape / Tile 不重置 Iteration / Selected；数据流关联高亮、Evidence 展开和源码查看可用；窄屏无组件横向溢出，Console 无 pageerror。没有全面重跑其他业务测试。

截图：[Iteration 0](previews/memory-planning/iteration-0.png)、[Iteration 1 / in0 仍选中](previews/memory-planning/iteration-1.png)、[Evidence](previews/memory-planning/evidence.png)、[窄屏](previews/memory-planning/narrow.png)。检查记录：[checks.json](previews/memory-planning/checks.json)。只改 ub-buffer-analysis.js / ub-buffer-analysis.css；主图、Mini、节点导航、连续滚动与真实证据文件未修改。未 commit、push 或部署。

## Single Tile 统一数据组织矩阵（2026-11-03）

Single Tile 详情仅保留输入 Shape、每 Tile 最多 3 行 / N 轴整行归约规则和一个可展开矩阵。每条 Tile 自身展示 Rows、有效行结构、Valid / Allocated；选中条目在原地展开三行结构，不再另画一张右侧 Tile 面板。尾 Tile 2 明确展示 Valid [1,200]、Allocated [3,8192] / 48 KiB，一行蓝色有效数据和两行灰色无效行。示意列宽不是实际列数比例，展开属性旁保留短标识。

删除独立 Register / N Tail 主视图、128-lane 图、128 / 72 / 56 拆分、右侧 anatomy 面板以及 Hover 重渲染临时容器。Hover 只有 CSS 反馈；点击条目切换 shared selectedTileId、更新矩阵并保留键盘焦点。证据保留分配尺寸、set_validshape、尾 Tile 行数源码及 DESIGN / 设计绑定，不再提供寄存器入口。主标题为 Single Tile · 数据组织，末尾仅保留一行跨 Tile Max / Sum 状态结论。

复用 SoftmaxTileModel.presets / tiles / setState；与 Memory Explorer 共享 Shape / Tile 身份。主图、Mini Lineage、滚动折叠及其他专业节点源码未改。已有 Memory Planning Explorer 改动保留。

快速 Chrome 检查：1280×900 / 600×900；Hover 后展开 DOM 身份不变；Tile 2 为 1 有效 + 2 无效行；Memory 往返保留 Tile 2；P0 [32,4096] 有 11 个 Tile、尾 Tile 10 有 2 行有效；键盘 Enter、窄屏无组件横向溢出、Console 无 pageerror。现有 IDE 回归脚本的标题期望同步为新标题，未全面重跑其他业务测试。

截图：[Tile 0](previews/tile-matrix/tile-0.png)、[Tile 2](previews/tile-matrix/tile-2.png)、[窄屏](previews/tile-matrix/narrow.png)。检查记录：[checks.json](previews/tile-matrix/checks.json)。未 commit、push 或部署。

## Single Tile · [1,128] unit 矩阵（2026-11-03）

本节替代上一版“可展开行块示意”表达。每个 Tile 始终显示三行真实矩阵组织；点击仅更新 Selected 和共享 Tile 状态，不再展开第二套条形结构。Hover 仅 CSS 反馈，矩阵 DOM 不重建。

最小格子为 [1,128]。Input [7,200] 的每个有效行显示 U0 的 128 列完整有效、U1 的 72 有效 / 56 无效（格内分割比例 72:56），以及未用但已分配的 U2、U63；U3–62 以一个明确标注“×60 已分配 · 未用”的省略格表示。单格与省略格不能被当作相同的数量宽度。纵向仍保留三行；尾 Tile 的两个无效行使用独立的无效行状态与 × 标记，不把它们称为未使用列。

上下文直接列出 Input、Allocated [3,8192]、Unit [1,128] 和 200 = 1×128 + 72 / 8192 = 64×128。每个 Tile 自带 Rows、Valid、Allocated 和 48 KiB；无需左右查看另一张 Tile。默认没有独立的 Register / N Tail 模块。

所有实际使用的 unit 都是独立单元格，不把已用区混入省略。Input [32,4096] 显示每行 32 个已用格，只折叠未用的 U33–62；U32、U63 保留。密集矩阵自身可横向滚动，整页不缩放。采用原 PTO 暗色、primary / foreground / surface 与排版 Token，没有引用参考图片的视觉样式；本轮消息未附图，按文字语义实施。

快速 Chrome 核对：Tile 0 / 1 / 2 的有效 / 无效行分别为 3/0、3/0、1/2；有效行完整格、部分格、未用区省略格数量正确；P0 11 个 Tile，每个满行 32 已使用 unit；Hover 不重绘或改变选中；1280 / 600 下无组件外横向溢出，Console 无 pageerror。原数据模型、Memory 状态、Lineage / Mini 与连续滚动代码未改。

截图：[Input [7,200]](previews/unit-matrix/input-7-200.png)、[选中尾 Tile](previews/unit-matrix/selected-tail.png)、[Input [32,4096]](previews/unit-matrix/input-32-4096.png)、[窄屏](previews/unit-matrix/narrow.png)。记录：[checks.json](previews/unit-matrix/checks.json)。未 commit、push 或部署。

## Single Tile · 连续二维矩阵（2026-11-03）

本节替代此前每 Tile 独立条目 / 卡片表达。主图现为一个 CSS Grid：X 轴为 Unit Index，Y 轴为物理行位置；Tile 标识纵向跨 3 行，不再有卡片 Header、额外展开区或 Cell 内重复文字。Tile 分组只通过矩阵连续边界、固定行列和左侧分组标识表达。

`SingleTileAnalysis.projection(index)` 与 DOM 分离：从 SoftmaxTileModel.tiles() 生成所有物理行，再按 full=floor(N/128)、tail=N%128、used=ceil(N/128)、allocated=8192/128 生成统一列布局及 Cell 状态。所有 Tile 共用列模型，所有普通格 24×24px，Partial 外部尺寸不变，内部宽度按 tail/128 精确分割。无效行也保留 24px 行高。

四种语义：Full 为完整填充；Partial 的 Masked 子区使用纹理；Allocated Unused 为空心中性格；Padding 整行采用弱填充与中线。状态不只依赖颜色。Unit 是 VF 逻辑数据组织单位，不表示永久独占的硬件寄存器。横向省略为固定 72px 的窄网格断裂区，包含压缩网格 / 断裂符号，范围和数量只在 X 轴标注，不在各行放长文字条。未用区末端 U63 保留，普通 Unit 尺寸跨 Shape 固定。

[7,200]：U0 Full；U1 内部 72/128 valid、56/128 masked；U2–U62 共 61 个 allocated-unused unit 压缩；U63 未用。共 9 物理行，7 有效 + 2 Padding。

[32,4096]：每行完整保留 U0–U31 共 32 Full Unit，无 Partial；U32–U62 共 31 个未用 Unit 压缩，U63 未用。共 33 物理行，32 有效 + 1 Padding。

Hover Cell 使用顶层、不可交互 Tooltip 显示 Unit、列范围、有效数与状态；Hover Row / Tile 仅更新样式。点击 Cell 或左侧 Tile 按钮更新 Selected 和共享 Tile 身份，不增删矩阵、不改变 Tile 高度、不重置滚动位置。选中边界独立于 Hover。矩阵不设独立纵向高度限制，大 Shape 随 Pane Body 自然滚动；仅矩阵横向滚动，Tile / Row 标识横向 Sticky 保持可见。

快速 Chrome 检查覆盖两个 Shape：行数 / 状态、压缩区间、全部普通格 24px 宽 / 高、Partial 填充比例 72/128、点击 Cell、Hover 不重绘、Memory 保持 Tile 2、P0 无 Partial、600px 窄屏横向滚动后行标识保持在矩阵左边界。Console 无 pageerror。检查记录：[checks.json](previews/continuous-matrix/checks.json)。未全面重跑其他业务测试。

截图：[Input [7,200]](previews/continuous-matrix/input-7-200.png)、[选中尾 Tile / Hover](previews/continuous-matrix/tail-hover.png)、[Input [32,4096]](previews/continuous-matrix/input-32-4096.png)、[窄屏横向滚动](previews/continuous-matrix/narrow-scrolled.png)。本轮只调整 Single Tile 两个组件文件和说明；保留已有 Memory 等本地改动，未改共享数据模型、Lineage / Mini、详情滚动及其他 Inspector。未 commit、push 或部署。

### 矩阵布局留白调整（2026-11-03）

根据实际 1280×900 工作区调整：旧矩阵仅 301×267px，普通格 24×24px、格间距 1px，坐标和 Tile 标签过密。现为 436×376px，普通格 40×32px、格间距 4px，Tile / Row 列宽 104 / 80px，轴区高 52px。压缩区固定 112px，仍显著短于所代表的 61 / 31 个普通格；不铺满剩余宽度。内容组间距改为 16px，去掉 Selected 摘要中重复的 Allocated 字段，恢复辅助坐标的 Token 字号。

1280 / 600px 实际浏览器检查：无组件外横向溢出；Partial 填充比仍为 0.5625；两个 Shape 的 unit 计算、共享状态及矩阵交互逻辑未改。大 Shape 继续使用矩阵横向滚动。截图：[调整前](previews/matrix-spacing/before-1280.png)、[调整后](previews/matrix-spacing/after-1280.png)、[窄屏](previews/matrix-spacing/after-600.png)。实测记录：`previews/matrix-spacing/after.json`。未 commit、push 或部署。
