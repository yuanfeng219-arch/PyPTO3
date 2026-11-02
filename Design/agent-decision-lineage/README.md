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
