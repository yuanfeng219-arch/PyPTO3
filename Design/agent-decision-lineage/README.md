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
