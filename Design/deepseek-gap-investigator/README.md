# DeepSeek V4 Trace Investigator · v0.12

对应 [Spec v0.13](../../Product_Planning/DeepSeekV4_运行异常反向追溯产品_Spec.md)，2026-09-11。已实现前端联动、工程只读导入及基于真实 Pass 结构的显式模拟补全；**尚未完成设备执行端到端闭环**。

## 运行

从仓库根目录提供 HTTP 服务：

```sh
rtk python3 -m http.server 8766 --bind 127.0.0.1
```

入口：[Trace Investigator](http://127.0.0.1:8766/Design/deepseek-gap-investigator/index.html)。v0.1 备份未修改。默认加载真实 Rank 0：9,480 个 X 事件、226 条进程线程泳道、4,884.62 µs。Worker 占用按时间区间并集统计，不合并 Scheduler。

[模拟 Pass Diff 入口](http://127.0.0.1:8766/Design/deepseek-gap-investigator/index.html?v=26&demo=compiler)：直接进入 01 / Rank 0 的 Step 04。Step 04 直接显示对照图，不再经过二次按钮。基线任务链取自真实 Pass 23，模拟变化复用真实 Pass 24 的 `gm_pipe_buffer` 语法；与当前 producer 的映射和耗时仍明确标为模拟。模拟仅通过此显式 Demo URL 进入，调查头部不再提供“真实证据／模拟补全”切换，也不显示“待取证”状态标签。03–06 使用显式模拟，主 Trace 和 01／02 的运行记录保持真实。

模拟场景：片上峰值 96 KiB 超过示例预算 64 KiB，内存分配在打包与发布之间插入 GM Store / Load；修改演示参数 `DEMO_PUBLISH_TILE_ROWS` 64→32，峰值降到 48 KiB，去掉 320 µs 搬运，等待 663→343 µs。两份模拟泳道在主工作区对比，事件 Inspector 也标为模拟。预算、参数、图和结果不是当前工程事实；真实致因 Pass 仍因证据不足而未确定。

## 已实现

- 单行工具栏；“诊断模式”默认开启并以黑底白字表示。关闭后恢复语义配色，开启后全部 Task 去色。黄色矩形同时编码时间与受影响 Core：具名调查只覆盖其 wait Task 所在 AIV，规则候选只覆盖参与共同空闲计算的 24 条 Worker AIC；前者实线、后者虚线。点击黄色空白打开调查，真实 Task 命中仍优先打开 Inspector；标签支持 Tab / Enter。
- 框选后先显示就地“调查此选区”；创建才打开表单。调查抽屉左侧列表常驻，收起后直接从黄色异常区间重新进入。
- 六步：确认异常 → 定位任务 → 排查原因 → 追溯编译变化 → 制定调优方案 → 运行实验与对比。
- 删除抽屉内独立泳道及其 Rank、对齐、核明细状态。保留两侧指标、相关任务表、生命周期、资源、上游、局部关系与七问。
- 统一 `revealTask`：切换目标 Rank、解除过滤、时间定位、纵向滚动、高亮和独立 Inspector。逻辑执行范围使用 Run / Rank / process / taskId / shortId / FuncId / 执行单元聚合；同一 r2t53 同时有 24 AIC + 48 AIV，选中 AIC 函数时不能混算为 72。FuncId 缺失时仅按相同原始名称回退，不做相似名称映射。支持全部 Worker 与精确实例选择，目标位于抽屉上方。
- Inspector 每次从关闭状态打开时恢复为 360px 最小宽度，随后仍可手动拉宽；未选中事件时显示当前 Rank 与当前筛选后的可见泳道、事件数、Run 时长、平均占用率和最活跃泳道，选中真实或模拟事件后切换为对象详情。
- Case 范围、Rank 和步骤不因浏览变化而改变。支持返回调查范围及定位前视口。
- 主工作区按上下文提供 Trace、跨 Rank 对照与实验对比。Step 04 移除默认 49→50 `InsertCommFence` 和全量 Pass 选择器；当前两个等待调查缺少生产者到达时序与 Task→编译子图映射，正文显示本次等待任务、已记录的直接前驱及返回上游排查的操作。点击任务继续联动主 Trace；进入步骤本身不移动视口。旧版保存的 Pass 编号不会恢复无关对照。
- Task 关联源码、候选参数源码和证据源码均使用原位折叠代码块，不再打开二级面板。
- 保留 52 份 Python IR 静态 AST 索引、104 个局部图与图渲染组件（Fit / hover）；当前诊断没有可追溯关联，不加载这些图。后续接入必须明确具体任务、编译节点及变化如何影响该任务，不按函数名或“有变化”自动选图。未执行任何被解析的 Python。
- 模型源码、IR 文本、Pass 实现区分命名；没有 Pass 实现映射时不冒充存在。
- 当前 DeepSeek 工程一键只读导入；读取并验证 49 个 Python 文件摘要，识别 decode_csa.py 的编译、golden 调用与 DFX CLI。另支持用户选择源码目录。识别信息自动填入实验定义，核验修改符号与当前值。
- 两份实际载入的独立 Trace 可在主工作区并排浏览、缩放和平移联动；按进程 / 泳道名称对照，缺失 lane 为空并标为未匹配。支持 Run 相对起点与用户选中事件锚点对齐。同一 Run 两个 Rank 明确叫“跨 Rank 对照”，不是优化 Diff。

## 未接通与证据边界

- **执行服务未连接**：PyPTO / 工具链版本、golden 辅助模块、输入和 Ascend 设备未验证。没有应用修改、候选编译、正确性实跑或重复采集；界面保留预检查阻塞，不伪造进度。
- 源码内容摘要只标识所导入文件，不证明它就是历史 baseline 的构建 revision；实验命令是静态识别或用户填写，未经执行验证。
- 当前数据没有第二次优化后的真实 Run，默认不展示合成候选。自动回收工程产物仍依赖 runner。
- 显式模拟模式的结果仅留在独立内存会话；不写入真实 Case／实验／测量字段，不持久化到 localStorage，不修改主 Trace。模拟以等待开始为零点，不与真实 Rank 1 事件拼接；只演示局部场景，正确性与真实端到端收益未验证。
- Runtime Ready / Enqueue / blocked reason、资源快照、Pass 决策理由仍缺失；界面统一使用“未采集／暂无法判断”，不向用户暴露内部证据状态枚举。
- IR 图只解析精确名称的函数 / scope，显示局部词法引用、控制嵌套和源码顺序。未分析循环携带依赖、别名、内存副作用或完整跨函数语义；源码顺序不代表运行时调度或耗时因果。解析不到的范围显示缺口，不把节点文本文法差异称为稳定结构匹配或首次变化 Pass。
- Pass IR 复用原工具的布局引擎及 PTO 节点 Pattern，新增 AST 格式适配；**没有接入原 Pass IR 的全部交互和编译器语义分析**。
- 跨 Run 暂无稳定逻辑 Task 对应和一对多编译映射，不生成确定性的任务新增／删除或 Copy 字节变化结论。仅有指标 JSON 时叫“指标对照”。
- 当前工程导入不写磁盘，内容仅留本页内存；刷新后需重新导入。Case 和实验定义仍保存至独立 localStorage。候选 Trace 刷新后需重选，不能只凭保存的文件名声称已加载。

## 候选产物与离线路径

设备端到端执行的预检查尚不通过，手工导入仅是降级路径。实验 ID 由页面创建。候选 Chrome Trace 顶层保留原 `traceEvents`，额外提供：

```json
{
  "run_manifest": {
    "run_id": "独立候选 Run ID",
    "rank": 0,
    "experiment_id": "当前实验定义的 ID",
    "case_id": "当前调查 ID"
  },
  "traceEvents": []
}
```

空数组不通过校验；必须载入有效原始事件。候选 ID 与基线相同、Case / 实验 ID 不符、Rank 不符均拒绝。页面为文件记录 SHA-256；身份来自导入者，不构成真实设备执行的独立证明。

指标结果继续用页面下载的 schema v2 模板。验证实验 / Case / Run / 修改对象、正确性声明与来源、相同运行条件、至少三次测量及跨 Run 匹配来源。3% 为门槛，不是收益预测；所有结果注明依据用户导入记录。

## 文件与复用契约

| 文件 / 部件 | v0.6 职责与消融 |
| --- | --- |
| `app.js` | 统一 Task 定位、Case / 浏览状态隔离、区间入口；删除仅更新 Inspector 的证据点击 |
| `render/trace.js` | 主 Trace 与真实对比共享渲染器；PTO Task bar / colormap / Tooltip；诊断模式只覆盖主 Trace |
| `render/workarea.js` | 主视图切换、真实 Trace 对照；保留显式输入图对的渲染组件，移除默认 Pass 选择和加载 |
| `build-workspace.py` | 非执行 AST 适配与工程文件索引，派生文件不覆盖原始数据 |
| `vendor/pass-ir-layout.js` | 原样复制自 /Users/yin/pto/js/layout.js（2026-09-10）；原源码未修改 |
| `investigation/project.js` | 源码目录、摘要、入口、修改符号与执行缺口；不执行导入命令 |
| `investigation/report.js` / `report.css` | 复用调查证据，删除独立 timeline 渲染器、事件状态与私有时间线样式 |
| `investigation/experiments.js` | 工程预检查、修改审查、离线结果门禁；指标不冒充泳道 |
| `investigation/simulation.js` | 经用户授权的 01 模拟证据、图、调优与结果；独立于真实数据及实验 |
| IDE Frame / Workbench split | 全窗口 body shell、底部非模态覆盖抽屉、右侧独立 Inspector |
| PTO cp-btn / card-demo / tab-control / btn | 导航、证据、上下文视图和动作；不创造新组件皮肤 |
| PTO Pass IR node Pattern | 真实 IR 节点与关系节点；无额外节点外框 |
| `components/ui.*` | 继续复用已授权 shadcn Field / Table / Alert / Dialog / Sheet / Command 适配，全部映射 PTO token |

所有 Pattern 均 direct embedding：本产品拥有数据与交互状态。没有 iframe、额外页面 shell、新视觉审批页或共享设计系统改动。全窗口 `ide-frame` 与 `data-surface="solid"` 保留；布局依照 Spec，默认使用 light 并支持 dark。调查抽屉固定为 `#F8F8F8` 的证据阅读面，不随全局主题改变。

Container decoration residue：无私有左侧装饰条或 inset-left shadow；生命周期连线为数据关系，IR 实线 / 虚线为数据 / 控制编码，outline 仅键盘焦点。正文 14px；普通 UI / 代码最低 12px，Canvas 内部文字由共享 Pattern 控制并提供 Tooltip 与 Inspector。

## 刷新与验证

```sh
rtk node Design/deepseek-gap-investigator/build-data.cjs
rtk python3 Design/deepseek-gap-investigator/build-workspace.py
rtk node Design/deepseek-gap-investigator/contract.test.cjs
rtk node Design/deepseek-gap-investigator/smoke.cjs
rtk node vendor/pto-design-system/scripts/audit-typography.mjs Design/deepseek-gap-investigator
```

Smoke 需要 Chrome / Playwright；必要时通过 NODE_PATH 指向工作环境依赖。默认访问 `http://127.0.0.1:8766/Design/deepseek-gap-investigator/index.html`，也可通过 `GAP_DEMO_URL` 指定已有服务。覆盖 1440×900、窄屏、主题、30px 单行标尺、按受影响 Core 限高的黄色入口、`#F8F8F8` 抽屉、Canvas 尺寸、逻辑任务与执行函数身份、跨 Rank 定位、实例、IR 来源、只读工程、实验阻塞与结果门禁。测试合成候选只在隔离浏览器内存在；封面生成前清空，不写入产品数据。

`cover.png` 为当前首屏，`investigation.png` 为联动 Task 与调查状态。旧 report 截图仅为历史素材，不作为 v0.6 验收要求。
