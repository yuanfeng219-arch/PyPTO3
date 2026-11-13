# Lab Tour 展示版 · v3

`../lab-tour-demo-v2/` 的独立新版本（2026-11-13 从 v2 复制，目录内对 `../lab-tour-demo-v2/` 的自引用已改为 `../lab-tour-demo-v3/`），v2 保持不变。讲解稿与 keynote 预览图未复制，launch.html 的预览图沿用 v2 的。

## v3 相对 v2

- **开场改为三页**（入口 `map.html`，launch.html 的 v3 卡片也指向它），页间用顶栏右侧的「01 地图 · 02 PyPTO · 03 演示」、下一页按钮或键盘 ← / →（PageUp / PageDown）翻页（`deck-nav.js`，读 `<body data-prev data-next>`）：
  1. `map.html` 新增：**昇腾计算 · 项目与产品支撑地图**。行是计算栈的层（模型与场景 / 服务与加速 / 框架与库 / 算子与编译 / 运行时与通信 / 硬件与互联），列是三条路径（主流生态接入 / 昇腾原生栈 CANN · Mind 系列 / PyPTO 开放栈 hw-native-sys），右侧一栏是横跨全栈的开发 · 调优 · 观测工具。标记：◆ 本次演示涉及（DeepSeek-V4、pypto、昇腾 NPU、PyPTO Toolkit）、● 已有体验原型（依据 `Design/` 下已有原型：Qwen3、pypto-serving、灵衢互联、PyPTO Studio）、○ 栈内相关。分层与产品取自 `Insight/Ascend-ecosystem2.md`、`Insight/hw-native-sys-仓库功能洞察报告.md`，结论句取自 PyPTO3.0 产品定位汇报大纲 P04。**这是按仓库资料起草的版本，团队实际支撑的项目 / 产品清单需要确认后替换。**
  2. `intro.html` 改为 **PyPTO 背景与用户问题与挑战**：左栏写 PyPTO 是什么（Tile 编程框架，切多大、交给谁、搬运和计算怎么衔接决定性能），保留 Tensor → Tile → 芯片三层图；右栏两类挑战各列三个具体难处——「写代码时 · 写得出来，看不见落点」→ 01 开发即可见，「上板之后 · 数据都在，结论得自己找」→ 03 让数据开口（痛点拆分取自 v2 目录的 11-13 领导版演示稿）。
  3. `index.html` 作为结尾页：右上角改为「← PyPTO 背景」，三个演示入口不变。
- **「开发即可见」前加一页 Agent 生成算子**（`agent.html`）：intro.html 卡片上的「01 开发即可见」与首页第 01 张卡都先进这一页，再进工作台。
  - 不带顶栏，主体占满一屏；样式直接引用工作台的样式表（`../csa-operator-lens/vendor/path.css`、`source-tags.css`、`styles/lens.css`，本目录 `header.css`、`demo-shell.css`、`workbench-polish.css`），28px 状态栏、源码行 `.cl` 与 `.tk-*` 着色都与工作台源码栏相同（着色规则抄自 `lens.js` 的 `hl`）。返回首页是最左图标栏的第一个图标；打开 Lens 走对话里的 `/lens qk_pv`（建议按钮或输入命令）。
  - 布局参考 Codex 桌面端的默认态：最左图标栏；会话侧栏（「Operator Agent」标题、新聊天、项目 deepseek_v4_flash_dspark 下的 qk_plan / kv_touch 历史线程、展开显示）；中间空白处居中一句「想在 deepseek_v4_flash_dspark 里写什么算子？」；底部输入框上方一条环境托盘（项目 · a2a3 本机模拟 · ···），输入框下排「+ / @decode_csa.py」与「Operator Agent 高 / 语音 / 发送」。发送后：项目下新增并选中「qk_pv · 稀疏注意力」线程（生成中转圈、完成变绿点），中间变成对话流（用户消息右侧气泡，Agent 正文 + 一行一个动作：读取 → 规划 → 编辑 +116 → 编译 → 对比 golden，进行中文字流光）；开始编辑时右侧「改动」面板展开（Codex 的审阅面板），显示 `decode_sparse_attn_csa.py` 新增行（+116，左侧绿条）与三项检查，可用对话栏右上角按钮收起 / 展开。1280px 以下改动展开时收起会话侧栏，1000px 以下去掉图标栏与侧栏，860px 以下上下排。
  - 输入框预填 qk_pv 的需求，点发送或 Enter 后回放约 10 秒；代码是 `decode_sparse_attn_csa.py` 198–313 行原文（`agent-source.js` 由 `../csa-operator-lens/js/source-snapshot.js` 抽出，与工作台 L2 源码栏同一段）。
  - 生成完成后，「改动」面板右上角的绿色「Operator Lens」按钮可用（生成前置灰），点它直接进下一页；对话里点 `/lens qk_pv` 建议或输入 `/lens qk_pv` 也会进下一页（先出 Lens 卡片再跳转）。
- **无限画布：L1 → L2 是同一块画布上的相机缩放**（`workbench-canvas.js` + `workbench-polish.css` 第 16 节；取代上一版“L2 下方面板默认收起”）。右侧可视化区变成一块带点阵底的画布：
  - 世界坐标里，L1 整层图占一屏；L2 的 Kernel 平面按 1/10 大小嵌在 L1 被双击的节点位置上。双击 Kernel 下钻时，相机从双击点开始放大约 10 倍（绕不动点做对数缩放，约 1s），L1 淡出、L2 淡入，停下时 L2 平面恰好 1:1；切回 L1（外层标签页的 L1 键）反向缩回。lens.js 自带的下钻动画与 View Transition 关掉，层级、页签、源码联动仍由 lens.js 管。
  - L2 平面上有三个对象：计算图卡片（Kernel 工作区，默认占满画面）；**「工作怎么分」卡片**——由计算图上的 **SPMD × 24 / FOR qk_it 外框**关联出来；**「放在哪」卡片**——由 **PIPELINE 里的阶段 ①–④** 关联出来（只 qk_pv 有）。两张卡片在计算图下方左右排开，各自从对象引一条虚线到计算图下沿的绿色入口按钮，再连到卡片。
  - 点入口按钮或底部导航条「计算图 / 工作怎么分 / 放在哪 / 全览」，相机平移（中途略拉远）到对应对象；聚焦卡片时同步切 lens.js 页签，源码栏随之换成整文件或 qk_pv 带标签摘录；没聚焦的卡片压暗，点它即聚焦。页签被别处切换（深链、“这一行在各层”跳到硅片落点）时相机跟过去。
  - L2 里滚轮缩放（以光标为中心）、拖动空白处或按住空格拖动平移；缩到很小自动回到 L1。深链直接落位：`#L2` 计算图、`#L2:assignment` / `#L2:eval` 等工作怎么分、`#L2:chip` 放在哪。
  - 只看计算图时撤掉“下方展开外两层 / 里层”的提示与高亮；动画用 rAF 推进，页面在后台时用 34ms 定时器兜底。
- **Agent → Lens 转场**：点「Operator Lens」（或 `/lens qk_pv` 走完后），改动面板复制一份从原位放大到整块标签页区域（0.46s，cubic-bezier(.2,.75,.15,1)），面板内容淡出、顶部 48px 显出标签条底色；会话栏、对话、状态栏同时淡出，图标栏不动。放大结束的画面与 `lens.html` 的第一帧一致，换页后标签条、导航栏的内容依次落下，工作台加载完成后淡入并轻微上浮（`lens.html#from-agent` 触发入场，进页后去掉这个标记）。减少动态效果时直接换页。
- **Operator Lens 标签页**（`lens.html`，Agent 页的下一页）：布局参考 Codex 内置浏览器的标签页——最左图标栏（同 Agent 页，第一个图标回首页），顶部标签条「Operator Lens · qk_pv」（× 关闭回到 `agent.html#done`），导航栏（后退回 Agent、前进置灰、刷新）+ 居中地址栏（随层级显示 `decode_csa · L1 整层` / `qk_pv · L2 Kernel`）+ 右侧「L1 整层 / L2 Kernel」分段控件、「详情」开关与绿色「编译并上板 →」，下面整块内嵌 `workbench.html?embed=lens`，**默认打开 L1**。
  - 嵌入模式（`workbench-embed.js`，只在 `?embed=lens` 下生效，单独打开工作台不变）：隐藏工作台自己的顶栏（返回键、品牌、路径、L1 / L2、详情、配置点、编译并上板），主体顶到 iframe 上沿；层级用 postMessage 与外层同步（外层发 `lens:go` 切层，工作台在层级 / 路径变化时回 `lens:level`），双击 Kernel 下钻或在外层点 L1 都会互相更新，file:// 下也可用；工作台里的离页链接在整页打开（`target=_top`），所以「编译并上板」按原流程进 loading。工作台的详情面板只能用「详情」键打开，所以这个键也挪到了外层导航栏（`lens:detail`），开关状态随工作台回报。
  - `agent.html#done` 跳过回放，直接停在“已生成”（讲解用）。生成过程是预置回放，状态栏注明。
- 「编译并上板」的 loading（`loading.html`）走完后**不再自动跳转**到 `result.html`，而是停在一页设备执行泳道：
  - 每行一个核（上 24 个 AIC、下 48 个 AIV），横轴为时间（每 0.5 ms 一格，末端标总时长 4.82 ms），色块是 kernel 的 block；**一种颜色一个 kernel**（64 个 kernel，按首次出现先后用黄金角在 OKLCH 色相环上错开，时间上相邻的 kernel 不撞色）。
  - 进入时从左往右约 1.4 s 画出（减少动态效果时直接画完）；悬停色块显示 kernel 名、核号、起止时间与耗时，并压暗其他 kernel。
  - 右上角「查看性能分析 →」进入 `result.html`，进入泳道时自动获得焦点。
  - `loading.html#lane` 跳过 loading 直接看泳道（讲解用）。
- 泳道数据 `loading-lane-data.js` 由 `result.html` 内嵌的 `lane-data`（perf-lane.v1，`_jit_l3_decode_csa_20260903_010617` · rank0）抽出：86 次 launch、4038 个 block，每个 block 记 core / 起止时间 / kernel 下标，约 70 KB；与性能页同一份数据。
- 窄屏下状态栏长文本不再把 loading 页撑宽。

## 继承自 v2 的说明

`../lab-tour-demo/` 的独立新版本，原版目录保持不变。相对上一版的三处改动：

1. 新增项目背景页 `intro.html`（内容取自 `../compile-board-demo/intro.html`，视觉与本目录 `index.html` 对齐），首页右上角「项目背景 →」进入。
2. 首页三幕的标题与描述改为体验设计点：下钻三处联动 / 编译不再黑盒 / 瓶颈主动指出；三个入口的去向不变。
3. 算子开发入口 `workbench.html` 改用 CSA 算子透镜的内容（`<base href="../csa-operator-lens/">`，与 `../compile-board-demo/workbench.html` 同源），顶栏控件与绿色「编译并上板」按钮沿用本目录的工作台样式。
4. 首屏默认取景按本 Demo 调过，只在本目录内生效，`../csa-operator-lens/` 不受影响：
   - 源码栏 `--code-width` 默认 34%（原 26%），代码行不再被截断；
   - 整层计算图改用本目录的 `graph.html` + `graph-host.js`（资源仍取自 `../csa-operator-lens/`），L1 缩放上限从 85% 压到 62%，首屏能看到前几个阶段而不是只剩第一个；
   - L2 上方计算图默认 46%（原 36%），「模块 Kernel 链」整条放得下；硅片落点仍是 34%。
5. `result.html` 顶栏下多一条默认结论：进页面就指出最值得分析的那一处，不再让一屏数据自己说话。一行 50px、无按钮——
   - 左边是谁：特征语义图标 + `qk_pv_aic + qk_pv_aiv` + 「关键路径 第 17 / 30 步」+ 特征标签「块负载不均」；
   - 右边三项读数：跨度 1.55 ms（路径上最长）、占关键路径 28%（配一条填充条，补注「新增 1.56 ms」）、最慢块 / 平均块 2.3×（平均块与最慢块按同一标尺画成两条对比条）；占比类特征则换成一条填充条。
   - 「它在路径哪儿」不另画迷你路径条，直接把下面那条真实的关键路径里对应的一段点亮（`#cp .cp-lead`，切视图或重画后自动补回）。

   结论本身不另算：取页面「优化候选」里已排好序的第 1 条（关键路径关联优先 + 分布异常补位），位置、跨度与特征名走 `window.ResultPathRows()`，占比取关键路径条自己写的「路径新增时间」除以顶栏执行窗口，数字和泳道、菜单一致；切 rank 会跟着重算（rank1 是第 16 / 29 步、1.67 ms、占 37%、2.4×）。开屏 5 秒的全局指标浮层正好压在这一行上，浮层在时它让出位置。实现只在 `result-polish.js` / `result-polish.css`，`../device-perf-lane/` 不动。
6. `result.html` 首屏视觉评审后改掉的三处，同样只在 `result-polish.js` / `result-polish.css`：
   - 泳道底图原本四千多个 block 同一个灰（#565656），哪里慢全靠浮标签指。改成按块宽（= 块耗时）在当前视图里分三档明度（`#4e4e4e` / `#5e5e5e` / `#747474`，分位点 75% 与 95%），热区自己浮出来；泳道每次重画、切 rank、缩放后重新分档，一次约 13 ms。
   - 页底加一条常驻图例：内容直接取页面自己那份 `#lane-lgnd`（原本藏在关闭的菜单里），压成一行 32px，长说明缩短、读图方法留给菜单里的完整版，再补一条「block 明暗 = 块耗时相对长短」。菜单那份仍在，所以这条对读屏隐藏。
   - 顶栏的「分析」是组标题不是按钮，改成 11px + 字距 + 左分隔线，不再像一颗禁用的胶囊。
7. 五个下钻页（workbench / compile / loading / result / all-passes）共用 `demo-shell.css` 作为视觉基线：底色（`--demo-page` #111111 / `--demo-canvas` #131313）、控件圆角 999px、卡片圆角 12px、浮起阴影 `0 7px 20px rgb(0 0 0/8%)`、焦点环 lime。返回首页键统一成 36px 圆形玻璃键，位置都在距左 20px、距顶 14px；整页 iframe 的包裹页用 `.demo-shell-home` 浮在画布上，内容顶到上沿的（all-passes）再加 `.demo-shell-inset` 留出 64px。各 Demo 内部组件不改。
8. `workbench.html` 视觉评审后的调整（2026-11-12），只在本目录：`workbench-polish.css` / `workbench-polish.js`，以及 `graph.html` / `graph-host.js`、`workbench.html` 内联样式；`../csa-operator-lens/` 不动。
   - L1 取景：阶段框（`.kf-pypa-graph__group`）及其伸出框外的标题算进取景范围，窄画布右侧标题不再被裁；缩放上限从 62% 提到 80%（下限 62%），首屏优先显示“上一阶段整框 + 当前节点”，放不下就从当前阶段标题开始排，阶段标题不再半截压在上沿。用户动过镜头后，画布变宽变窄不再自动取景。
   - 图元自带的底栏（图层 / 缩放 / Fit to View）原本压在提示条后面露出一截，现已隐藏（按钮仍在，供缩放 / 适应调用）；底部补一条图例：Tensor / 状态·Cache / 块内中间结果（黄）/ 计算·通信 / 阶段。
   - 页内控件（`.lens-seg` / `.lens-tabs` / 选择框 / 播放键）统一成顶栏同款胶囊；栏标题统一为 13px 正文字，不再一处等宽大写、一处中文粗体；顶栏路径改正文字 14px、当前层级高亮，窄宽度下省略。品牌名沿用五个下钻页共用的 `header.css`，未在单页改。
   - 强调色：lime 只留给主按钮、焦点环和硬件“GM / 数据搬运”语义；配置状态点、“真实”标签、L1 当前阶段源码行改为中性色。
   - L2 信息结构按“结构 → 分工 → 轮次 → 落点”重排：上方计算图是一个 Work Item 做什么（本体，不随下方切换），下方是它往两个方向的展开。
     - 下方页签按问题分两组：「工作怎么分 · 分工与轮次」（展开上图外两层 SPMD × 24 · FOR qk_it）与「放在哪 · 硅片落点」（展开里层阶段 ①–④）。原 Pipeline 并入前者，成为步骤条第 4 步：① Generation → ② Ordering → ③ Assignment → ④ Lane 轮次；lens.js 仍按 assignment / pipeline / chip 工作，隐藏的 pipeline 页签由步骤条触发。
     - 「Pipeline」改名「Lane 轮次」（页签、状态栏、说明），与计算图里一个 Work Item 内的 `PL.PIPELINE × 2` 区分开。
     - 计算图标出被展开的层：分工 / 轮次时点亮 SPMD 框与 FOR 循环框、压暗里层；落点时外层标注退后、阶段卡片描边；顶部一句「↓ 下方展开外两层 / 里层」。选中 Work Item 时仍以 lens.js 的路径高亮为准。
     - 「模块 Kernel 链」不再和计算图在画布顶部二选一，挪到顶栏路径旁（L1 模块与 L2 Kernel 之间）：「Kernel 链 N ▾」弹出迷你链，按源码顺序列出本模块派发的 Kernel，实线箭头是 TaskId 依赖，点 Kernel 即跳转；需要完整依赖图时点「在画布上看完整依赖链」。没有 Tile 计算图的 Kernel，画布上方仍直接显示完整链。
   - 「工作怎么分」图形化视图（`workbench-story.js`，只接 qk_pv）：顶部一条 5 节点流程条代替文字说明（图形 + 数字 + 两三个字，点节点切换）：① 1280 份活 → ② 真活 992｜空活 288 → ③ 24 个核轮流领 → ④ 54 轮 → ⑤ 分得好不好。同一批方块（一格 = 一份活）在五种排法之间移动，画布上只留短标签。⑤ 回答“计算逻辑写得好不好”：每个核一条长条，长度 = 工作量，所有核都等最慢的那个，右端齐 = 好、琥珀斜纹 = 空等；可切“KV 行数 / 份数”两种工作量口径与“现在 / 不排队”两种排法对照（rank 0：按 KV 行数 1.07× vs 1.04×，按份数 1.02× vs 1.04×）。KV 行数口径为估算（Synthetic 输入，未上板）。默认口径为 KV 行数。⑤ 另有「建议」排法与改进建议卡（窄画布时收成一条）：qk_plan 把真活按要搬的 KV 行数从多到少排队、每轮发牌换方向（蛇形），qk_pv 的发牌规则不变；rank 0 按 KV 行数 1.07× → 1.01×、空等 7% → 1%，按份数不变（1.02×）；rank 1 为 1.13× → 1.01×。代价是 qk_plan 多一次排序、同一 token 的活不再相邻，需上板确认。深链：`#L2:generation` / `#L2:ordering` / `#L2:eval`。
   - 硅片落点图整理（`workbench-chip.js` + `workbench-polish.css` 第 11 节，lens-tile.js 画完后补一层，联动不变）：AIV1 缓冲画成斜纹空心（已占地址、无有效数据）；默认不点亮三个大框，白边只在悬停或详情窗打开时出现，AIV1 琥珀虚线是唯一强调；184 可用线改中性灰，红色只留给超限；SPMD 条第 1 个 block 拉放大锥到“BLOCK 0 / 24”分组框；数据流补全（q → Q 绿、ori_kv / cmp_kv → qk_kv 紫、qk_order / bias → AIV0 与 AIV0 → mi / li 蓝，交叉处留缝，SPMD 条垫底让线从下面穿过）；AIV1 用虚线连到管道“只握手、无有效数据”；GM 卡点阵减淡；说明文字改正文字体；“数据：pass 36”移到全图右下角；Vec 柱加宽、段说明加引线。
   - 硅片落点图信息分层（`workbench-chip.js` + `workbench-polish.css` 第 12 节）：常驻只留区块名、L1 条、数据流、AIV0「高水位」、AIV1「空转」和数据来源注脚；其余 38 条说明（GM 形状、L0A/L0B/CUBE/L0C 细节、Vec 刻度与分段、AIV1 解释、管道槽数、连线标签）悬停所属区块时显示，点击钉住、再点取消；在源码悬停联动点亮的区块也会显示；读数行旁「全部标注」一次展开（讲解用）。顶部资源读数行常驻 Vec 与 AIV 两项，其余悬停该行展开。
   - 计算图与硅片图同一对象同一图元（`workbench-polish.js` unifyLogic + `workbench-chip.js` unify + CSS 第 13 节）：L1 上的 Q / KV 与计算图张量同为亮色网格、深色字，名字统一为 Q、KV（形状悬停看）；v2c 槽、C2V / V2C 管道槽与计算图 PIPELINE 的在飞格同为亮蓝方块，文字统一为“2 槽”；计算图阶段卡加执行单元竖条与角标（① 搬运 GM→L1 绿、②④ AIC · Cube 紫、③ AIV · Vec 蓝），硅片图 AIC / AIV 框加同色竖条；SPMD 两边同为竖条图标 + 24 格小方块条；Q / KV 进入 Q·Kᵀ 的连线按角色上色；写回统一为“写回 mi / li / oi”，硅片图上从整个 block 发出（qk_pv 302–304 行：mi / li 来自 AIV0，oi 来自 AIC L0C）。
   - 硅片落点图布局（`workbench-chip.js` 网格常量，用现有图元；上方计算图 / 下方内容区仍上下排）：最左一列 GM（自上而下 qk_order / bias、mi / li / oi、ori_kv、cmp_kv、q，每张卡与目标同高），与 BLOCK 框上下沿对齐；右边三行 —— AIV0 细长横条在上、AIC 居中、AIV1 细长横条在下（两个 AIV 夹着 AIC），三行左右对齐、行间距 14。AIC 一行从左往右：L1 竖条（v2c / KV / Q，低地址在上）→ L0B / L0A → CUBE → L0C（不再重复显示右侧 V2C 槽）。AIV 横条：标题在左上，Vec 横条 440 宽（地址从左往右），次要说明两列在右、每条前加同色色块对应分段。写回：L0C（oi）与 AIV0 UB（mi / li）汇入 AIV0 与 AIC 之间的横道回到 GM。
   - 硅片落点图连线化简（`workbench-chip.js` 位置常量 + 连线段）：每条关系一根线，靠摆位让线变直——GM 卡片与目标同高，qk_order → UB、ori_kv / cmp_kv → L1 KV 段、q → L1 Q 段都是水平直线；L1 改为低地址在上，v2c 槽正对 AIV0 UB 的 softmax 工作区，V2C 一条竖线；L0B / L0A 与 KV / Q 段同高，CUBE 盖住两条输入线，L0C 与 CUBE 同中线，AIC 内部全是水平直线；L0C 与两个 Vector 列上下对齐，C2V 一条竖线向上进 AIV0（两个管道槽骑在线上），AIV1 的握手虚线沿同一列向下；写回只拐一次（oi 从 L0C 上到横道左拐进 GM，mi / li 从 UB 竖直落到横道汇入）。原先 q / KV 的竖道、L0 → CUBE 的折线、C2V 绕右侧回 UB、V2C 绕 L1 左侧都去掉。同类搬运胶囊上下对齐（3 个 MTE2 一列、2 个 MTE1 一列）；CUBE 的形状说明挪到其上方空白处。
   - 硅片落点图：**图形与布局**参照 PTO 设计系统 `hardware-architecture-viewport`（存储块 / 实心计算块 / 搬运胶囊；CUBE 改为计算块，每个 AIV 加 Vector 计算块、UB ↔ Vector 双向箭头；L1→L0 标 MTE1、GM→L1 / GM→AIV0 标 MTE2、L0C 写回标 FixPipe、UB 写回标 MTE3），**颜色 / 字体 / 面板质感**沿用本页、与上方计算图一致（`workbench-chip.js` restyle + `workbench-polish.css` 第 15 节）：核面板同计算图阶段卡（#262628、圆角 8、无描边，AIV1 琥珀虚线）；存储块中性底 + 灰点阵，占用条按数据角色上色；计算块按执行单元着色（CUBE 用 Cube 紫、Vector 用 Vec 蓝，AIV1 的 Vector 压暗）；搬运胶囊同源码栏 GM 搬运标签（深底绿字等宽）。参考图里的 L2 Cache、Scalar、指令序列未加。GM 列的五份数据改用与上方计算图 / Kernel 链相同的张量图元（`PathGridGlyphs.tensor`）：q 为 Q 绿、ori_kv / cmp_kv 为 KV 紫、qk_order / bias 与 mi / li / oi 为 Kernel 链里的张量蓝，名字与形状常驻（qk_order 写作 `[QK_ITEMS] · INT32`，同上方图）。计算图阶段卡与硅片图 AIC / AIV 框上的执行单元色条已去掉（角标保留）。
   - 色板继承 Demo（`workbench-polish.css` 第 0 节）：照 `loading.html` 的 `.cb-app` 做法，在本页把透镜原版粉彩色换成首页品牌色——绿 `#88EC30`（GM 数据 / 主按钮 / 焦点环）、紫 `#A459FF`（Cube · AIC · K/V）、蓝 `#1953FC`（Vec · AIV）、琥珀 `#fbc568`（告警），文字 `#f5f5f5 / #b8b8b8 / #858585`，框架结构用首页中性冷灰 `#92939a`。饱和蓝紫作文字时改用浅一档（`--sky-ink #849bff` / `--lav-ink #c79bff`），大面积格子与硅片缓冲用降一档（`--*-mute`）。源码行内标签、Lane 格子、硅片画布、问题分析胶囊、Kernel 计算图的 Q / KV 张量都跟着换。L1 整层图沿用 PyPTO Compute Explorer 图元自己的配色（Tensor 蓝 `#80b7f7` 与首页 `--kn-blue #849bff` 同族），未改。
   - 硅片落点：资源读数并入“视图”页签行，去掉一整行标题；顶部加默认结论条（取状态栏告警与说明里的结论句），画布上用琥珀虚线圈出 AIV1；底部说明不再占行，只在出现悬停 / 选中读数时浮在画布下沿。硅片图在 1440×900 下高度约 327px → 370px。
   - L2 上方计算图：Assignment / Pipeline 默认 46% → 41%（模块 Kernel 链仍整条放得下），硅片落点 34% → 33%；Assignment 的 Lane 标注放大一档。
   - 详情窗默认停靠右侧、画布让位（宽屏 ≥1101px），拖动标题栏后变回自由浮窗，关闭再开回到停靠位；公式在窄栏里换行，不再横向截断。
   - 状态栏去掉与画布提示重复的“双击模块展开，双击 Kernel 进入 L2”。
   - L2「Kernel 计算图」（`#hpLogic`，由 `lens-tile.js` 渲染）：重画后由 `workbench-polish.js` 补装饰，几何与交互不变。控制结构统一用源码栏 PTO 框架标签的蓝灰（`--fw`），类型靠形状区分——SPMD 是实线框加背后两层错位卡片叠（并行 24 份）；FOR 是完整虚线框并标注每次处理一项（框内为循环体）；PIPELINE 是实线框加表头色带；原来孤立的「块无效时」画成 IF 虚线分支框，和「① 拼 KV 块」并列。嵌套由外到内逐级提亮，计算阶段 ①–④ 改为实心卡片 + 正文字标题，不再和等宽大写的结构关键字同一种样式。

# 昇腾计算可解释/可观测体系演示

入口为 `index.html`，提供算子开发、可视化编译和性能调优三个入口，性能调优直接进入第三幕 Device Perf Lane（`result.html`）。算子开发进入 `workbench.html`（整层计算图 → Kernel 工作区：Assignment / Pipeline / 硅片落点），点击「编译并上板」后经过模拟进度，进入设备性能分析。

编译与上板均为模拟。算子开发页面使用 CSA 编译产物，性能分析使用同一运行的预置 trace；页面交互不会触发重新编译，也不会改变结果数据。页面复用相邻 Demo 的资源（`../csa-operator-lens/`、`../device-perf-lane/`、`../hardware-native-ide-ui/` 等），需保留其相对目录结构；可直接打开或通过本地静态服务器访问。
