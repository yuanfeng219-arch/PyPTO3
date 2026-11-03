# AI Infra UI 接入与验证

日期：2026-10-29。入口保持 `../index.html`，不更新目录、不发布、不修改算子与模型数据。

## 实际复用

本次读取的调度入口为 AI_Infra checkout 的 `ai-infra-research/methods/ai-infra-investigation/SKILL.md`。按其路由检查了 catalog 中的 Paged Attention Graph、多核泳道、请求泳道与访存调查 Pattern。外观和浮层组织参照 `prototypes/logsumexp-case5/page.html`；没有引入该案例的数据、公式或调查候选。

| 层 | 来源与实现 |
|---|---|
| 领域模型与交互 | 保留本 Demo 原有 Assignment / Pipeline / Graph 控制器、布局、动画、对象 ID、选择映射、源码快照和文件导航。原有 `swimlane-task`、`model-graphviz` 与 IDE Frame 行为依赖保留。 |
| 计算图图元 | 复用正式 `patterns/paged-attention-graph/Design/Toolkit Studio/js/studio-graph-legend-skin.js`，以及图例库 `op-icons.js`。在 `compute-graph-scene.js` 的绘制完成位置接入；恢复原有展开目标属性，保持路由尺寸。未知 OP 使用通用函数符号，不回退成 Linear。 |
| 外围控件 | 复制 Spatial Systems UI 的设计变量及完整 Liquid Glass JS / CSS / MIT 许可；配置使用 control / panel 预设。依赖均为项目内相对路径。 |
| 宿主适配 | `styles/ai-infra-ui.css` 映射中性灰变量、胶囊 Tab、填充选中态与浮层。`js/ai-infra-ui.js` 提供图元接入、单一提示、动态可访问名称与按需帮助。 |

复制来源、上游 SHA-256、交付文件 SHA-256 和局部适配说明见 `../assets/ai-infra/sources.json`。图元样式取自上述 Graph Pattern 的 `styles/graph-style-preview.css`；没有改动正式资产库或安装的 Skill。

检查器以覆盖方式显示，避免重新分配主画布宽度。源码浏览仍使用原来的可调整分栏。Graph 的 Fit 补入现有分组边界和标题安全区；屏幕空间标签为图标留出位置。其余相机手势与状态存储保留。帮助窗口不写入模型或选择状态。

## 验证结果

桌面浏览器视口：1235 × 1042，当前 in-app browser。未接入设备测量；本次验证限于 UI 与既有逻辑。

| 路径 | 结果 |
|---|---|
| 默认源码 → Compute Explorer → 三种主视图 | 通过。Paged Attention 的 Assignment、Pipeline、Graph 均实际打开；Phase0 / Attention 切换可用。 |
| Task 0 → Block 内 → Page 8 | 通过。尾页保持 1 / 128 tokens、Stack 2 的身份和有效范围。 |
| 尾页 → Pipeline → Stack 2 QK_READY → Graph | 通过。保留 Task、Stack、SET / WAIT 证据和跨视图入口。 |
| Graph 节点选择 / 检查器开关 | 通过。选择 QK Matmul 前后的 viewport 宽高和 scene transform 完全一致；关闭后可继续操作。 |
| Graph 展开 / 收起、Shape 图层、缩放与 Fit | 通过。模块 aria-expanded、图层 aria-pressed 与实际内容更新；可读到 Shape 元数据。Fit 后分组标题在视口内。 |
| 空白画布拖拽 | 通过。scene transform 随平移更新。 |
| Decode CSA 跨文件符号导航 | 通过。从 `decode_csa.py` 的 `sparse_attn_csa` 打开 `decode_sparse_attn_csa.py`；从子文件进入的仍是 CSA 场景。 |
| CSA Rank / 场景 / Position 切换 | 通过。切换 Rank 1、Uniform 场景及 Position 7；Work Item 2 保持 invalid 路径，映射到 Lane 8 / iteration 21。 |
| CSA Work Item → Pipeline → Graph | 通过。Pipeline 显示 Validity Check / Neutral Write 分支；Graph 可展开 qk_pv，并查看 Shape 与源码证据。 |
| 帮助打开 / 关闭 / 复制 | 通过。实际显示复制成功反馈；关闭前后 scene style 与 Work Item 选择一致。失败反馈分支已做代码检查，未强制制造剪贴板故障。 |
| 浏览器控制台 | 已检查的路径没有 error / warn。 |

复用了 Toolkit Studio 中已有的 6 份 Node 测试，临时副本将读取路径改为本 Demo 的实际 JS；`paged-attention-intro` 在本 Demo 中对应 `paged-attention-model`。共 **28 / 28** 通过，覆盖 Assignment、Phase0、Pipeline 模型、READY 跨视图、Graph 状态与分页输入；临时结果位于 `/private/tmp/pypto-ai-infra-checks/results.txt`。没有把兄弟目录的测试通过当成本 Demo 的证据。

另外检查了本入口的本地资源闭包、全部 JS 语法及复制资产指纹。原有模型与源码快照未改动。

未验证：窄屏实际浏览、Safari / Firefox 材质回退、触控设备以及全部场景参数组合。本次没有声称全量交互、跨浏览器或设备性能验收完成。


## 2026-10-29 范围修正

独立入口为 `explorer.html`，直接进入 Assignment / Pipeline / Graph，复用原模型与控制器；保留 `index.html` 的 mock IDE 及源码入口，移除本适配器对 IDE 外壳的样式覆盖。独立页仍复用隐藏的宿主 DOM 以支持原控制器和源码跳转，不复制领域实现。

高亮采用 Spatial Systems UI 的 `--ss-selection: #f0f0f0` 对应近白填充和深色文字，去掉本地适配器引入的绿色。Assignment 阶段和范围控件回到顶部，取消画布 200px 左侧预留，左右页边距一致。Graph / Pipeline 的范围控件也进入顶部文档流。

本次验证：28/28 现有模型与控制器测试通过；浏览器验证独立入口、三个 Tab 切换、Pipeline Task 0 选择及跨 Graph 保留。此前全路径检查记录保留，不代表此次重新穷举所有路径。


## 2026-10-29 控件与主体第二轮自检

- 播放/暂停/重播图标直接抽取自正式 `patterns/npu-memory-atlas/index.html`，18px 图标、36px 点击区域，沿用原播放器；状态改变同步图标、ARIA 与 tooltip。未引入额外进度、速度或调度功能。
- 实际原控件按阶段/范围、播放/分配、Task/Stack、图层、视角整合进各视图底栏，底栏复用 Liquid Glass control。移动原节点保留现有监听器；Graph 对父级范围切换放行，修复移动后被 stopPropagation 拦截的问题。图层菜单向上展开，Graph fit 视口保留 104px 安全区。
- 正式 `visual-system.js` 实际参与 Assignment Task 卡片、内部 Task 标识与 Pipeline OP 胶囊渲染。Task 紧凑视图为数值容量编码留出独立区域，不把装饰网格当实际容量；原 Task / Stack / Page 身份与点击目标不变。来源与指纹见 assets/ai-infra/sources.json。
- Slot 不再使用 success/warning 色；统一中性色阶、线型与 Slot 编号，选择/关联用近白。Pipeline Canvas 的 Phase0/Output 色也在渲染器内改为中性灰。Graph 默认数据图元改为中性灰，清除 previous/operation/next 的状态外圈；图层仍控制原有数据、Shape、精度、存储、执行次数与同步内容。
- 浏览器验证：Assignment 暂停、手动分配、Task49 双击进入；Phase0 暂停/重播切换；Pipeline Task49/Stack1 选择与详情；Graph inter→intra→inter、Task/Stack 聚焦、Structure/Execution Weight、Shape 图层、Fit；Task Graph 底栏实测 scrollWidth=width=1246px，无横向溢出；没有控制台 error/warn。
- 28 项现有模型/交互状态测试通过。此次依然未验证 Safari/Firefox、触屏和全部窄屏组合；未提交、推送或修改全局 Skill。


## 2026-10-29 Liquid、功能与画布修正

此前底栏确有 SVG 折射滤镜，但宿主的 91% 不透明灰底覆盖了材质。本轮移除覆盖，实测底栏 rgba(26,26,26,.28)，backdrop-filter 为 url(#liquid-glass-*) saturate(1)，嵌套玻璃实例为 0。

移除三页的上下范围箭头组；保留原双击、对象入口及上下文返回。Pipeline 内部只显示一个返回按钮，Task 导航、播放和图层仍在底栏。未删除前后 Task 按钮，因为它们在分页检查中有独立意义。

主体相机：Pipeline 总览改为整张逻辑场景的二维平移与统一缩放；所有 Canvas 绘制和 DOM 点击区共用投影。Pipeline Task 内部与 Assignment Task 数据图从滚动边界改为独立 x/y 相机，支持任意方向平移、滚轮缩放、适应复位与每 Task 相机保留。Assignment 总览降低最小缩放，Graph 继续使用原无限画布相机；去掉内部卡片实心背板。缩放有数值保护范围，平移不锁定在内容边界。

Graph 主体恢复正式角色色：activation #B1EF6B、aggregate #F2C45F、metadata 使用 optimizer 中性 #B0B0B0。Q/K/V 与 KV Cache 作为激活数据，并不冒充模型权重或物理存储；Scores、Max、l_sum、Context 使用归约/中间结果角色；Probability 为激活。OP 使用正式灰色胶囊；近白仍用于 UI 选择和关联，不取代数据角色条。顶部提供语义图例。紧凑缩放下标签单行省略，避免固定屏幕字号互相覆盖。

实际浏览器验证（1280×720）：Pipeline 总览 Task0 坐标从 (304,167) 拖到 (374,207)；Task49 内部相机从 (315.828,12) 拖到 (415.828,72)，Assignment Task49 从 (385.429,12) 拖到 (495.429,62)。Pipeline fit 53%，滚轮缩放到224%；Task fit 后滚轮到213%。均能适应复位。验证 Task49 双击进入、Stack1 选择、跨 Graph 保留、明确返回全图；上下范围按钮已不在 DOM。浏览器没有 error/warn。现有28项模型/状态测试和7个修改脚本语法检查通过。Safari/Firefox、触屏和完整窄屏组合仍未验证。


## 2026-10-30 顶部与控件分离

去掉 footer 外层贯穿页面的 Liquid 实例与背景，footer 只作透明定位层。原操作控件集中到独立 ai-operation-dock；缩放组件作为 footer 的独立兄弟节点靠右。两组分别挂载系统 control 材质，内部的 segmented-control 不叠加黑色底或重复玻璃。实测外层 background 透明、filter none；操作/缩放分别使用独立 url(#liquid-glass-*) 滤镜。

顶部改为36px单行透明浮层：标题左侧、三Tab居中、源码名与说明右侧，清除继承的18px背景模糊和黑色背板。场景标题与公式压成一行，画布自页面y=0延伸。初始Fit预留顶部浮层安全区；不通过增加固定页头高度挤压画布。

浏览器检查 Assignment、Graph 切换与图层菜单/Shape/Fit；图元语义色保留。检查顶部 background 透明且 filter none，底部缩放与操作实际分为两个兄弟节点。脚本语法与 git diff --check 通过。


## 2026-10-30 对象特征与放大样本

新增本地共享渲染器 js/task-object-glyphs.js，为已确认的 Q Head / Stack / Page 对象生成图元；不是给通用方格附加任意装饰。Q Head 方格中为 Q，释义样本显示 Q0 等具体身份；Stack 方格分成 Page 区，填充来自各 Page 的实际有效 tokens，未用位置留空。源码模型 constants.BLOCK_SIZE=128、STACK_PAGES=4，故512-token分块正式命名为Stack。

Assignment 矩阵和放大样本调用同一 qHead / stack 渲染器。默认总览显示 Task ID、Q 数量和有内部Page分区的 Stack 方块；放大到225%后显示Q、S标识。释义面板初始展示真实 T0（1025 tokens：S0=512、S1=512、S2=1），点选其他Task同步；可收起并由底栏“图元”重新打开，不挤压或重置画布。

保留任务编号、分配模型、Q Head 映射和内部调查入口，切换仅改变 Task 位置与 Block 容器显隐。Q Head 原始几何标识保留供语义转场使用；转场填充也使用绿色数据token。此前的连续容量条已被对象化Stack图元替代。

验证：13项现有布局/分页检查通过；独立断言T0有5个Q、3个Stack、8个满Page、1个1-token尾Page、3个未用Page位。浏览器验证T51同步为Q15–Q19与6个512-token Stacks、225%后显示对象标识、关闭面板、双击T0进入内部并显示Page8为1/128 tokens、返回分组。无控制台error/warn。


## 正式图元样式复核 · 2026-10-30

- 依据正式库 inventory.json 的 activation / task / system-op / system-value 定义：激活为黄绿色，Task 调度条为亮绿色，蓝色属于具名存储空间；未将 KV 数据误标为物理存储。
- Graph 数据卡调用 VisualSystem.data 折叠表示，移除自行生成的背景矩阵；保留稳定节点 ID、端口、类别色及点击目标。
- Pipeline 节点为 Stack 计算阶段，不表示硬件单元。使用 OP 胶囊和正式 op-icons 的 MatMul、Softmax、归约图标；Slot 采用分类颜色，与节点类别分开。
- Task 标题及 Stack 列表左对齐；顶部 Tab 组采用 Liquid control；执行实例堆叠偏移缩为原来的40%，边线弱化，实例数量和选择目标保留。
- 核验 Task51 详情、Pipeline 和 Graph 路径；28项既有测试通过、无控制台 error。仍沿用原版长图 Fit 布局，不改变讲解顺序。

## 本 Demo 配色约定更新

按用户确认的区分方式，Q/KV 数据使用蓝色 #80B7F7，Task 调度图标保留绿色 #9EF04B；这是本 Demo 的宿主色板覆盖，不修改正式图元库的 activation 默认色。覆盖 Assignment、Anatomy、Pipeline、Graph、转场与图例。Block 归属选择器保留中性底色，面板圆角10px、按钮4px。

## 流水线性能修复

- 原热点：悬停重置 Canvas 尺寸并改写152个任务/执行块热区；中文化观察器对任何文本更新扫描全部视图；底栏 hidden 重复赋值可再次触发整理观察器。
- 处理：悬停通过 requestAnimationFrame 合并；仅绘制层更新，不重新测量/摆放热区；Canvas 仅在像素尺寸变化时重建；绘制复用一次 computedStyle 与热区引用；相同缩放文本、hidden 状态不重复写入。
- 中文化仅按帧处理变更子树，跳过未变化及隐藏视图的全量扫描。保留已有提示、选择、颜色、布局和缩放契约。
- 浏览器核验连续选择、提示和缩放，无控制台 error。5项流水线/上下文相关测试通过；未进行硬件级帧率基准，不宣称固定FPS提升。
