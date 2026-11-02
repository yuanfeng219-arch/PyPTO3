# CSA 独立视觉入口 · 实施记录

入口：`../csa.html`。原版目录不修改；复用现有 CSA 模型与控制器，不另建领域模型。

## 本轮

- 独立入口直接打开 decode_csa.py 场景；样式以 body.csa-visual 隔离。
- Assignment：8×3 Lane 卡片，黑底无边框；每个 Work Item 为正方形，蓝色 valid、灰色 invalid，选中用填充。
- 保留 Generation / Ordering / Assignment、播放重播、场景、Rank、Request、缩放和平移。
- 底部操作与缩放分组；隐藏重复场景说明，解释转入 CSA Info。
- Pipeline：复用 Lane×Iteration 全量工作图、valid/invalid 路径和内部依赖；统一控件及检查器呈现。
- Graph：复用共享 OP/Tensor 皮肤、图层、可见布局与源码检查器；移除旧 caption。
- 设置：CSA 专属讲解，默认关闭；仅覆盖，不改变画布尺寸与相机。
- launch 新增 CSA 优化版入口，保留用户已有变更。

## 验证

- 1280×720 浏览器：三主视图、三阶段切换、Graph Sparse Attention 展开、详情、Info、讲解开关。
- Generation / Ordering 均保留 1280 个 Work Item DOM 对象。
- Mixed Position 7 的 Work Item 2：invalid，qk_order[992]，Lane 8 / iteration 41。
- 8 组模型输入比较：Rank 0/1 × Mixed/Uniform × Position 7/8192，派生结果相同。
- 讲解开关前后画布 1232×469，SVG viewBox 相同。
- JS 语法检查与 git diff --check 通过；浏览器未观察到 console error。

## 边界

未新增原版未实现的计算阶段。没有设备 Trace，不给出耗时或性能提升结论。
窄屏、所有展开组合、源码跨文件返回及手动 override 全量回归仍未覆盖；自动播放策略沿用 CSA 原有播放器。
本轮没有提交或发布。

## 自检后修复（2026-10-31）

- 顶部阶段改为收起菜单，压缩辅助区域；统一完整产品名、设置及文件名位置。
- Generation / Ordering / Assignment 均为正方形；恢复 valid / invalid 图例，Lane 读数标注 V / I。
- 共享模型派生按场景缓存，hover 不重复构建全部队列；hover 不再中断自动播放。
- 同一批 1280 Work Items 保留 DOM / ID，阶段切换采用 640ms 位置与尺寸过渡、有限错峰和 520ms 相机插值。默认运行一次，保留暂停、重播与循环，减少动态效果时关闭自动运行及过渡。
- Pipeline 总览进入 Work Item 及返回增加身份标记的空间过渡；三主视图采用短过渡。内部 OP 使用显式类别图标，主文字与下方元信息分开。
- Graph 移除 CSA 的强制最小缩放和局部偏移；展开后 Fit 完整可见结构。图层与缩放直接挂到 footer，避免 Liquid Glass 滤镜创建的定位包含块。
- 数据开关检查：隐藏后仅保留 6 个节点；恢复、Fit 正常。深层展开 19 个节点均在 viewport 内。
- 1280×720、760×720 检查控件位置；默认、三阶段、valid/invalid Pipeline、Graph 展开、图层及讲解开关已检查。
- 讲解开关前后画布尺寸与 viewBox 相同。浏览器没有观察到 console error。
- 再次与原版对照 8 组场景：模型派生结果完全相同，队列无丢失或重复，Lane / iteration 不变量通过。
- 7 个相关 JS 文件语法检查及 git diff --check 通过。本轮未提交或发布。

## 2026-11-02 · 控件与流水线统一

- PA / CSA 动态 select 统一自定义 listbox，保持原 change 事件；文字、SVG chevron、外部点击、Escape 与方向键使用同一组件。
- 选中与 focus 用基础 fill 混白 10%，不添加蓝白 border、outline 或阴影；浏览器实测 border 0px、outline none。
- qk_pv 冗余按钮隐藏；CSA 阶段栏去掉内层容器，场景缩写为 Mixed / Uniform；讲解删除对象副标题并缩短正文。
- PA / CSA 内部流水线均采用 M/H/V 正交依赖路径，固定 userSpaceOnUse 箭头 5 单位。
- PA 关联 OP 及线条共享 Slot 语义色；CSA 对应关系延续批次色。PA 实测 72 条依赖、CSA 实测 13 条数据依赖均无 C/Q 曲线。
- 浏览器验证 Request 与 Rank 参数变更；当前 CSA 三个 native select 均隐藏，动态 Pipeline dropdown 同样完成接入。
- Spatial Systems UI review-rubric 增加控件及流水线自检，skill validator 通过。

## 2026-11-02 · 全画布与流水线相机

- CSA 与 PA 共用 operationGlyph renderer、类别图标、34 单位胶囊、节点下方元信息与语义填色；保留各自源码步骤和分支。
- CSA 顶部导航和对象说明不再参与画布排版；stage-inner 实测 1280×720、原点 (0,0)，对象信息背景为透明；工作项导航停靠左下。
- PA 内部流水线改为 viewBox 相机，滚轮直接缩放、拖拽平移，无需修饰键；保留 Fit、Task 切换和每 Task 相机。
- PA 实操：滚轮使 viewBox 从 (-116,-58.5,1280,720) 变为 (60.959,39.270,996.865,560.737)，拖拽后原点变为 (-16.921,0.330)。
- CSA 实操：滚轮使 viewBox 宽度从 2233.766 变为 1739.659，拖拽后原点从 (266.934,-199.342) 变为 (171.796,-240.116)。
- 默认 Fit 仅调整相机安全区，不缩减画布几何尺寸；顶部与底部 UI 独立覆盖。

### 2026-11-02 · Assignment 控件消融

- 删除“对象详情”：Lane / Work Item 选择直接打开详情，切换对象更新内容。浏览器验证 Lane 05 与 Work Item 0。
- 删除“返回全览”：其处理与“适应”相同，无独立能力；保留唯一“适应”入口。验证适应保留当前详情。
- 保留详情关闭：关闭同时清除选择；Escape 与点击空白均复用清除选择路径。已验证关闭和 Escape。
- 保留播放、重播、阶段、缩放与适应：分别承担动画、结构阶段与相机操作，不与详情入口重复。
- DOM 验证冗余按钮数量为 0；JS 语法检查通过。
