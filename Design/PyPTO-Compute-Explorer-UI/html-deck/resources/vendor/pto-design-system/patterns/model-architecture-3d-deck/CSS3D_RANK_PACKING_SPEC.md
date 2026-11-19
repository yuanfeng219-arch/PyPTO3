# Model Parallel Rank CSS3D 改造 Spec

## 0. 文档信息

| 字段 | 内容 |
| --- | --- |
| 文档状态 | Draft for design / architecture review |
| 文档版本 | 1.1 |
| 更新日期 | 2026-07-28 |
| 目标 Pattern ID | `model-parallel-rank-deck` |
| 现有模型 Pattern | `model-architecture-3d-deck` |
| 现有拓扑 Pattern | `rubik-cube-logical` |
| 首期目标规模 | 128 ranks |
| 首期模型 | `openPangu-2.0-Flash`，L0-L45 Decoder + Input/Output/MTP |
| 主渲染技术 | TypeScript + DOM/CSS3D + SVG |
| 明确排除 | Three.js/WebGL 作为目标组合视图的主 renderer |

本 Spec 定义如何将完整模型架构按照 TP、PP、EP、EDP 等并行关系切分，并将该 Rank 实际持有的原始 Layer DOM 放入对应的 Rank CSS3D 容器。

本文中的“塞入 Rank”不是缩略图、图标、纹理快照或视觉暗示，而是模型 Layer 本体在统一 CSS3D 场景中的真实空间归属。

---

## 1. 决策摘要

### 1.1 最终产品决策

目标页面统一使用 CSS3D：

```text
TypeScript Domain Model
        │
        ├── ModelSpec / LayerSceneSpec
        ├── ParallelTopology / RankManifest
        └── SceneState
                 │
                 ▼
        DOM/CSS3D Unified Scene
        ├── Rank transparent volumes
        ├── Original model Layer DOM
        ├── Original operator DOM
        ├── Original intra-layer SVG edges
        └── Projected inter-rank SVG overlay
```

不再采用以下方案：

- 不使用 Rank Payload Glyph；
- 不把完整模型替换为微型条带、点阵或图标；
- 不把模型 Layer 截图后作为纹理贴入 Rank；
- 不分别维护“CSS 详情模型”和“Three.js Rank 内近似模型”；
- 不为组合页面复制、手写或重新解释一份模型内部结构。

### 1.2 选择 CSS3D 的产品原因

CSS3D 是本改造的产品质量决策，而不只是实现偏好：

- 算子节点的圆角、渐变、文字、状态和 hover 能保持现有精度；
- SVG 边路由、虚线、残差路径和通信路径能够保持清晰；
- 原有 Layer DOM 可以复用，不需要翻译成近似 Mesh；
- 主题 token、字体、选中态和无障碍行为可以继续由设计系统控制；
- 用户进入 Rank 后看到的是同一个模型对象，不是另一套详情视图；
- 模型结构、并行分片和 Rank 空间归属可以在一个坐标系内观察。

### 1.3 核心边界

首期完整装载模式以 128 ranks 为产品上限和验收目标。

4K rank 配置可以进入拓扑摘要或按组筛选模式，但不属于“所有 Rank 同时装载完整 Layer DOM”的首期验收范围。页面不得暗示 128-rank CSS3D 性能结论可直接外推至 4K ranks。

---

## 2. 当前状态与问题

### 2.1 当前模型 3D Deck

`model-architecture-3d-deck` 当前具备：

- 46 个 Decoder Layer；
- Dense L0-L1 / MoE L2-L45；
- DSA/SWA、mHC、Block Post Layer；
- 完整 Attention、Dense/MoE、Residual 和 Communication 节点；
- Input Stem、Final Norm、LM Head、Logits、MTP Tail；
- iso / front / right 视图；
- CSS3D Layer Stack；
- DOMMatrix 投影的 PP wireframe、Annotation 和 Side Label；
- `onNodeSelect`、`setFrontLayer`、`setLayerExpansion` 等交互入口。

但其 PP、TP、EP、DP 目前主要是视觉标注。它不会生成“某 Rank 实际持有什么”的数据结果。

### 2.2 当前逻辑魔方

`rubik-cube-logical` 当前默认实际渲染：

```text
TP2 × PP4 × REP16 = 128 ranks
```

现有 renderer 使用 Three.js `InstancedMesh` 批量绘制 Rank。它可以表达 Rank 坐标、通信组、物理 placement 和不同拓扑投影，但不能让单个 instance 持有独立 DOM Layer 树。

此外，现有 `REP`、`DP`、`EP` folding 语义存在已记录的准确性问题，不能原样成为新组合视图的数据合同。

### 2.3 当前缺口

现有两个 pattern 之间缺少以下中间层：

```text
ModelSpec
  → Partition Rules
  → RankManifest[]
  → CSS3D Rank Scene
```

因此当前页面只能回答“Rank 在哪里、属于哪个组”，不能回答：

- Rank 37 持有哪些 Layer？
- 它持有 TP 的哪一个 tensor shard？
- 它持有哪些 Routed Experts？
- 哪些节点是复制的，哪些节点是切分的？
- PP Send/Recv 的源 Layer、目标 Layer 和目标 Rank 分别是什么？

---

## 3. 产品目标

### 3.1 目标

本次改造必须实现：

1. 将完整模型架构转换为 renderer 无关的 `ModelSpec`；
2. 根据并行拓扑生成确定性的 `RankManifest[]`；
3. 将每个 Rank 表现为透明 CSS3D 计算容器；
4. 将该 Rank 持有的原始 Layer DOM 放入容器内部；
5. 保留原模型的节点、Cluster、Edge、Expert Pool、文字和视觉状态；
6. 支持 Packed、Exploded、Enter Rank 三种空间阅读状态；
7. 支持模型对象与 Rank 的双向定位；
8. 支持 PP、TP、EP、EDP 通信组和跨 Rank 数据流高亮；
9. 在 128 ranks、标准演示配置下满足交互性能验收；
10. 明确区分官方事实、推导值和演示假设。

### 3.2 非目标

首期不包含：

- 4K ranks 全量展开完整 Layer DOM；
- 从 profiling trace 自动推断并行策略；
- 自动复现框架内部全部 tensor placement；
- 模拟真实训练数值；
- 用页面推导替代真实 rank group 配置；
- 改写模型本身的官方或源代码语义；
- 将 CSS3D renderer 泛化为任意模型编辑器；
- 在首期实现 VPP 动态流水调度动画；
- 在首期实现 CP token tile 的真实 runtime 流动。

---

## 4. 关键术语与不可混用概念

| 术语 | 本 Spec 定义 |
| --- | --- |
| Rank | 一个分布式进程或设备计算槽位的逻辑对象 |
| Rank Volume | 表示 Rank 的透明 CSS3D 空间容器 |
| Layer 本体 | 从共享 `LayerSceneSpec` 渲染出的完整 Layer DOM |
| Layer Instance | 某个 Rank 内某个 Layer 本体的实例 |
| Tensor Shard | TP 对参数或张量的分片描述 |
| Expert Shard | EP 对 Routed Expert 集合的分片描述 |
| Replica | DP/EDP 产生的相同模型分片副本 |
| MoGE Route Group | 模型路由概念，不等同于 EP 通信组 |
| EP Group | 参与专家参数或 token 分发的 Rank 集合 |
| RankManifest | 某 Rank 应持有的全部模型对象和通信身份 |
| REP | 仅用于布局的复合展示轴；目标默认可定义为 `EP × EDP`，不得无说明命名为 DP |

### 4.1 数据来源等级

所有 topology、model 和 placement 字段必须标记来源等级：

- `official`：官方资料或真实作业配置直接提供；
- `derived`：根据明确公式推导；
- `demo`：为了展示和布局设定。

UI 中至少在配置摘要和 Inspector 中展示来源类型。演示预设不得使用“官方真实训练拓扑”等措辞。

---

## 5. 首期演示基线

### 5.1 模型基线

首期使用当前 `openPangu-2.0-Flash` 3D Deck：

```text
Decoder Layers: L0-L45
Dense Layers: L0-L1
MoE Layers: L2-L45
Routed Experts: 256
Top-K: 8
PP Stage Ranges:
  PP0: L0-L11
  PP1: L12-L22
  PP2: L23-L34
  PP3: L35-L45
```

该模型不得与 Pangu Pro MoE 的 48 Layer、64 Routed Experts、TP8、EP2、PP5、VPP5 配置混合展示。

### 5.2 128-rank 演示拓扑

为保留当前 128 ranks 的第一版体量，同时纠正 EP/REP 语义，目标演示拓扑定义为：

```text
world_size = TP × PP × CP × EP × EDP
           = 2 × 4 × 1 × 8 × 2
           = 128

REP display axis = EP × EDP = 16
```

这是组合视图的演示拓扑，不宣称为 openPangu 官方训练配置。

目标默认 rank order：

```text
tp → pp → ep → edp
```

默认演示编码：

```text
rank = (((edp × EP + ep) × PP + pp) × TP + tp)
```

该公式只属于演示预设。接入真实作业时必须优先读取显式 `rankCoordinates` 和 `rankGroups`。

---

## 6. 用户体验与场景状态

### 6.1 页面结构

页面使用一个统一可旋转、平移、缩放的 CSS3D Scene：

```text
┌──────────────────────────────────────────────────────────┐
│ Model / Topology / Parallel / Source / Fit / Theme       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│               Unified CSS3D Rank Scene                   │
│                                                          │
│     Rank volumes + original Layers + communication       │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ Selection breadcrumb / Rank summary / interaction hints  │
└──────────────────────────────────────────────────────────┘
```

不得在主页面固定增加第二套独立模型 Inspector。Rank 内的 Layer 本身就是详情对象。

必要的字段说明可以使用浮层 Inspector，但 Inspector 不得重新绘制模型。

### 6.2 Packed 状态

目的：观察整个并行拓扑和模型分片的空间分布。

- 128 个 Rank Volume 按当前拓扑模式排列；
- Rank Volume 保持透明，可看到内部真实 Layer Stack；
- 所有 Layer DOM 仍存在，不替换为 glyph；
- 远距离文字可能因物理缩放而自然不可读，但不得换成摘要图标；
- Rank 外壳是主要点击目标，内部 operator pointer event 暂时关闭；
- 当前 PP/TP/EP/EDP 着色只作用于 Rank 外壳或边界，不覆盖模型语义色。

### 6.3 Exploded 状态

目的：比较多个 Rank 持有的模型分片。

- Rank 间距动画拉开；
- Rank Shell 的背景透明度下降，边框保持可见；
- Layer Stack 保持原始顺序；
- PP 相邻 Rank、同 TP Group、同 EP Group 可按选择显示连接；
- 支持只爆炸一个通信组，其他 Rank 保持原位和低强调；
- 不允许为了布局把 Layer 内节点重新排版。

### 6.4 Enter Rank 状态

目的：直接进入一个 Rank，阅读它持有的完整模型分片。

- 摄像机目标变为该 Rank Volume 的内容中心；
- Rank Shell 可降至极低透明度，但边界仍可识别；
- Layer 原始文字、节点 hover 和点击全部启用；
- 支持在 Rank 内沿 Layer Depth 浏览；
- 点击 Layer 可以沿用现有 Layer expansion / flip 交互；
- 顶部 breadcrumb 显示：`All Ranks / Rank 37 / PP2 / L23`；
- Escape 或 breadcrumb 返回 Exploded/Packed 状态；
- 退出后恢复进入前的相机姿态，而不是重新 Fit 整个场景。

### 6.5 Group Focus 状态

该状态是 Exploded 的子状态：

- TP Group：并排比较同一 operator 的不同 TP shard；
- PP Chain：按流水顺序排列相邻 Stage；
- EP Group：并排比较每个 Rank 的 Expert Shard；
- EDP Group：比较相同 Expert Shard 的副本；
- Group Focus 只改变 Rank 空间位置和强调，不改变其内部 Layer DOM。

---

## 7. 模型数据合同

### 7.1 ModelSpec

```ts
type SourceLevel = "official" | "derived" | "demo";

interface ModelSpec {
  id: string;
  label: string;
  sourceLevel: SourceLevel;
  decoderLayerCount: number;
  layerOrder: number[];
  layers: LayerSceneSpec[];
  input: StaticSceneSpec;
  output: StaticSceneSpec;
  routedExperts?: ExpertSpec[];
  sharedExperts?: ExpertSpec[];
  routing?: RoutingSpec;
}
```

### 7.2 LayerSceneSpec

`LayerSceneSpec` 是模型 Layer 的唯一结构事实源：

```ts
interface LayerSceneSpec {
  layerId: number;
  kind: "dense" | "moe";
  attentionKind: "dsa" | "swa";
  stageHints?: string[];
  blockPost: boolean;
  width: number;
  height: number;
  clusters: ModelClusterSpec[];
  nodes: ModelNodeSpec[];
  edges: ModelEdgeSpec[];
}
```

节点必须至少包含：

```ts
interface ModelNodeSpec {
  id: string;
  label: string;
  op: string;
  role: "operator" | "tensor" | "parameter" | "state" | "communication";
  geometry: { x: number; y: number; width: number; height: number };
  parallelPolicy: ParallelOwnershipPolicy;
  visualClasses?: string[];
}
```

### 7.3 原始视觉约束

从当前 `layerHtml()` 提取到 `LayerSceneSpec` 时必须保持：

- 节点 ID；
- 节点 label；
- 节点 op/role；
- x/y/width/height；
- Cluster 边界和标题；
- Edge source/target/kind/route；
- Dense 与 MoE 分支差异；
- Block Post 节点；
- DSA/SWA label；
- Expert Pool 的折叠/展开结构；
- mHC residual/state 路径。

不得先创建新模型 Schema，再凭视觉记忆重画现有 Layer。

---

## 8. 并行拓扑合同

### 8.1 ParallelTopology

```ts
interface ParallelTopology {
  worldSize: number;
  dimensions: {
    tp: number;
    pp: number;
    cp: number;
    ep: number;
    edp: number;
    vpp?: number;
  };
  rankOrder?: Array<"tp" | "pp" | "cp" | "ep" | "edp">;
  rankCoordinates?: RankCoordinate[];
  rankGroups?: RankGroup[];
  sourceLevel: SourceLevel;
}
```

### 8.2 RankCoordinate

```ts
interface RankCoordinate {
  rank: number;
  tp: number;
  pp: number;
  cp: number;
  ep: number;
  edp: number;
}
```

### 8.3 RankManifest

```ts
interface RankManifest {
  rank: number;
  coordinate: RankCoordinate;
  modelId: string;
  layerSegments: LayerSegment[];
  staticRoles: Array<"input" | "output" | "mtp">;
  nodeOwnership: Record<string, NodeOwnership>;
  expertOwnership: ExpertOwnership;
  communicationGroups: Record<string, string[]>;
  payloadSignature: string;
  sourceLevel: SourceLevel;
}
```

`payloadSignature` 用于判断两个 Rank 是否持有相同结构，不用于把多个 Rank 合并成一个 DOM 对象。

### 8.4 NodeOwnership

```ts
type NodeOwnership =
  | { kind: "owned" }
  | { kind: "replicated"; replicaAxis: "ep" | "edp" | "cp" }
  | { kind: "sharded"; axis: "tp" | "ep" | "cp"; index: number; count: number }
  | { kind: "remote" }
  | { kind: "not-applicable" };
```

Layer DOM 仍完整存在，但节点必须用 ownership state 表明该 Rank 对节点的拥有方式。

---

## 9. Parallel Planner 规则

### 9.1 总原则

`ParallelPlanner` 是纯函数模块：

```ts
partitionModel(
  model: ModelSpec,
  topology: ParallelTopology,
  rules: PartitionRuleSet
): PartitionPlan
```

它不得读取 DOM、CSS、Three.js Scene 或当前相机状态。

### 9.2 PP 规则

- PP 决定 Rank 持有的 Layer Segment；
- 首期使用模型配置中的 `stageRanges`，不使用 `round(layerCount / pp)` 临时计算；
- PP0 Rank 持有 Input Stem；
- 最后一个 PP Stage 持有 Final Norm、LM Head、Logits 和 MTP Tail；
- 每个真实 Layer 必须被一个物理 PP Stage 覆盖；
- 若未来支持 VPP，一个 Rank 可以持有多个不连续 `layerSegments`；
- PP Send/Recv 边必须连接前 Stage 最后一个计算 Layer 与后 Stage 第一个计算 Layer；
- 页面必须区分 Layer ID 和有效 Layer Slot。

### 9.3 TP 规则

- TP 不删除整个 Layer；
- Layer 的全部节点仍在 Rank 内可见；
- 被 TP 切分的节点显示 shard ownership；
- Norm、Residual State、Router 等对象是否复制必须来自 `parallelPolicy`；
- Communication 节点展示该 Layer 实际配置或演示配置中的 collective；
- 不得将 TP 永久写死为“Forward AllReduce”；
- 无真实作业数据时使用“TP collective demo”来源标签。

TP shard 的视觉状态由 pattern 内部类控制，例如：

```text
data-ownership="sharded"
data-shard-axis="tp"
data-shard-index="1"
data-shard-count="2"
```

消费者不得通过本地 CSS 修改节点内部 shard 视觉。

### 9.4 EP 规则

- EP 只切 Routed Expert 参数或相关 expert compute；
- EP 不等同于 MoGE Route Group；
- Expert Pool 仍使用原始 Cluster 和节点样式；
- 进入 Rank 后，Expert Pool 展开时只显示该 Rank 持有的 Expert；
- 首期演示可使用连续专家 ID 分配，但必须标记 `demo`；
- Router、Shared Expert、Combine/Add 的 ownership 由模型策略配置；
- All-to-All Rank Group 优先由显式 `rankGroups` 提供；
- 不得仅依靠 `REP / EP` 宣称整个作业只有若干 A2A Group。

首期 256 Expert / EP8 演示分配：

```text
EP0: E0-E31
EP1: E32-E63
...
EP7: E224-E255
```

### 9.5 EDP/DP 规则

- EDP/DP 产生副本，不产生新的模型切片；
- 相同 PP、TP、EP、CP 坐标，不同 EDP 的 Rank 应得到相同 `payloadSignature`；
- Replica 身份显示在 Rank Shell 或 Inspector，不覆盖 Layer 语义色；
- 梯度同步必须表达为反向阶段可能分桶并与计算重叠；
- 不得固定描述为只在 step 末执行。

### 9.6 CP 与 VPP

首期数据合同保留 CP/VPP，但默认 CP1、无 VPP 动画。

- CP 切分 Activation/Sequence，不等同于参数切分；
- VPP 影响 Layer Chunk 与调度，不增加物理 rank；
- 未实现的视觉能力必须显示 `not visualized`，不得静默丢失字段。

---

## 10. 统一 CSS3D 渲染架构

### 10.1 DOM 层级

目标 DOM 结构：

```html
<section class="pto-model-rank-deck" data-scene-mode="packed">
  <header class="pto-model-rank-deck__toolbar" data-stage-ui></header>

  <div class="pto-model-rank-deck__viewport">
    <div class="pto-model-rank-deck__camera">
      <div class="pto-model-rank-deck__world">
        <section class="pto-rank-volume" data-rank="37">
          <div class="pto-rank-volume__shell" aria-hidden="true">
            <i data-face="front"></i>
            <i data-face="back"></i>
            <i data-face="left"></i>
            <i data-face="right"></i>
            <i data-face="top"></i>
            <i data-face="bottom"></i>
          </div>

          <div class="pto-rank-volume__content">
            <section class="pto-model-deck__layer" data-layer="23"></section>
            <section class="pto-model-deck__layer" data-layer="24"></section>
            <!-- 继续放入本 Rank 的真实 Layer -->
          </div>
        </section>
      </div>
    </div>

    <svg class="pto-model-rank-deck__link-overlay"></svg>
    <div class="pto-model-rank-deck__labels"></div>
  </div>

  <aside class="pto-model-rank-deck__inspector"></aside>
</section>
```

### 10.2 Transform 层级

必须严格区分：

```text
viewport
  camera transform
    world transform
      topology position
        rank enter/explode offset
          rank content scale
            original layer transform
```

相机拖动期间只更新 `.camera` 或 `.world` 的单一 transform。不得逐 Rank、逐 Layer、逐节点更新 transform。

### 10.3 场景坐标

- 原始 Layer 坐标继续使用当前 720 × 1180 设计空间；
- Layer 内节点不得为适应 Rank 容器而重新排版；
- `depthGap` 继续来自模型配置；
- Rank Volume 尺寸由最大 Layer 宽高、Layer 数量和内容 scale 计算；
- 拓扑坐标与 Rank 内模型坐标分开存储；
- Packed/Exploded 只改变 Rank 外层 topology transform；
- Enter Rank 只改变 camera pose 和交互状态。

### 10.4 模型整体缩放

“原样”允许对整个 Layer Stack 使用统一几何缩放，但禁止内部重排：

```css
.pto-rank-volume__content {
  transform: scale3d(var(--rank-content-scale), var(--rank-content-scale), var(--rank-content-scale));
}
```

允许的变化：

- 整体 scale；
- 整体 translate/rotate；
- Rank Shell 的尺寸和透明度；
- ownership state；
- 选择、hover、通信高亮。

禁止的变化：

- 重新排列 Layer 内节点；
- 用简化节点替换真实节点；
- 删除普通节点只保留 Cluster；
- 把 SVG Edge 替换为示意直线；
- 把 Layer rasterize 成图片；
- 用 glyph 代替未选中的 Rank 内容。

---

## 11. Rank Volume 视觉规范

### 11.1 外壳

每个 Rank 使用六个 CSS3D Face 构成透明空间体：

- 填充使用 `--surface-*` 与维度色的低比例混合；
- 默认透明，内部 Layer 必须可见；
- 边框使用 `--border-default`；
- hover 使用 `--border-strong`；
- selected 使用一条清晰但不遮挡内部模型的高亮边；
- 不使用大面积饱和色填充；
- 不使用玻璃拟态模糊覆盖 Layer；
- 不使用阴影制造虚假层级。

### 11.2 Rank Label

Rank Label 固定表达：

```text
Rank 37
PP2 · TP1 · EP4 · EDP0
L23-L34 · E128-E159
```

来源类型显示在 Inspector，不在每个 Rank Label 上重复堆叠。

### 11.3 并行维度色

并行维度色只用于：

- Rank Shell 边界；
- Group Focus 连线；
- Shard 标记；
- 并行 badge；
- 选中 group 的局部强调。

不得覆盖算子节点语义色。用户必须能同时判断“这是 Attention”与“它属于 TP1”。

### 11.4 Ownership 视觉

- `owned`：保持原始节点外观；
- `sharded`：在节点内部增加非破坏性的 shard 分割线与 `1/2` 标记；
- `replicated`：增加小型 Replica 标志，不降低节点不透明度；
- `remote`：只用于临时对照视图，使用中性轮廓；
- `not-applicable`：不创建该对象实例，而不是创建虚假灰块。

ownership 状态必须由共享 pattern 内部实现，消费者不得自行覆盖节点 class。

---

## 12. Layer 本体复用规范

### 12.1 单一 renderer 源

模型 Layer 的 DOM 创建函数必须从当前 `layerHtml()` 重构为共享 renderer：

```ts
renderLayerScene(spec: LayerSceneSpec, context: LayerRenderContext): HTMLElement
```

以下页面必须调用同一个 renderer：

- 原 `model-architecture-3d-deck` preview；
- 新 `model-parallel-rank-deck`；
- 未来训练 sidecar 的 Layer focus；
- 任何 Rank 内 Layer 展示。

不得出现 `rankLayerHtml()` 第二套结构。

本条不能只靠代码评审判断。所有实现还必须通过第 24 章定义的原始 Layer 完整性保护协议与自动 Parity Gate。

### 12.2 上下文输入

```ts
interface LayerRenderContext {
  rankManifest?: RankManifest;
  interactionMode: "overview" | "rank" | "layer";
  selectedNodeId?: string;
  selectedLayerId?: number;
  theme: "dark" | "light";
}
```

Renderer 只能根据 context 添加状态，不得根据 Rank 重新计算模型结构。

### 12.3 Static Input/Output

- Input Stem 只进入 PP0 Rank；
- Final Norm、LM Head、Logits、MTP 只进入最后 PP Stage Rank；
- TP/EP ownership 必须依节点策略计算；
- Pipeline Boundary Link 从 Input 到 PP0、最后 Stage 到 Output；
- Static Scene 继续使用与 Layer 相同的节点、Edge 和 token 体系。

---

## 13. 连线系统

### 13.1 连线分层

连线分为三层：

1. Layer 内部 Edge；
2. Rank 内 Layer-to-Layer Edge；
3. Rank 间 Parallel/Communication Edge。

三层不得用同一种视觉强度同时全开。

### 13.2 Layer 内 Edge

- 完全复用当前 Layer SVG Edge；
- Edge 随 Layer DOM 一起进行 CSS3D transform；
- 保留 activation、communication、residual、parameter 等 kind；
- 不在 camera move 时重算 Layer 内路径。

### 13.3 Rank 内 Layer Spine

- 连接当前 Rank 内相邻 Layer；
- VPP 未启用时按 Layer ID 顺序连接；
- 使用模型语义中的 input/output anchor；
- 不得通过 `getBoundingClientRect()` 每帧测量全部 Layer；
- 几何应从 `LayerSceneSpec` 和 rank transform 推导；
- 只有 Layer expansion 改变局部深度时重新计算。

### 13.4 Rank 间连线

Rank 间连线使用 viewport 根级 SVG Overlay：

- PP：相邻 Stage 的 activation Send/Recv；
- TP：选中 operator 对应的 collective group；
- EP：Dispatch/Combine 对应的 All-to-All group；
- EDP/DP：梯度同步 group；
- Placement：如果存在真实 placement，可显示 HCCS/Scale-Up、网络域、Scale-Out 成本层级。

默认只显示：

- 当前选择对象相关的通信线；
- 当前 Group Focus 的全部成员线；
- 当前 Pipeline Chain 的相邻边。

不得默认同时绘制所有 TP/PP/EP/EDP group 的全部连线。

### 13.5 投影

Rank 间 Link Overlay 使用统一 `SceneProjection`：

```ts
projectWorldPoint(point: WorldPoint, pose: CameraPose): ScreenPoint
```

要求：

- 使用已知 world matrix 和 DOMMatrix；
- camera pose 变化时在单个 `requestAnimationFrame` 内批量更新；
- 不为每条边触发独立 layout read；
- Link hit target 与可见 stroke 分离；
- 深度较远的边可以降低 opacity，但不得伪造遮挡关系；
- Badge、Label、Edge 必须共享同一投影函数。

### 13.6 术语准确性

- 节点内 NPU 互联不得称为 UB；
- 昇腾节点内通信可在有依据时显示 HCCS / Scale-Up；
- 无真实 placement 时显示“演示映射”；
- TP collective 不固定为 AllReduce；
- DP gradient sync 描述允许 bucket overlap；
- HCCL 算法不得默认只给 Ring/Tree 并宣称为真实选择。

---

## 14. 相机与交互

### 14.1 相机状态

```ts
interface CameraPose {
  rx: number;
  ry: number;
  zoom: number;
  panX: number;
  panY: number;
  pivot: { x: number; y: number; z: number };
}
```

CSS3D camera 变换顺序必须固定，避免不同模式出现跳变：

```text
translate viewport center
→ scale zoom
→ rotateX
→ rotateY
→ translate negative pivot
→ translate pan
```

### 14.2 手势

- Packed/Exploded：拖动旋转；Command/Ctrl + 拖动平移；滚轮缩放；
- Enter Rank：默认拖动平移，按住修饰键旋转；
- 点击 Rank Shell 进入 Rank；
- 点击空白取消 group/node selection；
- 双击 Rank 快速 Enter Rank；
- Escape 按层级退出 Layer → Rank → Group → All Ranks；
- Fit 根据当前场景状态计算，不使用全局固定倍率。

### 14.3 Pointer Event 策略

为避免 Packed 状态数万个 operator 抢占 hit test：

- Packed：只有 Rank Shell 接收 pointer；
- Exploded：Rank Shell 与 Group Link 接收 pointer；
- Enter Rank：当前 Rank 的 Layer/Operator 接收 pointer；
- 非当前 Rank 内部设置 `pointer-events: none`，但保持完整渲染；
- Layer expansion 时只有当前 Layer 的 operator 接收 pointer；
- toolbar、Inspector 和 breadcrumb 始终位于 screen-space UI。

### 14.4 双向定位

必须支持：

```text
Layer → 所有持有该 Layer 的 Rank
Operator → 对应 TP/EP communication group
Expert → 持有该 Expert 的 Rank
Rank → 该 Rank 的 Layer、Tensor Shard、Expert Shard
Communication Edge → 源/目标 Rank 与源/目标 Operator
PP Stage → 该 Stage 的全部 Rank
```

选择变化不得自动重建全部 DOM。

---

## 15. Scene State

```ts
interface ModelRankSceneState {
  sceneMode: "packed" | "exploded" | "rank";
  topologyMode: "standard" | "pp" | "tp" | "ep" | "replica";
  camera: CameraPose;
  selectedRank: number | null;
  selectedLayer: number | null;
  selectedNode: { layer: number; nodeId: string } | null;
  selectedExpert: number | null;
  selectedGroup: string | null;
  theme: "dark" | "light";
}
```

URL 至少可以持久化：

```text
model
scene
topology
rank
layer
node
group
theme
```

分享链接恢复后必须得到同一 Rank、Layer 和相机语义；像素级 pose 可以作为可选字段。

---

## 16. 性能规范

### 16.1 性能原则

全 CSS3D 的目标不是减少模型内容，而是减少不必要的运行时工作：

- DOM 一次构建，状态切换以 class/data attribute 为主；
- camera move 只改顶层 transform；
- 不在 pointermove 中读取大量 layout；
- 不在每帧生成或销毁节点；
- 不在每帧重算 Layer 内 SVG Edge；
- Overlay 投影批量执行；
- pointer event 只开放给当前交互层级；
- Inspector 数据来自 `RankManifest`，不扫描 DOM；
- 主题切换使用 CSS variable，不逐节点写 inline color。

### 16.2 DOM 规模预算

首期必须在实现前记录实际节点数量：

```text
Rank Volumes
Layer Instances
Operator Nodes
Cluster Nodes
SVG Paths
Text Labels
Interactive Targets
```

不得仅使用 Layer 数量估算性能。

如果 128-rank 全量 DOM 超过性能门槛，允许的优化顺序是：

1. 合并无语义的装饰 DOM；
2. 删除重复但不可见的 pseudo/overlay 层；
3. 将静态装饰改为 CSS background；
4. 合并相同 SVG marker/definition；
5. 限制非当前 Rank 的 pointer hit testing；
6. 在不改变路径几何的前提下批量管理 Edge state；
7. 分批挂载 Rank，但在 Ready 前不得显示伪完成状态。

禁止的性能“优化”：

- 用图片替代 Layer；
- 用 glyph 替代 Layer；
- 删除未选中 Rank 的模型内容；
- 把节点简化成无 label 的方块并称为原样；
- 将完整模式偷偷降级为只有选中 Rank 才存在模型。

### 16.3 首期性能验收目标

基准环境以项目约定的本地 Chromium、1440 × 900 viewport、128 ranks 演示配置为准：

- 首次 Ready：目标 ≤ 4 秒；最大可接受 ≤ 6 秒；
- Packed 场景持续旋转：目标 ≥ 45 FPS；最低验收 ≥ 30 FPS；
- Exploded 场景持续旋转：最低验收 ≥ 30 FPS；
- Rank 进入动画：目标 520-700ms，期间无明显中间跳帧或布局闪烁；
- 选择 Rank：交互反馈 ≤ 100ms；
- 选择 Layer/Operator：交互反馈 ≤ 100ms；
- camera move 期间不得持续触发全部 Rank 的 layout read；
- 稳定状态不得有持续增长的 DOM、listener 或 animation frame。

如果最低门槛无法满足，必须提交性能数据和视觉等价的优化方案重新评审，不能自行恢复缩略符方案。

---

## 17. 无障碍与可读性

- Rank Volume 使用可聚焦元素或提供等价 focus target；
- Rank aria-label 包含 Rank、并行坐标、Layer Range 和 Expert Range；
- Operator 保留现有 button 语义；
- Enter/Space 可进入 Rank 或选择 Operator；
- Escape 按交互层级返回；
- focus ring 不得被 3D transform 隐藏；
- Inspector 提供屏幕空间的完整文字信息；
- 所有只靠颜色表达的 shard/group 状态必须同时提供文字或几何标记；
- `prefers-reduced-motion` 下取消飞行动画，改为短淡入和直接 pose 切换。

---

## 18. Pattern API

### 18.1 新组合 Pattern

目标全局入口：

```js
window.PtoModelParallelRankDeck.render(root, options)
```

### 18.2 Render Options

```ts
interface RenderOptions {
  model: ModelSpec | string;
  topology: ParallelTopology;
  partitionRules?: PartitionRuleSet;
  initialState?: Partial<ModelRankSceneState>;
  showChrome?: boolean;
  externallyManaged?: boolean;
  onRankSelect?: (manifest: RankManifest | null) => void;
  onLayerSelect?: (layer: number | null, rank?: number) => void;
  onNodeSelect?: (selection: NodeSelection | null) => void;
  onGroupSelect?: (group: RankGroup | null) => void;
  onStateChange?: (state: ModelRankSceneState) => void;
}
```

### 18.3 Controller API

```text
setSceneMode(mode)
setTopologyMode(mode)
setTheme(theme)
setPose(pose)
fit()
selectRank(rank)
enterRank(rank)
leaveRank()
selectLayer(layer, rank?)
selectNode(nodeId, layer, rank?)
selectExpert(expertId)
selectGroup(groupId)
setTopology(topology)
setPartitionRules(rules)
getRankManifest(rank)
getState()
refresh()
destroy()
```

所有 API 必须更新同一 `ModelRankSceneState`，不得让 DOM、URL 和 controller 各自持有冲突状态。

---

## 19. 文件与所有权规划

建议目标结构：

```text
patterns/model-architecture-3d-deck/
├── pattern.html
├── pattern.css
├── pattern.js
├── pattern.json
└── src/
    ├── model-spec.ts
    ├── layer-scene-spec.ts
    └── layer-renderer.ts

patterns/model-parallel-rank-deck/
├── pattern.html
├── pattern.css
├── pattern.js
├── pattern.json
└── src/
    ├── parallel-topology.ts
    ├── parallel-planner.ts
    ├── rank-manifest.ts
    ├── rank-layout.ts
    ├── scene-controller.ts
    ├── scene-projection.ts
    ├── rank-volume-renderer.ts
    └── link-overlay-renderer.ts
```

构建输出仍为无需 bundler 即可从 `pattern.html` 使用的普通 JavaScript。

### 19.1 Source of Truth

| 能力 | 唯一所有者 |
| --- | --- |
| 模型结构 | `model-architecture-3d-deck/src/model-spec.ts` |
| Layer DOM | `model-architecture-3d-deck/src/layer-renderer.ts` |
| 并行切分 | `model-parallel-rank-deck/src/parallel-planner.ts` |
| Rank 坐标与布局 | `model-parallel-rank-deck/src/rank-layout.ts` |
| CSS3D 相机 | `model-parallel-rank-deck/src/scene-controller.ts` |
| Rank 间连线 | `model-parallel-rank-deck/src/link-overlay-renderer.ts` |
| 设计系统展示 | `pattern.html` + `pattern.json` + registry |

现有 Three.js 逻辑魔方可以继续作为独立 topology pattern，但不得成为新 CSS3D 组合 pattern 的 renderer 依赖。可以复用经准确性修正后的纯布局思想和测试用例，不复制 Three.js scene code。

---

## 20. 改造阶段

### Phase 0：基线冻结

- 记录当前 3D Deck 三个视角截图；
- 记录 Dense/MoE Layer、Input、Output、Expert Pool 展开状态；
- 导出当前节点、边、Cluster 数量；
- 记录当前公开 API；
- 记录 1440 × 900 下的 load time、DOM count 和交互帧率；
- 将用户现有未提交修改视为基线的一部分，不覆盖。

交付：视觉基线、数据基线、性能基线。

### Phase 1：模型结构数据化

- 从 `layerHtml()` 提取 `ModelSpec` 与 `LayerSceneSpec`；
- 创建共享 `renderLayerScene()`；
- 原 3D Deck 改为消费共享 renderer；
- 保证原 preview 视觉和交互 parity；
- 不引入 Rank 逻辑。

退出条件：现有 Deck 与改造前视觉一致，模型结构只有一个事实源。

### Phase 2：Parallel Planner

- 实现 `ParallelTopology`、`RankCoordinate`、`RankManifest`；
- 实现 128-rank 演示拓扑；
- 实现 PP、TP、EP、EDP ownership；
- 生成显式 communication groups；
- 加入来源等级；
- 建立纯函数单元测试。

退出条件：无需 DOM 即可回答任意 Rank 持有什么。

### Phase 3：CSS3D Rank Scene

- 创建统一 viewport/camera/world；
- 创建 Rank Volume 和六面 shell；
- 实现 Packed 拓扑布局；
- 实现 drag rotate、pan、zoom、fit；
- 将 Rank Label 与 world projection 对齐；
- 暂不装入完整模型。

退出条件：128 个 CSS3D Rank 容器稳定排列和交互。

### Phase 4：原始 Layer 装载

- 根据 RankManifest 把原始 Layer DOM 放入 Rank；
- Input/Output 按 PP Stage 放置；
- 添加 TP/EP/Replica ownership state；
- 实现 Rank 内 Layer Spine；
- 验证所有 Layer/Expert 分配不丢失。

退出条件：任意 Rank 内可看到并检查完整原始模型分片。

### Phase 5：空间交互

- 实现 Exploded；
- 实现 Enter Rank 和相机恢复；
- 实现 Layer expansion；
- 实现模型与 Rank 双向选择；
- 实现 Group Focus；
- 实现 breadcrumb、URL state 和键盘退出层级。

退出条件：用户可从全局拓扑进入 Rank、进入 Layer，再原路返回。

### Phase 6：通信连线

- 实现统一 `SceneProjection`；
- 实现 PP boundary links；
- 实现 TP/EP/EDP selected-group links；
- 实现 Link tooltip 和选择；
- 接入 placement source level；
- 修正 UB、TP/DP phase、HCCL 算法等不准确文案。

退出条件：每条高亮通信线都可以解释源 Rank、目标 Rank、模型节点和来源。

### Phase 7：性能与 Pattern 收口

- 完成 DOM/paint/layout/profile；
- 按允许顺序优化；
- 完成 128-rank 性能验收；
- 创建 `pattern.json`；
- 更新 `patterns/patterns.json`；
- 更新 design-system preview；
- 更新 `references/DESIGN.md`；
- 执行最终浏览器 smoke test。

退出条件：新组合视图成为可复用 shared pattern，而不是产品页私有实现。

---

## 21. 测试与验证

### 21.1 Planner 单元测试

必须验证：

- `worldSize` 与显式 Rank 数一致；
- 每个 Rank 坐标唯一；
- 每个 Rank 都生成 Manifest；
- PP Stage 完整覆盖 L0-L45；
- 不同 PP Stage 不意外重复物理 Layer；
- PP0 持有 Input；
- 最后 PP Stage 持有 Output/MTP；
- TP shard index 覆盖 `0..tp-1`；
- 256 Experts 在每个 EP group 内完整且不重复分配；
- 相同 EP、不同 EDP 的 expert payload 相同；
- Replica Rank 的 `payloadSignature` 相同；
- Rank Group 成员数符合拓扑合同；
- 错误配置能够返回明确错误，而不是渲染半完成场景。

### 21.2 Renderer parity

对比改造前模型 Deck：

- Dense Layer 结构一致；
- MoE Layer 结构一致；
- 节点位置、尺寸、圆角、渐变一致；
- Cluster 边界一致；
- Edge route 一致；
- Expert Pool 展开一致；
- Dark/Light 一致；
- Node hover/selected 一致；
- Input/Output/MTP 一致。

### 21.3 Rank 装载验证

抽样至少覆盖：

- Rank 0；
- 每个 PP Stage 的首尾 Rank；
- 每个 TP index；
- 每个 EP index；
- 两个 EDP replica；
- Dense Layer Rank；
- MoE Layer Rank；
- Input Rank；
- Output/MTP Rank。

### 21.4 浏览器验收

canonical viewport：1440 × 900。

完成后执行一次最终 smoke test：

- 页面无 console error；
- 128 Rank 全部存在；
- Packed → Exploded → Enter Rank → 返回链路正常；
- 旋转、平移、缩放正常；
- 点击 Rank、Layer、Operator 正常；
- Dark/Light 正常；
- URL state 恢复正常；
- 最终页面无明显 Layer 穿插、Shell 遮挡、Edge 漂移或 label 错位。

---

## 22. 验收标准

### AC-01：不存在缩略替代物

任意 Rank 内展示的模型内容来自共享 `renderLayerScene()`。源码和 DOM 中不存在 `payload glyph`、Layer screenshot 或用于替代真实 Layer 的简化模型。

### AC-02：128 Rank 完整装载

128 个 Rank 都拥有合法 `RankManifest`，并装入其 PP Stage 对应的完整 Layer DOM。

### AC-03：原模型视觉 parity

进入任意 Rank 后，Layer 的节点、Cluster、Edge、文字和交互与独立 3D Deck 的同 Layer 保持视觉及语义一致。

### AC-04：PP 切分正确

PP0-L0-L11、PP1-L12-L22、PP2-L23-L34、PP3-L35-L45 的装载结果与配置完全一致，Input/Output 放置正确。

### AC-05：TP ownership 正确

TP 不删除 Layer。可切分节点显示正确 shard index；复制节点明确显示 replicated policy。

### AC-06：EP ownership 正确

MoE Layer 的 Expert Pool 只包含该 Rank 持有的 Routed Experts，且 EP 与 MoGE Route Group 不混用。

### AC-07：Replica 正确

不同 EDP、相同 PP/TP/EP 的 Rank 得到相同 payload signature，并保留独立 Rank 身份。

### AC-08：双向定位

Layer、Operator、Expert、Rank 和通信组可以双向定位，选择结果与 RankManifest 一致。

### AC-09：连线可解释

每条 Rank 间连线都能显示通信维度、源/目标 Rank、源/目标模型对象和来源类型。

### AC-10：统一 CSS3D

组合视图不创建 Three.js renderer、WebGL canvas 或模型纹理快照。模型、Rank 和交互对象位于同一 DOM/CSS3D Scene；屏幕空间连线使用 SVG Overlay。

### AC-11：交互层级正确

Packed、Exploded、Enter Rank 的 pointer target、相机姿态、返回行为和状态持久化符合本 Spec。

### AC-12：性能达标

128-rank canonical viewport 满足第 16.3 节最低性能门槛，且不存在持续内存增长。

### AC-13：准确性达标

页面区分 official、derived、demo；不使用 UB 指代节点内 NPU 互联；不把 EP 等同 MoGE Route Group；不把演示拓扑宣称为官方训练配置。

### AC-14：Pattern 收口

新组合视图具备完整 `pattern.html`、`pattern.css`、`pattern.js`、`pattern.json`，注册到 pattern registry，并在 design-system preview 中通过真实 renderer 展示。

### AC-15：结构零遗漏

基线与新 renderer 的 Layer、Node、Cluster、Edge、Static Input/Output/MTP 结构比较必须零 missing、零未批准 rename、零未批准 merge。新增 Rank ownership overlay 必须位于允许的扩展槽，不得改变 Core DOM。

### AC-16：几何与样式 Parity

在 1:1 自然尺度 parity harness 中，节点和 Cluster 几何误差不得超过 0.25 CSS px；关键 computed style 属性必须相等；Dark/Light、default/hover/selected、Expert collapsed/expanded 均必须覆盖。

### AC-17：基线不可静默更新

结构快照、视觉截图、style manifest 或允许差异清单发生变化时，测试必须失败。实现者不得通过自动更新 snapshot 让测试恢复绿色；更新基线必须附差异报告并获得用户明确批准。

### AC-18：独立对照有效

Parity 验证必须将改造前冻结 fixture 与新 renderer 对比。禁止让独立 Deck 和 Rank Deck 同时调用同一个已被改坏的新 renderer，再以“两边一致”作为正确证据。

---

## 23. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 128 Rank 产生大量 DOM | 旋转掉帧、内存升高 | 单一父级 transform、批量挂载、禁用非活动 pointer、删除非语义装饰 DOM |
| CSS3D 透明面遮挡文字 | Rank 内模型可读性下降 | Shell 低透明填充、无 blur、进入 Rank 后降低 Shell fill |
| Rank 深度与 Layer 深度冲突 | 视觉穿插 | 拓扑坐标与局部模型坐标分层、自动计算 Rank Volume bounds |
| SVG Overlay 漂移 | 通信线不落在节点 | 所有 overlay 共用 SceneProjection，不逐组件自建投影 |
| PP/EP 语义继续沿用旧公式 | 业务错误 | Planner 使用显式维度、EDP、Rank Groups 和来源标签 |
| 为性能重新引入简化模型 | 偏离产品目标 | AC-01 设为阻断项，任何视觉降级需重新评审 |
| 原 Deck 与 Rank Deck 分叉 | 两套模型逐渐不一致 | 共享 ModelSpec 与 renderLayerScene，禁止第二套 rankLayerHtml |
| 现有未提交修改被覆盖 | 用户工作丢失 | 改造前冻结基线，逐文件检查 dirty worktree，禁止覆盖无关修改 |

---

## 24. 原始 Layer 完整性保护协议

### 24.1 目的

本协议专门防止模型架构改造中最常见且最严重的失败：为了快速接入新场景，擅自简化、遗漏、重命名、合并、移动或重新着色原始 Layer 内容。

“复用原始 Layer”不能只是一句设计原则。它必须落实为：

```text
不可变迁移基线
  + 结构零差异检查
  + 几何零漂移检查
  + 样式状态检查
  + 独立视觉对照
  + 人工批准差异清单
  = 允许进入 Rank 改造阶段
```

该协议是阻断式门禁。任何一项未通过，不得以“整体效果接近”“后续再补”“性能需要”或“组合视图里看不清”为理由继续交付。

### 24.2 已知失败模式

实现与评审必须主动检查以下失败模式：

| ID | 失败模式 | 典型表现 | 是否阻断 |
| --- | --- | --- | --- |
| PF-01 | 代表节点替代完整结构 | 用 8-10 个代表节点重画完整 Sparse MLA / MoE Layer | 是 |
| PF-02 | 分支遗漏 | 只保留主干，遗漏 Q/KV 双分支、Dense/MoE 分支或 MTP | 是 |
| PF-03 | 节点合并 | 把两个真实算子合成一个“QKV”或“FFN”节点 | 是 |
| PF-04 | 节点改名 | 为了简洁修改源 label、ID、op role | 是 |
| PF-05 | Edge 简化 | 将曲线、Elbow、Residual、Parameter Edge 改成统一直线 | 是 |
| PF-06 | 几何漂移 | Rank 版本中修改节点坐标、尺寸、Cluster 范围或 Layer 高度 | 是 |
| PF-07 | 视觉近似 | 颜色键相同，但 gradient、opacity、border、shadow、radius 不同 | 是 |
| PF-08 | 状态遗漏 | 只验证默认态，遗漏 hover、selected、Light、Expert expanded | 是 |
| PF-09 | 特殊层泛化 | 将 Dense、MoE、DSA、SWA、Block Post 全部渲染成一种 Layer | 是 |
| PF-10 | Static Tail 遗漏 | 漏掉 Input、Embedding Weight、Final Norm、LM Head、Logits、MTP | 是 |
| PF-11 | Context 篡改 | `rankManifest` 进入 renderer 后重新构造或删除模型节点 | 是 |
| PF-12 | 假 Parity | 原 Deck 与 Rank Deck 同时调用同一错误 renderer，因此看起来一致 | 是 |
| PF-13 | Snapshot 洗白 | 测试失败后直接刷新基线或截图，不审查差异 | 是 |
| PF-14 | 性能名义降级 | 以 DOM 多为理由删除远端 Rank 的内部节点或 Edge | 是 |
| PF-15 | Consumer 覆盖 | 组合页面用局部 CSS 覆盖 Layer 内部 class | 是 |

### 24.3 Core DOM 定义

`Core DOM` 是必须与迁移前保持一致的 Layer 内部结构，包括：

- Layer 根节点及其语义 attributes；
- Layer Label 与 Layer metadata；
- Graph 容器；
- Cluster 节点、标题与 geometry；
- Operator/Tensor/Parameter/State/Communication 节点；
- Expert Pool 根节点和 Expert children；
- Layer 内 SVG、每条 path 及 marker；
- Node ID、Edge source/target、Edge kind；
- 影响视觉和交互的 class；
- aria-label、role、tabindex；
- Dense/MoE、DSA/SWA、Block Post 等 variant attributes。

以下内容不属于 Core DOM，但只能通过规定扩展槽添加：

- Rank 外层 wrapper；
- Rank ID 和并行坐标；
- `data-ownership`、`data-shard-axis`、`data-shard-index`、`data-shard-count`；
- ownership badge/line overlay；
- Rank Shell；
- Rank 间通信 overlay；
- 组合页面的 screen-space UI。

扩展内容不得插入到会改变原节点 flex/grid child ordering 的位置。优先使用 Layer 根级 sibling overlay、pseudo element 或明确的 `data-layer-extension-slot`。

### 24.4 不可变基线资产

Phase 0 必须在任何 renderer 重构前生成并冻结以下资产：

```text
tests/fixtures/model-architecture-3d-deck/
├── source-reference.html
├── structure-manifest.json
├── computed-style-manifest.dark.json
├── computed-style-manifest.light.json
├── interaction-manifest.json
├── screenshots/
│   ├── dark/
│   └── light/
└── BASELINE.md
```

#### 24.4.1 `source-reference.html`

- 保存改造前可独立运行的 reference fixture；
- 只用于迁移测试，不成为新的产品 renderer；
- 不随新 renderer 逻辑变化；
- 不允许从新 `LayerSceneSpec` 重新生成；
- 必须固定依赖版本、viewport、theme 和 query state。

#### 24.4.2 `structure-manifest.json`

必须记录全部 46 Layer 和 Static Scene，而不是只记录代表 Layer：

```ts
interface LayerStructureBaseline {
  layerId: number;
  variant: {
    kind: "dense" | "moe";
    attentionKind: "dsa" | "swa";
    blockPost: boolean;
    stage: number;
  };
  rootAttributes: Record<string, string>;
  clusters: ClusterBaseline[];
  nodes: NodeBaseline[];
  edges: EdgeBaseline[];
  domSignature: string;
  semanticSignature: string;
}
```

每个 Node 至少记录：

```text
id
label exact text
op
role
class list
x / y / width / height
aria-label
role / tabindex
data attributes
child semantic structure
```

每个 Edge 至少记录：

```text
source
target
kind
source anchor
target anchor
route mode
via points
exact normalized path
class list
marker attributes
```

每个 Cluster 至少记录：

```text
label exact text
x / y / width / height
semantic color variable
class list
```

#### 24.4.3 Computed Style Manifest

以下属性必须在 Dark 和 Light 两种主题下记录：

```text
display
position
left / top / width / height
padding
border width / style / color
border radius
background color / image
opacity
box shadow
filter
color
font family / size / weight / line-height
text align
transform
transform origin
z-index
pointer-events
```

对于 SVG 还必须记录：

```text
fill
fill-opacity
stroke
stroke-width
stroke-opacity
stroke-dasharray
marker-start / marker-end
vector-effect
```

不得只比较 semantic color key。相同色键但不同 opacity、gradient stop、border 或 shadow 仍视为视觉变化。

#### 24.4.4 Interaction Manifest

至少冻结：

- 默认状态；
- Node hover；
- Node focus-visible；
- Node selected；
- Layer hover；
- Layer selected；
- Expert Pool collapsed；
- Expert Pool expanded；
- iso/front/right；
- Dark/Light；
- PP/TP/EP/DP annotation states；
- Layer expansion 前后。

### 24.5 Baseline 生成治理

基线生成采用以下规则：

1. 只允许从改造前 reference fixture 生成首次基线；
2. 基线生成脚本与验证脚本分开；
3. 日常测试默认只能读取基线，不能写入；
4. CI/本地 check 不提供隐式 `--update`；
5. 基线文件变化必须单独出现在 diff 中；
6. 任何基线变化必须提供机器差异报告和视觉差异图；
7. 没有用户明确批准，不得接受新增、删除、rename、geometry 或 style 差异；
8. “测试已经更新”不是差异正确的证据；
9. 性能优化不得触发基线刷新；
10. Rank ownership overlay 变化只能更新 extension baseline，不能更新 Core DOM baseline。

### 24.6 迁移顺序隔离

为了避免“重构”和“新功能”混在一起掩盖内容变化，实施必须拆成三个可独立验证的阶段：

#### Gate A：Renderer Extraction Only

只允许：

- 把现有硬编码数据移入 `LayerSceneSpec`；
- 把现有 DOM 创建逻辑移入共享 `renderLayerScene()`；
- 改变函数边界和数据传递方式。

禁止：

- 新增 Rank；
- 新增 ownership；
- 修改样式；
- 修改 geometry；
- 修改 label；
- 修改 Edge；
- 顺手清理“看起来可以简化”的节点。

Gate A 必须先独立通过全部结构、样式、视觉和交互 parity。

#### Gate B：Rank Wrapper Only

只允许：

- 在共享 Layer DOM 外增加 Rank Volume；
- 增加整体 scale/translate/rotate；
- 增加 pointer-event gating；
- 增加 Rank Label 和 Shell。

禁止修改 Core DOM。Gate B 在 scale=1 的隔离 parity harness 中仍必须与基线一致。

#### Gate C：Ownership Extensions

只允许：

- 增加 ownership data attributes；
- 在明确 extension slot 中增加 shard/replica overlay；
- 增加 Rank 间 link；
- 增加 Inspector 数据。

任何需要修改 Core DOM 的 ownership 设计都必须返回设计评审，不得由实现者自行决定。

### 24.7 结构 Parity 检查

必须提供自动化脚本，例如：

```text
scripts/capture-model-deck-baseline.mjs
scripts/validate-model-deck-structure.mjs
scripts/validate-model-deck-styles.mjs
scripts/render-model-deck-parity.mjs
scripts/report-model-deck-parity.mjs
```

结构比较必须输出集合差异，不能只输出 hash mismatch：

```text
Layer L23
  nodes.expected: 31
  nodes.actual:   31
  nodes.missing:  []
  nodes.extra:    []
  nodes.renamed:  []

  edges.expected: 36
  edges.actual:   36
  edges.missing:  []
  edges.extra:    []
  edges.changed:  []
```

必须执行：

- 全 Layer node ID set equality；
- 全 Layer edge tuple set equality；
- Cluster set equality；
- exact label equality；
- exact op/role equality；
- variant equality；
- Static Input/Output/MTP equality；
- DOM child semantic ordering equality；
- aria/interaction attribute equality。

结构字段默认零容差。

允许新增的 Rank extension 节点必须由 allowlist 精确匹配，例如：

```ts
const allowedExtensions = [
  "[data-layer-extension-slot] > .pto-layer-ownership-overlay",
  ".pto-rank-volume__shell",
  ".pto-model-rank-deck__link-overlay"
];
```

不得使用宽泛规则，例如忽略所有 `.overlay`、所有新增 child 或所有 `data-*`。

### 24.8 几何 Parity 检查

几何比较必须在独立 parity harness 中执行：

- viewport 固定；
- browser/version 固定；
- device scale factor 固定；
- Layer natural scale = 1；
- 无 Rank outer transform；
- 字体加载完成；
- animation/transition 关闭；
- 等待两帧稳定后采样。

比较对象：

- Layer bbox；
- 每个 Cluster bbox；
- 每个 Node bbox；
- Expert Pool collapsed/expanded bbox；
- 每条 Edge path normalized geometry；
- Static Input/Output bbox；
- Layer depth transform；
- label anchor 与基线距离。

默认容差：

```text
left/top/width/height: ≤ 0.25 CSS px
transform matrix component: ≤ 0.0005
SVG path coordinate: ≤ 0.25 CSS px
text baseline anchor: ≤ 0.5 CSS px
```

超过容差必须列出具体 Layer、Node/Edge 和字段。

### 24.9 Style Parity 检查

Style Parity 不能只检查 CSS 文件文本，因为 selector 作用域、继承和 Rank wrapper 都可能改变 computed style。

必须对相同对象逐状态比较 computed style：

```text
reference fixture object
vs
new renderer isolated object
vs
new renderer inside Rank at natural scale
```

比较矩阵：

| Theme | View | Object | State |
| --- | --- | --- | --- |
| Dark | iso | Dense Node | default/hover/selected |
| Dark | iso | MoE Node | default/hover/selected |
| Dark | front | Expert Pool | collapsed/expanded |
| Dark | right | Layer/Node side | default/hover |
| Light | iso | Tensor/Parameter | default/selected |
| Light | front | Expert Pool | collapsed/expanded |
| Light | right | Residual/State | default/selected |

以下差异一律阻断，除非进入批准差异清单：

- fill/background；
- gradient stop；
- opacity；
- border；
- radius；
- shadow；
- filter；
- typography；
- label alignment；
- selected ring；
- hover emphasis；
- Light theme tensor/parameter neutral fill。

### 24.10 Visual Parity 检查

结构和 style 检查通过后，再执行截图对照。截图不是结构测试的替代品。

必须覆盖以下 reference cases：

```text
L0  Dense + DSA + Block Post
L1  Dense + SWA
L2  MoE + SWA
L3  MoE + DSA
L4  MoE + Block Post
L45 MoE + DSA + final boundary
Input Stem
Output + MTP Tail
Expert Pool expanded
Node selected
Layer hover
Dark theme
Light theme
iso / front / right
```

每个 case 输出：

- reference；
- actual isolated renderer；
- actual inside Rank at normalized pose；
- pixel-diff heatmap；
- changed pixel ratio；
- 最大差异区域坐标；
- 对应 DOM selector。

Pixel diff 仅允许抗锯齿和字体栅格化造成的小范围差异。默认目标：

```text
changed pixel ratio ≤ 0.30%
geometry silhouette difference = 0
missing visible object = 0
unexpected visible object = 0
```

如果同一浏览器、字体和设备倍率下仍超过阈值，必须人工审查，不能自动接受。

### 24.11 独立验证原则

以下验证方式无效：

- 原 Deck 与 Rank Deck 都调用新 renderer，然后互相截图比较；
- 只比较新 renderer 的两次输出；
- 只验证 selected Layer；
- 只验证 L0 或一个代表 Layer；
- 只比较 node 数量，不比较 ID 和 Edge；
- 只比较 Dark，不比较 Light；
- 只检查 screenshot，不检查 DOM；
- 只检查源码 diff，不检查实际渲染；
- 改完后才生成所谓“原始基线”。

有效验证必须保留改造前冻结 fixture，并让基线来源与新 renderer 实现相互独立。

### 24.12 批准差异清单

如果确实需要改变原始 Layer，必须创建显式差异项：

```ts
interface ApprovedParityDifference {
  id: string;
  scope: string;
  baseline: unknown;
  proposed: unknown;
  reason: string;
  userApproval: string;
  approvedAt: string;
}
```

批准流程：

1. 单独展示原始与拟改效果；
2. 说明业务原因，不能只写“代码更简单”；
3. 说明受影响 Layer、Node、Edge、Theme 和交互状态；
4. 获得用户明确批准；
5. 加入精确 allowlist；
6. 更新 baseline version 和差异历史；
7. 不得扩大到同类节点或其他 Layer。

没有批准记录的差异一律视为 regression。

### 24.13 Parity Report

每个实现阶段必须生成 `PARITY_REPORT.md`，至少包含：

```text
Baseline version
Source commit / file checksum
Target files
Layer count comparison
Node/Edge/Cluster comparison
Geometry comparison
Computed style comparison
Interaction state comparison
Visual diff summary
Approved differences
Unapproved differences
Final gate status
```

最终结果必须明确显示：

```text
Core missing nodes: 0
Core extra nodes: 0
Changed labels: 0
Changed node roles: 0
Missing edges: 0
Changed edge routes: 0
Unapproved geometry changes: 0
Unapproved style changes: 0
Unapproved interaction changes: 0
```

只给出“测试通过”或“看起来一致”不满足交付要求。

### 24.14 实现权限边界

实现者默认拥有：

- 抽取数据结构；
- 创建共享 renderer；
- 增加 Rank 外层容器；
- 增加明确扩展槽；
- 添加不改变 Core DOM 的 ownership overlay；
- 修复由改造引入的 regression。

实现者默认不拥有：

- 删除或合并模型节点；
- 修改模型 label；
- 改变模型节点语义；
- 修改原始 geometry；
- 改变 Edge route；
- 更换视觉风格；
- 简化非当前 Rank；
- 以性能为由改变模型内容；
- 自动接受或刷新 baseline；
- 把未验证的模型事实写入 Layer。

超出默认权限的动作必须停止并请求用户批准。

### 24.15 阻断条件

出现以下任意条件，当前阶段必须判定失败：

- 一个 Core Node 缺失；
- 一个 Core Edge 缺失或端点变化；
- 一个未批准 label/role/ID 变化；
- 一个特殊 Layer 被错误泛化；
- Input/Output/MTP 任一对象遗漏；
- Expert Pool 结构或展开行为变化；
- Dark/Light 任一主题未验证；
- default/hover/selected 任一状态未验证；
- Rank wrapper 改变 Layer computed style；
- parity test 使用新 renderer 自己作为 reference；
- baseline 在无批准情况下更新；
- 性能优化删除、隐藏或替换模型本体；
- 无法提供可读的差异报告。

---

## 25. Definition of Done

只有同时满足以下条件，改造才算完成：

- 用户能在一个 CSS3D 场景中看到 128 个 Rank；
- 每个 Rank 内部存在其真实 Layer DOM，而不是缩略替代物；
- 进入 Rank 后能够操作原始算子和连线；
- Parallel Planner 可以独立解释每个 Rank 的模型归属；
- PP、TP、EP、EDP 语义及来源标签通过产品与分布式训练评审；
- 模型 Deck 与 Rank Deck 共用同一个 Layer renderer；
- 改造前 reference fixture、structure/style/interaction manifests 已冻结；
- Gate A、Gate B、Gate C 分别通过，未把 renderer 抽取与新视觉混在同一未验证步骤；
- `PARITY_REPORT.md` 显示 Core Node/Edge/Label/Geometry/Style/Interaction 零未批准差异；
- 所有 baseline 变化均有用户明确批准记录；
- 128-rank 性能达到最低门槛；
- 所有验收标准通过；
- registry、preview、DESIGN 和 Pattern 合同完成更新；
- 最终浏览器 smoke test 通过；
- 不存在用 Three.js、纹理截图、payload glyph 或局部复制 renderer 绕过本 Spec 的实现。
