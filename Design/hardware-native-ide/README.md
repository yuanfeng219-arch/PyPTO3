# Hardware-Native Lens

IDE 内的单页设计原型，用一个真实算子演示昇腾 Hardware-Native 体系的两件事：

- **可表达** —— Tensor/Tile、任务与依赖、并行与同步、内部布局等硬件语义写在源码里，因此可以被读取、被标注、被可视化。
- **约束前移** —— 在编码时展示示例硬件布局模型、性能估算以及算子和运行时契约；实际设备合法性与性能仍需结合目标、编译产物和运行结果验证。

入口：[`index.html`](index.html)（也在仓库根的 `launch.html` 列表里）。

默认先展示「执行路径」：围绕 QK 容量示例，将代码、KV Cache → Right buffer → Cube → Vector 的数据路径和方案比较放在同一视图。节点选择联动源码节选；两种方案分别展示分页契约和分段计算的影响。顶部可打开原 Lens 工作台，并从工作台返回执行路径；两种视图的演示配置相互独立。执行路径使用 910B 容量参考与双缓冲假设，不代表编译分配或设备结果。

## 取材

全部结构、常量、依赖关系与约束来自 [`../Toolkit Studio/decode_layer.py`](../Toolkit%20Studio/decode_layer.py)
（Qwen3-14B fused decode，FP32 跨层 carry）。编辑器里是节选，**行号与源文件一致**，折叠区标出跳过的行数，便于回溯原文。

参考 Demo `Design/Toolkit Studio/` 的其余能力（模型结构图、Pass 流水线、Correctness Lab、运行详情、SOLO 模式等）在本设计中全部去掉，只保留 IDE 主页面这一条线。

## 页面结构

结构沿用 `vendor/pto-design-system` 的 `ide-frame` + `workbench-shell` pattern；视觉语言参考
`repo/AI_Infra` 的 demo（见下节）。

| 区域 | 内容 |
| --- | --- |
| 左栏 | 语义目录，按片上存储与布局、计算单元与性能、任务与运行时、模型数据分组 |
| 中栏 | `decode_layer.py` 节选，顶部透镜开关控制行内语义标注与 scope 色带，并显示约束 gutter 标记 |
| 右栏 | 10 条示例规则和估算的实时检查与详情，区分硬件模型、性能估算和算子／运行时契约 |
| 底部 | 按工作负载、模型维度、运行时和算子策略分组的 11 个配置项；模型维度通常由模型决定，此处可调用于演示 |
| 状态栏 | 当前配置下的约束总判定 |

### 一个选中模型

左栏目录条目、源码里的高亮符号和右栏约束行共同控制当前选择：

- 选中**语义对象** —— 在对应代码行下方展开关键事实和 SVG 可视化；多个代码落点可在卡片中切换。右栏高亮相关约束，其余约束压暗但仍可点。没有关联约束时显示完整约束列表。
- 选中**约束** —— 右栏下钻显示规则、当前判定、硬件事实和代码落点，并链接相关语义对象；可返回原约束列表，保留筛选和滚动位置。
- **总览** —— 顶部显示四个关键派生量，下方列全部约束。

约束区可按违反、有代价、已满足筛选：选中语义对象时筛选复位（约束区在讲的东西换了），在约束之间跳转时筛选保留（同一列表内的导航）。选中条目会定位源码，右侧面板收起时会自动展开。底部「快速体验」可从默认配置切换到容量与分页、切片对齐或并行分配场景；每个场景先恢复默认值再应用对应改动。手动调整时显示相对默认配置的修改项数。顶部重置恢复配置、透镜、筛选和总览。

源码区默认开启「配置代入」：已知配置和派生常量在表达式引用处显示为具体值，悬停可查看原变量名；关闭后恢复显示变量引用。变量定义的左侧、字符串、注释和依赖运行时的数据不作代入。显示值来自演示模型，不是运行时采样，也不修改原始源文件。

源码区提供容量与分块、算子能力和有效形状三个 Issue 参考入口。容量场景关联 QK / PV 源码，可调整 `SEQ_TILE` 和示意分块大小，逐块播放循环并观察双缓冲估算。QK 分段生成分数，合并并遮罩后执行 Softmax，PV 按段累加；图中的循环为伪代码，不改写源码。容量公式假设两个完整 BF16 右操作数块同时驻留，并采用 #1595 中 910B 的 64 KiB 作为参考；#630 的原始 DeepSeek 案例未注明设备。估算不代表目标设备的实际编译分配。有效形状场景关联 `valid_len`、`set_validshape` 与 `fillpad`，使用可调整的示例上下文长度和块索引展示有效区与填充区；越界工作表项仅为模拟，不能据此判定当前源码存在故障。两种场景均为演示，不是实时源码检查，也不计入当前配置的违规数量。

## 视觉与代码渲染

视觉参考 `repo/AI_Infra`（`ai-infra-research/reports/llm-compute/*`、`atlas-950-explorer`）的做法：

- 近黑灰阶：`#111` 底 / `#131313` 画布 / `#171717` 面板 / `#202020` 抬升，1px 细线 `#353535`，平面无阴影
- 各观察视角使用独立强调色区分；约束判定使用状态色。
- 标签一律等宽小字 + 宽字距大写；选中态为浅底深字（`#e0e0e0` / `#111`）

代码渲染按常规编辑器口径分词后再按上下文着色，而不是只认关键字：

| 类别 | 判定方式 |
| --- | --- |
| 函数调用 / 定义 | 标识符后紧跟 `(`；`def` 之后的名字另算 |
| 关键字实参 | 标识符紧贴 `=`（PEP8 下赋值两侧有空格，实参没有），如 `name_hint=` `deps=` |
| 类型 / dtype | 首字母大写，或在 dtype 表内（`FP32` `BF16` `TASK_ID` …） |
| 常量 | 全大写标识符 |
| 模块 / 属性 | `pl` 限定符与点号链上的非调用成员 |
| 其余 | 关键字、字面量、装饰器、字符串、数字、运算符、标点各自分色 |

点号链（`pl.Level.CORE_GROUP`）整段作为一个可点击对象，内部各段仍分别着色。

## 五个观察视角

| 透镜 | 在代码里的落点 | 可视化 |
| --- | --- | --- |
| **Tensor / Tile** | `pl.Tensor[[...], dtype]`、`pl.create_tensor`、切片表达式、`pl.matmul` 的 A/B/C | 张量切块图、tile 去向与字节数、AIC/AIV 流水 |
| **任务与依赖** | `pl.manual_scope()`、`pl.at(level=pl.Level.CORE_GROUP)`、`deps=[...]`、`pl.Array[_, pl.TASK_ID]` | 任务边界、任务句柄和显式依赖边 |
| **并行与同步** | `pl.parallel`、`pl.spmd`、`pl.pipeline`、grid-stride、`atomic=pl.AtomicType.Add` | 核上分布、流水交接和 Split-K 原子累加 |
| **内部布局** | `Q_HEAD_PAD` / `set_validshape` / `fillpad`、`cache_row` 算术、split-K 切片宽度 | 填充行 vs 有效行、逻辑块→物理页、切片与 slab 对齐 |
| **数值与精度** | Attention 与残差路径的 FP32 / BF16 转换 | dtype 转换边界与累加输出 |

依赖与 Task ID、Split-K 原子累加、K/V 页、Cube/Vector 交接和有效区域处理均可从目录或代码标注进入。流水与交接图表达源码意图，不是实测时间线；源码说明该版本在目标设备上预期存在代码生成失败。

嵌套 scope 以左侧色带表示（`manual_scope` → `pl.at` / `pl.parallel` / `pl.spmd` → `pl.pipeline`）。

## 设参数时的就地预警

约束前移最靠前的那一步不在编译、也不在上板，而在**用户给参数填值的那一刻**。
每条约束都声明了自己依赖哪几个旋钮（`model.js` 里每条约束的 `knobs` 字段），
反过来就知道当前这个值让哪条约束不成立。

- **未满足** —— 标签旁出现 `!`，下拉框描边转红，正下方展开一条说明：状态、约束标题、一句话结论；
- **需关注** —— 标签旁只留一个可点的 `△`（悬停给出结论），下拉框描边转琥珀色，不占版面；
- 两者都可点开：右栏打开该约束的详情与原因图，源码定位到它的落点 ——
  从「哪个值不行」到「为什么」只有一跳。

两档分开是为了不淹掉信号：默认配置本来就有两条「需关注」的估算
（`l0b-capacity` 正好占满 64 KiB 参考容量、`cube-m-fractal` 填充 68.75%），
如果它们也展开成说明块，真正的预警就看不见了。

预警只按当前配置计算，不编译、不上板；源码里的波浪线和状态栏同时重算。

### 原因可视化 · Vec(UB) 预算条

`ub-usable-headroom` 这条约束带一张专属的原因图，抽象自
[pto-isa #170](https://github.com/hw-native-sys/pto-isa/issues/170)：一条 Vec UB 地址空间，
上面按地址排出跨块 carry 与流水各级的实际占位，并标出名义上限、可用上限和顶部保留带；
越过上限的那一块标红。下面是三行算式 —— 当前高水位、分配器的预算（名义容量）、
真正能用的（名义 − 保留）。**这两个数不一样，就是全部原因**：分配器拿名义 192 KB 做判断，
而顶部 8 KB 已被运行时占着（`ReserveBufferOp` / `PTOInferValidatePipeInitPass`），
落进去的 tile 读写的是别人的数据，地址本身合法，越界中断也不会触发。
图下两个按钮分别定位到 L764 的 `stage` 和越界那个中间量的源码行。

**抽象自哪里。** 原 issue 是 Ascend910B1 / cann-9.0.0 上一个纯 Vector kernel：`alloc_tile` 高水位
188.8 KB，在名义 192 KB 之内，设备输出 112 个 NaN；同一 kernel 把 tile 列宽改小、高水位降到
183.3 KB 后输出干净。NaN 全在最高地址的累加器（`m_oi_nope`），低地址干净。
本 demo 把它挂到 `fa_fused` 内层流水的 softmax 链（L764 的 `stage`，L775–L786 的中间量）。

**各档的数。** 默认配置高水位 96.4 KB，占可用容量 52%，不触发预警。
`SEQ_TILE` 256 + `FA_STAGE` 2 时高水位 184.4 KB —— 越过可用上限 384 B，距名义上限还差 7.6 KB，
正好落在那条 8 KB 宽的带里，越界的是 `exp_scores_bf16`（L786）。
`FA_STAGE` 降到 1 回到 96.3 KB。`FA_STAGE` 3 以上超出名义容量，变成分配器自己会报错的那一档。

Vec 高水位按「每个中间量各占一块、按出现顺序排布、每个驻留级一份」估算，
是否复用、就地改写和实际分配顺序需按编译产物核对；保留区大小随平台与 CANN 版本变化。

## 十条示例检查

每条标明规则和依据类型。容量与均衡是基于示例假设的估算，不直接代表实际编译结果或硬件性能。

1. `cube-m-fractal` —— Q 头打包行数对齐 cube M fractal 16（填充占比过半时给警告）
2. `page-identity` —— `SEQ_TILE == runtime.page_size`
3. `l0b-capacity` —— K tile 双缓冲 ≤ Right(L0B) 64 KB
4. `ub-usable-headroom` —— Vec 高水位 ≤ `UB_NOMINAL − UB_RESERVED`（184 KB，而非 192 KB）；
   落在两个上限之间时分配器放行而运行时读写保留区，带专属原因图，见上节
5. `slab-1to1` —— `QKV_K_SLICE == DOWN_TN`（`deps=[prev_normed_tids[q_ks]]` 成立的几何前提）
6. `down-chunk-align` —— `HIDDEN % DOWN_TN == 0` 且 `DOWN_TN % 256 == 0`
7. `splitk-divides-k` —— `HIDDEN % QKV_OK == 0`
8. `gp-head-cover` —— `GP_SIZE == NUM_KV_HEADS`（work table 只编码 `(b, block)`）
9. `spmd-even-split` —— `(KV heads × batch) % ROPE_CORES == 0`
10. `block-balance` —— grid-stride 的整数打包偏差 ≲ 1.25×

## 怎么演示「约束前移」

### 片上容量决策闭环

在「快速体验」中选择「容量与分页」，右栏直接打开 Right buffer 容量详情。超出示例参考容量时，可比较缩小逻辑块和保持逻辑块、探索内部再分块两个方向。前者先预览容量和分页契约，再由用户应用到页面配置；后者打开源码旁的 QK 分块示意。应用仅修改演示参数，分块伪代码也不会修改源文件。

验证区列出编译分配、数值正确性和设备性能三个未执行的步骤。容量数字基于两个完整 BF16 块同时驻留的假设和 910B 的 64 KiB 参考容量，不是当前目标的实际编译分配；案例 #630 是独立案例，不代表本示例的运行结果。

底部旋钮改一个值，源码里的形状、Inspector 的布局图、波浪线与状态栏会同时重算。几条立刻出效果的路径：

- `SEQ_TILE` 改 256 → `l0b-capacity` 红（双缓冲 128 KB 超出 64 KB），`page-identity` 红（逻辑块跨页）
- `runtime.page_size` 改 64 → `page-identity` 红，paged 图上画出跨页区域
- `QKV_OK` 改 4 → `slab-1to1` 红，split-K 对齐图的连线变红虚线
- `HEAD_DIM` 改 64 + `DOWN_TN` 保持 1024 → `down-chunk-align` 红（2560 不被整除）
- `GP_SIZE` 改 4 → `gp-head-cover` 红（work table 缺头维度）
- `ROPE_CORES` 改 24 → `spmd-even-split` 红（128 items 不被 24 整除）
- `NUM_KV_HEADS` 改 4 → Q 填充占比升到 75%，`cube-m-fractal` 转警告
- `SEQ_TILE` 改 256 + `PAGE_SIZE` 改 256 → `SEQ_TILE` 和 `FA_STAGE` 两个旋钮下方同时出现
  `ub-usable-headroom` 预警：高水位 184.4 KB 越过可用上限 384 B 但没超名义上限；
  点预警看预算条，`FA_STAGE` 降到 1 即解除

## 文件

```
index.html          IDE 页面骨架
styles/app.css      AI_Infra 配色映射、编辑器装饰、代码着色、SVG 配色
js/model.js         旋钮、派生量、十条约束、源码节选与语义标注、符号词典
js/source-extra.js  从原文件摘录的计算主链路，按原始行号与语义标注合并
js/ai-infra-visual-system.js  复用 AI Infra 图例库的 SVG 图元与关系渲染器
js/viz.js           8 个 SVG 可视化 + 15 个概念面板
js/app.js           分词与上下文着色、渲染编排、选中与联动
```

图元来源：[AI Infra 图例与 Pattern 库](https://yinyucheng0601.github.io/AI_Infra/ai-infra-research/reports/llm-compute/diagram-legend-library/index.html?v=69860fd1f593)，本地副本来自 `repo/AI_Infra/ai-infra-research/reports/llm-compute/diagram-legend-library/visual-system.js`。当前在 MatMul、Split-K 累加和显式依赖视图中调用数据、算子、加法、任务和关系图元；图中的数据角色色与页面透镜分类色分别表达不同含义。

无构建步骤、无外部依赖，直接用浏览器打开 `index.html` 即可。
