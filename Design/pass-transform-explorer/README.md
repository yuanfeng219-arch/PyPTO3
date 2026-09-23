# Pass Transform Explorer

把 PyPTO 编译器从「输入算子、输出二进制的黑盒」变成一条可以逐站检查的流水线。

工具直接解析 `Data/**/passes_dump/*.py`——每个 Pass 执行后的真实 IR 快照——回答三个问题：

| 问题 | 视图 |
| --- | --- |
| 变化发生在哪个 Pass？ | **Pass 时间线**：按实际改动行数排序，空操作 Pass 直接置灰 |
| 具体改了哪几行？ | **代码 Diff**：按函数拆分，token 级高亮，并排 / 统一双模式 |
| 这个 Pass 到底优化了什么？ | **变化概览**：从 AST 实测出的结构增量 + **结构图**：五种视角的前后对照 |

设计依据是仓库里已有的结论 [`Design/llvm-flow/Pass_Diff_对话记录_20260908.md`](../llvm-flow/Pass_Diff_对话记录_20260908.md)：
文本 IR Diff 是事实来源，Graph 负责快速理解结构，两者同步；并且**不存在一种计算图能解释所有 Pass**，
所以每个 Pass 自带「推荐视角」。

## 运行

数据是生成的（约 50 MB，已 gitignore），第一次使用需要先构建：

```bash
node Design/pass-transform-explorer/build.mjs
```

然后用任意静态服务器打开 `Design/pass-transform-explorer/index.html`，或从 `launch.html` 进入。
（页面通过 `<script>` 标签按需加载快照，`file://` 下通常也能直接打开，但静态服务器更可靠。）

```bash
npx http-server . -p 4178 -c-1
```

构建约 6 秒，产出：

| 产物 | 大小 | 内容 |
| --- | --- | --- |
| `data/index.js` | ~1.5 MB | Pass 时间线、实测增量、证据卡片 |
| `data/docs.js` | ~210 KB | 从 `repo/pto/docs/zh-cn/dev/passes/` 提取的 Pass 说明 |
| `data/<run>/NN.js` | ~50 MB | 每份 IR 快照的原文，按需加载 |
| `lib/bundle.js` | ~75 KB | 解析器 / 分析器的浏览器版 |

`build.mjs --no-src` 只重建索引，跳过快照（改分析逻辑时用它更快）。

## 独立 demo（单文件，无需构建和服务器）

要把工具发给别人、或在没有这个仓库的机器上打开：

```bash
node Design/pass-transform-explorer/build-demo.mjs
```

产出 `demo.html`——**一个文件，双击即开**。样式、解析器、应用代码、Pass 索引、Pass 文档和
全部 94 份 IR 快照都 gzip + base64 内嵌在里面：

| 命令 | 产出 | 大小 | 内容 |
| --- | --- | --- | --- |
| `node build-demo.mjs` | `demo.html` | 6.0 MB | 两份 run，94 份快照（46.96 MB IR） |
| `node build-demo.mjs --runs decode_fwd_layers --out demo-lite.html` | 自定义 | 1.7 MB | 单份 run，42 份快照 |

快照是**逐份压缩、按需解压**的，所以打开时只付索引的代价（约 110 KB），
不是 47 MB 的 IR。切到某个 Pass 的 Diff 或结构图时才解压对应的两份快照（每份约 10 ms）。

需要浏览器支持 `DecompressionStream`：Chrome / Edge 80+、Firefox 113+、Safari 16.4+。
不支持时会显示明确的提示而不是白屏。

demo 里没有仓库，所以 Pass 源码路径和文档来源会显示成带 tooltip 的纯文本而不是死链接；
其余功能与完整版完全一致（同一份 `app.js` 和 `lib/bundle.js`）。

## 为什么 Diff 和图在浏览器里现算

`lib/*.mjs` 既是构建期代码，也被打包进 `lib/bundle.js` 供页面使用——**同一份解析器**。
页面加载一份快照原文后自己解析、自己 diff、自己建图，因此任意函数、任意视角、任意 Pass
都能全保真查看，而不需要把几十万行预计算结果冻进数据包。解析一份 5000 行快照约 50ms，最近 8 份缓存在内存里。

## 五种结构图

节点按 union 布局**一次性排好**，再按状态着色（新增 / 删除 / 属性改变 / 未变）。
分别给前后两版单独布局会让插入一个节点就把整张图挪位，真正的变化反而被淹没。

| 视角 | 节点 | 最适合看 |
| --- | --- | --- |
| 调用 / 作用域 | 函数 | `InlineFunctions`、`Outline*`、`ExpandMixedKernel` |
| 控制流 | for / if / with 区域 | `UnrollLoops`、`LowerPipelineLoops`、`SkewCrossCorePipeline` |
| 数据流 | SSA def-use | `ConvertTensorToTileOps`、`LowerCompositeOps`、`ResolveBackendOpLayouts` |
| 任务 DAG | `pl.submit` / `pl.at` 及依赖边 | `AutoDeriveTaskDependencies`、通信相关 Pass |
| 内存布局 | 缓冲区、空间、地址 | `InitMemRef`、`MemoryReuse`、`AllocateMemoryAddr` |

任务依赖是抽象解释 `pl.array.create` / `update_element` 链恢复出来的——依赖在 IR 里是通过
`_submit_deps_buf` 数组传递的，不解释这些语句就拿不到真实的 task DAG。

## 「高频改写」如何工作

有些 Pass 不改任何结构指标，只是就地改写每条语句——`AllocateMemoryAddr` 填 MemRef 偏移、
`ConvertToSSA` 追加版本后缀、`CanonicalizeIOOrder` 把 `pipeline` 降成 `range`。
这类 Pass 如果只报「735 行变化」等于什么都没说。

所以构建期会把被改写的行按 token 重叠度**配对**（相似度低于 0.5 的算纯增删，不硬凑），
再对每一对做 token 级 diff，聚合出最高频的替换。例如：

```
AllocateMemoryAddr   735 行就地改写，最高频 `0` → `1024`（119 处）
DeriveCallDirections  42 行就地改写，最高频 插入 `, attrs={"arg_directions": [...]}`
MemoryReuse          tile.alloc 668 → 211，合计 11.0 MiB → 4.59 MiB
```

## 自检

```bash
node lib/test.mjs      # 220 项单元检查：解析、类型、diff、改写配对、markdown、bundle 导出
node lib/validate.mjs  # 94 份快照全量解析，要求 0 行未覆盖、0 处表达式失败
node lib/smoke.mjs     # 6321 个函数 × 5 种视角，检查 id 唯一性、悬空边、内存一致性
```

`validate.mjs` 是这套工具可信度的底座：dump 文件是 Python 的一个极规则子集，
解析器必须对 **273,630 条语句一行不漏**，任何分析才谈得上准确。

## 已知边界

- `repo/pto` 镜像里缺少部分 Pass 的文档与源码（`BlockNzTensorViews`、`InsertCommFence`、
  `LowerPipelineToSlots` 等）。这些 Pass 的说明面板会明确标注「镜像里没有」，实测证据不受影响。
- MemRef 的 `size` 为 0 且 shape 含动态维时，内存视图显示「动态」而不是「0 B」——
  编译期确实算不出字节数，不应伪装成 0。
- 数据流视角每个函数最多 600 个节点、控制流 900 个，超出会标注「已截断」。
- 两份 run 的 Pass 序列不同（`l3_decode_csa` 52 个、`decode_fwd_layers` 42 个），
  这是编译配置差异，不是数据缺失。

## 新增一份 run

在 `build.mjs` 顶部的 `RUNS` 数组里加一项，指向仓库内的 `passes_dump` 目录即可：

```js
{ id: 'my_run', title: '…', subtitle: '…', dir: 'Data/…/passes_dump' }
```

Pass 名到文档 / 源码的映射走大小写与下划线无关的归一化（`FlattenTileNdTo2D` ↔
`flatten_tile_nd_to_2d`），新 Pass 通常不需要额外配置；
阶段分组和推荐视角在 `lib/passinfo.mjs` 里，未登记的 Pass 会落到合理默认值。
