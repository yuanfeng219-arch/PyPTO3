# Hardware-Native Kit

把 `Design/hardware-native-ide/` 两个 demo（Lens / 主线）里那套「源码 ⇄ 算子逻辑 ⇄ 硅片落点」三区联动，从一个算子的手写页面，抽成**换一个 kernel 只写一份 manifest**的工程化套件。

入口：[`index.html`](index.html)。自带两个 kernel，顶栏可切换。支持 MCP Apps 的 Agent 客户端可通过 [MCP App 接入说明](mcp-app/README.md) 在对话中嵌入界面。

> 本目录是新增的，不改动 `Design/hardware-native-ide/` 下的任何文件。两边类名前缀不同（`kb-` vs `insp-` / `ub-`），可同页共存。

---

## 1. 为什么要抽一层

原 demo 能跑，但三样东西写死在页面里，换 kernel 等于重写：

| 对象 | 原 demo 的做法 | 换 kernel 时的代价 |
| --- | --- | --- |
| 源码定位 | 节点表里写死行号（`path.js` 里约 44 处字面行号） | 上游插一行，全部静默错位，页面不报错 |
| 逻辑图 | 节点坐标与 12 条 SVG 路径手摆 | 节点数一变就得重排版 |
| 容量判定 | 按 softmax 的形状硬编码 | 换算子等于重写算术 |

Kit 把这三样分别换成：**名字锚点 + 诊断**、**依赖/阶段自动布局（可选指定展示布局）**、**形状 × dtype × 驻留份数的通用模型**。

---

## 2. 文件结构

```
Design/hardware-native-kit/
├── index.html              宿主页：引脚本、跑体检、mount
├── kit/
│   ├── platform.js         平台描述：片上容量口径 + 硅片区域盒子树
│   ├── core.js             纯函数：锚点解析 / 容量模型 / 两种自动布局
│   ├── view.js             渲染三区 + 联动索引（不认识任何具体 kernel）
│   └── kit.js              公共入口：mount() / check()
├── kernels/
│   ├── fa_fused.js         decode_layer.py L728–798（Qwen3-14B 融合 decode）
│   └── matmul_64.js        repo/pto/examples/kernels/03_matmul.py L32–41
└── styles/kit.css          AI_Infra 视觉语言，类名全部 kb- 前缀
```

一句话分工：**platform 描述芯片，manifest 描述 kernel，core 算，view 画。**
加一个 kernel 只动 `kernels/`，加一块芯片只动 `kit/platform.js`。

---

## 3. 加一个 kernel：写一份 manifest

```js
window.HNKitKernels.my_kernel = {
  id: 'my_kernel', entry: 'my_kernel',
  file: 'path/to/source.py',
  platform: 'a2a3',

  consts: { TILE: 128 },                       // 旋钮初值 + 固定事实
  knobs:  [{ id: 'TILE', label: 'TILE', options: [64, 128, 256] }],
  derive: function (c) { return { PAD: Math.ceil(c.TILE / 16) * 16 }; },

  source: [{ no: 32, t: '    x = pl.load(a, [0, 0], [{{TILE}}, 64])' }, /* … */],

  tensors: [{ id: 'a', label: 'a' }],          // 填 GM 槽位
  frames:  [{ id: 'f', scope: 'pl.pipeline', label: '…', residency: 2 }],
  phases:  [{ id: 'p1', label: '① 搬运', nodes: ['ld'] }],
  nodes:   [{ id: 'ld', assigns: 'x', label: 'a → L1', frame: 'f', regions: ['gm:a'] }],
  buffers: [{ assigns: 'x', space: 'vec', shape: ['PAD', 'TILE'], dtype: 'fp32', frame: 'f' }]
};
```

### 字段口径

**定位（不写行号）** — 每个 `node` / `frame` / `buffer` 用下面任一字段锚定源码行：

| 字段 | 匹配 | 用在 |
| --- | --- | --- |
| `assigns: 'x'` | `^\s*x\s*(:…)?=`（非 `==`） | 绝大多数中间量 |
| `also: ['y']` | 同上，追加若干行 | 一个节点还依赖别处的标量 |
| `call` / `calls` | 文本含 `pl.store(` | 没有赋值目标的调用 |
| `scope` | 同上 | `with pl.spmd(` / `pl.pipeline(` 这类框 |
| `lines: [784]` | 字面行号 | **逃生口**，尽量不用 |

这个 DSL 里中间量都是单赋值 `x = pl.foo(...)`，所以名字是可靠锚点。锚点一个都没命中会进 `diagnostics`，页面状态栏和 `check()` 都会报——**源码变了会被发现，而不是静默高亮错的行**。

**容量** — `buffers` 不写字节数，只写形状和类型：

```
字节 = Π shape × dtype.bytes × 驻留份数
驻留份数 = buffer 所在 frame 的 residency（resident: true 的恒为 1）
```

`residency` 可以是数字，也可以是 `function(consts)`——`fa_fused` 的 `f-pipe` 用的是 `Math.min(c.FA_STAGE, c.GP_SIZE)`，所以调 pipeline stage 能直接看到高水位翻倍。`shape` 里的字符串从 `consts` + `derive()` 取值；引用了未定义的常量同样进 `diagnostics`。

**布局** — `nodes` 只给 `after`（依赖）和 `phases`（语义分段），不给任何坐标：

- 列 = 阶段内的依赖深度（指向阶段外的依赖不计，所以每段从第 0 列起）
- 行 = 同深度堆叠；`phases[].wrap` 控制折行（默认 4）
- `frames` 框自动罩住属于它和它子 frame 的节点，`parent` 表示嵌套
- 可选 `graphLayout` 指定节点、阶段与框的位置；默认的 `fa_fused` 使用它对齐主线页面，其他 kernel 仍使用自动布局

**联动** — `nodes` 表就是联动的单一真相源。一个节点 = 源码行 + 硅片区域（`regions`）+ 片上 buffer。悬停任意一区反查这张表，再同时点亮另外两区；点击钉住。`regions` 里的 id 来自平台盒子树（`cube`、`l0b`、`ub`、`ch-v2c`…），GM 张量写 `gm:<tensor id>`。buffer 所在空间对应的区域会自动补上，不用重复写。

**参数代入** — 源码文本、标签、tensor 备注里的 `{{CONST}}` 会替换成当前值。锚点匹配跑在原始文本上，所以代入不影响定位。

---

## 4. 加一块芯片：写一份 platform

`kit/platform.js` 两部分：

```js
spaces: { vec: { label: 'Vec · UB', nominal: 192*1024, reserved: 8*1024 } }
```

`reserved > 0` 的空间才会出现 **`silent`** 那一档——高水位越过「可用上限」但没越过「名义容量」，分配器按名义容量放行，运行时读写落进保留区。四档判定：

| 档位 | 条件 | 含义 |
| --- | --- | --- |
| `ok` | ≤ 92% 可用 | 未触发 |
| `tight` | > 92% 可用 | 接近上限 |
| `silent` | > 可用，≤ 名义 | 分配器放行，但越过可用上限 |
| `over` | > 名义 | 分配器自己会报错 |

`chip` 是个盒子树（`dir`/`w`/`h`/`flex`/`kind`），坐标由 `core.layoutBoxes()` 算。a2a3 页面使用与主线一致的固定示意坐标，内容和容量仍从平台与 manifest 读取；其他平台沿用盒子树渲染。`kind` 取 `frame`（虚框）/ `mem` / `unit` / `chan` / `cores` / `bar`（地址条）/ `slots`（由 manifest 的 `tensors` 填充）。`modeled: false` 的空间显示「本例未建模」，也不参与总判定。

---

## 5. `check()`：manifest 体检

```js
HNKit.check(kernel)
// → { kernel: 'fa_fused', ok: true, diagnostics: [], worst: 'ok' }
```

`index.html` 在 mount 前对每个 kernel 跑一遍，不通过就打 `console.warn`。这个函数不碰 DOM，可以直接拿去 CI——源码一改，锚点失配会先在这里报出来。

---

## 6. 自带的两个 kernel

### `fa_fused` · decode_layer.py L728–798

Qwen3-14B 融合 decode 的 paged flash-attention，三层嵌套：`pl.spmd(24)` → `pl.range` grid-stride → `pl.pipeline(GP_SIZE, stage=…)`。四段：取页 / 打分 / online softmax / 加权累加。

默认配置（`SEQ_TILE=128, stage=2`）下 Vec 高水位 96.4 KB 是 `ok`，L0B 的 K 双缓冲刚好 64 KB 占满，是 `tight`。

调到 `SEQ_TILE=256, stage=2` 会同时命中两档，这正好是四档判定要区分的两种失败：

| 空间 | 结果 | 说明 |
| --- | --- | --- |
| L0B · Right | `over`，超 64 KB | K 双缓冲 2 × 64 KB = 128 KB > 名义 64 KB。**分配器自己会报错**——看得见的失败。 |
| Vec · UB | `silent`，184.4 KB | 越过可用上限 384 B，距名义上限还有 7.6 KB。分配器按名义容量放行，编译干净、`rc=0`，最高地址那块读写垃圾。越界的是 `exp_scores_bf16`（L786）。 |

`stage=1` 时 Vec 回到 96.3 KB，`stage=3` 则是 272.5 KB 的 `over`。
（`SEQ_TILE` 在原算子里等于 `BLOCK_SIZE=128`，256 是推演值，不是现有配置。）

口径说明（manifest 的 `note` 里也有）：

- K 与 V 复用同一块 Right(L0B)，所以只声明 `k_tile`；`v_tile` 节点有落点、没有独立声明。
- online carry（`o_i`/`m_i`/`l_i` 累加量）在本节选之外，但常驻 UB、参与高水位，作为无锚点的 `resident` buffer 声明。
- `q_padded` 在源码里是 GM 切片，这里按它作为 CUBE 左操作数落在 L0A 建模。

### `matmul_64` · 03_matmul.py L32–41

最小对照组：10 行源码、单层 `pl.at(CORE_GROUP)`、6 个节点。它在这里的作用是证明"换 kernel 只写 manifest"——同一套 core 和 view，没有一行为它特化。`TILE=256` 时 L0A/L0B 各 128 KB，是 `over`；L0C 刚好 256 KB，是 `tight`。源码未写 dtype，按 BF16 入 / FP32 出估算。

---

## 7. 适用范围与风险

- **是编码期估算，不是编译结论。** 容量数字来自 manifest 声明的形状与 frame 驻留份数；是否复用、是否就地、真实分配顺序要按编译产物（`passes_dump/36_after_AllocateMemoryAddr`）核对。页面上「片上空间判定」标题旁写明了「编码期估算 · 未经编译 · 未上板」。
- **容量口径随平台与 CANN 版本变化。** `a2a3` 的 Vec 保留 8 KB 取自 pto-isa#170；其余沿用示例口径，接别的目标前要核对 `kit/platform.js`。
- **锚点依赖单赋值风格。** 同名变量被多次赋值时，锚点会命中多行（`oi_tmp` 就命中 L790 与 L794，这是期望行为）；如果源码改成分支内重复赋值，需要改用 `lines` 逃生口或拆节点。
- **未经浏览器验证。** 本目录的页面与样式是静态写出并逐项手工核算的（两个 manifest 的字节数、四档判定边界、盒子树尺寸分配都手算过），但尚未在浏览器中打开检查布局与交互，需人工验收。

## Codex 本地插件

本仓库提供 `pypto-local` 本地插件目录，插件名为 `hardware-native-kit`。在仓库根目录运行：

```powershell
codex plugin marketplace add D:\project\PyPTO3
codex plugin add hardware-native-kit@pypto-local
```

本机已安装时，可在 Codex 插件目录中找到并启用 Hardware-Native Kit，然后在新的本地聊天中请求“使用 Hardware-Native Kit 展示 fa_fused”。插件通过本地 STDIO 启动 `mcp-app/server.mjs`，需要本机 Node.js，且插件安装副本须包含 `mcp-app` 的 npm 依赖；安装前可在 `mcp-app` 运行 `npm ci --legacy-peer-deps`。界面是否内嵌取决于客户端对本地 MCP Apps 的支持，需在新聊天实际验收。独立页面仍可使用 `index.html`。
