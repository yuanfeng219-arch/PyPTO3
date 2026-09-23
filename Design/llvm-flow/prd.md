# PyPTO Control Flow Explorer

## 从源码控制意图，到编译控制流，再到真实执行流

### 01. 产品定义

PyPTO 官方编译链是 `Python DSL → IR → Pass Pipeline → CodeGen`，最终一条路径生成 AICore 的 `.pto`，另一条路径生成运行在 AICPU 上的 orchestration C++；Runtime 再根据这些产物构建 task dependency graph，并协调 Host、AICPU、AICore 执行。([GitHub][1])

因此 Control Flow 可视化不应该只存在于编译后的某个 `.cpp` 文件，而应该形成一条连续关系：

```text
开发态                         编译态                         Runtime
Source Control Intent          Compiler Control Flow          Execution Flow

pypto.loop                     IR / Basic Block               Runtime Task
pypto.loop_unroll      →       Pass Transformation     →      Dispatch / Execute
dynamic shape                  Branch / Back Edge             Dependency / Wait
pass options                   Generated Paths                Scheduler / Orch

     Graph                         Graph                         Graph
     ↓                             ↓                             ↓
“我想怎么执行”                 “编译器变成什么”               “实际上怎么执行”
```

**这是同一个问题在三个阶段的不同答案。** 核心价值：

> 用一套连续的 Flow Visualization，把 PyPTO 开发者的“控制意图 → 编译变换 → 生成结构 → 实际执行”串成可追踪、可比较、可诊断的一条证据链。LLVM-FLOW 是整个产品的视觉母体——开发态是它的语义化变形，编译态基本保留原形，Runtime 再变形成带执行状态和时间证据的 Flow。

---

### 02. End-to-End 用户体验路径

| 阶段 | 用户在做什么 | 用户真正的问题 | 可视化如何变形 | 用户下一步 |
| ---- | ---- | ---- | ---- | ---- |
| **① 开发态：控制意图** | 写 PyPTO Python 算子 | 我的 loop / unroll / dynamic shape 会形成怎样的执行路径？参数怎么调？ | **Source Flow**：节点=Loop/Branch/Control Region；边=控制关系；Inspector 解释参数与路径 | 调整源码参数，或“编译预览” |
| **② 编译态：控制流变换** | 编译 Python → PyPTO IR，并经过 Pass | 编译器有没有按意图变换？哪个 Pass 改了控制结构？ | **Transformation Flow**：Before / After CFG；节点=IR Control Region / Basic Block；边=branch/back-edge；自动 Diff | 接受变化、调整源码/Pass，或继续 CodeGen |
| **③ CodeGen：生成控制结构** | PyPTO IR → `.pto` + orchestration C++ | 控制结构最终生成了哪些调度路径？ | **Generated Flow**：节点=generated task/dispatch region；边=生成后的调度关系 | 继续运行，或定位异常 CodeGen |
| **④ Runtime：真实执行流** | 算子已在设备上运行 | 编译路径实际怎么跑？哪里等待、哪里没执行、哪条最慢？ | **Runtime Flow**：节点=Task / Scheduler phase / Orchestrator phase；边=dependency / dispatch；叠加时间 | 回到对应源码 Loop / Pass 调整 |

完整的 end-to-end 闭环：写代码 → 理解控制意图 → 编译 → 验证 Transformation → CodeGen → 验证生成路径 → Run → 观察真实执行流 → 回到源代码修改。

---

### 03. 第一阶段：开发态不是 CFG，而是 Source Control Flow

PyPTO 源码已经天然提供了这一层信息。例如：

```python
for _ in pypto.loop(0, 1, 1, name="LOOP_RESHAPE", idx_name="dummy"):
    ...

for tIdx, unrollLength in pypto.loop_unroll(
    0, t, 1,
    name="IndexerPrologQuantQuantLoop",
    unroll_list=unroll_list
):
    ...
```

这里用户写的不是普通 Python loop，而是在声明一种**编译控制意图**。选中 `loop_unroll` 时，可直接沿用 LLVM-FLOW 的 node-edge 语言：

```text
                  Dynamic t
                     │
                     ▼
        IndexerPrologQuantQuantLoop
                     │
              loop_unroll
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Full path   Tail path   ...
```

此时节点不是 `%17` Basic Block，而是 **Semantic Control Region**。Inspector：

```text
IndexerPrologQuantQuantLoop · 循环展开

作用       沿动态维度 t 执行 Query / Key / Weight 计算
范围       0 → t，步长 1
展开策略   unroll_list = [...]
动态依赖   t = x_in.shape[0]

可能影响
• 生成多少种执行路径
• 尾部数据如何处理
• 循环控制开销
• 生成代码规模

下一步   [模拟 Shape] [调整展开策略] [编译预览]
```

这一阶段解决的是：**我写的控制逻辑意味着什么？** 还不需要编译。

---

### 04. Dynamic Shape 应该直接作用在 Flow 上

源码把多个 Tensor 第 0 维标记为 dynamic，`t = x_in.shape[0]` 直接成为 `loop_unroll` 的 upper bound。产品可让开发者在源码阶段直接输入不同 shape，Flow 随之变化：

```text
t = 32                      t = 33
Loop                        Loop
 └── Full Path              ├── Full Path
                            └── Tail Path

[ 0 ─────────────── 32 ][33]
        Full             Tail
```

这时可视化不只是“解释 loop API”，而成为**控制逻辑模拟器**，可回答：哪些 shape 会触发额外路径？`unroll_list` 是否覆盖关键 shape？哪些输入会走 tail？

---

### 05. `pass_options` 也应该挂在 Flow 上

源码当前直接设置一组 compiler options：`pypto.set_pass_options(...)`。产品不需要在开发态假装知道这些参数最终一定产生怎样的 IR，而应表达：

```text
Source Flow

Loop A
  │
  │ affected by
  │
[Pass option]
  │
  ▼
Potential compiler transformation
```

Inspector 明确区分：**当前已知**——这个 option 会进入 Compiler Pass Pipeline；**尚未验证**——最终是否改变 Control Flow。并提供 `[编译并查看影响]`，自然引导到第二阶段。

---

### 06. 「编译预览」是整个体验的关键转折点

用户完成源码设置后点击 **编译预览**，后台运行 PyPTO compile 并打开 `dump_passes`。PyPTO Runtime API 原生支持 dump 每个 PyPTO pass 后的 IR，也支持 dump PTOAS 每个 pass 的 IR。([GitHub][2])

```text
Source Intent
      ↓ compile preview
Initial IR → Pass 01 → IR 01 → Pass 02 → IR 02 → Pass 03 → IR 03
```

此时 LLVM-FLOW 进入主舞台。

---

### 07. 编译态：Before / After 模型 + Pass Timeline

原 LLVM-FLOW 的核心能力是比较优化前后的 CFG 并匹配相同 Basic Block。([GitHub][3]) PyPTO 版本：

```text
             Pass: Loop Unroll
Before Pass                       After Pass
     %1                               %1
      │                                │
     %2                               %2
    /  \                             / | \
   T    F                           ... ...
```

关键改进：**用户不是随便选两个 IR 比较，而是沿 Pass Timeline 查看 transformation。**

```text
Pass Timeline
01 Normalize
      │
02 Loop Unroll        ← 当前
      │
03 SSA
      │
04 Tiling
      │
05 Memory ...
```

点击 `Loop Unroll`：`Before Loop Unroll  ↔  After Loop Unroll`

---

### 08. Pass 对比的价值：形成诊断链

不是为了告诉用户“6 个 block 变成了 9 个”，而是形成诊断链：

> 源码意图 `loop_unroll(...)` → Loop Unroll Pass → 控制流变化（1 loop → 多条执行路径）→ 判断（符合预期 / 与源码配置不符）

| 用户问题 | Pass Diff 怎么回答 |
| ---- | ---- |
| **我的 `loop_unroll` 生效了吗？** | 找到 Loop Unroll Pass，直接看 Before/After |
| **这条额外路径从哪来的？** | 找到第一次新增该 control region 的 Pass |
| **为什么最终 controlFlow 很复杂？** | 沿 Pass Timeline 找到结构开始膨胀的位置 |
| **修改参数到底改变了什么？** | A/B 编译后比较对应 Pass 输出 |

价值：**把“最终结果不对”变成“第 N 个 Pass 开始和我的意图不同”。**

---

### 09. Inspector 必须建立 Source ↔ Pass 的解释

选中编译后新增的节点：

```text
%17 · 新增控制块

身份      Loop Unroll 生成的控制块
来源      IndexerPrologQuantQuantLoop · Python 252–253
变化      优化前不存在 → Loop Unroll 后新增
生成原因  由当前循环展开策略产生
影响      增加一条执行路径，用于处理特定 iteration range

下一步   [定位源码] [查看展开配置] [查看下一处变化]
技术详情  LLVM / PyPTO IR ...
```

核心：**用户永远不应只看到一个孤立的 `%17`**，它必须能回答“你是从我哪段 Python 来的？”。

---

### 10. CodeGen：从「Transformation Flow」变成「Generated Flow」

PyPTO CodeGen 官方设计强调 **IR → Generated Code 严格 1:1 mapping**，优化和结构 transformation 应在之前的 Pass 完成。([GitHub][4]) 因此 CodeGen 阶段不再问“编译器为什么优化”，而问：**这个已经确定的 IR 控制结构，最终生成了什么？**

PyPTO 有两条 CodeGen 路：

```text
Optimized PyPTO IR
        │
        ├───────────────┐
        ▼               ▼
     InCore         Orchestration
        │               │
       .pto             C++
        │               │
     AICore            AICPU
```

仍可使用相同图形语言：

```text
Compiler Flow              Generated Flow
Loop Region
    │
    ▼
Control Block   ───────→   Task Submit
                            │
                            ▼
                         AICore Task
                            │
                            ▼
                         Next Task
```

点击两侧建立映射：`IR %17  ↔  Generated orchestration region`

`controlFlow.cpp → CFG` 就在这一层，是整个体验链中的 **CodeGen Evidence View**。它应该回答：**我的 PyPTO control intent 经过 Pass Pipeline 后，最终生成了怎样的 orchestration control flow？**

```text
Python   pypto.loop_unroll(...)
            │
            ▼
Compiler  Loop Unroll
            │
            ▼
Generated  Path 32 → Path 16 → Path 8 → ...
```

用户点击 generated CFG 中一个节点，可一路回溯：

```text
Generated CFG Node  ↑  Compiler Pass  ↑  PyPTO IR  ↑  Python loop_unroll
```

---

### 11. Runtime：第三个 Flow 叫「执行流」

`simpler` Runtime 负责 task dependency graph execution，以及 Host ↔ AICPU ↔ AICore 协调。([GitHub][1]) PyPTO Runner 已可采集 chip swimlane：

- Level 1：AICore task start/end
- Level 2：增加 AICPU dispatch / finish
- Level 3：增加 scheduler phases
- Level 4：增加 orchestrator phases ([GitHub][2])

Runtime 阶段继续使用 LLVM-FLOW 图形语言，节点再次变形：

```text
Runtime Execution Flow
Task 14 → Dispatch → AICore Task → Task 15
```

边不再是 LLVM branch，而是 `dependency / dispatch / completion`，并把时间编码进去：

```text
              12 μs
Task A ─────────────────→ Task B
            dependency

Task A
dispatch ──  wait  ───── execute ── finish
             4.2 μs         17 μs
```

Runtime Flow 最重要的不是“看 Timeline”，而是重新连回前两个阶段：

```text
Source                Compile                 Runtime
loop_unroll
    │
    ▼
Loop Region
                      %17
                       │
                       ▼
                    %24 / %31
                                                Task 37
                                                  │
                                                  ▼
                                                Task 41
```

点击 Runtime Task 41，Inspector：

```text
Task 41 · AICore

运行状态   执行 18.3 μs / 等待 4.1 μs
来源       Generated Control Block %31
形成于     Loop Unroll Pass
对应源码   IndexerPrologQuantQuantLoop · line 252
当前发现   该 Tail Path 占本次执行总时间 21%

下一步    [查看编译结构] [定位 loop_unroll]
```

---

### 12. 三个阶段严格共享一套视觉语法

这是整个产品最重要的设计原则。

| Visual Grammar | 开发态 | 编译态 | Runtime |
| ---- | ---- | ---- | ---- |
| **Node** | Loop / Branch / Control Region | Basic Block / IR Control Region | Task / Runtime Phase |
| **Edge** | Source control relation | Branch / Back Edge | Dependency / Dispatch |
| **T/F** | 源码条件成立/不成立 | IR conditional branch | 一般不使用 |
| **Diff** | 参数 A/B | Pass Before/After | 编译预测 / 实际运行 |
| **Inspector** | 意图/参数影响 | Transformation 原因 | 实际执行结果 |
| **Cross-link** | Python line | IR / Pass | Runtime Task |

三个页面看起来应是同一个工具——`SOURCE FLOW → TRANSFORMATION FLOW → EXECUTION FLOW`——而不是三套产品。

---

### 13. 用户真正的 E2E 故事

最终用户旅程：

**Step 1 · 写代码** — 选中 `pypto.loop_unroll(...)`，看到 Source Flow：Loop / Dynamic upper bound t / Unroll configuration / Potential execution paths。用户调整参数。

**Step 2 · 编译预览** — 点击 **编译并查看控制流**，出现 `Source Flow → Compiler Flow`（Loop A → Loop A'）。系统提示 Loop Unroll Pass 新增 5 个控制块，点击查看 Before / After。

**Step 3 · Review Pass**

```text
Before        After
 %12          %12
  │           / \
 %17         ...
```

Inspector：当前变化来自源码第 252 行的 `loop_unroll`，符合当前展开策略。用户 **确认并继续 CodeGen**。

**Step 4 · CodeGen** — `Compiler Flow → Generated Flow`（%17 / %24 → RootStitch / Task）。用户确认 control structure 已正确生成 orchestration path。

**Step 5 · Run** — 同一张图进入 Execution Flow：

```text
Task A   8 μs
   ↓
Task B   wait 4 μs
   ↓
Task C   19 μs
```

Task B 被标记 **运行时异常等待**，点击一路反查：`Task B ↑ Generated Block ↑ Loop Unroll Pass ↑ Python line 252`，最终 Action：**回到展开策略**。

---

### 14. 产品真正的创新点

不是“把 Loop、CFG、Runtime 都画出来”，而是让同一个控制语义拥有一条连续 identity：

```text
Source Control Intent → Compiler Transformation → Generated Control Flow → Runtime Execution
```

因此用户始终可以回答：**我写的这条控制逻辑，编译器把它变成什么了，它最终又是怎么执行的？**

---

[1]: https://github.com/hw-native-sys/pypto/blob/main/docs/en/dev/00-ecosystem.md?utm_source=chatgpt.com "pypto/docs/en/dev/00-ecosystem.md at main · hw-native-sys/pypto · GitHub"
[2]: https://github.com/hw-native-sys/pypto/blob/main/python/pypto/runtime/runner.py?utm_source=chatgpt.com "pypto/python/pypto/runtime/runner.py at main · hw-native-sys/pypto · GitHub"
[3]: https://github.com/kc-ml2/llvm-flow?utm_source=chatgpt.com "GitHub - kc-ml2/llvm-flow: An open-source interactive visualization tool for comparing IR CFGs · GitHub"
[4]: https://github.com/hw-native-sys/pypto/blob/main/docs/en/dev/codegen/00-pto_codegen.md?utm_source=chatgpt.com "pypto/docs/en/dev/codegen/00-pto_codegen.md at main · hw-native-sys/pypto · GitHub"
