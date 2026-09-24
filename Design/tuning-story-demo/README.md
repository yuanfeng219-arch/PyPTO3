# 调优故事九幕 · Tuning Story

把 DeepSeek-V4 解码压缩链路的**一次真实性能调优**做成可走查的产品 demo：九个场景，每个场景回答同一组问题——
用户在做什么、看什么数据、卡在哪里、产品应该提供什么。

打开入口：`Design/tuning-story-demo/index.html`，也在 `launch.html` 的「内存与性能」分类里。
本地打开需要经过 http 服务（相对引用了 `vendor/pto-design-system`），例如 `.claude/launch.json` 里的 `design-static`。

---

## 与算子调优控制台的关系

| | 算子调优控制台 V1 / V2 | 本 demo |
| --- | --- | --- |
| 工作单元 | 一条瓶颈条目 / 一个 Investigation | 一个**场景**（用户在旅程中的一个位置） |
| 回答 | 这个瓶颈怎么查、怎么验 | 整条旅程里，每一步**卡在哪里、缺什么能力** |
| 数据 | `operator-tuning-console/data.js`（构建器产物） | 本目录 `story-data.js`（人工整理，附出处） |

两者互不依赖，也不共享文件。控制台演示的是「做」，这里演示的是「为什么难做」，用于体验评审与设计对齐。

---

## 九个场景

| 阶段 | 场景 | 关键问题 |
| --- | --- | --- |
| 准备 | T1 建任务、立基线 | 现在有多快？怎么测才算数？ |
| 定位 | T2 整图定位热点 | 时间花在哪个 scope？哪段时间核在空转？ |
| 定位 | T3 诊断调度与依赖 | 是派发开销，还是依赖把并行变成了串行？ |
| 归因与修改 | T4 修正源码结构 | 是哪一行写法造成的？怎么改？ |
| 归因与修改 | T5 核对编译过程 | 我写的意图生效了吗？编译器提示了什么？ |
| 归因与修改 | T6 硬件资源预算 | 再加大一档放得下吗？数据是怎么搬的？ |
| 归因与修改 | T7 核内下钻 | 单个 task 的时间花在搬运还是计算上？ |
| 验证 | T8 实验与验证 | 真的更快吗？是噪声吗？精度对吗？ |
| 交付 | T9 交付与沉淀 | 改动会影响谁？经验在什么条件下成立？ |

每个场景有 2–3 个视图（共 20 个），例如 T5 的「意图核对 / 五层对照 / 提示分诊」。

## 界面分区

| 区域 | 内容 |
| --- | --- |
| 顶栏 | case chip + 阶段进度（五段、九点，当前场景高亮） |
| Explorer | 按阶段分组的九个场景，每条带关键问题 |
| 中心 | 场景视图，用 tab 切换 |
| Inspector | 关键问题 / 用户在做什么 / 卡点 / 机会点（P0–P2）/ 对象与原型 / 数据来源（带 fact·infer 角标） |
| 底部 Dock | Visualization：整图泳道总览，当前场景的聚焦 scope 高亮、其余压暗；Terminal：该场景真实的命令与输出片段 |
| 播放条 | 逐场景播放（4.2s / 场景）、上一步 / 下一步 / 重播 / 进度条 |
| 状态条 | case、rank、wall、AIC/AIV 占用、scope 数、pass 数、提示数、当前情绪标签 |

键盘：`←` `→` 切换场景。右上角可切 light / dark。

---

## 数据来源

所有数字来自两处真实材料，没有为了演示而改写：

**本地上板产物**
- `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/` — 2026-09-03 DSpark `decode_csa` 构建：
  泳道（66 个 scope、wall 4879.8 µs、AIC 32.2% / AIV 37.5%）、`deps.json`（86 task / 138 张量 / 338 边）、
  52 个 pass dump、124 个 `.pto`、230 条性能提示、STRACE 分层耗时、`debug/run.py`。
- `Data/_jit_decode_fwd_layers_20260625_184941/report/memory_after_AllocateMemoryAddr.txt` — 内存分配报告
  （Vec 上限 184 KB、Mat 512 KB、Right 64 KB，含每个 buffer 的地址与生命周期）。

**公开记录**
- pypto-lib [#314](https://github.com/hw-native-sys/pypto-lib/issues/314)（基线与分步调优）、
  [#628](https://github.com/hw-native-sys/pypto-lib/pull/628) / [#641](https://github.com/hw-native-sys/pypto-lib/pull/641)（已合入的两个 perf PR）、
  pypto [#2309](https://github.com/hw-native-sys/pypto/issues/2309)（b_trans 的反例）。
- 社区泳道调优日志 §2 §6 §8 §14 §19 §20（索引见 pypto-lib [#828](https://github.com/hw-native-sys/pypto-lib/issues/828)）。

Inspector 的「数据来源」区对每条证据标注可信度：`fact`（实测或公开记录）/ `infer`（多源推断）。

### 边界

- 九个场景来自**同一模型的不同文件与时间点（5–9 月）**，不是一次连续开发；本地数据（9 月）晚于多数调优（6 月）。
- 性能数字绑定 a2a3（910B）、特定 shape 与版本，只用于说明方法，不构成性能结论。
- op-sim（`*.clean.json`）与 PMU（`pmu.csv`）本地没有样本，T7 的数值引自公开记录。
- 「机会点」是产品建议，不是已有能力。

---

## 设计系统合规

- **Shell**：`patterns/ide-frame`，`data-host="standalone"`，继承共享渐变、80% 半透明 pane 与 backdrop blur，未做页面私有替换。
- **消费的 pattern**：`workbench-shell`（拖拽分屏，由 ide-frame 委托）、`floating-playback-control`（播放条，
  经 `data-ide-floating-playback` 挂载；页面只在同一批控件上追加场景切换逻辑，未复刻外壳）、
  `swimlane-task`（泳道任务条与 hover 提示，调用 `drawTaskBar` / `createTaskColormap` / `initHoverTooltip`）。
- **组件**：`btn` / `btn-ghost` / `btn-icon`、`tab-control-item`、`stat-chip`、`inspector-section*`、`panel` 相关类均来自 `css/style.css`。
- **`styles.css` 只含页面级布局**，颜色一律引用 token；未新建按钮、badge、card 体系，未保留边框型容器装饰。
- **字号**：正文 14px（`--type-body`），密集 UI ≥ 12px，11px 仅用于 micro label（泳道轨道名、优先级徽标、来源角标、状态条）。
  `node scripts/audit-typography.mjs` 在设计系统根目录通过。
- **data-viz 例外**：泳道任务条颜色来自共享 `createTaskColormap`；条形图的 low/mid/high 使用 `--danger` / `--warning` / `--success`，属于数据编码。

## 文件

```
index.html       ide-frame 外壳与 pane 结构
story-data.js    九个场景的数据与出处（唯一需要改数据的地方）
story.js         场景渲染、播放条接线、泳道绘制
styles.css       页面级布局（不含视觉语言）
```
