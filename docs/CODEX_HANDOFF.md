# Toolkit Studio Numerical Accuracy Handoff

## 1. 当前目标

完成 Toolkit Studio Run 视图中的 Numerical Accuracy 调试体验。Correctness 是一级概念；Numerical Accuracy 是 finding subtype。当前用两个 UX fixture 演示两条互斥的定位路径：

- Run #106：运行时数据 / 排序问题。数值校验逐 Pass 通过，设备输出不一致；从首个分歧 Tensor 进入 Execution。
- Run #109：编译语义问题。Host IR 数值校验在 `ExpandMixedKernel` 首次分歧，尚未进入设备执行；从首个数值分歧 Pass 进入 Compilation。

这是 UX Demo。数据、任务、Pass 和指标只表达交互与信息层级，不可作为 PyPTO 工程事实。

## 2. 已完成的修改

- 新增集中式 Correctness profile：`run_106` 与 `run_109`，避免 renderer 内散落 mock 状态。
- Correctness 顶部加入紧凑的 Numerical Accuracy gates：Reference、Tolerance、Expected Difference、Structural Verification、Numerical Validation、Device Result。
- Correctness Result 区域统一为四维诊断定位：`Symptom × Failure Mode × Cause Domain × First Divergence`。
- #106 表达为 `输出不一致 / Data / Runtime / Dataflow / Tensor · attention_out / T37`，保留语义图、runtime expansion、timeline、tooltip、#182/#197 overlap 和 Execution CTA。
- #109 表达为 `输出不一致 / Semantic / Compiler / Pass · ExpandMixedKernel`，隐藏 #106 专属 runtime graph/timeline，展示 Compilation CTA。
- 新增 `deviceResult` 字段，使整体 Correctness FAIL 与设备结果解耦：#106 为 `MISMATCH`，#109 为 `NOT EVALUATED`。
- 用户可见术语已统一为「首个数值分歧 Pass」。
- #106 的 Correctness CTA 可进入 Execution，保留 Run #106，定位 #182/#197 overlap 证据并显示轻量 context strip。
- Compilation 已有 Numerical Validation 展示、Pass detail 和 Before/After IR diff；#109 直接复用 `compiler_semantic_error` fixture，在 `ExpandMixedKernel` 自动展开。
- #109 从 Correctness CTA 和直接点击 Compilation 都走现有 `PTO_COMPILATION` renderer；workspace 复用现有 `PTO_IR_KERNELS`、`PTO_IR_PIPELINE`、`PTO_DECODE_LAYER_SOURCE`，不复制 42 Pass 或 45 Kernel。
- Overview 为 #106/#109 显示 Numerical finding；#106 → #107 对比已增加正确性恢复叙事。

## 3. 尚未完成的问题

- 两条 #109 Compilation 入口尚未完成实际浏览器点击 smoke check；语法检查已通过。
- 浏览器自动化连接曾连续超时/重置，无法验证视觉、焦点滚动和 CTA 点击。
- 若后续接入真实 artifact，需要定义 profile adapter；当前不要从 demo fixture 推断真实 validate_ir 或 runtime 行为。

## 4. 关键文件及职责

| 文件 | 职责 |
| --- | --- |
| `Design/Toolkit Studio/js/correctness-diagnostic-data.js` | 唯一的 Correctness demo profile 数据源；`run_106`、`run_109`、`deviceResult`、diagnosis 四维字段。|
| `Design/Toolkit Studio/js/task-history.js` | Run History、Overview、Correctness 页面渲染、Run 间导航、`PTO_RUN_CONTEXT` 组装；包含 #106/#109 的轻量接线。|
| `Design/Toolkit Studio/js/ir-compilation-view.js` | 现有 Compilation workspace renderer、Numerical Validation fixture、Pass detail、IR diff 和 first-divergence 自动 focus。|
| `Design/Toolkit Studio/styles/diagnostic-semantic-trace.css` | Correctness gates、Result 与诊断定位区的视觉样式。|
| `Design/Toolkit Studio/styles/task-history.css` | Overview numerical finding、Correctness comparison 等 Run History 样式。|
| `Design/Toolkit Studio/index_v2-slz.html` | 脚本加载顺序与 cache-busting 版本；`ir-compilation-view.js` 必须先于 correctness data 加载。|

## 5. 当前架构与数据流

```text
Run selection
  -> task-history.js: diagnosisViewForRun(runId)
  -> PTO_CORRECTNESS_DIAGNOSTICS profile
  -> Correctness gates + 四维诊断定位

#106: runtime_data_error
  -> Execution entry / timeline focus

#109: compiler_semantic_error
  -> correctnessCompilationEntry()
  -> PTO_RUN_CONTEXT.numericalValidation
  -> PTO_COMPILATION.setNumericalContext()
  -> ExpandMixedKernel 自动展开，复用现有 Pass detail / IR diff
```

`renderCompilationTab(panel, r)` 是 #109 的关键入口：当 `r.id === 'run_109'`，它从 `PTO_CORRECTNESS_DIAGNOSTICS` 取 `compiler.numericalValidation`，注入现有 Compilation renderer。它只覆盖数值诊断语义；Compilation 的 Kernel、Source、pipeline 和 IR 内容仍来自已加载的真实 workspace 数据。`PTO_RUN_CONTEXT.runId` 保持 `run_109`，页面不应显示 source artifact timestamp 作为当前 Run。

## 6. 重要设计决策

- Correctness Finding 固定为四个正交维度：Symptom、Failure Mode、Cause Domain、First Divergence。`compiler_semantic_error` 不是和 Numerical Accuracy 并列的类型，而是 #109 的 `Semantic` failure mode。
- `Device Result` 只能读 `profile.deviceResult`，不能从 `result.verdict` 推导。#109 在 Host IR validation 已定位，设备执行状态必须是 `NOT EVALUATED`。
- Structural Verification 与 Numerical Validation 严格分开：前者表示 IR 合法；后者表示当前 Pass 后 Host IR 结果与 Golden 的数值一致性。
- 不新增 Precision tab、第二个 Pass viewer、第三张 graph、通用 router 或 data framework。通过现有 tab、selection、`PTO_RUN_CONTEXT` 和 Compilation renderer 串接。
- #109 不复制 Compilation data。复用现有 artifact workspace，只以 numerical fixture 覆盖 Pass MATCH/MISMATCH 语义。
- #106 与 #109 的主路径分别落到 Execution 与 Compilation，避免把 runtime ordering 图硬套给 compiler semantic case。

## 7. 不能破坏的行为

- UI 一级标题始终是「正确性」；两案 subtitle 都是「数值精度（Numerical Accuracy）」。
- #106：Structural PASS、Numerical Validation `42 / 42 Pass`、Device Result `MISMATCH`；首个分歧 Tensor 为 `attention_out / T37`；#182/#197 排序证据、semantic graph、tooltip、runtime timeline 和 Execution CTA 必须继续可用。
- #109：Structural PASS、Numerical Validation FAIL、Device Result `NOT EVALUATED`；首个数值分歧 Pass 为 `ExpandMixedKernel`；前序 Pass MATCH、该 Pass FIRST DIVERGENCE、后续 Pass MISMATCH；Compilation 自动 focus 并仍能查看 Source、Pass detail 和 IR diff。
- #109 两条入口都必须完整渲染：直接点击 Compilation；Correctness CTA「在「编译」中查看首个数值分歧 Pass」。
- #106/#107 Overview、Run Compare、Compilation/Execution 主视图不能回归；真实 Run 的 Compilation 不能自动套用 #109 fixture。
- 可见文案不得出现「首个异常 Pass」或「异常 Tensor」。

## 8. 已知 bug / technical debt

- `run_109` 是有意的 special case，位于 `renderCompilationTab()` 和历史 Run 的 Compilation 分支；这是 UX demo 的最小接线，尚未抽象成通用 artifact fallback。
- `compilerSemanticError109.compiler.numericalValidation` 在脚本初始化时从 `window.PTO_COMPILATION.numericalFixtures` 获取。依赖 HTML 现有加载顺序；若变更脚本顺序，#109 会退化为 `not_collected` fallback。
- #109 的 workspace 填充数据与其 numerical story 并不要求完全同源。展示时数值 fixture 优先，其他 Kernel/Source/IR 内容来自复用 workspace。
- 当前 source file 的顶部注释仍可能称 numerical fixtures「不会自动贴到 Run 上」；#109 已成为例外，若继续维护该能力应同步改正该注释。
- 浏览器自动化在本环境中超时，无法确认 `requestAnimationFrame` 后的 focus/scroll 和视觉布局。

## 9. 下一步建议（按优先级）

1. 恢复可用浏览器环境后，手动/自动 smoke 两条 #109 Compilation 入口，并检查 Run 标题仍为 #109、`ExpandMixedKernel` 展开、context strip 文案和 #106 回归。
2. 修正 `ir-compilation-view.js` 中 fixtures 不自动贴 Run 的过时注释，明确 #109 是 UX fixture exception。
3. 检查 `index_v2-slz.html` 的脚本版本和顺序，确保 `ir-compilation-view.js` 早于 `correctness-diagnostic-data.js`。
4. 若继续扩案例，先扩 profile 字段和 renderer 分支；保持同一四维诊断表达，不建立新的 diagnosis taxonomy 页面。
5. 只有真实 backend artifact 确定后，再设计 profile adapter，把 DFX/pass-validation/runtime evidence 映射进当前 schema。

## 10. 启动、测试、build 命令

Toolkit Studio 是静态页面。进入仓库后可启动本地静态服务：

```bash
cd /Users/songchenfei/Documents/pypto项目/PyPTO3
python3 -m http.server 4173
```

打开 `http://127.0.0.1:4173/Design/Toolkit%20Studio/index_v2-slz.html`，验证 #106/#109 路径。

本次 JavaScript 最小检查：

```bash
node --check 'Design/Toolkit Studio/js/task-history.js'
node --check 'Design/Toolkit Studio/js/correctness-diagnostic-data.js'
node --check 'Design/Toolkit Studio/js/ir-compilation-view.js'
git diff --check
```

完整 PyPTO 构建与测试（仅在改动核心工程代码时执行）：

```bash
pip install -e ".[dev]"
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python -m pytest tests/ut/ -n auto --maxprocesses 8 -v
```
