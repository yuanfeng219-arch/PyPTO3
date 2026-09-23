# Operator Lab Design QA

## Comparison target

- Source visual truth: the three selected ImageGen directions, used as a composite interaction model rather than a single pixel-perfect screen:
  - `C:/Users/cyf12/.codex/generated_images/01a060d7-8640-76e0-8b9f-95d1fee57191/exec-67f13291-90a3-4546-aba8-5caf8763c7eb.png` — trace-first debugging
  - `C:/Users/cyf12/.codex/generated_images/01a060d7-8640-76e0-8b9f-95d1fee57191/exec-9025f4dd-022d-4b24-982e-9609d40cfd0d.png` — baseline/candidate tuning
  - `C:/Users/cyf12/.codex/generated_images/01a060d7-8640-76e0-8b9f-95d1fee57191/exec-c73bf11f-5b9a-4ce5-8c20-8b0673a71c49.png` — evidence canvas analysis
- Source pixels: 1487 × 1058 each; generated at the requested desktop 1440 × 1024 class.
- Implementation: `http://127.0.0.1:8765/Design/operator-debug-tuning-lab/`
- Browser screenshots: [qa-overview.png](D:/project/PyPTO3/Design/operator-debug-tuning-lab/qa-overview.png), [qa-compare.png](D:/project/PyPTO3/Design/operator-debug-tuning-lab/qa-compare.png), [qa-analysis.png](D:/project/PyPTO3/Design/operator-debug-tuning-lab/qa-analysis.png); each is 1265 × 712 CSS pixels at the in-app browser viewport; device scale was not overridden (1x screenshot normalization).
- Tested states: execution overview, Run comparison, divergence analysis.

## Review

The implementation intentionally composes the three visual directions into a single workflow. The overview is the default post-run surface, comparison is entered from the Run workflow, and analysis is entered from the correctness gate or the abnormal-analysis nav item. This matches the requested context transition and avoids putting all evidence on the first screen. The richer pass now includes a Task trace slice, AIC/AIV occupancy, pass snapshot sizing, 20-run performance distribution, generated-code diff, source-line focus, and Tensor slice comparison.

### Required fidelity surfaces

- Fonts and typography: compact system UI with monospace values for IDs, hashes and generated-code previews. Hierarchy, weights, and small labels are consistent across all three workspaces.
- Spacing and layout rhythm: persistent left navigation, 3-column debug workspace, 2-column tuning workspace, and stacked overview cards preserve the source concepts while adapting to a 1265px desktop viewport.
- Colors and visual tokens: graphite surfaces with cyan primary actions, violet relationships, green validated states, amber warnings, and red divergence states are applied consistently.
- Image quality and asset fidelity: the source concepts contain no required photographic/illustrative assets. The implementation uses UI-native text and controls; no placeholder imagery is needed.
- Copy and content: workflow copy references the observed Run structure (`passes_dump`, `ptoas`, `kernels`, `dfx_outputs`, `perf_hints`, `Task 142`, and `kernel_aiv_07.cpp`) and keeps debug/tuning actions explicit.
- Data provenance: counts, callable names, file groups, performance hint type, and memory-space warning are grounded in `Data/_jit_decode_fwd_layers_20260625_184941`; Baseline/Candidate measurements and Tensor diff values are prototype mock data derived from that shape.

### Primary interactions tested

- `运行正确性检查` changes the overview from pending validation to `发现 1 处首个偏差` and updates the correctness gate.
- `对比 Runs` switches to Baseline/Candidate metrics and generated artifact diff tabs.
- `异常分析` switches to the evidence canvas; selecting Tensor, Task, Kernel, Memory, Artifact, and Trace nodes updates the inspector.
- `执行证据浏览器` switches between Task trace, Kernel lanes, and Pass snapshots; Task/Kernel filtering and row selection are interactive.
- `性能分布` switches between Latency and Throughput views with different candidate/baseline series and deltas.
- `异常现场详情` exposes Source, Tensor diff, and Time window tabs alongside the evidence canvas.
- Primary actions show clear confirmation toasts for replay, promote, rollback, export, pin, and reproduction actions.

### Findings

No actionable P0/P1/P2 findings remain. The three source screens are intentionally reinterpreted as contextual workspaces instead of copied as separate routes. Remaining differences are P3 polish: the generated concepts use more simulated micro-chart detail than the first demo, while this prototype prioritizes clickable task transitions and real artifact terminology.

## Implementation checklist

- [x] Default Run overview is the post-execution landing state.
- [x] Correctness gate can transition into an explicit first-divergence state.
- [x] Run comparison keeps Baseline/Candidate evidence and a correctness gate visible.
- [x] Divergence analysis links Tensor → Task → Kernel → artifacts → runtime evidence.
- [x] Responsive layout is provided for narrow viewports.
- [x] Browser-rendered states captured and console errors checked (none).

## Follow-up polish

- Add real artifact file loading and syntax highlighting when the product moves from prototype to implementation.
- Add a saved Run picker and a real candidate creation flow.

final result: passed
