# Explore ↔ Design 关联审查

仅表达设计依据与约束；不是实验、执行时序或逐步性能测量。

| Source | Target | Relation | Evidence |
|---|---|---|---|
| d-kb | d-vf | informs | EXPLORE_REPORT.md §4.2 / §5.1；DESIGN.md §1 Vector 选择；KB_SELECTION.json optional_patterns: patterns/vec-row-reduce-broadcast.md；DESIGN_BINDINGS.json patterns/vec-row-reduce-broadcast.md / R01 / R03；KB_USAGE.json invariants[reference=patterns/vec-row-reduce-broadcast.md] |
| d-kb | d-tiling | informs | EXPLORE_REPORT.md §4.2 / §5.1；DESIGN.md §0 归约轴容量结论 / §2；KB_SELECTION.json optional_patterns: patterns/vec-row-reduce-broadcast.md；DESIGN_BINDINGS.json patterns/vec-row-reduce-broadcast.md / R01 / R03；KB_USAGE.json invariants[reference=patterns/vec-row-reduce-broadcast.md] |
| d-kb | d-tiling | evaluated_not_triggered | EXPLORE_REPORT.md §4.2 / §5.1；DESIGN.md §0 归约轴容量结论；KB_SELECTION.json optional_patterns: patterns/online-softmax-tail.md；DESIGN_BINDINGS.json patterns/online-softmax-tail.md / R04；KB_USAGE.json invariants[reference=patterns/online-softmax-tail.md] |
| d-kb | d-buffer | informs | EXPLORE_REPORT.md §4.2 / §5.1；DESIGN.md §2–§3 / §6；KB_SELECTION.json optional_patterns: patterns/buffer-reuse-lifetime.md；DESIGN_BINDINGS.json patterns/buffer-reuse-lifetime.md / R01 / R02 / R03；KB_USAGE.json invariants[reference=patterns/buffer-reuse-lifetime.md] |
| d-kb | d-precision | informs | EXPLORE_REPORT.md §4.2 / §5.1；DESIGN.md §1 Vector 选择；KB_SELECTION.json optional_patterns: patterns/online-softmax-tail.md；DESIGN_BINDINGS.json patterns/online-softmax-tail.md / R02；KB_USAGE.json invariants[reference=patterns/online-softmax-tail.md] |
| d-api | d-vf | informs | EXPLORE_REPORT.md §3.3 / §4.2–4.3；DESIGN.md §1 API 映射 |
| d-api | d-precision | constrains | EXPLORE_REPORT.md §3.3 / §4.2–4.3；DESIGN.md §1 Vector 选择 |
| d-api | d-tiling | constrains | EXPLORE_REPORT.md §3.3 / §4.2–4.3；DESIGN.md §2 / §7；KB_SELECTION.json optional_patterns: patterns/vec-row-reduce-broadcast.md；DESIGN_BINDINGS.json patterns/vec-row-reduce-broadcast.md / R02 / R03；KB_USAGE.json invariants[reference=patterns/vec-row-reduce-broadcast.md] |
| d-constraint | d-tiling | constrains | EXPLORE_REPORT.md §5.2 / §6.1 / §7；DESIGN.md §2–§3 |
| d-constraint | d-buffer | constrains | EXPLORE_REPORT.md §5.2 / §6.1 / §7；DESIGN.md §3 |
| d-constraint | d-schedule | constrains | EXPLORE_REPORT.md §5.2 / §6.1 / §7；DESIGN.md §4–§5 |

九条待核对关系均有可溯源依据，因此没有因证据不足删除其中任何一条。API→Single Tile 仅限定寄存器/数据对齐与尾 Mask，不声称单靠 API 推导 TILE_ROWS=3。另补充知识→精度及知识→Tile 的 not_triggered 关系，共 11 条语义关系、10 对节点。

不建立：知识检索→所有 Pattern 已执行、双缓冲→有效流水重叠、硬件约束→固定实际核数、未触发流式归约→编译失败等无证据推断。

坐标复用 FlowGeometry.itemBox；在两张卡片边缘设置与子项对应的右/左 Port，仅在卡片间隙绘制单调贝塞尔曲线。同节点对合并连线，解释和 Evidence 仍分别保留。

Pointer enter 与键盘 Focus 使用同一临时预览；leave、blur、Escape、重新绘制清理。点击后 Inspector 的“Explore ↔ Design”可展开查看完整证据。原始选中与滚动位置不由 Hover 修改。

浏览器验证记录：previews/relations-validation.json；截图前缀 relations-，包含两个桌面视口。沿用现有 IDE Frame 与深色状态语言，未引入页面框架或共享 Design System 修改。
