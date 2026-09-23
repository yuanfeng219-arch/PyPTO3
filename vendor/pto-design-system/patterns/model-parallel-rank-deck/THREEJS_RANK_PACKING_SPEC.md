# Model Parallel Rank Three.js Packing Spec

Status: Implemented  
Supersedes: `../model-architecture-3d-deck/CSS3D_RANK_PACKING_SPEC.md` for the combined 128-Rank view only  
Canonical model source: `../model-architecture-3d-deck/pattern.js`  
Canonical parallel planner: `./pattern.js`

## 1. Product outcome

Render 128 Rank compute containers as one ordered row before a parallel dimension is selected. Once PP, TP, EP, or EDP is selected, reveal the complete openPangu payload inside its assigned Rank containers and keep the combined view interactive in one Three.js WebGL scene. DOM is limited to controls, Inspector, tooltip, accessibility text, and other product chrome.

The model is partitioned by PP, TP, EP, CP, and EDP. A Rank is a transparent compute container, not a replacement for the model. Every physical Layer assigned by the planner remains a complete Layer payload.

## 2. Non-negotiable source-integrity contract

1. Do not replace a Layer with a glyph, screenshot, texture, summary slab, thumbnail, or representative Layer.
2. Every source Node, Cluster, and Edge emitted by `PtoModelArchitecture3dDeck.renderLayerScene` must have a corresponding semantic record in `ModelSceneSpec`.
3. Node IDs, labels, operator kinds, layer variants, source geometry, edge kinds, descriptions, and ownership metadata must not be renamed, merged, or omitted by the Three.js compiler.
4. Overview LOD may suppress text and reduce opacity. It must not delete a semantic record or discard its GPU instance.
5. Selecting a Rank and Layer must expose the complete original inventory in the Inspector and make every source node individually selectable.
6. Dense, MoE, DSA, SWA, block-post, Input, Output, and MTP variants are separate source structures and must remain distinguishable.
7. The frozen source-integrity test for `model-architecture-3d-deck` remains blocking. The Three.js migration must not update its hashes.

## 3. Runtime architecture

```text
PtoModelArchitecture3dDeck public renderer -> ModelSceneSpec
ParallelPlanner / RankManifest             -> PlacementSpec
ModelSceneSpec + PlacementSpec             -> ThreeSceneCompiler
ThreeSceneCompiler                         -> GPU batches + SceneIndex
SceneIndex                                 -> Picking + Inspector
```

There is one renderer, one scene, and one camera. Creating or cloning one Three.js Group per Rank is forbidden.

## 4. ModelSceneSpec

`ModelSceneSpec` is renderer-neutral and complete. It contains:

- Model configuration and source level.
- One template for every decoder Layer.
- Input and Output/MTP static templates.
- Node records: stable ID, label, op kind, source rectangle, layer/static role, source metadata.
- Cluster records: stable ID, label, source rectangle, semantic kind.
- Edge records: stable ID, source, target, edge kind, and full sampled source path.
- Source counts and integrity diagnostics.

The browser compiler may obtain these records by reading the canonical public renderer into detached DOM. Detached DOM must be released after compilation and must never become the combined scene renderer.

## 5. Parallel placement

Default demo topology:

- World Size: 128
- TP: 2
- PP: 4
- CP: 1
- EP: 8
- EDP: 2
- Pipeline ranges: L0–L11, L12–L22, L23–L34, L35–L45
- Routed experts: 256; 32 experts per EP shard

The mapping is demo/derived unless an externally validated topology is supplied. UI copy must not describe it as the official openPangu deployment topology.

The ungrouped state orders Rank 0–127 on one X-axis row. Grouped layouts reserve X for PP stage order. Layer planes occupy YZ and overlap along X, so PP0–PP3 and their assigned Layer ranges share one depth axis and combine into a whole-model deck in side view. The default camera starts from an isometric pose and uses a 16° long-lens perspective projection, preserving the diagram-like composition while retaining subtle depth foreshortening.

## 6. GPU batch contract

- Rank bodies: one `InstancedMesh`.
- Rank outlines: one colored `LineSegments` buffer.
- Layer cards: one `InstancedMesh`.
- Clusters: one or a small fixed number of `InstancedMesh` batches.
- Operators: one `InstancedMesh` per geometry/material semantic family, never one Mesh per operator.
- Edges: one `LineSegments` buffer per edge kind.
- Expert marks: instanced and created only from Expert ownership data.
- Text: shared atlas or Inspector DOM; never one canvas texture per duplicated operator.

Each operator instance maps through `SceneIndex` to:

`rank:{rank}/layer:{layer}/node:{nodeId}`

Static objects use `rank:{rank}/static:{role}/node:{nodeId}`.

## 7. LOD and visibility

Ungrouped overview:

- Render all 128 Rank shells in one row with no visible model payload.
- Keep every Layer, Cluster, Node, and Edge record compiled in the GPU scene and `SceneIndex`; visibility changes must not delete semantic records.
- Selecting any PP, TP, EP, or EDP dimension reveals the complete payload in the resulting grouped layout.

Grouped overview:

- All Rank, Layer, Cluster, Node, and Edge records exist in the GPU scene.
- The first assigned Layer in every Rank keeps its semantic solid colors. Remaining Layers stay structurally complete but use a light-blue, low-opacity ghost treatment so repeated transparency does not accumulate into a dirty gray volume.
- Repeated Layer-card and Cluster fill planes are not painted in overview because alpha-stacking those large surfaces creates a false black volume. Their semantic records remain complete in SceneIndex; nodes and edges remain the visible Layer structure.
- Operator text is suppressed in the 128-Rank overview. Rank focus creates labels only for the first assigned, solid Layer, preserving every original node label without paying the global texture cost. Labels use one fixed black type size with no outline, sit directly on the front node faces, and never billboard toward the camera. Inspector selection must not move these labels onto a translucent rear Layer.
- Rank hover uses a single tooltip.

Parallel dimensions are cumulative rather than mutually exclusive. Active dimensions form one canonical hierarchy in PP → TP → EP → EDP order. Every grouped transform reserves X for PP stage order, matching the Layer deck depth axis. The first active dimension must retain its original single-mode transform exactly. In particular, PP alone produces four adjacent Pipeline Stage groups whose Layer decks align along X. Every newly added dimension must then produce a visible rigid-body Rank transform inside each parent group, not merely add or recolor borders: adding TP pulls the TP0 and TP1 sub-cubes apart inside every PP parent; removing TP returns them to their exact PP-only coordinates. Adding EP keeps the centers and dimensions of all four PP frames unchanged while regrouping their members and creating eight EP child frames inside every PP frame, with four Ranks in each PP × EP leaf. Clicking an active dimension again removes only that hierarchy level and recomputes the remaining layout. Selecting a Rank focuses its concrete communication group on the deepest active axis; non-members retain only a near-background container outline and their payload rendering is suppressed without deleting SceneIndex records.

The default scene uses the exploded spacing and does not expose packed/exploded/enter-Rank toolbar buttons. The former ALL/PP/TP/EP/EDP tab strip is replaced by one centered header panel: it states the current Rank scope and offers multi-select PP, TP, EP, EDP, and disabled CP1 actions under “继续按”. Active actions remain pressed; clicking one a second time removes it. Clicking the current-scope value clears every grouping dimension and restores the complete 128-Rank view. Adding or removing a dimension changes the physical Rank layout, not only color: the 128 Rank shells animate as whole containers while payload batches briefly fade, then the complete payload reappears at the target layout. Every active level receives its own nested full-domain frames.

Selected Rank:

- Non-selected Rank payloads are visually muted.
- Selected Rank remains complete and pickable.
- The first assigned Layer remains solid; every other Layer uses the light-blue ghost treatment at 40% opacity in Rank focus.
- TP-owned operators retain their original label and append the concrete `TP index/count`; their solid node tone is mixed with the corresponding TP shard color.
- EP-owned Expert Pool, Dispatch, and Combine nodes append the concrete EP shard. Expert Pool also exposes the exact owned expert range, such as `E96–E127`, and uses the corresponding EP shard tone.
- EDP replicas intentionally keep identical Layer structure. The fixed Layer payload header exposes `EDP index/count` so replica identity is visible without inventing a false structural difference.
- Inspector lists every Layer assigned to the Rank.

Selected Layer:

- Inspector lists every original node in source order with its original label.
- Selected Layer is emphasized without changing its geometry.
- Node selection shows exact semantic ID, ownership, source role, and description when available.

## 8. Interaction contract

- Left-button drag: continuously orbit through 360° on both axes. The model follows the drag direction, and the camera-up tangent crosses both poles without clamping or snapping.
- Wheel/trackpad: zoom.
- Click Rank: select it and inspect its parallel relationship.
- Click empty canvas space in the global view: clear Rank/group selection and restore the full topology emphasis.
- Double-click Rank: enter Rank focus.
- Click a Layer in Inspector: select Layer.
- Click a visible operator in Rank focus or its Inspector entry: select Node.
- Escape: Node -> Rank focus -> group -> global selection, in that order.
- PP/TP/EP/EDP controls highlight the selected Rank's derived communication group.

## 9. Performance budgets

At the default 128-Rank topology and canonical model:

- Rank draw calls: 1 body batch plus 1 outline batch.
- Total steady-state draw calls: <= 80.
- Initial interactive time at 1440 × 900 on the target Mac: <= 3 seconds.
- Static isometric overview: target 55–60 FPS.
- Orbit/zoom: minimum 45 FPS.
- Rank click to Inspector update: <= 50 ms.
- No continuous render loop while the scene is idle.
- No per-operator DOM. Canvas label textures are created only for the first solid Layer in Rank focus and cached by exact source label and width.

Current CSS3D-derived regression baseline:

- Rank manifests: 128
- Layer placements: 1,472
- Static placements: 64
- Node and Edge counts are derived from the canonical renderer at build time and snapshot in runtime diagnostics. Hard-coded counts may not be used to conceal source changes.

## 10. Acceptance criteria

- AC-01: 128 Rank manifests render in one WebGL canvas.
- AC-02: Every planned Layer placement exists in `SceneIndex`.
- AC-03: Per-template Node/Cluster/Edge counts equal the canonical renderer output.
- AC-04: No `pto-rank-volume` or duplicated Layer DOM exists in the combined viewport.
- AC-05: Dense, ordinary MoE, DSA, SWA, block-post, Input, Output, and MTP samples pass source-inventory comparison.
- AC-06: Selecting any Rank exposes the exact planned layer and expert ranges.
- AC-07: Selecting a Layer exposes every original Node label and Node ID.
- AC-08: Overview operator labels do not create DOM or texture-per-instance cost.
- AC-09: Default view is a 16° long-lens perspective isometric pose with Rank 0–127 ordered in one row and no visible payload until a split is selected.
- AC-10: Source-integrity, planner, scene-integrity, and browser smoke tests pass.

## 11. Migration rule

The CSS3D implementation remains available only as a historical baseline until AC-01 through AC-10 pass. After acceptance, the Three.js renderer is the default `model-parallel-rank-deck` implementation. Do not delete the historical spec or silently update the frozen source hashes.
