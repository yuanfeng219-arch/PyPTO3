/* STAGE2_REVISED_BEGIN — Stage 2-local React extension; no separate workspace. */
(function () {
  "use strict";
  const DATA = /* STAGE2_CASE_DATA */ window.LLVMCfgEvidence.DATA;
  const CFG_PACKED = /* STAGE2_CFG_DATA */ window.LLVMCfgEvidence.CFG_PACKED;
  const B = window.LLVMStage2Bridge;
  const R = B.React, h = R.createElement;
  const S = { caseId: DATA.case.id, sourceFn: "l3_decode_csa", sourceMode: "code", reviewed: false, pass: 2, view: "graph", query: "", filter: "all", inputQuery: "",
    beforeFn: "indexer_topk_group_wave", afterFn: "indexer_topk_group_wave", full: false,
    dual: false, selection: null, returnTo: null, loaded: false, error: "", codeFull: false };
  let sourceLines = [], cfg = null, loading = null;
  const graphCache = new Map();
  const viewCache = new Map();
  const textStatus = { changed: "文本变化", identical: "完全一致", unavailable: "快照缺失" };
  const kindLabel = { statement: "语句", statements: "顺序语句块", loop: "循环条件", branch: "条件分支", scope: "词法 scope", scope_exit: "scope 正常出口", entry: "函数入口", exit: "函数出口", return: "返回", submit: "提交点", break: "break", continue: "continue", boundary: "上下文边界" };
  const emit = () => window.dispatchEvent(new Event("llvm-stage2-change"));
  const set = patch => { Object.assign(S, patch); emit(); };
  const pair = () => B.stage === "source" ? { id: DATA.case.id + ":frontend", before: 0, after: 0, textStatus: "identical", name: "Frontend" } : DATA.comparisons.find(p => p.ordinal === S.pass);
  const snapshot = side => pair()[side];
  const fnName = side => B.stage === "source" ? S.sourceFn : S[side + "Fn"];
  const graph = side => cfg?.graphs[snapshot(side)]?.[fnName(side)];
  const fnInfo = side => DATA.snapshots[snapshot(side)].functions.find(f => f.name === fnName(side));
  const matches = () => B.stage === "source" ? [] : DATA.matches.filter(m => m.passOrdinal === S.pass);
  const entity = id => DATA.entities.find(e => e.id === id);
  const nodeId = (side, n, fn = fnName(side)) => [DATA.case.id, pair().id, side, fn, "node", n.id].join("|");
  const button = (label, action, active = false, extra = {}) => h("button", { type: "button", onClick: action, "aria-pressed": active, ...extra }, label);
  const tab = (label, action, active) => h("button", { type:"button", role:"tab", onClick:action, "aria-selected":active, tabIndex:active?0:-1 }, label);
  const small = text => h("small", null, text);
  const p = text => h("p", null, text);
  const section = (title, ...children) => h("section", { className: "s2-body", style: { padding: 0 } }, h("h3", null, title), ...children);
  function pane(role, title, body, extra = "") {
    const isCenter = role === "editor-preview";
    return h(isCenter ? "section" : "aside", {
      className: "pto-ide-frame__pane s2-pane " + (isCenter ? B.classes.centerPane : role === "explorer" ? B.classes.sidebar : B.classes.inspector) + " " + extra,
      "data-ide-pane": role, id: isCenter ? undefined : "llvmcfg-" + role + "-pane"
    }, title ? h("header", { className: "pto-ide-frame__pane-header" }, h("h2", { className: "pto-ide-frame__pane-title" }, title), h("span", { className: "pto-ide-frame__pane-meta" }, B.stage === "source" ? "STAGE 1" : B.stage === "transform" ? "STAGE 2" : B.stage === "codegen" ? "STAGE 3" : "STAGE 4")) : null, body);
  }
  async function unpack(packed) {
    if (!window.DecompressionStream) throw new Error("当前浏览器缺少 gzip 解码能力；请使用支持 DecompressionStream 的浏览器读取内置证据。");
    const bytes = Uint8Array.from(atob(packed), c => c.charCodeAt(0));
    return JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text());
  }
  function load() {
    if (loading) return loading;
    loading = Promise.all([unpack(DATA.sourcesPacked), unpack(CFG_PACKED)]).then(([recipes, graphs]) => {
      const split = text => text.match(/[^\n]*\n|[^\n]+$/g) || [];
      let lines = split(recipes[0]);
      sourceLines = [lines.map(l => l.replace(/\r?\n$/, ""))];
      for (const recipe of recipes.slice(1)) {
        const next = []; let cursor = 0;
        for (const [start, end, insert] of recipe) { next.push(...lines.slice(cursor, start), ...split(insert)); cursor = end; }
        next.push(...lines.slice(cursor)); lines = next;
        sourceLines.push(lines.map(l => l.replace(/\r?\n$/, "")));
      }
      cfg = graphs; set({ loaded: true });
    }).catch(error => { set({ error: error.message }); });
    return loading;
  }
  function caseSelector() {
    return null;
  }
  function pickPass(ordinal) {
    const target = DATA.comparisons.find(p => p.ordinal === ordinal);
    const preferred = ordinal === 9 ? "decode_csa_test" : ordinal === 2 || ordinal === 3 ? "indexer_topk_group_wave" : "l3_decode_csa";
    const choose = side => DATA.snapshots[target[side]].functions.some(f => f.name === preferred) ? preferred : DATA.snapshots[target[side]].functions[0]?.name || "";
    graphCache.clear();
    viewCache.clear();
    set({ pass: ordinal, beforeFn: choose("before"), afterFn: choose("after"), selection: null, returnTo: null, full: false, dual: false, codeFull: false });
  }
  function selectFunction(side, name) {
    // Same-name navigation is not evidence of semantic identity.
    const other = side === "before" ? "after" : "before";
    const patch = { [side + "Fn"]: name, selection: null, returnTo: null };
    if (DATA.snapshots[snapshot(other)].functions.some(f => f.name === name)) patch[other + "Fn"] = name;
    set(patch);
  }
  // General purpose, not observed effects. Reference: https://www.pypto.ai/pypto/zh/dev/passes/00-pass_manager/
  const passDescriptions = {
    "InlineFunctions": "把可内联函数的计算展开到调用位置。",
    "UnrollLoops": "展开指定循环，将迭代转换为连续的计算语句。",
    "CtrlFlowTransform": "将分支与循环整理为结构化控制流。",
    "ConvertToSSA": "为每次变量赋值建立独立版本，明确数据依赖。",
    "Simplify": "化简表达式与冗余计算，整理中间表示。",
    "NormalizeStmtStructure": "统一语句块结构，整理嵌套与顺序语句。",
    "FlattenCallExpr": "将嵌套调用拆成独立语句，明确求值顺序。",
    "OutlineHierarchyScopes": "将层级作用域提取为独立函数与调用。",
    "OutlineIncoreScopes": "把核内计算区域提取为可单独编译的函数。",
    "OutlineClusterScopes": "将集群作用域提取为独立的分组函数。",
    "ConvertTensorToTileOps": "将核内张量运算转换为硬件可执行的分块运算。",
    "OptimizeOrchTensors": "优化编排层张量，减少可避免的分配并整理视图。",
    "LowerCompositeOps": "将复合运算拆解为更基础的操作。",
    "FlattenTileNdTo2D": "将高维分块转换为硬件支持的二维分块。",
    "BlockNzTensorViews": "为 NZ 布局张量整理分块视图。",
    "LegalizeTileCast": "将分块类型转换调整为后端支持的形式。",
    "AutoTileMatmulL0": "对矩阵乘法自动分块，以适配 L0 存储与计算。",
    "CanonicalizeTileSlice": "规范分块切片，统一偏移和视图表达。",
    "InferTileMemorySpace": "推导各分块应使用的硬件存储空间。",
    "InsertMxScaleAddr": "为 MX 量化运算补充缩放数据的地址信息。",
    "ResolveBackendOpLayouts": "按目标后端要求确定操作的数据布局。",
    "LowerAutoVectorSplit": "将自动向量拆分意图转换为具体的拆分操作。",
    "ExpandMixedKernel": "将混合计算核拆分为矩阵核、向量核及协作调用。",
    "InjectGMPipeBuffer": "为跨核数据管道补充全局内存缓冲区。",
    "SplitVectorKernel": "按拆分配置划分向量核的计算任务。",
    "StampTfreeSplit": "为缓冲释放操作补齐对应数据管道的拆分信息。",
    "NormalizeReturnOrder": "统一返回值的排列顺序。",
    "SkewCrossCorePipeline": "错开跨核流水线阶段，使生产与消费交叠执行。",
    "LowerPipelineToSlots": "将适用的流水循环转换为多个缓冲槽轮转。",
    "LowerPipelineLoops": "将流水循环展开为具体的执行阶段。",
    "CanonicalizeIOOrder": "统一输入输出顺序，便于后续接口处理。",
    "MaterializeTensorStrides": "补充张量各维步长，明确内存寻址方式。",
    "InitMemRef": "为变量建立内存引用，准备存储规划。",
    "MaterializeSemanticAliases": "显式建立共享存储的别名关系。",
    "MemoryReuse": "让生命周期不重叠的数据复用存储空间。",
    "AllocateMemoryAddr": "为内存引用分配具体的存储地址。",
    "FoldNoOpReshape": "消除不需要实际数据搬运的冗余形状变换。",
    "FuseCreateAssembleToSlice": "将可合并的创建与拼装操作转换为切片操作。",
    "DeriveCallDirections": "推导函数参数的输入、输出及读写方向。",
    "AutoDeriveTaskDependencies": "根据数据读写关系推导任务间依赖。",
    "ExpandManualPhaseFence": "将手动阶段同步标记展开为具体依赖。",
    "SynthesizeAllReduceSignals": "为全归约通信生成所需的同步信号。",
    "MaterializeCommDomainScopes": "显式建立通信域作用范围。",
    "LowerHostTensorCollectives": "将主机侧张量集合通信转换为底层调用。",
    "MaterializeDistTensorCtx": "为分布式张量补充运行所需的上下文。",
    "LegalizeGraphBoundary": "规范计算图边界，使跨边界接口符合后端要求。",
    "MaterializeRuntimeScopes": "将运行时作用域转换为显式执行结构。",
    "ClassifyIterArgCarry": "区分循环迭代间传递的数据与依赖。",
    "InsertCommFence": "在通信边界补充必要的同步屏障。",
    "MaterializeValidShapeSymbols": "为有效数据形状补齐显式符号。"
};
  function explorer() {
    const visible = DATA.comparisons;
    const params = DATA.inputs.filter(i => `${i.name} ${i.direction} ${i.dtype} ${JSON.stringify(i.shape)}`.toLowerCase().includes(S.inputQuery.toLowerCase()));
    return pane("explorer", "Transformation", h("div", { className: "pto-ide-frame__pane-body s2-body" },
      h("div", { className: "s2-input-summary" }, h("strong", null, "l3_decode_csa"), small("a2a3 / Ascend910B"), h("code", null, "x_hc: [2, T_DYN, 4, 4096] FP32"), small("动态维实际运行值未知 · 编译器 commit 未确认")),
      h("details", null, h("summary", null, "编译输入 · 55 个参数"), h("div", { className: "s2-params" },
        h("input", { "aria-label": "搜索输入参数", placeholder: "名称 / 方向 / dtype / shape", value: S.inputQuery, onChange: e => set({ inputQuery: e.target.value }) }),
        small("distributed_meta.json 与 00 frontend 声明已核对；-1 保留为元数据动态维，不使用调试脚本的 1。"), small(`${params.length} / 55 个参数`),
        ...params.map(i => h("div", { className: "s2-param", key: i.name }, h("strong", null, i.name), h("code", null, `${i.direction} · ${i.dtype} · ${JSON.stringify(i.shape)}`), small("metadata shape: " + JSON.stringify(i.metadataShape)), h("code", null, i.declaration), small(`00 frontend · L${i.start}–${i.end}`))))),
      section("Pass 时间线"),
      h("div", { className: "s2-pass-list", "aria-label": "相邻快照 Pass 列表" }, visible.map(c => {
        const available = [c.before,c.after].every(index => Object.values(cfg?.graphs[index] || {}).some(g=>g.status === "analyzed"));
        const description = passDescriptions[c.name] || "整理当前阶段的中间表示，为后续编译准备输入。";
        return h("button", {
          key:c.id, type:"button", disabled:!available, "aria-current":c.ordinal === S.pass ? "true" : "false",
          title:description + (available ? "" : "（尚未接入查看）"), onClick:available ? ()=>pickPass(c.ordinal) : undefined
        }, h("span",null,String(c.ordinal).padStart(2,"0")), h("span",{className:"s2-pass-content"},
          h("span",{className:"s2-pass-heading"},h("strong",null,c.name),c.textStatus === "changed" ? h("span",{className:"s2-diff-tag",title:"前后快照文本有变化"},"diff") : null),
          h("small",{className:"s2-pass-description",title:description},description)));
      }))));
  }
  function selectedNode() {
    if (!S.selection || S.selection.kind !== "node") return null;
    return cfg?.graphs[S.selection.snapshot]?.[S.selection.fn]?.nodes?.find(n => n.id === S.selection.node);
  }
  function selectedEntities() { return selectedNode()?.entities || []; }
  function linkedEntities() {
    const selected = selectedEntities();
    return matches().filter(m => [...m.before, ...m.after].some(id => selected.includes(id))).flatMap(m => [...m.before, ...m.after]);
  }
  function isLinked(selectedId, candidateId) {
    if (!S.selection || S.selection.kind !== "node") return false;
    for (const side of ["before", "after"]) {
      const n = graph(side)?.nodes?.find(n => nodeId(side,n) === candidateId);
      if (!n) continue;
      if (pair().textStatus === "identical" && S.selection.fn === fnName(side)) return selectedNode()?.start === n.start && selectedNode()?.kind === n.kind;
      return n.entities.some(id => linkedEntities().includes(id));
    }
    return false;
  }
  function selectNode(side, n) {
    if (n.kind === "boundary") { set({ full: true }); return; }
    const expand = !graphModel(side)?.nodes.some(item => item.id === n.id);
    set({ full: S.full || expand, selection: { caseId: DATA.case.id, pairId: pair().id, kind: "node", side, snapshot: snapshot(side), fn: fnName(side), node: n.id, id: nodeId(side,n) } });
  }
  function selectById(side, id) { const n = graph(side)?.nodes?.find(n => nodeId(side,n) === id); if (n) selectNode(side,n); else if (id.includes("|boundary|")) set({ full: true }); }
  function selectEdge(title, id) {
    const side = title === "dev" ? "before" : "after";
    const model = graphModel(side), edge = model?.edges.find(e => String(e._gvid) === id);
    if (!edge) return;
    set({ selection: { caseId: DATA.case.id, pairId: pair().id, kind: "edge", side, snapshot: snapshot(side), fn: fnName(side), edge, id: pair().id + "|" + side + "|edge|" + id } });
  }
  function focusEntity(id) {
    const e = entity(id); if (!e) return;
    const side = e.snapshot === snapshot("before") ? "before" : e.snapshot === snapshot("after") ? "after" : null;
    if (!side) return;
    const info = DATA.snapshots[e.snapshot].functions.find(f => f.start <= e.start && f.end >= e.end);
    if (!info) return;
    S[side + "Fn"] = info.name;
    const nodes = graph(side)?.nodes || [];
    const n = nodes.find(n => n.start === e.start && n.entities.includes(id)) || nodes.find(n => n.entities.includes(id));
    if (n) selectNode(side,n); else emit();
  }
  // Human-readable summaries use syntax and verified regions, never inferred runtime behavior.
  function nodeMeaning(n,side) {
    const text=n.text || "", tags=[], actions=[];
    if (/alloc_window_buffer\s*\(/.test(text)) actions.push("分配窗口与同步缓冲区");
    if (/\b(?:matmul|gemm)\s*\(/.test(text)) actions.push("计算矩阵乘积");
    if (/\b(?:reshape|view|transpose)\s*\(/.test(text)) actions.push("调整张量形状或维度");
    if (/\b(?:load|read)\s*\(/.test(text)) actions.push("读取数据");
    if (/\b(?:store|write)\s*\(/.test(text)) actions.push("写入数据");
    const control={entry:["开始执行函数","从这里进入当前函数。"],exit:["函数执行结束","当前函数的正常流程在这里结束。"],return:["返回计算结果","返回后结束当前函数。"],loop:["逐项执行循环","每次处理一项；没有剩余项时离开循环。"],branch:["判断执行路径","条件成立与不成立分别进入两条后续路径。"],scope:["进入计算区域","执行此区域内的操作。"],scope_exit:["离开计算区域","区域内操作结束，继续后续流程。"],break:["提前退出循环","跳到当前循环之后的操作。"],continue:["进入下一轮循环","跳过本轮剩余操作，回到循环判断。"],boundary:["查看后续上下文","点击展开完整函数，查看省略的流程。"],submit:["提交计算任务","提交目标："+(n.submitTarget || "见原始代码")+"。"]};
    let [title,description]=control[n.kind] || [actions.slice(0,2).join(" · ") || "执行顺序操作",actions.length?actions.join("，")+"。":`顺序执行此处的 ${n.statementCount || 1} 条语句；具体变量与表达式可在源码中查看。`];
    if (/alloc_window_buffer\s*\(/.test(text) && !control[n.kind]) description="为数据窗口及配套同步信号准备存储空间；容量和变量见展开源码。";
    if(n.kind==="loop") {
      const variable=text.match(/for\s+(.+?)\s+in\s/), condition=text.match(/^\s*while\s+(.+?):/);
      if(variable) { title=`遍历 ${variable[1]}`; description="继续：处理下一项。结束：离开此循环。"; }
      if(condition) { title="按条件重复执行"; description="条件成立时继续执行，否则离开循环。"; }
      tags.push("循环");
    }
    if(n.kind==="branch")tags.push("条件分支");
    const g=graph(side);
    const incoming=(g?.edges || []).filter(e=>e.target===n.id && ["True","False"].includes(e.label));
    incoming.forEach(e=>tags.push(`路径：${e.label==="True"?"成立":"不成立"}`));
    const m=matches().find(m=>n.entities.some(id=>m[side].includes(id)));
    if(m?.id==="leaf-unroll") {
      const index=m.after.findIndex(id=>n.entities.includes(id));
      tags.push(side==="before"?`展开前 · 对应 ${m.after.length} 个副本`:`展开副本 ${index+1}/${m.after.length}`);
      if(side==="after"&&index>=0)tags.push(`同源区域 · group_leaf=${index}`);
    } else if(m) tags.push("已核验对应区域");
    return {title,description,tags:[...new Set(tags)],group:m?.id || ""};
  }
  function renderNode(info) {
    return h("span",{className:"llvmcfg-semantic-node","data-region":info.group || undefined},
      h("small",{className:"llvmcfg-semantic-node__id"},info.eyebrow),
      h("strong",null,info.title),h("span",{className:"llvmcfg-semantic-node__description"},info.description),
      h("span",{className:"llvmcfg-semantic-node__tags"},...info.tags.map(tag=>h("i",{key:tag},tag))));
  }
  const edgeCaption = label => ({"有下一项":"继续","迭代结束":"结束","True":"成立","False":"不成立",return:"返回",break:"退出",continue:"下一轮"}[label] ?? label);
  function graphModel(side) {
    const g = graph(side); if (g?.status !== "analyzed") return null;
    const key = `${B.stage}|${S.pass}|${side}|${fnName(side)}|${S.full}`;
    if (graphCache.has(key)) return graphCache.get(key);
    let nodes = g.nodes, edges = g.edges;
    const relevant = matches().flatMap(m => m[side]);
    const seeds = nodes.filter(n => n.entities.some(id => relevant.includes(id)));
    let focused = !S.full && seeds.length > 0;
    if (focused) {
      const keep = new Set(seeds.map(n => n.id));
      const initial = new Set(keep);
      edges.forEach(e => { if (initial.has(e.source)) keep.add(e.target); if (initial.has(e.target)) keep.add(e.source); });
      nodes = nodes.filter(n => keep.has(n.id));
      let nextId = Math.max(...g.nodes.map(n => n.id)) + 1;
      const boundaries = new Map();
      const boundary = id => {
        if (boundaries.has(id)) return boundaries.get(id);
        const original = g.nodes.find(n => n.id === id), b = nextId++;
        nodes.push({ id:b, kind:"boundary", start:original.start, end:original.end, text:"展开未显示的函数上下文", entities:[] });
        boundaries.set(id,b); return b;
      };
      edges = edges.filter(e => keep.has(e.source) || keep.has(e.target)).map(e => ({ ...e, source:keep.has(e.source)?e.source:boundary(e.source), target:keep.has(e.target)?e.target:boundary(e.target), label:e.label }));
      focused = nodes.some(n => n.kind === "boundary");
    }
    const model = { name:fnName(side), directed:true, strict:false, label:"PyPTO lexical normal-flow", _subgraph_cnt:0,
      objects:nodes.map(n => ({ _gvid:n.id, name:"Node"+n.id, label:`{%${String(n.id)}:\\l${n.text.replace(/[{}]/g,"").replace(/\n/g,"\\l")}\\l}`, shape:"record" })),
      edges:edges.map(e => ({ _gvid:e.id, tail:e.source, head:e.target, label:edgeCaption(e.label), originalId:e.id })), nodes, focused };
    graphCache.set(key,model); return model;
  }
  function renderGraph(side) {
    const g = graph(side), model = graphModel(side);
    if (!model) return h("div", { className:"s2-empty", key:side }, h("strong",null,side === "before" ? "Before" : "After"), p(g?.status === "unsupported" ? `此函数的结构解析暂不支持：${g.reason}` : "此快照/函数的图形分析尚未接入。"), button("查看真实代码 Diff",()=>set({view:"code"})), small("未绘图不代表没有结构变化。"));
    const cacheKey = [B.stage,S.pass,side,fnName(side),S.full,S.selection?.id].join("|");
    if (viewCache.has(cacheKey)) return viewCache.get(cacheKey);
    const presentations = Object.fromEntries(model.nodes.map(n => [n.id, {
      ...nodeMeaning(n,side),
      same: B.stage !== "source" && pair().textStatus === "identical" && n.kind !== "boundary",
      selectionId:n.kind === "boundary" ? pair().id+"|"+side+"|boundary|"+n.id : nodeId(side,n),
      eyebrow:`${kindLabel[n.kind] || "操作"} · %${n.id} · L${n.start}${n.end!==n.start?"–"+n.end:""}`
    }]));
    const result = h("div", { className:"s2-graph-side", key:side, "data-s2-side":side },
      h(R.Suspense, { fallback:h("div",{className:"s2-empty"},"正在加载原有 CFG 引擎…") }, h(B.Graph, {
        key:`${S.pass}|${side}|${fnName(side)}|${S.full}`, llvmJson:model, llvmJson_compare:model, llvmOutput:[], title:side === "before"?"dev":"host",
        paneLabel:B.stage === "source" ? "00 FRONTEND" : side === "before" ? "BEFORE" : "AFTER",
        variant:"reviewCfg", nodePresentations:presentations, selectedBlockId:S.selection?.id, onSelectBlock:(_,id)=>selectById(side,id), readOnly:false, showMinimap:true
      })));
    if (viewCache.size > 12) viewCache.clear();
    viewCache.set(cacheKey,result);
    return result;
  }
  function codeRanges(side) {
    const selected = S.selection;
    if (selected?.kind === "node" && selected.side === side && selected.fn === fnName(side)) { const n=selectedNode(); return n?[[n.start,n.end]]:[]; }
    if (pair().textStatus === "identical" && selected?.kind === "node" && selected.fn === fnName(side)) { const n=selectedNode();return n?[[n.start,n.end]]:[]; }
    return linkedEntities().map(entity).filter(e=>e.snapshot===snapshot(side)).map(e=>[e.start,e.end]);
  }
  function selectLine(side, line) {
    const candidates = (graph(side)?.nodes || []).filter(n=>n.kind!=="exit" && n.kind!=="scope_exit" && n.start<=line && n.end>=line);
    const n = candidates.sort((a,b)=>(a.end-a.start)-(b.end-b.start))[0];
    if (n) selectNode(side,n); else set({ selection:{caseId:DATA.case.id,pairId:pair().id,kind:"line",side,snapshot:snapshot(side),fn:fnName(side),line,id:pair().id+"|"+side+"|line|"+line} });
  }
  function renderCode(side) {
    const info = fnInfo(side), lines = sourceLines[snapshot(side)] || [], ranges = codeRanges(side);
    const start = info?.start || 1, end = info?.end || lines.length;
    const ops = pair().opcodes.filter(o=>o[0]!=="equal"), changed=new Set(), visible=new Set();
    ops.forEach(o=>{ const a=side==="before"?o[1]:o[3], b=side==="before"?o[2]:o[4]; for(let i=a+1;i<=b;i++)changed.add(i); for(let i=Math.max(start,a-2);i<=Math.min(end,b+3);i++)visible.add(i); });
    ranges.forEach(([a,b])=>{for(let i=Math.max(start,a-3);i<=Math.min(end,b+3);i++)visible.add(i);});
    if(S.codeFull || pair().textStatus==="identical" || !visible.size) for(let i=start;i<=end;i++)visible.add(i);
    const rows=[]; let previous=start-1;
    [...visible].filter(i=>i>=start&&i<=end).sort((a,b)=>a-b).forEach(line=>{
      if(line>previous+1)rows.push(h("div",{className:"s2-code-gap",key:"gap"+line},`… ${line-previous-1} 行未变化上下文 · 可展开完整函数`));
      rows.push(h("div",{key:line,id:`s2-code-${side}-${line}`,className:"s2-code-line"+(changed.has(line)?" is-diff":"")+(ranges.some(([a,b])=>line>=a&&line<=b)||S.selection?.kind==="line"&&S.selection.side===side&&S.selection.line===line?" is-focused":"")},
        h("button",{type:"button",onClick:()=>selectLine(side,line),title:`选择 ${side} L${line} 并关联图对象`, "aria-label":`选择 ${side} 第 ${line} 行`},line),h("code",null,lines[line-1]||" ")));
      previous=line;
    });
    if(previous<end)rows.push(h("div",{className:"s2-code-gap",key:"end"},`… ${end-previous} 行未变化上下文`));
    return h("section",{className:"s2-code-side",key:side},h("header",null,`${side.toUpperCase()} · ${DATA.snapshots[snapshot(side)].file} · ${fnName(side)} · L${start}–${end}`),...rows);
  }
  function locateCode() {
    set({ view:"code" });
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
      for(const side of ["before","after"]) {
        const line=codeRanges(side)[0]?.[0] || (S.selection?.side===side?S.selection.line:null);
        if(line)document.getElementById(`s2-code-${side}-${line}`)?.scrollIntoView({block:"center",inline:"nearest"});
      }
    }));
  }
  function structureState() {
    const gs=[graph("before"),graph("after")];
    return gs.some(g=>g?.status==="unsupported")?"结构：所选函数含未支持语法":gs.every(g=>g?.status==="analyzed")?"结构：所选函数正常流已提取":"结构：所选快照/函数未分析";
  }
  function center() {
    const c=pair(), identical=c.textStatus==="identical", sides=identical&&!S.dual?["before"]:["before","after"];
    const status=S.error || (!S.loaded?"正在解码内置证据；不访问开发目录…":"");
    return pane("editor-preview",null,h(R.Fragment,null,
      h("div",{className:"s2-header"},
        h("div",{className:"s2-commandbar"},
          h("div",{className:"s2-view-tabs",role:"tablist","aria-label":"Pass 视图"},
            tab("控制流",()=>set({view:"graph"}),S.view==="graph"),tab("代码 Diff",()=>set({view:"code"}),S.view==="code")),
          h("div",{className:"s2-tools s2-actions"},
            identical?button(S.dual?"恢复单图":"双栏核对",()=>set({dual:!S.dual}),S.dual):null,
            S.view==="code"?button(S.codeFull?"仅变化与选择":"完整函数代码",()=>set({codeFull:!S.codeFull}),S.codeFull):null,
            S.view==="code"?button("定位所选代码",locateCode):null)),
        h("div",{className:"s2-function-row"},...["before","after"].map(side=>h("label",{key:side},side.toUpperCase()+" 函数",h("select",{value:fnName(side),onChange:e=>selectFunction(side,e.target.value)},...DATA.snapshots[snapshot(side)].functions.map(f=>h("option",{key:f.name,value:f.name},f.name)))))),
        matches().length?h("div",{className:"s2-tools"},...matches().flatMap(m=>[...m.before,...m.after]).map(id=>button(entity(id).name,()=>focusEntity(id),selectedEntities().includes(id),{key:id}))):null,
        identical?small("整份快照逐字节一致；单图共享原始结构。不代表 Pass 未执行或 Verify 通过。"):null),
      h("div",{className:"s2-canvas"},status?h("div",{className:"s2-empty",role:S.error?"alert":"status"},status):h("div",{className:(S.view==="graph"?"s2-graphs":"s2-code-grid")+(sides.length===1?" s2-single":"")},...sides.map(side=>S.view==="graph"?renderGraph(side):renderCode(side)))),
      S.view==="code"?h("div",{className:"s2-footnote"},"原始文本差异，不等同于结构变化。点击行号关联图对象；原始绝对行号保留。"):null));
  }
  function openTarget() {
    const n=selectedNode(), side=S.selection?.side, target=n?.submitTarget;
    if(!target||!side)return;
    if(!DATA.snapshots[snapshot(side)].functions.some(f=>f.name===target))return;
    const saved={selection:S.selection,fn:fnName(side),side,full:S.full};
    S[side+"Fn"]=target; S.returnTo=saved; S.full=true;
    const entry=graph(side)?.nodes?.find(n=>n.kind==="entry");
    if(entry)selectNode(side,entry);else set({selection:null});
  }
  function goBack() { const saved=S.returnTo;if(saved)set({[saved.side+"Fn"]:saved.fn,selection:saved.selection,full:saved.full,returnTo:null}); }
  function inspector() {
    const c=pair(), sel=S.selection, n=selectedNode();
    const match=matches().find(m=>[...m.before,...m.after].some(id=>selectedEntities().includes(id)));
    const description=DATA.changes.find(change=>change.passOrdinal===S.pass);
    const content=[];
    if(!sel)content.push(section("当前比较",p(description?.title || `${c.name} · ${textStatus[c.textStatus]}`),p(description?.explanation || "文本比较已完成；此 Pass 的语义对应关系尚未分析。"),small("选择函数、图对象或代码行核对证据。")));
    else {
      content.push(section("当前选择",h("strong",null,`${sel.side.toUpperCase()} · ${sel.fn}`),small(sel.kind==="node"?`${kindLabel[n?.kind]} · L${n?.start}–${n?.end}`:sel.kind==="line"?`L${sel.line} · 尚无图对象映射`:"控制流边")));
      if(n) {
        const meaning=nodeMeaning(n,sel.side);
        content.push(section("作用",h("strong",null,meaning.title),p(meaning.description),small(meaning.tags.join(" · "))));
        content.push(section("对应关系",p(c.textStatus==="identical"?"整份快照一致；同一函数、语句类型与位置可直接对应。":match?"所选语句位于已核验的一对多区域内。高亮表示区域对应，不代表每条语句都已建立一对一身份。":"该对象尚无匹配/分析结果；同名函数也不自动视为语义等价。")));
        if(match)content.push(section("区域核验依据",p(match.id==="leaf-unroll"?"将 group_leaf 分别替换为 0、1 后，循环体全部 AST 语句与两个 After 片段唯一匹配。":"原 scope 计算体与提取函数体（不含新增 return）AST 一致；唯一提交目标与 10 个位置参数一致。"),...([...match.before,...match.after].map(id=>{const e=entity(id);return button(`${e.snapshot===c.before?"Before":"After"} · ${e.name} · L${e.start}–${e.end}`,()=>focusEntity(id),false,{key:id});})),small("跨快照对应通过高亮与定位呈现，不是图中的控制流边。")));
        content.push(section("原始代码",h("pre",{className:"s2-selection-code"},sourceLines[sel.snapshot]?.slice(n.start-1,n.end).join("\n")),button("在 Before / After 代码中定位",locateCode)));
        if(n.submitTarget)content.push(section("提交关系",p(`显式 pl.submit → ${n.submitTarget}`),small("此关系不同于 CFG 后继；不推断同步调用、Kernel 或 Runtime 执行次数。"),DATA.snapshots[sel.snapshot].functions.some(f=>f.name===n.submitTarget)?button("进入提交目标函数",openTarget):small("目标函数快照未提供。")));
      } else if(sel.kind==="edge") {
        const g=graph(sel.side), original=g?.edges.find(e=>e.id===sel.edge.originalId), a=g?.nodes.find(n=>n.id===original?.source), b=g?.nodes.find(n=>n.id===original?.target);
        content.push(section("后继关系",p(`L${a?.start??"?"} → L${b?.start??"?"}`),p(original?.label || "正常顺序后继"),small("仅属于当前快照；未建立跨侧边匹配。若画布端点为上下文边界，可展开完整函数核对。"),a?button("定位起点语句",()=>selectNode(sel.side,a)):null,b?button("定位终点语句",()=>selectNode(sel.side,b)):null));
      } else content.push(section("代码证据",p("该行尚无已提取图对象；不回退到无关案例说明。"),button("定位原始行",locateCode)));
    }
    if(S.returnTo)content.push(button("← 返回原提交点",goBack));
    content.push(section("分析边界",p(structureState()),small("异常流、上下文管理器内部行为及后端调度未分析。未建立对应关系的对象保持未匹配。"),small("Verify / Runtime / 性能数据均未关联。")));
    content.push(h("details",{key:"raw"},h("summary",null,"原始身份与文件证据"),h("dl",null,
      ...[["case",DATA.case.id],["artifact",DATA.case.artifact],["pair",c.id],["selection",sel?.id||"未选择"],["Before",DATA.snapshots[c.before].file],["Before SHA-256",DATA.snapshots[c.before].sha256],["After",DATA.snapshots[c.after].file],["After SHA-256",DATA.snapshots[c.after].sha256],["mapping",match?.method||"无对象级匹配"],["metadata SHA-256",DATA.case.metadataHash]].flatMap(([k,v])=>[h("dt",{key:k},k),h("dd",{key:k+"value"},v)]))));
    return pane("inspector","证据与诊断",h("div",{className:"pto-ide-frame__pane-body s2-body s2-inspector"},...content));
  }
  // Bind case data into the original components. Do not replace pane renderers.
  function sourceData() {
    const f=DATA.snapshots[0].functions.find(f=>f.name===S.sourceFn);
    return {lines:sourceLines[0]||[],controlLine:selectedNode()?.start||f.start,
      controlLines:cfg?.graphs[0]?.[S.sourceFn]?.nodes.filter(n=>["loop","branch"].includes(n.kind)).map(n=>n.start)||[],
      summary:S.error||(!S.loaded?"正在读取真实源码…":"完整 frontend 源码 · 1793 行 · DeepseekV4")};
  }
  function bindSourceProps(props) {
    if(props.stage!=="source")return props;
    return {...props,workflow:S.reviewed?"baselineReady":"initial",runMessage:S.error||"当前为真实 frontend 快照；审查读取静态控制流，不运行编译器。",
      onRun:()=>{if(S.loaded&&!S.error){set({reviewed:true,selection:null});props.onLeftMode?.("review");}}};
  }
  function mapOriginal(tree,replace) {
    if(!R.isValidElement(tree))return tree;
    const changed=replace(tree);if(changed!==tree)return changed;
    if(tree.props.children===undefined)return tree;
    return R.cloneElement(tree,{},R.Children.map(tree.props.children,c=>mapOriginal(c,replace)));
  }
  const hasClass=(el,name)=>String(el.props.className||"").split(" ").includes(B.classes[name]);
  function originalControls() {
    const c=B.classes;
    const draft=S.constraintDraft||{t:16,factors:[16,8,4,2,1]};
    const edit=patch=>set({constraintDraft:{...draft,...patch},constraintEdited:true});
    return h("section",{className:c.reviewControls},
      h("div",{className:c.panelIntro},h("span",null,"控制流审查已完成"),h("strong",null,"调整场景与源码策略"),p("修改只进入候选草稿，不改变当前真实源码和画布。")),
      h("div",{className:c.controlGroup},h("div",{className:c.controlHeading},h("label",{htmlFor:"shape-t"},"审查场景 · 输入长度"),h("em",{className:c.scenarioBadge},"不修改源码")),
        h("div",{className:c.shapeInput},h("code",null,"输入长度："),h("input",{id:"shape-t",type:"number",min:1,max:512,value:draft.t,onChange:e=>edit({t:Math.min(512,Math.max(1,Math.trunc(Number(e.target.value))||1))})}),h("code",null,"token")),
        h("div",{className:c.scenarioButtons},...[1,16,31,32,33,64].map(value=>h("button",{key:value,type:"button",className:draft.t===value?c.active:"",onClick:()=>edit({t:value})},value))),
        p("候选输入长度对应当前 x_hc 的 T_DYN 维；初始 16 是演示草稿，不是已观测运行值。")),
      h("div",{className:c.controlGroup},h("div",{className:c.controlHeading},h("label",null,"源码策略 · unroll_list"),h("em",{className:c.sourceBadge},"候选草稿")),
        h("div",{className:c.factorButtons},...[32,16,8,4,2,1].map(factor=>{const enabled=draft.factors.includes(factor);return h("button",{key:factor,type:"button",className:enabled?c.active:"","aria-pressed":enabled,onClick:()=>edit({factors:enabled?draft.factors.filter(f=>f!==factor):[...draft.factors,factor].sort((a,b)=>b-a)})},"×",factor);})),
        p("保留展开规格选择交互；尚未绑定本案例的源码策略，不写入代码或生成新 Diff。")),
      h("div",{className:c.readonlyGroup},h("span",null,"循环意图 · 只读"),h("dl",null,...[["范围","待绑定目标循环"],["步长","待绑定目标循环"],["源码",S.sourceFn]].map(([k,v])=>h("div",{key:k},h("dt",null,k),h("dd",null,v))))),
      h("div",{className:c.constraintList},...[["输入参数","55 个 · 已核对"],["动态边界","T_DYN"],["候选状态","未编译验证"],["Runtime","未关联"]].map(([k,v])=>h("span",{key:k},h("b",null,k),h("code",null,v)))));
  }
  function adaptExplorer(tree,props) {
    return mapOriginal(tree,el=>{
      if(props.stage==="source") {
        if(hasClass(el,"reviewControls"))return originalControls();
        if(hasClass(el,"runCard")&&S.reviewed)return R.cloneElement(el,{className:`${B.classes.runCard} ${S.constraintEdited?B.classes.runCardDirty:""}`},
          h("span",null,S.constraintEdited?"候选草稿":"基线已就绪"),h("strong",null,S.constraintEdited?"审查配置已修改":"DeepseekV4 · 静态基线已就绪"),
          p("候选配置尚未接入重新编译；当前画布和 Stage 2 仍显示已有真实证据。"),
          h("button",{type:"button",onClick:()=>B.navigate?.("transform")},h("svg",{className:B.classes.icon,viewBox:"0 0 24 24","aria-hidden":"true"},h("path",{d:"m9 7 8 5-8 5z"})),"查看 Pass 编译"));
        if(hasClass(el,"runCard"))return R.cloneElement(el,{},R.Children.map(el.props.children,c=>{
          if(!R.isValidElement(c))return c;
          if(c.type==="strong"&&S.reviewed)return R.cloneElement(c,{},"DeepseekV4 · 静态基线已就绪");
          if(c.type==="button")return R.cloneElement(c,{disabled:!S.loaded||!!S.error});
          return c;
        }));
      } else if(hasClass(el,"stageMenu"))return R.cloneElement(el,{},h("span",{className:B.classes.sectionLabel},props.stage==="codegen"?"GENERATED PATHS":"OBSERVED EXECUTION"),h("div",{className:B.classes.sidebarFinding},h("b",null,"DeepseekV4"),"当前阶段证据尚未接入，不显示旧样例记录。"));
      return el;
    });
  }
  function adaptCenter(tree,props) {
    return mapOriginal(tree,el=>{
      if(hasClass(el,"graphArea")&&(props.stage!=="source"||S.reviewed))return R.cloneElement(el,{},props.stage==="source"?renderGraph("before"):h("div",{className:B.classes.emptyCanvas},h("span",null,props.stage==="codegen"?"03":"04"),h("strong",null,"DeepseekV4 · 当前阶段证据尚未接入"),p("不使用旧样例的生成路径或执行数量。")));
      if(props.stage!=="source"&&typeof el.type==="function"&&el.props.scenario)return h("header",{className:`pto-ide-frame__pane-header ${B.classes.canvasHeader}`},h("div",null,h("span",null,props.stage==="codegen"?"GENERATED ORCHESTRATION":"OBSERVED EXECUTION"),h("h1",null,"DeepseekV4"),p("当前案例证据待接入")));
      return el;
    });
  }
  function originalSelection(fallback) {
    if(B.stage==="transform")return fallback;
    const n=selectedNode(),downstream=B.stage!=="source";
    return {id:S.selection?.id||DATA.case.id,confidence:"exact",kind:downstream?"证据待接入":n?kindLabel[n.kind]:"源码审查",title:downstream?"DeepseekV4":S.sourceFn,
      subtitle:downstream?"当前阶段未关联，不使用旧样例证据。":"真实 frontend 快照 · 静态正常控制流",
      identity:[["案例","DeepseekV4 / l3_decode_csa"],["源码","passes_dump/00_frontend.py"],["位置",n?`L${n.start}–${n.end}`:"1793 行 / 24 个函数"],["平台","a2a3 / Ascend910B"]],
      role:n?nodeMeaning(n,"before").description:"核对 55 个真实编译参数及源码控制意图。",
      change:downstream?"尚未接入":S.reviewed?"静态控制流已载入；不代表编译器或 Runtime 已运行。":"等待运行控制流审查。",
      impact:"x_hc: [2, T_DYN, 4, 4096] FP32；动态维实际值未知。",
      nextStep:S.reviewed?"沿原顶部阶段导航进入编译变换，查看同一案例的 Pass Diff。":"查看左侧完整源码，然后运行控制流审查。",
      rawEvidence:[["artifact",DATA.case.artifact],["source SHA-256",DATA.snapshots[0].sha256],["metadata SHA-256",DATA.case.metadataHash],["Verify / Runtime","未关联"]]};
  }
  window.LLVMStage2={active:stage=>stage==="transform",caseSelector,sourceData,bindSourceProps,adaptExplorer,adaptCenter,originalSelection,
    explorer,center,inspector,isLinked,selectEdge,renderNode,
    canNavigate:stage=>stage==="source"||S.reviewed,
    guardStage:stage=>{if(stage!=="source"&&!S.reviewed)return true;set({selection:null,returnTo:null});return false;},
    footer:()=>[h("span",{key:"stage"},`${B.stage.toUpperCase()} · DeepseekV4 CSA`),h("span",{key:"pair"},B.stage==="transform"?`${String(S.pass).padStart(2,"0")} ${pair().name}`:"00 frontend · 同一编译案例"),h("span",{key:"identity"},"动态输入值未知"),h("span",{key:"verification"},"Verify / Runtime 未关联")],
    // Read-only/static test contract; also makes unsupported cases inspectable.
    contract:{data:DATA,state:S,load,pickPass,graph:()=>cfg,sourceLines:()=>sourceLines,graphModel,nodeMeaning,edgeCaption,nodeId,isLinked,selectNode,focusEntity,selectLine,openTarget,goBack}
  };
  const configureSplitDefaults = () => {
    const split=document.querySelector?.('[data-ide-split="standalone-main"]');
    if(!split)return;
    split.dataset.storageKey="llvmcfg-main-split-v3";
    split.dataset.pixelSizes="320,0,360";
  };
  configureSplitDefaults();
  load();
  emit();
})();
/* STAGE2_REVISED_END */
