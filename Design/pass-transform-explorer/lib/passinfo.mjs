// Pass metadata: phase grouping, the graph lens that best explains each pass,
// and the prose pulled out of the PyPTO pass docs in `repo/pto`.

/** Pipeline phases, in the order the compiler runs them. */
export const PHASES = [
  { id: 'frontend', label: '前端规范化', hint: '内联、展开、结构化控制流、SSA 化，把用户写法收敛成规范 IR' },
  { id: 'outline', label: '作用域外提', hint: '把层级 / 核内 / Cluster 作用域抽成独立函数，分出编排与计算' },
  { id: 'lowering', label: 'Tensor → Tile 下降', hint: '张量语义下降到 Tile 语义，选 layout、定内存空间、拆分算子' },
  { id: 'pipeline', label: '核与流水线成形', hint: '拆分 AIC/AIV 核、注入缓冲、错位跨核流水、展开流水循环' },
  { id: 'memory', label: '内存实体化', hint: '补 stride、建 MemRef、生命周期复用、分配片上地址' },
  { id: 'task', label: '任务与依赖', hint: '推导参数方向与任务依赖，生成 task DAG' },
  { id: 'dist', label: '分布式与通信', hint: '集合通信下降、通信域作用域、信号与栅栏' },
  { id: 'runtime', label: '运行时收尾', hint: '图边界合法化、运行时作用域与动态 shape 符号' },
];

const PHASE_OF = {
  frontend: ['InlineFunctions', 'UnrollLoops', 'CtrlFlowTransform', 'ConvertToSSA', 'NormalizeStmtStructure', 'FlattenCallExpr', 'SplitChunkedLoops', 'InterchangeChunkLoops'],
  outline: ['OutlineHierarchyScopes', 'OutlineIncoreScopes', 'OutlineClusterScopes'],
  lowering: ['ConvertTensorToTileOps', 'OptimizeOrchTensors', 'LowerCompositeOps', 'FlattenTileNdTo2D', 'BlockNzTensorViews', 'LegalizeTileCast', 'AutoTileMatmulL0', 'CanonicalizeTileSlice', 'InferTileMemorySpace', 'InsertMxScaleAddr', 'ResolveBackendOpLayouts', 'LowerTransposeLoadParamLayout', 'LowerAutoVectorSplit'],
  pipeline: ['ExpandMixedKernel', 'InjectGMPipeBuffer', 'SplitVectorKernel', 'StampTfreeSplit', 'NormalizeReturnOrder', 'SkewCrossCorePipeline', 'LowerPipelineToSlots', 'LowerPipelineLoops', 'CanonicalizeIOOrder'],
  memory: ['MaterializeTensorStrides', 'InitMemRef', 'MaterializeSemanticAliases', 'MemoryReuse', 'AllocateMemoryAddr', 'FoldNoOpReshape', 'FuseCreateAssembleToSlice'],
  task: ['DeriveCallDirections', 'AutoDeriveTaskDependencies', 'ExpandManualPhaseFence', 'ClassifyIterArgCarry'],
  dist: ['SynthesizeAllReduceSignals', 'MaterializeCommDomainScopes', 'LowerHostTensorCollectives', 'MaterializeDistTensorCtx', 'InsertCommFence'],
  runtime: ['LegalizeGraphBoundary', 'MaterializeRuntimeScopes', 'MaterializeValidShapeSymbols'],
};

/**
 * Which structural view explains a given pass best. Taken from the pass-type /
 * graph-type mapping worked out in `Design/llvm-flow/Pass_Diff_对话记录_20260908.md`:
 * no single "IR graph" explains every pass, so each pass names its own lens.
 */
const LENS_OF = {
  call: ['InlineFunctions', 'FlattenCallExpr', 'OutlineHierarchyScopes', 'OutlineIncoreScopes', 'OutlineClusterScopes', 'ExpandMixedKernel', 'SplitVectorKernel', 'LowerAutoVectorSplit', 'LegalizeGraphBoundary', 'MaterializeRuntimeScopes', 'MaterializeCommDomainScopes'],
  control: ['UnrollLoops', 'CtrlFlowTransform', 'ConvertToSSA', 'NormalizeStmtStructure', 'SplitChunkedLoops', 'InterchangeChunkLoops', 'SkewCrossCorePipeline', 'LowerPipelineToSlots', 'LowerPipelineLoops', 'ClassifyIterArgCarry', 'StampTfreeSplit'],
  dataflow: ['Simplify', 'ConvertTensorToTileOps', 'OptimizeOrchTensors', 'LowerCompositeOps', 'FlattenTileNdTo2D', 'BlockNzTensorViews', 'LegalizeTileCast', 'AutoTileMatmulL0', 'CanonicalizeTileSlice', 'ResolveBackendOpLayouts', 'LowerTransposeLoadParamLayout', 'FoldNoOpReshape', 'FuseCreateAssembleToSlice', 'NormalizeReturnOrder', 'CanonicalizeIOOrder', 'MaterializeValidShapeSymbols'],
  memory: ['InferTileMemorySpace', 'InsertMxScaleAddr', 'InjectGMPipeBuffer', 'MaterializeTensorStrides', 'InitMemRef', 'MaterializeSemanticAliases', 'MemoryReuse', 'AllocateMemoryAddr'],
  task: ['DeriveCallDirections', 'AutoDeriveTaskDependencies', 'ExpandManualPhaseFence', 'SynthesizeAllReduceSignals', 'LowerHostTensorCollectives', 'MaterializeDistTensorCtx', 'InsertCommFence'],
};

const PHASE_INDEX = invert(PHASE_OF);
const LENS_INDEX = invert(LENS_OF);

function invert(table) {
  const out = new Map();
  for (const [key, names] of Object.entries(table)) for (const n of names) out.set(n, key);
  return out;
}

export function phaseOf(pass) {
  if (pass === 'frontend') return 'frontend';
  return PHASE_INDEX.get(pass) || 'lowering';
}

export function lensOf(pass) {
  return LENS_INDEX.get(pass) || 'dataflow';
}

/** `FlattenTileNdTo2D` -> `flattentilendto2d`, for matching doc / source files. */
export function normalizeName(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
}

/**
 * Pull the prose out of a pass doc.
 *
 * The docs are not uniform - some use `## 概述` / `## 算法`, others `## 概览` /
 * `## 行为`, and a few start with a BOM - so every `##` section is kept in
 * document order rather than only the ones a fixed list knows about. Nothing a
 * pass author wrote is silently dropped.
 */
export function extractDoc(markdown) {
  const lines = markdown.replace(/^﻿/, '').split(/\r?\n/);
  const title = (lines.find((l) => /^#\s/.test(l)) || '').replace(/^#\s+/, '').trim();

  const blocks = [];
  let current = null;
  let preamble = [];
  let buf = [];
  let inFence = false;

  const flush = () => {
    const body = buf.join('\n').trim();
    if (current === null) preamble = buf.slice();
    else if (body) blocks.push({ heading: current, body });
    buf = [];
  };

  for (const line of lines) {
    if (line.trim().startsWith('```')) inFence = !inFence;
    if (!inFence && /^##\s/.test(line)) {
      flush();
      current = line.replace(/^##\s+/, '').trim();
      continue;
    }
    if (!inFence && /^#\s/.test(line)) continue;
    buf.push(line);
  }
  flush();

  const tagline = preamble
    .join('\n')
    .split(/\n\s*\n/)
    .map((s) => s.trim())
    .find((s) => s && !s.startsWith('>') && !s.startsWith('|') && !s.startsWith('```')) || '';

  return { title, tagline, timing: extractTiming(blocks), blocks };
}

/** The "run this pass here" note that most overview sections carry. */
function extractTiming(blocks) {
  for (const b of blocks) {
    const m = b.body.match(/\*\*使用时机\*\*[：:]([\s\S]*?)(?:\n\n|$)/);
    if (m) return m[1].trim();
  }
  return '';
}
