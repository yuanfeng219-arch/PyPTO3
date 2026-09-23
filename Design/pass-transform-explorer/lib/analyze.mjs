// Structural analysis of one parsed IR snapshot.
//
// Everything the UI shows - op histograms, memory tables, loop nests, task
// DAGs, def-use graphs - is derived here from the parsed AST, never from
// regexes over the dump text.

import { dotted, literal, walkExpr, walkStmts } from './pyir.mjs';

const DTYPE_BYTES = {
  FP32: 4, FLOAT32: 4, F32: 4,
  BF16: 2, FP16: 2, F16: 2, HALF: 2,
  FP8_E4M3: 1, FP8_E5M2: 1, FP8: 1, HIFLOAT8: 1,
  INT64: 8, UINT64: 8, INDEX: 8,
  INT32: 4, UINT32: 4,
  INT16: 2, UINT16: 2,
  INT8: 1, UINT8: 1, BOOL: 1,
  TASK_ID: 4,
};

const DIR_WRAPPERS = { Out: 'out', InOut: 'inout', In: 'in' };

/** Bytes for one element of `dtype`, or 0 when unknown. */
export function dtypeBytes(dtype) {
  if (!dtype) return 0;
  const key = String(dtype).replace(/^pl\./, '').toUpperCase();
  return DTYPE_BYTES[key] ?? 0;
}

/**
 * Decode a type annotation node into a structured descriptor.
 * Handles pl.Tensor / pl.Tile / pl.Scalar / pl.Array / pl.Ptr / pl.Tuple and
 * the pl.Out[...] / pl.InOut[...] direction wrappers.
 */
export function parseType(node) {
  if (!node) return null;
  let dir = null;
  let cur = node;

  for (let guard = 0; guard < 4; guard++) {
    if (cur.k !== 'sub') break;
    const name = dotted(cur.on).replace(/^pl\./, '');
    if (!DIR_WRAPPERS[name]) break;
    dir = DIR_WRAPPERS[name];
    cur = cur.index[0];
  }

  const ctor = dotted(cur.k === 'sub' ? cur.on : cur).replace(/^pl\./, '');
  const out = { ctor, dir, shape: null, dtype: null, memref: null, space: null, inner: null };
  if (cur.k !== 'sub') return out;

  const idx = cur.index || [];

  if (ctor === 'Tuple') {
    out.inner = idx.map(parseType);
    return out;
  }
  if (ctor === 'Array') {
    out.shape = [literal(idx[0])];
    out.dtype = idx[1] ? dotted(idx[1]).replace(/^pl\./, '') : null;
    return out;
  }
  if (ctor === 'Scalar') {
    out.dtype = idx[0] ? dotted(idx[0]).replace(/^pl\./, '') : null;
    return out;
  }

  // Tensor / Tile: [shape], dtype, MemRef(...), [Mem.Space]
  for (const item of idx) {
    if (item.k === 'list') { out.shape = item.items.map((x) => literal(x)); continue; }
    const name = dotted(item).replace(/^pl\./, '');
    if (item.k === 'call' && name === 'MemRef') {
      const a = item.args;
      out.memref = {
        buffer: a[0] ? (a[0].k === 'str' ? a[0].v : dotted(a[0])) : null,
        offset: a[1] ? numericArg(a[1]) : null,
        size: a[2] ? literal(a[2]) : null,
      };
      continue;
    }
    if (name.startsWith('Mem.')) { out.space = name.slice(4); continue; }
    if (name.startsWith('TensorLayout.')) { out.layout = name.slice(13); continue; }
    if (!out.dtype && item.k !== 'call') out.dtype = name;
  }
  return out;
}

function numericArg(node) {
  // pl.const(0, pl.INT64) -> 0 ; plain 0 -> 0 ; symbolic -> null
  if (node.k === 'call' && dotted(node.fn).replace(/^pl\./, '') === 'const') return literal(node.args[0]);
  const v = literal(node);
  return typeof v === 'number' ? v : null;
}

/** Element count of a shape, or null when any dimension is dynamic. */
export function shapeElems(shape) {
  if (!Array.isArray(shape) || !shape.length) return null;
  let n = 1;
  for (const d of shape) {
    if (typeof d !== 'number' || d < 0) return null;
    n *= d;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Program analysis
// ---------------------------------------------------------------------------

const COMM_OPS = new Set(['pld.system.notify', 'pld.system.wait', 'pl.notify', 'pl.wait']);

/**
 * Analyze a parsed program into per-function facts plus program-wide rollups.
 */
export function analyzeProgram(prog, sourceLines) {
  const functions = prog.functions.map((fn) => analyzeFunction(fn, sourceLines));
  const byName = new Map(functions.map((f) => [f.name, f]));

  const totals = {
    functions: functions.length,
    stmts: 0,
    ops: {},
    loops: 0,
    loopKinds: {},
    allocs: 0,
    allocBytes: {},
    tasks: 0,
    taskEdges: 0,
    comm: 0,
    values: { Tensor: 0, Tile: 0, Scalar: 0, Array: 0, Ptr: 0, other: 0 },
    memSpaces: {},
    maxNest: 0,
    lines: prog.lineCount,
    dynamicSymbols: prog.globals.length,
  };

  for (const f of functions) {
    totals.stmts += f.stmtCount;
    for (const [op, n] of Object.entries(f.opHist)) totals.ops[op] = (totals.ops[op] || 0) + n;
    totals.loops += f.loops.length;
    for (const l of f.loops) totals.loopKinds[l.kind] = (totals.loopKinds[l.kind] || 0) + 1;
    totals.allocs += f.allocs.length;
    for (const a of f.allocs) totals.allocBytes[a.space] = (totals.allocBytes[a.space] || 0) + (a.size || 0);
    totals.tasks += f.tasks.length;
    totals.taskEdges += f.tasks.reduce((s, t) => s + t.deps.length, 0);
    totals.comm += f.commCount;
    for (const [k, n] of Object.entries(f.valueKinds)) {
      if (k in totals.values) totals.values[k] += n;
      else totals.values.other += n;
    }
    for (const [k, n] of Object.entries(f.memSpaces)) totals.memSpaces[k] = (totals.memSpaces[k] || 0) + n;
    totals.maxNest = Math.max(totals.maxNest, f.maxNest);
  }

  return {
    name: prog.name,
    className: prog.className,
    functions,
    byName,
    totals,
    globals: prog.globals.map((g) => ({ name: g.name, src: g.rhsSrc, line: g.line })),
  };
}

function analyzeFunction(fn, sourceLines) {
  const opHist = {};
  const loops = [];
  const allocs = [];
  const calls = [];
  const valueKinds = {};
  const memSpaces = {};
  const buffers = new Map(); // memref buffer name -> {space, size, offsets:Set, uses}
  let stmtCount = 0;
  let maxNest = 0;
  let commCount = 0;
  let validShapeCount = 0;
  let order = 0;

  const stmtIndex = []; // flat, in program order, for lifetime windows

  walkStmts(fn.body, (s, chain) => {
    stmtCount++;
    const depth = chain.length;
    maxNest = Math.max(maxNest, depth);
    s._ord = order++;
    stmtIndex.push(s);

    if (s.op) {
      opHist[s.op] = (opHist[s.op] || 0) + 1;
      if (COMM_OPS.has(s.op)) commCount++;
      if (s.op === 'pl.tile.set_validshape') validShapeCount++;
    }

    if (s.kind === 'for') {
      loops.push({
        kind: s.loopKind || 'range',
        trip: s.tripCount,
        stage: s.stage,
        line: s.line,
        depth,
        carries: s.initValues ? s.initValues.length : 0,
        bodyStmts: countStmts(s.body),
      });
    }

    if (s.kind === 'assign' && s.op === 'pl.tile.alloc') {
      const space = dotted(s.expr.args[0]).replace(/^pl\.Mem\./, '');
      allocs.push({ ptr: s.targets[0], space, size: literal(s.expr.args[1]), line: s.line });
    }

    // Calls to sibling functions, however they are spelled.
    if (s.expr) {
      walkExpr(s.expr, (e) => {
        if (e.k !== 'call') return;
        const name = dotted(e.fn);
        if (name.startsWith('self.')) {
          calls.push({ callee: name.slice(5), via: 'direct', line: s.line });
        } else if (name === 'pl.submit' || name === 'pl.spmd_submit') {
          const target = dotted(e.args[0]).replace(/^self\./, '');
          if (target) calls.push({ callee: target, via: name.slice(3), line: s.line });
        }
      });
    }

    // Type-derived facts.
    if (s.kind === 'assign' && s.type) {
      const t = parseType(s.type);
      if (t) {
        const kinds = t.ctor === 'Tuple' && t.inner ? t.inner.filter(Boolean) : [t];
        for (const k of kinds) {
          valueKinds[k.ctor] = (valueKinds[k.ctor] || 0) + 1;
          if (k.space) memSpaces[k.space] = (memSpaces[k.space] || 0) + 1;
          if (k.memref && k.memref.buffer) {
            const b = buffers.get(k.memref.buffer) || {
              name: k.memref.buffer, space: k.space || null, size: 0, derived: 0, dynamic: false,
              offsets: new Set(), first: s._ord, last: s._ord, uses: 0,
            };
            b.space = b.space || k.space || null;
            b.size = Math.max(b.size, k.memref.size || 0);
            // A MemRef size of 0 on a symbolic shape means "not statically
            // known", not "empty" - keep the two apart so the memory view does
            // not claim a dynamic DDR tensor occupies nothing.
            const elems = shapeElems(k.shape);
            if (elems == null) b.dynamic = true;
            else b.derived = Math.max(b.derived, elems * dtypeBytes(k.dtype));
            if (k.memref.offset != null) b.offsets.add(k.memref.offset);
            b.last = s._ord;
            b.uses++;
            buffers.set(k.memref.buffer, b);
          }
        }
      }
    }
  });

  const tasks = collectTasks(fn);

  const params = fn.params.map((p) => {
    const t = parseType(p.type);
    return {
      name: p.name,
      dir: (t && t.dir) || 'in',
      ctor: t ? t.ctor : null,
      shape: t ? t.shape : null,
      dtype: t ? t.dtype : null,
      space: t ? t.space : null,
      memref: t ? t.memref : null,
      typeSrc: p.typeSrc,
    };
  });

  const src = sourceLines ? sourceLines.slice(fn.decoLine - 1, fn.endLine) : null;

  return {
    name: fn.name,
    kind: (fn.deco.type || '').replace(/^pl\.FunctionType\./, '') || null,
    level: (fn.deco.level || '').replace(/^pl\.Level\./, '') || null,
    role: (fn.deco.role || '').replace(/^pl\.Role\./, '') || null,
    attrs: fn.deco.attrs,
    params,
    stmtCount,
    opHist,
    loops,
    allocs,
    calls,
    tasks,
    valueKinds,
    memSpaces,
    commCount,
    validShapeCount,
    maxNest,
    buffers: [...buffers.values()].map((b) => ({
      name: b.name,
      space: b.space,
      size: b.size || b.derived,
      dynamic: b.dynamic && !b.size,
      offsets: [...b.offsets].sort((x, y) => x - y),
      first: b.first,
      last: b.last,
      uses: b.uses,
    })),
    line: fn.line,
    decoLine: fn.decoLine,
    endLine: fn.endLine,
    srcLineCount: fn.endLine - fn.decoLine + 1,
    src,
    body: fn.body,
    defSrc: fn.defSrc,
  };
}

function countStmts(body) {
  let n = 0;
  walkStmts(body, () => { n++; });
  return n;
}

// ---------------------------------------------------------------------------
// Task DAG
// ---------------------------------------------------------------------------

/**
 * Recover the submitted-task graph of one function.
 *
 * Two IR shapes carry tasks, depending on how far the pipeline has run:
 *   early  `with pl.at(level=..., name_hint=..., deps=[buf]) as tid:`
 *   late   `ret: pl.Tuple[..., pl.Scalar[pl.TASK_ID]] = pl.submit(self.f, ..., deps=[buf])`
 *
 * Dependencies are passed as arrays built up by `pl.array.create` /
 * `pl.array.update_element`, so we interpret those abstractly to recover the
 * set of task ids each dependency buffer holds.
 */
function collectTasks(fn) {
  const arrays = new Map(); // array SSA name -> Set(task-id value names)
  const tupleOf = new Map(); // tuple SSA name -> task node id (for ret[N] extraction)
  const tasks = [];
  const byTid = new Map(); // task-id value name -> task node id
  const seqOf = new Map(); // label -> ordinal, so ids survive line-number drift

  const nextId = (prefix, label) => {
    const n = seqOf.get(label) || 0;
    seqOf.set(label, n + 1);
    return prefix + ':' + label + '#' + n;
  };

  const resolve = (node) => {
    const out = new Set();
    if (!node) return out;
    const items = node.k === 'list' || node.k === 'tuple' ? node.items : [node];
    for (const it of items) {
      const name = dotted(it);
      if (arrays.has(name)) for (const v of arrays.get(name)) out.add(v);
      else if (name) out.add(name);
    }
    return out;
  };

  walkStmts(fn.body, (s) => {
    if (s.kind === 'assign' && s.op === 'pl.array.create') {
      arrays.set(s.targets[0], new Set());
      return;
    }
    if (s.kind === 'assign' && s.op === 'pl.array.update_element') {
      const base = dotted(s.expr.args[0]);
      const set = new Set(arrays.get(base) || []);
      const val = dotted(s.expr.args[2]);
      if (val) {
        if (arrays.has(val)) for (const v of arrays.get(val)) set.add(v);
        else set.add(val);
      }
      arrays.set(s.targets[0], set);
      return;
    }
    if (s.kind === 'assign' && s.op === 'pl.array.get_element') {
      const base = dotted(s.expr.args[0]);
      arrays.set(s.targets[0], new Set(arrays.get(base) || []));
      return;
    }

    if (s.kind === 'with' && s.ctxName === 'pl.at') {
      const kw = s.ctx.kwargs || {};
      const label = kw.name_hint ? String(literal(kw.name_hint)) : s.as || 'scope';
      const id = nextId('at', label);
      tasks.push({
        id,
        label,
        level: kw.level ? dotted(kw.level).replace(/^pl\.Level\./, '') : null,
        shape: 'at',
        tid: s.as || null,
        deps: [...resolve(kw.deps)],
        line: s.line,
        stmts: countStmts(s.body),
      });
      if (s.as) byTid.set(s.as, id);
      return;
    }

    if (s.kind === 'assign' && (s.op === 'pl.submit' || s.op === 'pl.spmd_submit')) {
      const callee = dotted(s.expr.args[0]).replace(/^self\./, '');
      const id = nextId('submit', callee);
      tasks.push({
        id,
        label: callee,
        level: null,
        shape: s.op === 'pl.spmd_submit' ? 'spmd' : 'submit',
        tid: null,
        deps: [...resolve(s.expr.kwargs.deps)],
        line: s.line,
        stmts: 0,
      });
      tupleOf.set(s.targets[0], id);
      return;
    }

    // `tid = ret__tmp_v0[1]` binds the task-id element of a submit result.
    if (s.kind === 'assign' && s.expr && s.expr.k === 'sub') {
      const base = dotted(s.expr.on);
      if (tupleOf.has(base)) {
        const t = parseType(s.type);
        if (t && (t.dtype === 'TASK_ID' || (t.ctor === 'Scalar' && t.dtype === 'TASK_ID'))) {
          byTid.set(s.targets[0], tupleOf.get(base));
          const node = tasks.find((x) => x.id === tupleOf.get(base));
          if (node) node.tid = s.targets[0];
        }
      }
    }
  });

  // Rewrite dependency value names into task node ids where we can.
  for (const t of tasks) {
    t.deps = [...new Set(t.deps.map((d) => byTid.get(d)).filter(Boolean))].filter((d) => d !== t.id);
  }
  return tasks;
}

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

/** Program-level call / scope graph: one node per function, edges for calls. */
export function callGraph(an) {
  const nodes = an.functions.map((f) => ({
    id: f.name,
    label: f.name,
    kind: f.kind || 'Func',
    level: f.level,
    role: f.role,
    stmts: f.stmtCount,
    loops: f.loops.length,
    tasks: f.tasks.length,
    line: f.decoLine,
  }));
  const seen = new Map();
  for (const f of an.functions) {
    for (const c of f.calls) {
      const key = JSON.stringify([f.name, c.callee, c.via]);
      const e = seen.get(key);
      if (e) { e.count++; continue; }
      seen.set(key, { from: f.name, to: c.callee, via: c.via, count: 1 });
    }
  }
  return { nodes, edges: [...seen.values()] };
}

/**
 * Control-structure tree of one function: nested for / if / with regions, with
 * straight-line statement runs collapsed into "block" leaves. Node ids are
 * structural paths so the same region matches across passes.
 */
export function controlTree(f, limit = 900) {
  const nodes = [];
  const edges = [];
  let truncated = false;

  const visit = (body, parentId, path) => {
    let runStart = null;
    let runCount = 0;
    let runOps = {};
    let seq = 0;

    const flush = () => {
      if (!runCount) return;
      const id = path + '/blk' + seq;
      const top = Object.entries(runOps).sort((a, b) => b[1] - a[1]).slice(0, 3);
      nodes.push({
        id,
        type: 'block',
        label: runCount + ' stmt',
        detail: top.map(([k, v]) => shortOp(k) + '×' + v).join(' '),
        line: runStart,
        weight: runCount,
      });
      edges.push({ from: parentId, to: id });
      seq++;
      runStart = null;
      runCount = 0;
      runOps = {};
    };

    for (const s of body) {
      if (nodes.length > limit) { truncated = true; return; }
      if (s.kind === 'for' || s.kind === 'if' || s.kind === 'with') {
        flush();
        const key = s.kind === 'for' ? (s.loopKind || 'range')
          : s.kind === 'with' ? (s.ctxName || 'with').replace(/^pl\./, '')
            : 'if';
        const id = path + '/' + key + '#' + seq;
        seq++;
        let label = key;
        let detail = '';
        if (s.kind === 'for') {
          label = key + '(' + (s.tripCount ?? '?') + ')';
          detail = [s.stage != null ? 'stage=' + s.stage : '', s.targetSrc].filter(Boolean).join('  ');
        } else if (s.kind === 'if') {
          label = 'if';
          detail = s.condSrc.slice(0, 60);
        } else {
          label = key;
          detail = s.as ? 'as ' + s.as : s.ctxSrc.slice(0, 60);
        }
        nodes.push({ id, type: s.kind, label, detail, line: s.line, weight: countStmts(s.body) + countStmts(s.orelse || []) });
        edges.push({ from: parentId, to: id });
        visit(s.body, id, id);
        if (s.orelse && s.orelse.length) {
          const eid = id + '/else';
          nodes.push({ id: eid, type: 'else', label: 'else', detail: '', line: s.line, weight: countStmts(s.orelse) });
          edges.push({ from: id, to: eid });
          visit(s.orelse, eid, eid);
        }
        continue;
      }
      if (runStart === null) runStart = s.line;
      runCount++;
      if (s.op) runOps[s.op] = (runOps[s.op] || 0) + 1;
    }
    flush();
  };

  const rootId = 'fn:' + f.name;
  nodes.push({ id: rootId, type: 'fn', label: f.name, detail: [f.kind, f.level, f.role].filter(Boolean).join(' / '), line: f.line, weight: f.stmtCount });
  visit(f.body, rootId, rootId);
  return { nodes, edges, truncated, root: rootId };
}

function shortOp(op) {
  return String(op).replace(/^pl\./, '').replace(/^tensor\./, 't.').replace(/^tile\./, 'T.').replace(/^array\./, 'a.');
}

/**
 * Def-use dataflow graph of one function. One node per assignment; edges from
 * every SSA value the right-hand side reads. Params become source nodes.
 */
export function dataflowGraph(f, limit = 600) {
  const nodes = [];
  const edges = [];
  const defOf = new Map();
  const seen = new Map(); // target name -> occurrences, keeping ids line-independent
  let truncated = false;

  for (const p of f.params) {
    const id = 'p:' + p.name;
    defOf.set(p.name, id);
    nodes.push({
      id,
      type: 'param',
      label: p.name,
      op: p.dir,
      ctor: p.ctor,
      shape: p.shape,
      dtype: p.dtype,
      space: p.space,
      line: f.line,
    });
  }

  walkStmts(f.body, (s) => {
    if (nodes.length > limit) { truncated = true; return; }
    if (s.kind !== 'assign' || !s.targets.length) return;
    const t = s.type ? parseType(s.type) : null;
    const occ = (seen.get(s.targets[0]) || 0);
    seen.set(s.targets[0], occ + 1);
    const id = 'v:' + s.targets[0] + (occ ? '#' + occ : '');
    nodes.push({
      id,
      type: 'value',
      label: s.targets[0],
      op: s.op ? shortOp(s.op) : 'expr',
      rawOp: s.op,
      ctor: t ? t.ctor : null,
      shape: t ? t.shape : null,
      dtype: t ? t.dtype : null,
      space: t ? t.space : null,
      buffer: t && t.memref ? t.memref.buffer : null,
      line: s.line,
    });
    const reads = new Set();
    walkExpr(s.expr, (e) => {
      if (e.k === 'name') reads.add(e.id);
    });
    for (const r of reads) {
      const from = defOf.get(r);
      if (from && from !== id) edges.push({ from, to: id });
    }
    for (const tgt of s.targets) defOf.set(tgt, id);
  });

  return { nodes, edges, truncated };
}

/** Task DAG for one function (nodes already carry resolved dependency ids). */
export function taskGraph(f) {
  const ids = new Set(f.tasks.map((t) => t.id));
  return {
    nodes: f.tasks.map((t) => ({
      id: t.id,
      label: t.label,
      type: t.shape,
      level: t.level,
      line: t.line,
      weight: t.stmts,
    })),
    edges: f.tasks.flatMap((t) => t.deps.filter((d) => ids.has(d)).map((d) => ({ from: d, to: t.id }))),
  };
}

/** Buffer lifetime / placement table for one function. */
export function memoryView(f) {
  const bySpace = new Map();
  for (const a of f.allocs) {
    const arr = bySpace.get(a.space) || [];
    arr.push({ name: a.ptr, space: a.space, size: a.size, line: a.line, kind: 'alloc' });
    bySpace.set(a.space, arr);
  }
  const allocNames = new Set(f.allocs.map((a) => a.ptr));
  for (const b of f.buffers) {
    const space = b.space || (allocNames.has(b.name) ? null : 'DDR');
    const arr = bySpace.get(space) || [];
    const existing = arr.find((x) => x.name === b.name);
    if (existing) {
      existing.offsets = b.offsets;
      existing.first = b.first;
      existing.last = b.last;
      existing.uses = b.uses;
      existing.viewSize = b.size;
    } else {
      arr.push({
        name: b.name, space, size: b.size, dynamic: b.dynamic, offsets: b.offsets,
        first: b.first, last: b.last, uses: b.uses, kind: 'memref',
      });
    }
    bySpace.set(space, arr);
  }
  return [...bySpace.entries()]
    .map(([space, items]) => ({
      space: space || 'unspecified',
      bytes: items.reduce((s, x) => s + (x.size || 0), 0),
      dynamicCount: items.filter((x) => x.dynamic).length,
      items: items.sort((a, b) => (b.size || 0) - (a.size || 0)),
    }))
    .sort((a, b) => b.bytes - a.bytes || b.items.length - a.items.length);
}
