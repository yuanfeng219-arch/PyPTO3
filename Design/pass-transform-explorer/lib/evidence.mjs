// Turn a before/after pair of analyzed snapshots into measured evidence:
// what actually changed, in the compiler's own units (functions, ops, buffers,
// loops, tasks) rather than in lines of text.

import { tokenEdits } from './diff.mjs';

const OP_LABEL = (op) => String(op).replace(/^pl\./, '');

/**
 * Aggregate the substitutions a pass performs across every rewritten line.
 *
 * Several passes leave every structural metric untouched and only rewrite
 * pieces of each statement - AllocateMemoryAddr fills in MemRef offsets,
 * ConvertToSSA appends version suffixes, ResolveBackendOpLayouts flips a
 * layout tag. Counting the token substitutions is what makes those passes
 * legible instead of showing up as "735 lines changed, nothing else moved".
 */
export function rewritePatterns(rows, limit = 14) {
  const blocks = [];
  let del = [];
  let add = [];
  const flush = () => {
    if (del.length || add.length) blocks.push({ del, add });
    del = [];
    add = [];
  };
  for (const r of rows) {
    if (r.tag === '=') { flush(); continue; }
    if (r.tag === '-') {
      if (add.length) flush();
      del.push(r.line);
    } else {
      add.push(r.line);
    }
  }
  flush();

  const subs = new Map();
  let paired = 0;
  let pureAdd = 0;
  let pureDel = 0;

  for (const blk of blocks) {
    const { pairs, unmatchedDel, unmatchedAdd } = pairLines(blk.del, blk.add);
    pureDel += unmatchedDel;
    pureAdd += unmatchedAdd;
    for (const [before, after] of pairs) {
      paired++;
      for (const e of tokenEdits(before, after)) {
        if (!e.old && !e.new) continue;
        const from = cleanFragment(e.old) || '∅';
        const to = cleanFragment(e.new) || '∅';
        if (from === '∅' && to === '∅') continue;
        const key = JSON.stringify([from, to]);
        const hit = subs.get(key);
        if (hit) { hit.count++; continue; }
        subs.set(key, { from, to, count: 1, example: { before: before.trim(), after: after.trim() } });
      }
    }
  }

  const top = [...subs.values()].sort((a, b) => b.count - a.count).slice(0, limit);
  return { paired, pureAdd, pureDel, distinct: subs.size, top };
}

const SIM_TOKENS = /[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?/g;

/** Token-overlap similarity of two lines, in [0, 1]. */
function similarity(a, b) {
  const ta = a.match(SIM_TOKENS) || [];
  const tb = b.match(SIM_TOKENS) || [];
  if (!ta.length && !tb.length) return 1;
  const bag = new Map();
  for (const t of ta) bag.set(t, (bag.get(t) || 0) + 1);
  let common = 0;
  for (const t of tb) {
    const n = bag.get(t);
    if (n) { common++; bag.set(t, n - 1); }
  }
  return (2 * common) / (ta.length + tb.length);
}

const PAIR_THRESHOLD = 0.5;
const PAIR_SEARCH_CAP = 160;

/**
 * Pair deleted lines with the added lines that replaced them.
 *
 * Index-order pairing is wrong whenever a block deletes n lines and adds m
 * unrelated ones - it then reports bogus substitutions in both directions
 * (`array` -> `tensor` *and* `tensor` -> `array`). Lines are matched greedily
 * by token overlap instead, and anything below the similarity floor is counted
 * as a genuine insertion or deletion rather than an edit.
 */
function pairLines(dels, adds) {
  const pairs = [];
  if (dels.length * adds.length <= PAIR_SEARCH_CAP * PAIR_SEARCH_CAP) {
    const used = new Set();
    for (const d of dels) {
      let best = -1;
      let bestScore = PAIR_THRESHOLD;
      for (let j = 0; j < adds.length; j++) {
        if (used.has(j)) continue;
        const s = similarity(d, adds[j]);
        if (s > bestScore) { bestScore = s; best = j; }
      }
      if (best >= 0) { used.add(best); pairs.push([d, adds[best]]); }
    }
    return { pairs, unmatchedDel: dels.length - pairs.length, unmatchedAdd: adds.length - pairs.length };
  }
  // Very large blocks: fall back to positional pairing, still similarity-gated.
  const n = Math.min(dels.length, adds.length);
  let matched = 0;
  for (let i = 0; i < n; i++) {
    if (similarity(dels[i], adds[i]) >= PAIR_THRESHOLD) { pairs.push([dels[i], adds[i]]); matched++; }
  }
  return { pairs, unmatchedDel: dels.length - matched, unmatchedAdd: adds.length - matched };
}

/** Drop a dangling dotted-path tail left behind by token alignment. */
function cleanFragment(s) {
  return String(s || '').replace(/[\s,]*\bpl\.$/, '').replace(/\s+$/, '');
}

/** A one-line, human-readable account of the dominant rewrite. */
export function describeRewrites(rw) {
  if (!rw || !rw.top.length) return '';
  const head = rw.top[0];
  const share = rw.paired ? Math.round((head.count / rw.paired) * 100) : 0;
  const what = head.from === '∅' ? `插入 \`${head.to}\``
    : head.to === '∅' ? `删除 \`${head.from}\``
      : `\`${head.from}\` → \`${head.to}\``;
  return `${rw.paired} 行被就地改写，最高频的改写是 ${what}（${head.count} 处${share ? `，覆盖 ${share}% 的改写行` : ''}）`;
}

export function fmtBytes(n) {
  if (n == null) return '-';
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(n < 10240 ? 1 : 0) + ' KiB';
  return (n / 1048576).toFixed(n < 10485760 ? 2 : 1) + ' MiB';
}

function delta(a, b) { return (b || 0) - (a || 0); }

function histDelta(before, after) {
  const keys = new Set([...Object.keys(before), ...Object.keys(after)]);
  const rows = [];
  for (const k of keys) {
    const d = delta(before[k], after[k]);
    if (d !== 0) rows.push({ key: k, before: before[k] || 0, after: after[k] || 0, delta: d });
  }
  rows.sort((x, y) => Math.abs(y.delta) - Math.abs(x.delta));
  return rows;
}

/** Compare the function tables of two snapshots. */
export function compareFunctions(prev, next) {
  const rows = [];
  const prevNames = new Set(prev.functions.map((f) => f.name));
  const nextNames = new Set(next.functions.map((f) => f.name));

  for (const f of prev.functions) {
    if (nextNames.has(f.name)) continue;
    rows.push({ name: f.name, status: 'removed', kind: f.kind, stmtsBefore: f.stmtCount, stmtsAfter: 0, linesBefore: f.srcLineCount, linesAfter: 0 });
  }
  for (const f of next.functions) {
    if (!prevNames.has(f.name)) {
      rows.push({ name: f.name, status: 'added', kind: f.kind, stmtsBefore: 0, stmtsAfter: f.stmtCount, linesBefore: 0, linesAfter: f.srcLineCount });
      continue;
    }
    const p = prev.byName.get(f.name);
    const same = p.src && f.src && p.src.length === f.src.length && p.src.join('\n') === f.src.join('\n');
    rows.push({
      name: f.name,
      status: same ? 'same' : 'changed',
      kind: f.kind,
      kindBefore: p.kind,
      stmtsBefore: p.stmtCount,
      stmtsAfter: f.stmtCount,
      linesBefore: p.srcLineCount,
      linesAfter: f.srcLineCount,
    });
  }
  const rank = { added: 0, removed: 1, changed: 2, same: 3 };
  rows.sort((a, b) => rank[a.status] - rank[b.status]
    || Math.abs(b.stmtsAfter - b.stmtsBefore) - Math.abs(a.stmtsAfter - a.stmtsBefore)
    || a.name.localeCompare(b.name));
  return rows;
}

/**
 * Build the evidence cards for one pass transition. Every card is derived from
 * measured deltas; a card is only emitted when it has something to say.
 */
export function buildEvidence(prev, next, funcRows, rewrite) {
  const a = prev.totals;
  const b = next.totals;
  const cards = [];

  // Below a handful of paired lines there is no *pattern* to report, only an
  // anecdote - the diff view already shows those lines.
  if (rewrite && rewrite.top && rewrite.top.length && rewrite.paired >= 3) {
    cards.push({
      id: 'rewrite',
      title: '高频改写',
      tone: 'change',
      headline: describeRewrites(rewrite),
      subs: rewrite.top,
      stats: { paired: rewrite.paired, pureAdd: rewrite.pureAdd, pureDel: rewrite.pureDel, distinct: rewrite.distinct },
    });
  }

  // --- function structure ---
  const added = funcRows.filter((r) => r.status === 'added');
  const removed = funcRows.filter((r) => r.status === 'removed');
  const changed = funcRows.filter((r) => r.status === 'changed');
  if (added.length || removed.length) {
    cards.push({
      id: 'functions',
      title: '函数结构',
      tone: added.length >= removed.length ? 'add' : 'remove',
      headline: `函数 ${a.functions} → ${b.functions}` +
        (added.length ? `，新增 ${added.length} 个` : '') +
        (removed.length ? `，移除 ${removed.length} 个` : ''),
      items: [
        ...added.slice(0, 12).map((r) => ({ tone: 'add', label: r.name, note: `${r.kind || 'Func'} · ${r.stmtsAfter} 语句` })),
        ...removed.slice(0, 12).map((r) => ({ tone: 'remove', label: r.name, note: `${r.kind || 'Func'} · 原 ${r.stmtsBefore} 语句` })),
      ],
      more: Math.max(0, added.length - 12) + Math.max(0, removed.length - 12),
    });
  }

  // --- kind / level retagging (ExpandMixedKernel, SplitVectorKernel, ...) ---
  const retagged = funcRows.filter((r) => r.status === 'changed' && r.kindBefore && r.kind && r.kindBefore !== r.kind);
  if (retagged.length) {
    cards.push({
      id: 'retag',
      title: '函数类型改写',
      tone: 'change',
      headline: `${retagged.length} 个函数被重新标注执行单元`,
      items: retagged.slice(0, 10).map((r) => ({ tone: 'change', label: r.name, note: `${r.kindBefore} → ${r.kind}` })),
    });
  }

  // --- op migration ---
  const ops = histDelta(a.ops, b.ops);
  if (ops.length) {
    const up = ops.filter((r) => r.delta > 0).slice(0, 8);
    const down = ops.filter((r) => r.delta < 0).slice(0, 8);
    cards.push({
      id: 'ops',
      title: '算子迁移',
      tone: 'change',
      headline: `${ops.length} 种算子的数量发生变化（语句 ${a.stmts} → ${b.stmts}）`,
      items: [
        ...up.map((r) => ({ tone: 'add', label: OP_LABEL(r.key), note: `${r.before} → ${r.after}`, delta: r.delta })),
        ...down.map((r) => ({ tone: 'remove', label: OP_LABEL(r.key), note: `${r.before} → ${r.after}`, delta: r.delta })),
      ],
      more: Math.max(0, ops.length - up.length - down.length),
    });
  }

  // --- tensor vs tile ---
  const vt = delta(a.values.Tensor, b.values.Tensor);
  const vl = delta(a.values.Tile, b.values.Tile);
  if (vt || vl) {
    cards.push({
      id: 'values',
      title: '值语义',
      tone: vl > 0 ? 'change' : 'neutral',
      headline: `Tensor ${a.values.Tensor} → ${b.values.Tensor}，Tile ${a.values.Tile} → ${b.values.Tile}`,
      rows: [
        { label: 'Tensor 值', before: a.values.Tensor, after: b.values.Tensor },
        { label: 'Tile 值', before: a.values.Tile, after: b.values.Tile },
        { label: 'Scalar 值', before: a.values.Scalar, after: b.values.Scalar },
      ].filter((r) => r.before !== r.after),
    });
  }

  // --- memory ---
  const spaces = new Set([...Object.keys(a.allocBytes), ...Object.keys(b.allocBytes)]);
  const memRows = [...spaces]
    .map((s) => ({ label: s, before: a.allocBytes[s] || 0, after: b.allocBytes[s] || 0 }))
    .filter((r) => r.before !== r.after);
  if (memRows.length || a.allocs !== b.allocs) {
    const totalBefore = Object.values(a.allocBytes).reduce((s, x) => s + x, 0);
    const totalAfter = Object.values(b.allocBytes).reduce((s, x) => s + x, 0);
    cards.push({
      id: 'memory',
      title: '片上内存',
      tone: totalAfter < totalBefore ? 'remove' : 'add',
      headline: `tile.alloc ${a.allocs} → ${b.allocs}，合计 ${fmtBytes(totalBefore)} → ${fmtBytes(totalAfter)}`,
      rows: memRows.map((r) => ({ ...r, fmt: 'bytes' })),
    });
  }

  const msRows = [...new Set([...Object.keys(a.memSpaces), ...Object.keys(b.memSpaces)])]
    .map((s) => ({ label: s, before: a.memSpaces[s] || 0, after: b.memSpaces[s] || 0 }))
    .filter((r) => r.before !== r.after);
  if (msRows.length) {
    cards.push({
      id: 'memspace',
      title: '内存空间归属',
      tone: 'change',
      headline: `${msRows.length} 个内存空间的驻留值数量变化`,
      rows: msRows,
    });
  }

  // --- control flow ---
  const loopRows = [...new Set([...Object.keys(a.loopKinds), ...Object.keys(b.loopKinds)])]
    .map((k) => ({ label: k, before: a.loopKinds[k] || 0, after: b.loopKinds[k] || 0 }))
    .filter((r) => r.before !== r.after);
  if (loopRows.length || a.maxNest !== b.maxNest) {
    const kindText = loopRows.map((r) => `${r.label} ${r.before} → ${r.after}`).join('，');
    const headline = a.loops !== b.loops
      ? `循环 ${a.loops} → ${b.loops}` + (kindText ? `（${kindText}）` : '')
      : kindText
        ? `循环总数不变，但形态改变：${kindText}`
        : `最大嵌套深度 ${a.maxNest} → ${b.maxNest}`;
    cards.push({
      id: 'control',
      title: '控制流',
      tone: 'change',
      headline,
      rows: [...loopRows, ...(a.maxNest !== b.maxNest ? [{ label: '最大嵌套深度', before: a.maxNest, after: b.maxNest }] : [])],
    });
  }

  // --- tasks ---
  if (a.tasks !== b.tasks || a.taskEdges !== b.taskEdges) {
    cards.push({
      id: 'tasks',
      title: '任务图',
      tone: b.tasks >= a.tasks ? 'add' : 'remove',
      headline: `任务 ${a.tasks} → ${b.tasks}，依赖边 ${a.taskEdges} → ${b.taskEdges}`,
      rows: [
        { label: '任务节点', before: a.tasks, after: b.tasks },
        { label: '依赖边', before: a.taskEdges, after: b.taskEdges },
      ],
    });
  }

  // --- communication ---
  if (a.comm !== b.comm) {
    cards.push({
      id: 'comm',
      title: '通信原语',
      tone: b.comm > a.comm ? 'add' : 'remove',
      headline: `notify / wait ${a.comm} → ${b.comm}`,
      rows: [{ label: 'notify / wait', before: a.comm, after: b.comm }],
    });
  }

  // --- dynamic shape symbols ---
  if (a.dynamicSymbols !== b.dynamicSymbols) {
    cards.push({
      id: 'dyn',
      title: '动态 shape 符号',
      tone: b.dynamicSymbols > a.dynamicSymbols ? 'add' : 'remove',
      headline: `模块级 pl.dynamic 符号 ${a.dynamicSymbols} → ${b.dynamicSymbols}`,
      rows: [{ label: 'pl.dynamic', before: a.dynamicSymbols, after: b.dynamicSymbols }],
    });
  }

  // Rank cards by how much of the pass they actually explain, so the headline
  // is the dominant effect rather than whichever metric happens to be listed
  // first. A pass that only renames values should lead with its rewrites, not
  // with a one-edge wobble in the task graph.
  for (const c of cards) c.weight = cardWeight(c, a, b, added.length + removed.length, ops.length);
  cards.sort((x, y) => y.weight - x.weight);

  const headline = cards.length
    ? cards[0].headline
    : (changed.length ? `${changed.length} 个函数内部被改写，汇总指标不变` : '本 Pass 未改变 IR');

  return { cards, headline };
}

function cardWeight(card, a, b, fnChurn, opKinds) {
  switch (card.id) {
    case 'functions': return 10000 + fnChurn * 10;
    case 'retag': return 6000 + (card.items ? card.items.length : 0) * 10;
    case 'ops': {
      // A handful of op-count wobbles explains little; a broad migration
      // (Tensor -> Tile touches dozens of op kinds) explains almost everything.
      const stmtDelta = Math.abs(b.stmts - a.stmts);
      return opKinds * 60 + Math.min(stmtDelta, 1500);
    }
    case 'memory': {
      const before = Object.values(a.allocBytes).reduce((s, x) => s + x, 0);
      const after = Object.values(b.allocBytes).reduce((s, x) => s + x, 0);
      const rel = before ? Math.abs(after - before) / before : (after ? 1 : 0);
      return 1500 + Math.round(rel * 2000) + Math.abs(b.allocs - a.allocs) * 20;
    }
    case 'tasks': return Math.abs(b.tasks - a.tasks) * 200 + Math.abs(b.taskEdges - a.taskEdges) * 5;
    case 'comm': return Math.abs(b.comm - a.comm) * 60;
    case 'control': return 150 + Math.abs(b.loops - a.loops) * 30 + Math.abs(b.maxNest - a.maxNest) * 120;
    case 'values': return Math.abs(b.values.Tile - a.values.Tile) * 3 + Math.abs(b.values.Tensor - a.values.Tensor) * 3;
    case 'memspace': return 200;
    case 'dyn': return Math.abs(b.dynamicSymbols - a.dynamicSymbols) * 40;
    case 'rewrite': return Math.min(card.stats.paired, 900) + (card.stats.pureAdd + card.stats.pureDel) * 0.2;
    default: return 0;
  }
}

/** Flat metric snapshot used by the timeline. */
export function metricSnapshot(an) {
  const t = an.totals;
  return {
    lines: t.lines,
    stmts: t.stmts,
    functions: t.functions,
    loops: t.loops,
    allocs: t.allocs,
    allocBytes: Object.values(t.allocBytes).reduce((s, x) => s + x, 0),
    tasks: t.tasks,
    taskEdges: t.taskEdges,
    comm: t.comm,
    tensors: t.values.Tensor,
    tiles: t.values.Tile,
    maxNest: t.maxNest,
    ops: Object.values(t.ops).reduce((s, x) => s + x, 0),
  };
}
