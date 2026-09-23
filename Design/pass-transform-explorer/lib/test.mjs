// Unit checks for the pieces that are easy to get subtly wrong.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { parseDump, parseExpression, dotted, literal, splitTop, findTop } from './pyir.mjs';
import { parseType, analyzeProgram, controlTree, dataflowGraph } from './analyze.mjs';
import { diffLines, toHunks, countChanges, wordDiff, tokenEdits } from './diff.mjs';
import { rewritePatterns } from './evidence.mjs';
import { md, mdInline, escapeHtml } from './markdown.mjs';
import { extractDoc, normalizeName } from './passinfo.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../../..');

let pass = 0;
let fail = 0;
function ok(name, cond, detail) {
  if (cond) { pass++; return; }
  fail++;
  console.log(`  FAIL ${name}${detail ? ' :: ' + detail : ''}`);
}
function eq(name, got, want) {
  ok(name, JSON.stringify(got) === JSON.stringify(want), `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

// ── expression parsing ────────────────────────────────────────────────
{
  const e = parseExpression('pl.tensor.matmul(a, pl.tensor.slice(w, [16, 256], [0, k]), a_trans=False, out_dtype=pl.FP32)');
  eq('call name', dotted(e.fn), 'pl.tensor.matmul');
  eq('arg count', e.args.length, 2);
  eq('kwarg a_trans', literal(e.kwargs.a_trans), false);
  eq('kwarg out_dtype', dotted(e.kwargs.out_dtype), 'pl.FP32');
  eq('nested slice shape', literal(e.args[1].args[1]), [16, 256]);
}
{
  const e = parseExpression('pl.cast(group_base__ssa_v0, pl.INDEX) + peer_tp_inline2472__idx_v0');
  eq('binop', e.k, 'bin');
  eq('binop op', e.op, '+');
}
{
  // Strings containing brackets and commas must not confuse the splitter.
  eq('splitTop respects strings', splitTop('a, "x,y", [1, 2]', ',').length, 3);
  eq('findTop skips brackets', findTop('f(a: b) : c', ':'), 8);
}

// ── type decoding ─────────────────────────────────────────────────────
{
  const t = parseType(parseExpression('pl.Tile[[16, 256], pl.BF16, pl.MemRef(mem_vec_2, pl.const(8192, pl.INT64), 16384), pl.Mem.Vec]'));
  eq('tile ctor', t.ctor, 'Tile');
  eq('tile shape', t.shape, [16, 256]);
  eq('tile dtype', t.dtype, 'BF16');
  eq('tile space', t.space, 'Vec');
  eq('tile memref', t.memref, { buffer: 'mem_vec_2', offset: 8192, size: 16384 });
}
{
  const t = parseType(parseExpression('pl.Out[pl.Tensor[[16, 5120], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 327680)]]'));
  eq('out direction', t.dir, 'out');
  eq('out ctor', t.ctor, 'Tensor');
  eq('out buffer', t.memref.buffer, 'mem_ddr_0');
}
{
  const t = parseType(parseExpression('pl.Tensor[[t_dim__ssa_v0, 16384], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 0)]'));
  eq('dynamic dim kept symbolic', t.shape[0], 't_dim__ssa_v0');
  eq('dynamic memref size', t.memref.size, 0);
}

// ── statement structure ───────────────────────────────────────────────
{
  const src = [
    '# pypto.program: demo',
    'import pypto.language as pl',
    'T_DYN = pl.dynamic("T_DYN")',
    '@pl.program',
    'class demo:',
    '    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)',
    '    def f(self, x: pl.Tensor[[4, 4], pl.FP32]) -> pl.Tensor[[4, 4], pl.FP32]:',
    '        p: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 1024)',
    '        for i, (acc,) in pl.pipeline(8, stage=2, init_values=(x,)):',
    '            if i > 0:',
    '                acc: pl.Tensor[[4, 4], pl.FP32] = pl.tile.add(acc, x)',
    '            else:',
    '                acc: pl.Tensor[[4, 4], pl.FP32] = pl.tile.mul(acc, x)',
    '        return x',
  ].join('\n');
  const prog = parseDump(src, 'demo');
  eq('program name', prog.name, 'demo');
  eq('globals', prog.globals.map((g) => g.name), ['T_DYN']);
  eq('function count', prog.functions.length, 1);

  const fn = prog.functions[0];
  eq('deco type', fn.deco.type, 'pl.FunctionType.AIV');
  eq('params', fn.params.map((p) => p.name), ['self', 'x']);
  eq('top-level stmts', fn.body.map((s) => s.kind), ['assign', 'for', 'return']);

  const loop = fn.body[1];
  eq('loop kind', loop.loopKind, 'pipeline');
  eq('loop trip', loop.tripCount, 8);
  eq('loop stage', loop.stage, 2);
  eq('loop carries', loop.initValues, ['x']);
  eq('if/else captured', [loop.body[0].body.length, loop.body[0].orelse.length], [1, 1]);

  const an = analyzeProgram(prog, src.split('\n'));
  const f = an.functions[0];
  eq('alloc space', f.allocs[0].space, 'Vec');
  eq('alloc size', f.allocs[0].size, 1024);
  eq('role', f.role, 'SubWorker');
  eq('stmt count', f.stmtCount, 6);

  const ct = controlTree(f);
  ok('control tree has the pipeline node', ct.nodes.some((n) => n.label === 'pipeline(8)'));
  ok('control tree has an else branch', ct.nodes.some((n) => n.type === 'else'));

  const df = dataflowGraph(f);
  ok('dataflow links x into tile.add', df.edges.some((e) => e.from === 'p:x' && e.to.startsWith('v:acc')));
}

// ── diff ──────────────────────────────────────────────────────────────
{
  const a = ['a', 'b', 'c', 'd'];
  const b = ['a', 'x', 'c', 'd'];
  const rows = diffLines(a, b);
  eq('diff counts', countChanges(rows), { add: 1, del: 1 });
  eq('diff keeps order', rows.map((r) => r.tag).join(''), '=-+==');

  const hunks = toHunks(rows, 1, 10, 20);
  eq('hunk count', hunks.length, 1);
  eq('hunk start lines', [hunks[0].aStart, hunks[0].bStart], [10, 20]);
}
{
  // Repeated near-identical lines: the unique anchor must win over a naive LCS.
  const a = ['x = load(0)', 'y = load(0)', 'z = load(0)', 'UNIQUE_TAIL'];
  const b = ['x = load(0)', 'y = load(0)', 'NEW = load(9)', 'z = load(0)', 'UNIQUE_TAIL'];
  const rows = diffLines(a, b);
  eq('anchored insert', countChanges(rows), { add: 1, del: 0 });
}
{
  const w = wordDiff('t: pl.Mem.Vec = load(a)', 't: pl.Mem.Mat = load(a)');
  ok('word diff marks only the changed token', w.left.filter((r) => r[0]).map((r) => r[1]).join('') === 'Vec');
}
{
  const e = tokenEdits('x = f(a, 0)', 'x = f(a, 4096)');
  eq('token edit', e.map((r) => [r.old, r.new]), [['0', '4096']]);
}
{
  // Unrelated deleted/added lines must not be reported as a substitution.
  const rows = diffLines(['alpha = one(1)', 'beta = two(2)'], ['gamma = three(3)', 'delta = four(4)']);
  const rw = rewritePatterns(rows);
  eq('no bogus pairing', rw.paired, 0);
  eq('counted as pure add/del', [rw.pureAdd, rw.pureDel], [2, 2]);
}
{
  const rows = diffLines(['a = f(x, 0)', 'b = f(y, 0)'], ['a = f(x, 512)', 'b = f(y, 512)']);
  const rw = rewritePatterns(rows);
  eq('substitution found', [rw.top[0].from, rw.top[0].to, rw.top[0].count], ['0', '512', 2]);
}

// ── markdown ──────────────────────────────────────────────────────────
{
  eq('escape', escapeHtml('<a & "b">'), '&lt;a &amp; &quot;b&quot;&gt;');
  ok('inline code', mdInline('use `pl.tile.load`').includes('<code>pl.tile.load</code>'));
  ok('inline bold', mdInline('**核心**').includes('<strong>核心</strong>'));
  ok('unsafe link neutralised', mdInline('[x](javascript:alert(1))').includes('href="#"'));

  // The horizontal rule used to stall the block loop forever.
  ok('horizontal rule terminates', md('para\n\n---\n\nmore').includes('<hr>'));
  ok('bare numeric line terminates', md('1.5x faster than before').includes('<p>'));
  ok('table', md('| a | b |\n| --- | --- |\n| 1 | 2 |').includes('<td>1</td>'));
  ok('fenced code', md('```cpp\nint x;\n```').includes('int x;'));
  ok('ordered list', md('1. first\n2. second').startsWith('<ol>'));
}
{
  // Every real pass doc must render, and must terminate.
  const dir = path.join(REPO, 'repo/pto/docs/zh-cn/dev/passes');
  if (fs.existsSync(dir)) {
    let rendered = 0;
    for (const file of fs.readdirSync(dir).filter((f) => f.endsWith('.md'))) {
      const raw = fs.readFileSync(path.join(dir, file), 'utf8');
      const doc = extractDoc(raw);
      const html = doc.blocks.map((b) => md(b.body)).join('');
      ok(`doc has sections: ${file}`, doc.blocks.length > 0);
      ok(`doc has a title: ${file}`, Boolean(doc.title));
      ok(`doc renders: ${file}`, typeof html === 'string');
      rendered++;
    }
    ok('rendered every pass doc', rendered > 30, `${rendered} docs`);
  }
}

// ── name mapping ──────────────────────────────────────────────────────
{
  eq('normalize FlattenTileNdTo2D', normalizeName('FlattenTileNdTo2D'), normalizeName('flatten_tile_nd_to_2d'));
  eq('normalize InitMemRef', normalizeName('InitMemRef'), normalizeName('init_memref'));
  eq('normalize SynthesizeAllReduceSignals', normalizeName('SynthesizeAllReduceSignals'), normalizeName('synthesize_allreduce_signals'));
}

// ── the browser bundle really exposes what app.js calls ───────────────
{
  const bundlePath = path.join(HERE, 'bundle.js');
  if (fs.existsSync(bundlePath)) {
    const sandbox = { window: {} };
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(bundlePath, 'utf8'), sandbox);
    const lib = sandbox.window.PTXLib;
    const needed = ['parseDump', 'analyzeProgram', 'diffLines', 'toHunks', 'countChanges', 'wordDiff',
      'callGraph', 'controlTree', 'dataflowGraph', 'taskGraph', 'memoryView', 'fmtBytes',
      'md', 'mdInline', 'escapeHtml'];
    for (const n of needed) ok(`bundle exports ${n}`, typeof lib[n] === 'function');
    ok('bundle has no NUL bytes', !fs.readFileSync(bundlePath, 'utf8').includes('\0'));
  } else {
    ok('bundle exists', false, 'run `node build.mjs` first');
  }
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
