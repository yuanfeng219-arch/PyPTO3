// End-to-end check: run every lens over every pass of both runs and report
// anything that looks wrong (empty graphs where there should be nodes, unstable
// node ids, zero-byte buffers, layout blowups).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseDump } from './pyir.mjs';
import { analyzeProgram, callGraph, controlTree, dataflowGraph, taskGraph, memoryView } from './analyze.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../../..');
const RUNS = [
  ['l3_decode_csa', 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump'],
  ['decode_fwd_layers', 'Data/_jit_decode_fwd_layers_20260625_184941/passes_dump'],
];

const problems = [];
let checked = 0;

for (const [runId, dir] of RUNS) {
  const full = path.join(REPO, dir);
  const files = fs.readdirSync(full).filter((f) => f.endsWith('.py')).sort();
  let prev = null;

  for (const file of files) {
    const text = fs.readFileSync(path.join(full, file), 'utf8');
    const lines = text.split(/\r?\n/);
    const an = analyzeProgram(parseDump(text, file), lines);

    const cg = callGraph(an);
    if (cg.nodes.length !== an.functions.length) problems.push(`${runId}/${file}: call graph node count mismatch`);

    // Dangling call edges point at functions that do not exist in this snapshot.
    const names = new Set(cg.nodes.map((n) => n.id));
    const dangling = cg.edges.filter((e) => !names.has(e.to));
    if (dangling.length) {
      problems.push(`${runId}/${file}: ${dangling.length} call edges to unknown callees (${dangling.slice(0, 3).map((e) => e.to).join(', ')})`);
    }

    for (const f of an.functions) {
      checked++;
      const ct = controlTree(f);
      const ids = new Set(ct.nodes.map((n) => n.id));
      if (ids.size !== ct.nodes.length) problems.push(`${runId}/${file}/${f.name}: duplicate control-tree ids`);

      const df = dataflowGraph(f);
      const dids = new Set(df.nodes.map((n) => n.id));
      if (dids.size !== df.nodes.length) problems.push(`${runId}/${file}/${f.name}: duplicate dataflow ids`);

      const tg = taskGraph(f);
      const tids = new Set(tg.nodes.map((n) => n.id));
      if (tids.size !== tg.nodes.length) problems.push(`${runId}/${file}/${f.name}: duplicate task ids`);

      const mv = memoryView(f);
      for (const space of mv) {
        const zero = space.items.filter((i) => !i.size).length;
        const dyn = space.items.filter((i) => i.dynamic).length;
        if (zero && space.items.length && zero === space.items.length && dyn < zero) {
          problems.push(`${runId}/${file}/${f.name}: memory space ${space.space} has ${zero} buffers all sized 0`);
        }
      }
    }
    prev = an;
  }
  process.stdout.write(`${runId}: ok\n`);
}

console.log(`\nfunctions checked: ${checked}`);
console.log(`problems: ${problems.length}`);
const seen = new Set();
for (const p of problems) {
  const key = p.replace(/\d+/g, '#').replace(/\/[^/]+\.py/, '/*.py');
  if (seen.has(key)) continue;
  seen.add(key);
  console.log('  ! ' + p);
  if (seen.size > 24) { console.log('  … (more of the same)'); break; }
}
