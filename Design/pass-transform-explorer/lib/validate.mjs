// Parser self-check: parse every dump file in both runs and report any line
// that the structured parser failed to account for.
import fs from 'node:fs';
import path from 'node:path';
import { parseDump, parserErrors, resetParserErrors, walkStmts } from './pyir.mjs';

const ROOTS = [
  'D:/project/PyPTO3/Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump',
  'D:/project/PyPTO3/Data/_jit_decode_fwd_layers_20260625_184941/passes_dump',
];

let totalFiles = 0;
let totalStmts = 0;
let totalUnaccounted = 0;
const opNames = new Map();
const unaccountedSamples = [];

resetParserErrors();

for (const root of ROOTS) {
  const files = fs.readdirSync(root).filter((f) => f.endsWith('.py')).sort();
  for (const f of files) {
    const text = fs.readFileSync(path.join(root, f), 'utf8');
    const prog = parseDump(text, f);
    totalFiles++;

    const covered = new Set();
    for (const fn of prog.functions) {
      covered.add(fn.line);
      covered.add(fn.decoLine);
      walkStmts(fn.body, (s) => {
        covered.add(s.line);
        totalStmts++;
        if (s.op) opNames.set(s.op, (opNames.get(s.op) || 0) + 1);
      });
    }

    text.split(/\r?\n/).forEach((line, i) => {
      const t = line.trim();
      if (!t || t.startsWith('#')) return;
      if (t.startsWith('import ') || t.startsWith('from ') || t.startsWith('class ') || t === '@pl.program') return;
      if (t === 'else:') return;
      if (covered.has(i + 1)) return;
      if (prog.globals.some((g) => g.line === i + 1)) return;
      totalUnaccounted++;
      if (unaccountedSamples.length < 20) unaccountedSamples.push(`${f}:${i + 1}  ${t.slice(0, 140)}`);
    });
  }
}

console.log(`files parsed        : ${totalFiles}`);
console.log(`statements parsed   : ${totalStmts}`);
console.log(`unaccounted lines   : ${totalUnaccounted}`);
console.log(`expression failures : ${parserErrors().length}`);
for (const e of parserErrors().slice(0, 10)) console.log(`  ! ${e.where} ${e.msg} :: ${e.src.slice(0, 120)}`);
for (const s of unaccountedSamples) console.log(`  ? ${s}`);

const ops = [...opNames.entries()].sort((a, b) => b[1] - a[1]);
console.log(`\ndistinct ops: ${ops.length}`);
console.log(ops.slice(0, 45).map(([k, v]) => `${k}=${v}`).join('  '));
