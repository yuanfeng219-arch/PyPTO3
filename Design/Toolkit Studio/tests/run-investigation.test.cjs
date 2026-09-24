const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
function load() {
  const context = vm.createContext({window:{location:{search:""}}, console, URLSearchParams});
  for (const name of ['ir-kernels-data.js','ir-compilation-view.js','correctness-diagnostic-data.js','run-investigation-data.js']) vm.runInContext(fs.readFileSync(path.join(__dirname,'../js',name),'utf8'),context);
  return context.window;
}
test('strict Run identity and valid graph references',()=>{
 const w=load(), api=w.PTO_RUN_INVESTIGATION;
 assert.equal(api.build('unknown'),null);
 for(const run of ['run_106','run_109']) {
  const g=api.build(run), ids=new Set(g.nodes.map(n=>n.id));
  for(const e of g.edges) {assert.ok(ids.has(e.source));assert.ok(ids.has(e.target));assert.ok(e.relation);}
  for(const n of g.nodes) for(const ref of n.evidenceRefs) assert.ok(g.evidenceById[ref]);
  assert.doesNotMatch(JSON.stringify(g),/mock|fixture|一键修复/i);
 }
});
test('distinct errors and missing reference downgrade',()=>{
 const w=load(), p=w.PTO_CORRECTNESS_DIAGNOSTICS.profiles.run_106, api=w.PTO_RUN_INVESTIGATION;
 let g=api.build('run_106');
 assert.equal(g.evidenceById.output.rows[0][1],p.result.maxAbs);
 assert.equal(g.evidenceById['first-divergence'].rows[0][1],p.tensors.find(t=>t.id==='attention_out').maxAbs);
 p.tensors.find(t=>t.id==='attention_out').ref=false;
 g=api.build('run_106');assert.equal(g.nodes.find(n=>n.id==='first-attention-out').state,'unresolved');assert.match(g.judgment,/不足/);
 delete p.runtime; assert.doesNotThrow(()=>api.build('run_106'));
});
test('navigation rejects cross Run and invalid objects; preserves overlap',()=>{
 const w=load(), api=w.PTO_RUN_INVESTIGATION, a=api.build('run_106').action;
 assert.equal(api.resolveAction('run_109',a),null);
 assert.equal(api.resolveAction('run_106',{...a,taskIds:['999']}),null);
 assert.equal(api.resolveAction('run_106',{...a,range:{from:3,to:1}}),null);
 assert.deepEqual(JSON.parse(JSON.stringify(api.resolveAction('run_106',a).taskIds)),['182','197']);
 assert.equal(api.resolveAction('run_106',a).range.from,238);
});
test('compiler uses recorded status and degrades without pass evidence',()=>{
 const w=load(), api=w.PTO_RUN_INVESTIGATION, f=w.PTO_COMPILATION.numericalFixtures.compiler_semantic_error;
 const g=api.build('run_109'); assert.equal(g.evidenceById['compiler-numerical-validation'].rows[0][1], 'mismatch');
 assert.equal(g.evidenceById['compiler-tolerance'].rows[0][0],f.tolerance.rtol);
 f.passes.splice(0); assert.match(api.build('run_109').judgment,/不足/);
});
