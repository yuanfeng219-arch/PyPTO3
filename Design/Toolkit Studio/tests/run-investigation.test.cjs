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
test('Run #106 exposes two investigation cases with explicit horizontal spines',()=>{
 const w=load(), api=w.PTO_RUN_INVESTIGATION;
 const cases=api.cases('run_106');
 assert.deepEqual(JSON.parse(JSON.stringify(cases.map(item=>item.title))),['输出偏差','排序风险']);
 assert.deepEqual(JSON.parse(JSON.stringify(cases.map(item=>item.marker))),['FAIL · 正确性','WARNING · Runtime']);
 assert.deepEqual(JSON.parse(JSON.stringify(cases.map(item=>item.location))),['attention_out → Task #182','B2 · Task #182 / #197']);
 assert.equal(cases[0].description.includes('attention_out'),true);
 assert.equal(cases[1].description.includes('WAR'),true);
 for (const item of cases) {
  const spine=item.investigationData.nodes.filter(node=>Number.isFinite(node.spineIndex)).sort((a,b)=>a.spineIndex-b.spineIndex);
  assert.equal(spine.length,7);
  assert.deepEqual(JSON.parse(JSON.stringify(spine.map(node=>node.spineIndex))),[0,1,2,3,4,5,6]);
  for (const node of item.investigationData.nodes) {
   assert.ok(node.evidenceRefs.every(ref=>item.investigationData.evidenceById[ref]));
  }
 }
 assert.deepEqual(JSON.parse(JSON.stringify(api.cases('unknown'))),[]);
});
test('Run #109 compiler investigation uses an explicit horizontal spine and peripheral branches',()=>{
 const w=load(), graph=w.PTO_RUN_INVESTIGATION.build('run_109');
 const spine=graph.nodes.filter(node=>Number.isFinite(node.spineIndex)).sort((a,b)=>a.spineIndex-b.spineIndex);
 assert.deepEqual(JSON.parse(JSON.stringify(spine.map(node=>node.id))),[
  'compiler-validation','validation-split','expand-mixed-kernel','adjacent-pass-ir','compiler-diagnosis'
 ]);
 assert.deepEqual(JSON.parse(JSON.stringify(spine.map(node=>node.spineIndex))),[0,1,2,3,4]);
 assert.equal(graph.nodes.find(node=>node.id==='structure').lane,'above');
 assert.equal(graph.nodes.find(node=>node.id==='device-scope').lane,'below');
 assert.equal(graph.edges.some(edge=>edge.source==='device-scope'&&edge.target==='compiler-diagnosis'),false);
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
