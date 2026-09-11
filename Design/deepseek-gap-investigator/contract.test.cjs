const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
vm.runInThisContext(fs.readFileSync(path.join(__dirname,'adapters/trace.js'),'utf8'));
const sample={traceEvents:[{ph:'M',name:'process_name',pid:4,args:{name:'Worker View'}},{ph:'M',name:'process_name',pid:3,args:{name:'Scheduler View'}},...[3,4].map(pid=>({ph:'M',name:'thread_name',pid,tid:1,args:{name:'AIC_0'}})),{ph:'X',pid:4,tid:1,ts:0,dur:10},{ph:'X',pid:4,tid:1,ts:5,dur:10},{ph:'X',pid:3,tid:1,ts:0,dur:20}]};
const model=TraceModel.parse(sample,{runId:'test',rank:0});assert.equal(model.lanes.length,2);assert.equal(model.lanes.find(l=>l.worker).busy,15);assert.deepEqual(TraceModel.gaps(model.lanes.filter(l=>l.worker),0,20,1),[[15,20]]);
const data=JSON.parse(fs.readFileSync(path.join(__dirname,'pass-graphs.json')));assert.equal(data.passes.length,52);
for(const pass of data.passes)for(const graph of Object.values(pass.graphs)){
 const ids=new Set(graph.nodes.map(n=>n.id));assert.equal(ids.size,graph.nodes.length);
 const source=fs.readFileSync(path.join(__dirname,'../../Data/DeepseekV4',graph.file),'utf8').split('\n');
 for(const n of graph.nodes){assert.ok(n.source.line>0&&n.source.line<=source.length);assert.ok(n.text&&n.signature);}
 for(const e of graph.edges){assert.ok(ids.has(e.source)&&ids.has(e.target));assert.ok(['data','control'].includes(e.kind));assert.ok(e.source_ref.line>0);}
}
const fenceGraph=data.passes.find(pass=>pass.index===50).graphs.a2a;
const fence=fenceGraph.nodes.find(node=>node.label==='pl.system.cacheinvalid');
assert.ok(fence,'InsertCommFence output must include cache invalidation');
assert.ok(fenceGraph.edges.some(edge=>edge.target===fence.id&&edge.symbol==='源码顺序'),'side-effect-only nodes must show their lexical order');
const manifest=JSON.parse(fs.readFileSync(path.join(__dirname,'project-manifest.json')));assert.equal(manifest.files.length,49);assert.equal(manifest.traces.length,2);assert.equal(manifest.execution,'unbound');
for(const file of ['app.js','render/workarea.js','investigation/project.js','investigation/report.js','investigation/workflow.js','investigation/experiments.js','investigation/simulation.js'])new vm.Script(fs.readFileSync(path.join(__dirname,file),'utf8'),{filename:file});
assert.ok(!fs.readFileSync(path.join(__dirname,'investigation/report.js'),'utf8').includes('mountTimeline'));
const context={document:{getElementById:()=>null},requestAnimationFrame:fn=>fn()};context.window=context;vm.createContext(context);
for(const file of ['data.js','components/ui.js','investigation/report.js','investigation/workflow.js','investigation/simulation.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,file),'utf8'),context);
for(const preset of ['a2a','token']){
 const current={preset,rank:0,step:4,report:{pass:50}},A={activeCase:()=>current,state:{simulationMode:false},workarea:{clearSimulation(){},renderPassComparison(){}},persist(){},renderCase(){}};
 A.workflow=context.InvestigationWorkflow(A);
 A.simulation=context.InvestigationSimulation(A);
 const report=A.workflow.report,html=report.renderStep(),task=context.INVESTIGATION_DATA.cases[preset].ranks[0].tasks[A.workflow.meta[preset].wait];
 if(preset==='a2a'){
  assert.ok(html.includes('Pass Diff')&&html.includes('gm_pipe_buffer'),'supported Step 04 must render the Pass diff directly');
  assert.ok(!html.includes('simulation-open-pass'),'Step 04 must not require a second open button');
 }else{
  assert.ok(html.includes(task.name)&&html.includes(task.id),'Step 04 must retain the current anomaly task when no mock mapping exists');
  assert.ok(html.includes('尚未定位到相关的编译变化'));
 }
 assert.doesNotMatch(html,/stepPassComparison|data-pass-select|InsertCommFence|cacheinvalid/,'saved Pass choices cannot substitute for a relevant compilation target');
 report.mount();
 assert.equal(report.action({dataset:{action:'report-upstream'}}),true);assert.equal(current.step,3);assert.equal(current.tab,'upstream');
}
const realData=JSON.stringify(context.INVESTIGATION_DATA),mockCase={case_id:'test-mock',preset:'a2a',rank:0,step:4},mockApp={activeCase:()=>mockCase,state:{simulationMode:true}};
const simulation=context.InvestigationSimulation(mockApp),pair=simulation.pair(),result=simulation.buildResult();
assert.ok(simulation.active());assert.equal(pair.b.nodes.length-pair.a.nodes.length,1);
assert.ok(pair.b.edges.some(e=>e.source==='sim-gm-pipe-buffer'&&e.target==='after-submit')&&pair.b.edges.some(e=>e.source==='after-submit'&&e.target==='after-publish-tid'),'mock GM buffer must feed the real-shaped publish task dependency chain');
assert.ok(pair.b.nodes.find(n=>n.id==='sim-gm-pipe-buffer').source.file.endsWith('24_after_InjectGMPipeBuffer.py'),'mock node must retain its real Pass syntax reference');
for(const side of ['baseline','candidate']){
 const part=result[side],model=TraceModel.parse(part.trace,{runId:'SIM-'+side,rank:0});
 assert.equal(model.duration,part.scene_us);assert.equal(model.tasks.find(t=>t.rawName.includes('等待信号')).duration,part.wait_us);
 assert.ok(part.trace.traceEvents.filter(e=>e.ph==='X').every(e=>e.args.provenance==='simulation'));
}
assert.equal(result.baseline.wait_us-result.candidate.wait_us,320);assert.equal(result.baseline.scene_us-result.candidate.scene_us,320);
assert.equal(result.correctness,'not-executed');assert.equal(JSON.stringify(context.INVESTIGATION_DATA),realData);assert.equal(mockCase.experiment,undefined);
mockCase.preset='token';assert.equal(simulation.active(),false);
console.log('PASS: source/trace contracts, unrelated Pass excluded, mock causality/timing/provenance isolated');
