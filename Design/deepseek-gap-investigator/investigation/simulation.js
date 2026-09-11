/* Explicit product-demo fixtures. No mock field is merged into a Case, source
 * archive, real Trace, experiment definition, or measured result. */
window.InvestigationSimulation=function(A){
 const {esc,icon,badge,button,card,table,code}=InvestigationUI;
 const sessions=new Map(),MiB=1024*1024;
 const fixture={id:'SIM-COMPILER-01',origin:'simulation',tileBefore:64,tileAfter:32,budgetKiB:64,peakBeforeKiB:96,peakAfterKiB:48,copyBytes:32*MiB,waitBefore:663,waitAfter:343};
 const supported=()=>A.activeCase()?.preset==='a2a'&&A.activeCase()?.rank===0;
 const active=()=>supported()&&A.state.simulationMode===true;
 function syncLocation(){
  if(A.state.simulationMode&&!supported()){A.state.simulationMode=false;A.workarea.clearSimulation();}
  const url=new URL(location.href);if(active())url.searchParams.set('demo','compiler');else url.searchParams.delete('demo');
  if(url.href!==location.href)history.replaceState(null,'',url.href);
  document.title=active()?'Trace Investigator · 模拟 Pass Diff | DeepSeek V4':'Trace Investigator · 真实证据 | DeepSeek V4';
 }
 const session=()=>{const id=A.activeCase().case_id;if(!sessions.has(id))sessions.set(id,{defined:false,result:null});return sessions.get(id);};
 const us=n=>A.workflow.us(n);
 function context(){return active()?'<div class="simulation-context" role="note"><strong>模拟补全</strong><span>主 Trace 为真实记录；03–06 的原因、图和结果为演示数据。</span></div>':'';}
 function heading(title,text){return `<div class="step-heading">${icon('flask')}<div><h2>${title}</h2><p>${text}</p></div></div>`;}
 const pass23='_jit_l3_decode_csa_20260903_010617/passes_dump/23_after_ExpandMixedKernel.py';
 const pass24='_jit_l3_decode_csa_20260903_010617/passes_dump/24_after_InjectGMPipeBuffer.py';
 const edge=(source,target,symbol='')=>({source,target,kind:'data',symbol,source_ref:{simulation:true}});
 function node(id,label,text,type='op',signature=id,line=3449,file=pass23,basis='真实 DeepSeekV4 快照'){
  return {id,label,text,type,signature,source:{simulation:true,function:'decode_csa',file,line,basis}};
 }
 function pair(){
  const inputs=node('before-inputs','inputs + deps','attention_grouped_inline1276__ssa_v0: pl.Tensor[[2048, 4096], pl.BF16]\ndeps=[qk_tid_inline1229__ssa_v0, attn_rope_tid_inline1294__ssa_v0]','tensor','inputs:pack-publish',3447);
  const submit=node('before-submit','pl.spmd_submit','pl.spmd_submit(self.csa_merge_pack_publish_spmd, attention_grouped_inline1276__ssa_v0, …, deps=[qk_tid_inline1229__ssa_v0, attn_rope_tid_inline1294__ssa_v0], core_num=48)','op','submit:pack_publish:without-gm',3449);
  const publishTid=node('before-publish-tid','publish_tid · TASK_ID','publish_tid_inline1260__ssa_v0 = ret__tmp_v0_45[0]','outcast','task:publish_tid',3450);
  const wait=node('before-wait','pl.submit(a2a_wait)','pl.submit(self.o_group_a2a_wait, tp_rank__ssa_v0, attention_signal__ssa_v0, deps=[publish_tid_inline1260__ssa_v0])','op','submit:o_group_a2a_wait',3451);
  inputs.symbol='inputs + deps';publishTid.symbol='publish_tid';
  const inputsAfter=node('after-inputs',inputs.label,inputs.text,inputs.type,inputs.signature,3447);
  const pipeBuffer=node('sim-gm-pipe-buffer','gm_pipe_buffer','gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32] = pl.tensor.create([1], dtype=pl.FP32, layout=pl.TensorLayout.ND, manual_dep=True)\n\n真实 Pass 24 会以这一形式为部分 SPMD 任务增加 GM pipe buffer；将其映射到当前 producer 以及 320 µs 代价属于模拟。','tensor','tensor:gm_pipe_buffer',3389,pass24,'真实 Pass 24 语法 + 模拟因果映射');
  pipeBuffer.layoutRow=1;
  const submitAfter=node('after-submit','pl.spmd_submit + GM','pl.spmd_submit(self.csa_merge_pack_publish_spmd, attention_grouped_inline1276__ssa_v0, …, gm_pipe_buffer_0, deps=[qk_tid_inline1229__ssa_v0, attn_rope_tid_inline1294__ssa_v0], core_num=48)\n\n模拟该附加依赖使发布路径增加 320 µs。','op','submit:pack_publish:with-gm',3390,pass24,'真实 Pass 24 调用形态 + 模拟 producer 映射');
  const publishTidAfter=node('after-publish-tid',publishTid.label,publishTid.text,publishTid.type,publishTid.signature,3450);
  const waitAfter=node('after-wait',wait.label,wait.text,wait.type,wait.signature,3451);
  inputsAfter.symbol='inputs + deps';pipeBuffer.symbol='gm_pipe_buffer_0';publishTidAfter.symbol='publish_tid';
  const graph=(nodes,edges)=>({scope:'SIM-PRODUCER-01',nodes,edges,warnings:[],simulation:true});
  return {
   before:{label:'23 · after_ExpandMixedKernel'},
   after:{label:'24 · after_InjectGMPipeBuffer · 模拟映射'},
   a:graph([inputs,submit,publishTid,wait],[edge(inputs.id,submit.id,'inputs + deps'),edge(submit.id,publishTid.id,'TASK_ID'),edge(publishTid.id,wait.id,'deps')]),
   b:graph([inputsAfter,pipeBuffer,submitAfter,publishTidAfter,waitAfter],[edge(inputsAfter.id,submitAfter.id,'inputs + deps'),edge(pipeBuffer.id,submitAfter.id,'gm_pipe_buffer'),edge(submitAfter.id,publishTidAfter.id,'TASK_ID'),edge(publishTidAfter.id,waitAfter.id,'deps')])
  };
 }
 function stepThree(){
  return heading('上游多了一次 GM 往返搬运，数据晚到 320 µs','模拟记录：消费任务在等待期间尚未就绪；信号到达后 5 µs 启动。这里的瓶颈在数据到达之前。')+
   card('模拟等待记录',table(['相对时间','发生的事','说明'],[
    ['0 µs','开始等待发布信号','以本示例等待开始为零点；不是实际跨 Rank 时钟'],
    ['200 → 520 µs','生产者把中间结果写入 GM，再读回','额外搬运 32 MiB，持续 320 µs'],
    ['520 → 647 µs','发布数据','必须先完成上面的回读'],
    ['663 µs','信号到达，消费任务依赖就绪','等待解除'],
    ['664 / 666 / 668 µs','进入队列 / 下发 / 实际开始','就绪后 5 µs 启动，不存在长时间就绪排队']
   ]))+
   card('为什么继续检查编译',`<p>模拟映射将额外搬运关联到 <code>SIM-PRODUCER-01 → gm_pipe_buffer</code>。因此下一步直接对照生产任务在 Pass 23 与 Pass 24 的依赖链。</p><p>模拟资源记录：核和调度队列在依赖就绪后可用；片上空间不足发生在上游生产者。</p>${button('04 · 对照 Pass 变化','simulation-step','btn-solid','data-target-step="4"')}`);
 }
 function stepFour(){
  return `<div class="simulation-pass-step">${heading('Pass Diff · 模拟新增 GM buffer 后，发布依赖链变长','基线任务链来自真实 DeepSeekV4 Pass 23；Pass 24 的 gm_pipe_buffer 写法来自真实归档。高亮节点、与当前 producer 的映射和 320 µs 代价为模拟数据。')}<div id="simulationPassGraph" class="pass-comparison-direct" aria-label="基于真实 Pass 结构的模拟编译变化对照"></div><div class="inline simulation-pass-legend">${badge('模拟因果','warning')}<span>真实结构：pack_publish → publish_tid → o_group_a2a_wait</span><span>模拟变化：InjectGMPipeBuffer 为 pack_publish 增加 GM buffer 输入</span></div><div class="simulation-causal-chain" aria-label="Pass 变化到泳道异常的因果链"><span><strong>01</strong> 新增 gm_pipe_buffer</span><i>→</i><span><strong>02</strong> pack_publish +320 µs</span><i>→</i><span><strong>03</strong> publish_tid 晚完成</span><i>→</i><span><strong>04</strong> wait 依赖晚解除</span></div><p class="small-meta">依据：真实任务链见 Pass 23 L3447–3451；真实 GM pipe buffer 语法见 Pass 24 L3389–3390。归档中 Pass 24 没有修改 csa_merge_pack_publish；这里的关联仅用于补全产品流程。</p></div>`;
 }
 function stepFive(){
  return heading('减小生产者分块，让中间结果留在片上','本示例只修改一个参数：每个分块从 64 行减到 32 行。模拟峰值从 96 KiB 降到 48 KiB，低于 64 KiB 预算。')+
   card('模拟修改',`<dl class="definition"><dt>参数</dt><dd><code>DEMO_PUBLISH_TILE_ROWS</code> · 演示参数</dd><dt>修改</dt><dd>64 → 32</dd><dt>预期影响</dt><dd>去掉 32 MiB 的 GM 往返搬运，使发布与消费端启动提前</dd><dt>需要检查</dt><dd>更小分块是否增加循环和调度开销，结果是否正确</dd></dl><details class="source-disclosure"><summary>展开示例配置 · 非工程源码</summary>${code({start:1,line:2,lines:['# 模拟配置；并非当前 PyPTO 工程参数','- DEMO_PUBLISH_TILE_ROWS = 64','+ DEMO_PUBLISH_TILE_ROWS = 32']})}</details><div class="inline">${button(session().defined?'06 · 验证模拟方案':'采用此模拟方案','simulation-define','btn-solid')}</div>`)+
   '<p class="small-meta">此模拟固定其余开销，用于验证产品交互；实际修改是否有效，需要重新编译和设备实测。</p>';
 }
 function trace(optimized=false){
  const lanes=['AIV_0','DMA_0','AIV_1','AIC_0'];
  const events=[{ph:'M',name:'process_name',pid:4,args:{name:'Worker View'}},...lanes.map((name,tid)=>({ph:'M',name:'thread_name',pid:4,tid,args:{name}}))];
  const event=(tid,name,start,duration)=>({ph:'X',pid:4,tid,name,ts:start,dur:duration,args:{provenance:'simulation',scenario_id:fixture.id}});
  events.push(event(0,'模拟 · 打包',0,200));
  if(!optimized)events.push(event(1,'模拟 · Store 16 MiB',200,176),event(1,'模拟 · Load 16 MiB',376,144));
  events.push(event(0,'模拟 · 发布数据',optimized?200:520,127),event(2,'模拟 · 等待信号',0,optimized?fixture.waitAfter:fixture.waitBefore),event(3,'模拟 · 下游计算',optimized?348:668,42));
  return {provenance:'simulation',scenario_id:fixture.id,traceEvents:events};
 }
 function buildResult(){
  return {provenance:'simulation',scenario_id:fixture.id,experiment_id:'SIM-EXP-01',edit:{symbol:'DEMO_PUBLISH_TILE_ROWS',before:64,after:32},
   baseline:{wait_us:fixture.waitBefore,scene_us:710,copy_bytes:fixture.copyBytes,peak_kib:96,trace:trace()},
   candidate:{wait_us:fixture.waitAfter,scene_us:390,copy_bytes:0,peak_kib:48,trace:trace(true)},correctness:'not-executed'};
 }
 function stepSix(){
  const s=session(),r=s.result;
  return heading(r?'模拟中，搬运减少，等待缩短 320 µs':'验证减小分块能否缩短等待',r?'两份模拟 Trace 使用同一时钟、同一计算量；只移除示例中的额外搬运。':'方案：DEMO_PUBLISH_TILE_ROWS 64 → 32。生成模拟结果后，可在主工作区对照两份模拟泳道。')+
   card('模拟实验 · SIM-EXP-01',`<p>${s.defined?'已采用减小分块方案。':'请先在 05 采用模拟方案。'}此示例不执行开发工程，也不运行设备。</p><div class="inline">${s.defined?button(r?'重新生成模拟结果':'生成模拟结果','simulation-run','btn-solid'):button('05 · 采用调优方案','simulation-step','btn-solid','data-target-step="5"')}${r?button('查看模拟泳道对比','simulation-diff','btn-sm'):''}</div>`)+
   (r?card('模拟结果',table(['指标','模拟修改前','模拟修改后'],[
    ['本次等待',us(r.baseline.wait_us),us(r.candidate.wait_us)],
    ['额外 GM 搬运','32 MiB','0 MiB'],['片上峰值','96 KiB','48 KiB'],
    ['局部场景总长',us(r.baseline.scene_us),us(r.candidate.scene_us)],
    ['正确性 / 真实端到端收益','未执行 / 未测量','未执行 / 未测量']
   ])):'');
 }
 function renderStep(step){return [stepThree,stepFour,stepFive,stepSix][step-3]?.()||'';}
 function mount(){const root=document.getElementById('simulationPassGraph');if(root)A.workarea.renderPassComparison(root,pair());}
 function openDiff(){
  const r=session().result;if(!r)return;
  const models=['baseline','candidate'].map(side=>TraceModel.parse(r[side].trace,{runId:`${fixture.id}-${side}`,rank:0,name:`模拟${side==='baseline'?'修改前':'修改后'}`,source:'显式模拟数据'}));
  A.workarea.compareSimulation(...models);A.closeInvestigation();
 }
 function action(el){
  if(!active())return false;
  switch(el.dataset.action){
   case 'simulation-step':A.activeCase().step=Number(el.dataset.targetStep);break;
   case 'simulation-define':session().defined=true;A.activeCase().step=6;break;
   case 'simulation-run':if(session().defined)session().result=buildResult();break;
   case 'simulation-diff':openDiff();return true;
   case 'simulation-return':A.workarea.show('trace');A.openInvestigation();return true;
   default:return false;
  }
  A.renderCase();return true;
 }
 return {supported,active,syncLocation,context,renderStep,mount,action,fixture,pair,buildResult};
};
