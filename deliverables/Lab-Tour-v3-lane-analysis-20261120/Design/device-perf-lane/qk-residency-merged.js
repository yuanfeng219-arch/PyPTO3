// #2080-inspired demonstration, bound to the archived QK task. Original input is never mutated.
window.QK_RESIDENCY = (() => {
  const id='8589934645', kernels=['qk_pv_aic','qk_pv_aiv'];
  const e=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const us=n=>n.toFixed(2)+' μs';
  const isTarget=t=>t?.taskId===id&&kernels.includes(t.leaf);
  function hero(raw) {
    return raw.launches.find(t=>t.leaf==='qk_pv_aic').blocks.items.reduce((a,b)=>a.computeUs>b.computeUs?a:b);
  }
  function withLongBlock(raw){
    const rank=JSON.parse(JSON.stringify(raw)),task=rank.launches.find(t=>t.leaf==='qk_pv_aic'),block=hero(rank);
    const round=n=>Math.round(n*100)/100,total=(items,key)=>round(items.reduce((sum,item)=>sum+item[key],0));
    const original=block.computeUs,cut=block.endUs,extra=round(original*.8);
    rank.demoLongBlock={core:block.core,originalComputeUs:original,computeUs:round(original+extra)};
    // Extend one block in the presentation copy. Move the later context rather
    // than leave successors at their original completion times.
    for(const launch of rank.launches){
      if(launch===task){
        for(const b of launch.blocks.items){if(b===block){b.computeUs=round(b.computeUs+extra);b.slotUs=round(b.slotUs+extra);b.endUs=round(b.endUs+extra);}else if(b.endUs>cut){b.startUs=round(b.startUs+extra);b.endUs=round(b.endUs+extra);}}
      }else if(launch.endUs>cut){for(const b of launch.blocks.items){b.startUs=round(b.startUs+extra);b.endUs=round(b.endUs+extra);}}
      launch.startUs=Math.min(...launch.blocks.items.map(b=>b.startUs));launch.endUs=Math.max(...launch.blocks.items.map(b=>b.endUs));launch.spanUs=round(launch.endUs-launch.startUs);
      if(launch===task){const values=launch.blocks.items.map(b=>b.computeUs).sort((a,b)=>a-b),mean=total(launch.blocks.items,'computeUs')/values.length;launch.computeUs=total(launch.blocks.items,'computeUs');launch.slotUs=total(launch.blocks.items,'slotUs');Object.assign(launch.blocks,{durs:values,min:values[0],max:values.at(-1),mean:round(mean),p50:values[Math.floor((values.length-1)*.5)],p90:values[Math.floor((values.length-1)*.9)],spread:round(values.at(-1)/mean)});launch.tailWasteUs=round(values.at(-1)*values.length-launch.computeUs);}
      launch.gbPerS=launch.spanUs?round(launch.bytes/launch.spanUs/1000):null;
    }
    for(const core of rank.cores){const blocks=rank.launches.flatMap(t=>t.blocks.items.filter(b=>b.core===core.id));for(const key of ['computeUs','setupUs','slotUs'])core[key]=total(blocks,key);core.waitUs=round(Math.max(0,core.slotUs-core.computeUs-core.setupUs));}
    rank.windowUs=round(rank.windowUs+extra);const k=rank.kpi;k.windowUs=rank.windowUs;
    for(const key of ['computeUs','setupUs','slotUs','tailWasteUs'])k[key]=total(rank.launches,key);
    k.capacityUs=round(k.windowUs*k.totalCores);k.occupancy=k.computeUs/k.capacityUs;k.setupShare=k.setupUs/k.computeUs;k.tailWasteRatio=k.tailWasteUs/k.computeUs;
    for(const ct of ['AIC','AIV']){const cores=rank.cores.filter(c=>c.ct===ct),entry=k.byType[ct];for(const key of ['computeUs','setupUs','slotUs','waitUs'])entry[key]=total(cores,key);entry.computeShare=entry.computeUs/(entry.cores*k.windowUs);entry.slotShare=entry.slotUs/(entry.cores*k.windowUs);}
    let previous=0;for(const item of rank.criticalPath){const launches=rank.launches.filter(t=>t.taskId===item.taskId);if(launches.length){item.startUs=Math.min(...launches.map(t=>t.startUs));item.endUs=Math.max(...launches.map(t=>t.endUs));item.spanUs=round(item.endUs-item.startUs);item.computeUs=total(launches,'computeUs');}item.pathContributionUs=round(Math.max(0,item.endUs-previous));previous=Math.max(previous,item.endUs);}
    k.criticalPathUs=round(rank.criticalPath.at(-1).endUs-rank.criticalPath[0].startUs);
    let end=-Infinity,single=0;for(const item of rank.criticalPath.filter(p=>p.onCore&&p.blockNum===1).sort((a,b)=>a.startUs-b.startUs)){single+=Math.max(0,item.endUs-Math.max(item.startUs,end));end=Math.max(end,item.endUs);}k.singleBlockOnPathUs=round(single);k.singleBlockOnPathShare=single/k.criticalPathUs;
    return rank;
  }
  let header,panel,micro,reference,callbacks,ctx,pendingCore=null;
  let hoistRun=null,lastSkewKey='';
  const state={step:0,iter:0,object:'q',stage:'qk',core:null,leaf:null,gate:'safe',pipelinePass:'memory'};

  const stageInfo={qk:{label:'QK',leaf:'qk_pv_aic',op:'mm'},softmax:{label:'Softmax',leaf:'qk_pv_aiv',op:'softmax'},pv:{label:'PV',leaf:'qk_pv_aic',op:'pv'},dependency:{label:'跨核协作',leaf:'qk_pv_aic',op:'wait_p'},ooutput:{label:'O 写回',leaf:'qk_pv_aic',op:'write_o'},returnp:{label:'P 返回',leaf:'qk_pv_aiv',op:'send_p'},output:{label:'m、l 写回',leaf:'qk_pv_aiv',op:'write_ml'}};
  const opStage={q:'qk',kv:'qk',l0:'qk',mm:'qk',loop:'qk',pv:'pv',softmax:'softmax',scale:'softmax',exp:'softmax',reduce:'softmax',send_score:'dependency',wait_p:'dependency',recv_p:'dependency',send_p:'returnp',write_o:'ooutput',write_ml:'output'};
  const stageFor=op=>state.leaf==='qk_pv_aic'&&['scale','exp','reduce','send_p','write_ml'].includes(op)?'dependency':opStage[op]||'qk';
  const steps=['问题对象','计算过程','Pass 分析','检查条件','待验证项'];
  const passFor=op=>['q','kv','loop'].includes(op)?'memory':['l0','write_o'].includes(op)?'lower':['mm','pv','wait_p'].includes(op)?'skew':['send_score','recv_p','send_p'].includes(op)?'exchange':'split';
  const beforeCode=[['loop','for query, head in fixed_query_heads:'],['loop','    for kv_panel in range(32):'],['q','        qk_q_tile = load(q_flat, [head_row, 0], Mat)'],['kv','        qk_kv = load(current_kv_panel, ..., Mat)'],['l0','        q_left = extract(qk_q_tile, ..., Left)'],['mm','        qk_raw = matmul(q_left, kv_right)']];
  const afterCode=[['loop','for query, head in fixed_query_heads:'],['q','    qk_q_tile = load(q_flat, [head_row, 0], Mat)'],['loop','    for kv_panel in range(32):'],['kv','        qk_kv = load(current_kv_panel, ..., Mat)'],['l0','        q_left = extract(qk_q_tile, ..., Left)'],['mm','        qk_raw = matmul(q_left, kv_right)']];
  function pair(){const original=ctx.raw.launches.find(t=>t.leaf===state.leaf);if(!original)return null;let b=original.blocks.items.find(b=>b.core===state.core);if(!b)b=original.blocks.items.reduce((a,b)=>a.computeUs>b.computeUs?a:b);state.core=b.core;return{original,b};}
  const next=(n,text)=>'<button type="button" class="ctl rc-next" data-qk-step="'+n+'">'+text+' →</button>';
  function code(rows,title){return'<article class="rc-card rc-code"><h4>'+title+'</h4><div>'+rows.map(([id,text],i)=>'<button type="button" data-qk-object="'+id+'" class="'+(state.object===id?'selected':'')+'" aria-pressed="'+(state.object===id)+'"><em>'+(i+1)+'</em><code>'+e(text)+'</code></button>').join('')+'</div></article>';}
  function selectedEvent(p){
    const names={q:'MTE2 · Q 加载',kv:'MTE2 · KV 加载',l0:'MTE1 · L0 提取',mm:'CUBE · QK 计算',pv:'CUBE · PV 计算',softmax:'VEC · Softmax',scale:'VEC · 缩放',exp:'VEC · exp',reduce:'VEC · 行归约',send_score:'AIC → AIV · 发送分数',wait_p:'AIC · 等待 P',recv_p:'AIC ← AIV · 接收 P',send_p:'AIV → AIC · 返回 P',write_o:'AIC · O 写回',write_ml:'AIV · m、l 写回',loop:'循环作用域'};
    const matches=inlineEvents(p.b,state.leaf).filter(ev=>ev.iter===state.iter&&(ev.op===state.object||state.object==='softmax'&&ev.stage==='softmax'));
    const event=matches.length?{startUs:matches[0].startUs,endUs:matches.at(-1).endUs}:null;
    return '<button type="button" class="qk-linked-selection '+(state.object==='q'&&state.iter>0?'is-repeat':'')+'" data-qk-reveal title="'+(event?us(event.startUs)+'–'+us(event.endUs):'循环作用域')+'" aria-label="定位核内流水中选中的搬运或计算事件"><span class="qk-selection-dot"></span><span><b>'+names[state.object]+'</b><small>分片 '+state.iter+' · 第 '+(state.iter+1)+' 轮'+(matches[0]?.related?' · 关联 AIV':'')+'</small></span><em>↗ 泳道</em></button>';
  }
  function blockSummary(p){
    const aic=state.leaf==='qk_pv_aic',b=p.b;
    const tile=(name,shape,size)=>'<div class="qk-input-tile"><b>'+name+'</b><code>'+shape+'</code><small>'+size+'</small></div>';
    const inputs=state.stage==='dependency'?tile('分数 → AIV','32 × 128 · FP32','16 KiB')+tile('P ← AIV','32 × 128 · BF16','8 KiB'):state.stage==='ooutput'?tile('O','32 × 512 · FP32','64 KiB'):state.stage==='output'?tile('m','32 × 1 · FP32','128 B')+tile('l','32 × 1 · FP32','128 B'):state.stage==='pv'?tile('P','32 × 128 · BF16','8 KiB')+tile('V / KV','128 × 512 · BF16','128 KiB'):aic?tile('Q','32 × 512 · BF16','32 KiB')+tile('KV','128 × 512 · BF16','128 KiB'):tile('QK 分数','32 × 128 · FP32','16 KiB')+tile('P · exp 权重','32 × 128 · BF16','8 KiB');
    const flow=aic?[['qk','QK','Q × Kᵀ','AIC'],['dependency','跨核协作','发送分数 / 接收 P','AIC ↔ AIV'],['pv','PV','P × V','AIC'],['ooutput','O 写回','局部 O','AIC']]:[['softmax','Softmax','缩放 / exp / 归约','AIV'],['returnp','P 返回','BF16 → AIC','AIV'],['output','m、l 写回','局部统计','AIV']];
    const core='<label>Core <select class="ctl" data-qk-core aria-label="选择观测 block">'+p.original.blocks.items.map(item=>'<option value="'+item.core+'"'+(item.core===state.core?' selected':'')+'>'+item.core+'</option>').join('')+'</select></label>';
    const stages=flow.map(([key,name,math,device],i)=>'<button type="button" data-qk-stage="'+key+'" aria-pressed="'+(state.stage===key)+'" title="'+name+' · '+math+' · '+device+'" class="qk-compute-stage '+(state.stage===key?'current':'')+'"><b>'+name+'</b><small>'+math+'</small></button>'+(i<flow.length-1?'<span aria-hidden="true">→</span>':'')).join('');
    const tileLabel=state.stage==='dependency'?'分片传递':state.stage==='ooutput'||state.stage==='output'?'写回 tile':state.stage==='pv'?'PV 输入 tile':aic?'QK 输入 tile':'Softmax 输入 / 输出 tile';
    return '<section class="qk-block-summary" aria-label="所选 block 的计算与耗时"><div class="qk-block-scope">'+core+'<div class="qk-block-duration"><span>执行</span><strong>'+us(b.computeUs)+'</strong></div></div><details class="qk-compute-details" data-qk-section="compute"><summary>计算过程</summary><div class="qk-compute-flow" aria-label="当前 block 计算过程">'+stages+'</div></details><details class="qk-summary-details"><summary>详细信息</summary><div class="qk-summary-context">'+ctx.rank.rank+' · Task #'+id+'</div><dl class="qk-block-metrics">'+[['Setup',us(b.setupUs)],['同组均值',us(p.original.blocks.mean)]].map(([label,value])=>'<div><dt>'+label+'</dt><dd>'+value+'</dd></div>').join('')+'</dl><div class="qk-block-range">设备时间 <span>'+us(b.startUs)+' → '+us(b.endUs)+'</span></div><div class="qk-summary-label">'+tileLabel+'</div><div class="qk-inputs">'+inputs+'</div></details></section>';

  }
  function stopHoist(){
    window.QK_PIPELINE_PASSES?.stop();window.QK_PASS_EVOLUTION?.stop();
    const run=hoistRun;if(!run)return;hoistRun=null;
    if(run.frame)cancelAnimationFrame(run.frame);
    for(const animation of run.animations){animation.onfinish=null;animation.cancel();}
    run.ghost?.remove();run.host?.removeAttribute('aria-busy');
  }
  function playHoist(){
    stopHoist();
    const host=panel.querySelector('.qk-structure-pair'),source=host?.querySelector('.before .qk-loop .qk-load'),target=host?.querySelector('.after .qk-outside .qk-load');
    if(!source||!target||window.matchMedia?.('(prefers-reduced-motion: reduce)').matches||typeof source.animate!=='function')return;
    const run={host,ghost:null,animations:[],frame:0};hoistRun=run;
    run.frame=requestAnimationFrame(()=>{
      run.frame=0;if(hoistRun!==run)return;
      const box=host.getBoundingClientRect(),from=source.getBoundingClientRect(),to=target.getBoundingClientRect();
      if(!from.width||!to.width||!from.height||!to.height){stopHoist();return;}
      const ghost=source.cloneNode(true);run.ghost=ghost;
      for(const attr of [...ghost.attributes])if(attr.name.startsWith('data-')||attr.name.startsWith('aria-'))ghost.removeAttribute(attr.name);
      ghost.className='qk-op qk-load selected qk-hoist-ghost';ghost.setAttribute('tabindex','-1');ghost.setAttribute('aria-hidden','true');
      Object.assign(ghost.style,{position:'absolute',left:(from.left-box.left)+'px',top:(from.top-box.top)+'px',width:from.width+'px',height:from.height+'px',margin:'0',transformOrigin:'top left'});
      host.append(ghost);host.setAttribute('aria-busy','true');
      const duration=1100,options={duration,easing:'linear',fill:'both'},dx=to.left-from.left,dy=to.top-from.top;
      const animate=(node,keyframes)=>{if(!node)return null;const animation=node.animate(keyframes,options);run.animations.push(animation);return animation;};
      const arrive='translate('+dx+'px,'+dy+'px) scale('+(to.width/from.width)+','+(to.height/from.height)+')';
      const flight=animate(ghost,[{transform:'translate(0,0)',opacity:0,offset:0},{transform:'translate(0,0)',opacity:1,offset:.14,easing:'ease-in-out'},{transform:'translate('+(dx*.5)+'px,'+(dy-22)+'px)',opacity:1,offset:.47,easing:'ease-out'},{transform:arrive,opacity:1,offset:.78},{transform:arrive,opacity:0,offset:.9},{transform:arrive,opacity:0,offset:1}]);
      animate(target,[{opacity:0,offset:0},{opacity:0,offset:.76},{opacity:1,offset:.9},{opacity:1,offset:1}]);
      for(const node of host.querySelectorAll('.after .qk-reuse,.after .qk-rounds'))animate(node,[{opacity:.2,offset:0},{opacity:.2,offset:.77},{opacity:1,offset:.95},{opacity:1,offset:1}]);
      animate(source,[{opacity:1},{opacity:.45,offset:.15},{opacity:1,offset:.32},{opacity:1}]);
      animate(host.querySelector('.qk-hoist-link'),[{opacity:.2},{opacity:1,offset:.45},{opacity:1}]);
      const stages=panel.querySelectorAll('.qk-hoist-process>span');
      stages.forEach((node,index)=>animate(node,[{opacity:.35},{opacity:1,offset:[.12,.44,.9][index]},{opacity:1}]));
      const event=document.querySelector('#lane [data-qk-inline-event="q"][data-qk-inline-iter="'+state.iter+'"]');
      animate(event,[{opacity:1},{opacity:.65,offset:.2},{opacity:1,offset:.4},{opacity:1}]);
      flight.onfinish=()=>{if(hoistRun===run)stopHoist();};
    });
  }
  function loopDiagram(){
    const node=(op,label,sub,extra='')=>'<button type="button" data-qk-object="'+op+'" class="qk-op '+extra+(state.object===op?' selected':'')+'" aria-pressed="'+(state.object===op)+'"><b>'+label+'</b><small>'+sub+'</small></button>';
    const rounds=after=>'<div class="qk-rounds" aria-label="'+(after?'候选复用关系':'重复加载轮次')+'">'+Array.from({length:32},(_,i)=>'<button type="button" data-qk-iter="'+i+'" data-qk-object="q" aria-label="第 '+(i+1)+' 轮 '+(after&&i>0?'复用同一 Q':'加载 Q')+'" class="'+(after?'reuse':'repeat')+(state.object==='q'&&state.iter===i?' selected':'')+'" aria-pressed="'+(state.object==='q'&&state.iter===i)+'">'+(after?(i===0?'L':'R'):'L')+(i+1)+'</button>').join('')+'</div><small class="qk-sequence-caption">'+(after?'L：加载 · R：复用':'L：GM → Mat 加载')+'</small>';
    return '<section class="qk-structure" aria-label="Q 加载从 KV 循环内移到循环外的候选 Pass 变化"><div class="qk-hoist-process"><span>① 识别固定 Q</span><b>→</b><span>② 加载移出循环 ↗</span><b>→</b><span>③ Mat 驻留复用</span></div><div class="qk-structure-pair"><svg class="qk-hoist-link '+(state.object==='q'?'selected':'')+'" viewBox="0 0 40 110" aria-label="Q 加载上移至循环外" role="img"><defs><marker id="qk-hoist-head" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="currentColor"/></marker></defs><path d="M2 104 C38 104 2 9 36 9" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#qk-hoist-head)"/></svg><article class="qk-scope before"><h4>Pass 前 · 循环内</h4><small>同一 query / head</small><div class="qk-outside empty">循环外：尚无 Q 加载</div><div class="qk-loop"><h5>↻ KV 循环 ×32</h5>'+node('q','Q 加载 · GM → Mat','每轮加载同一面板 · 32 KiB','qk-load repeat')+'<div class="qk-flow-down">↓ 每轮执行</div>'+node('kv','KV 加载','随 KV 分片改变')+node('l0','L0 提取','Mat → Left / Right')+node('mm','CUBE · QK','执行矩阵计算')+'</div>'+rounds(false)+'</article><article class="qk-scope after"><h4>候选 Pass 后 · 循环外</h4><small>固定 query / head 作用域</small><div class="qk-outside">'+node('q','↑ Q 加载 · GM → Mat','移到循环外 · 只加载 1 次','qk-load resident')+'</div><div class="qk-loop"><h5>↻ KV 循环 ×32</h5>'+node('q','复用已驻留的 Q','第 '+(state.iter+1)+' 轮 → 同一 Mat 面板','qk-reuse')+'<div class="qk-flow-down">↓ 每轮执行</div>'+node('kv','KV 加载','仍在循环内')+node('l0','L0 提取','仍在循环内')+node('mm','CUBE · QK','数学计算保持不变')+'</div>'+rounds(true)+'</article></div><div class="qk-selection-link">选中第 '+(state.iter+1)+' 轮 ↔ 两侧节点与泳道同步高亮</div></section>';
  }
  function chooseStage(stage){
    const info=stageInfo[stage];if(!info||info.leaf!==state.leaf)return;
    state.stage=stage;state.object=info.op;state.pipelinePass=passFor(info.op);state.step=state.leaf==='qk_pv_aic'?2:1;
    callbacks.redraw();panel.scrollTop=0;
  }
  const subscript=n=>String(n).replace(/\d/g,d=>'₀₁₂₃₄₅₆₇₈₉'[Number(d)]);
  function pairOrder(){
    const first=Math.floor(state.iter/2)*2;
    return '<div class="qk-pair-order" aria-label="最终编译代码的 CUBE 分片顺序"><span>组 '+(first/2+1)+'</span>'+[['mm',first],['mm',first+1],['pv',first],['pv',first+1]].map(([op,iter],i)=>'<button type="button" class="'+(state.object===op&&state.iter===iter?'on':'')+'" data-qk-object="'+op+'" data-qk-iter="'+iter+'">'+(op==='mm'?'QK':'PV')+subscript(iter)+'</button>'+(i<3?'<span>→</span>':'')).join('')+'</div>';
  }
  function dependencyEvidence(p){
    const nodes=[['mm','AIC','QK'],['send_score','AIC → AIV','发送分数'],['scale','关联 AIV','缩放 + bias'],['exp','关联 AIV','减最大值 / exp'],['reduce','关联 AIV','行求和 / 转 BF16'],['send_p','AIV → AIC','返回 P'],['recv_p','AIC','接收 P'],['pv','AIC','PV']];
    const trace='<div class="qk-dependency-steps">'+nodes.map(([op,side,label])=>'<button type="button" class="'+(state.object===op?'on':'')+'" data-qk-object="'+op+'"><small>'+side+'</small><b>'+label+'</b></button>').join('<span aria-hidden="true">→</span>')+'</div>';
    const outputs='<div class="qk-output-branches"><span>分别写回</span><button type="button" class="ctl" data-qk-object="write_o">AIC · O</button><button type="button" class="ctl" data-qk-object="write_ml">AIV · m、l</button></div>';
    const wait='<button type="button" class="ctl qk-wait-selection '+(state.object==='wait_p'?'on':'')+'" data-qk-object="wait_p">等待 P · 对应分片返回后才能执行 PV</button>';
    if(state.step===1||state.step===2)return selectedEvent(p)+'<div class="qk-stage-analysis"><h3>第 '+(state.iter+1)+' 轮 · AIC / AIV 分片依赖</h3>'+pairOrder()+'<button type="button" class="ctl" data-qk-pipeline-pass="skew">Pass 分析 · P28 ↗</button>'+wait+trace+outputs+'<dl><div><dt>分数</dt><dd>AIC → AIV · 32 × 128 FP32 · 16 KiB</dd></div><div><dt>P</dt><dd>AIV → AIC · exp 权重 32 × 128 BF16 · 8 KiB</dd></div><div><dt>写回</dt><dd>AIC：局部 O · AIV：局部 m、l</dd></div></dl><a href="../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/next_levels/decode_csa_test/kernels/aic/qk_pv_aic.cpp" target="_blank" rel="noopener">最终代码 · QK / PV 与 push / pop ↗</a></div>';
    return '<div class="qk-stage-analysis"><h3>'+(state.step===3?'依赖检查':'待验证项')+'</h3><ul><li>QK 分数发送后，AIV 才能消费对应分片</li><li>P 返回且 AIC 接收后，PV 才能消费</li><li>tfree 释放管道槽位，检查槽位复用顺序</li><li>采集真实 push / pop 及等待事件，确认等待时长与物理核对应</li></ul></div>';
  }
  function stageEvidence(p){
    if(state.stage==='dependency')return dependencyEvidence(p);
    const info=stageInfo[state.stage],soft=state.stage==='softmax',output=state.stage==='output',o=state.stage==='ooutput',ret=state.stage==='returnp';
    const sequence=soft?[['scale','缩放 + bias'],['exp','减行最大值 / exp'],['reduce','行求和 / 转 BF16']]:output?[['write_ml','m、l 写回']]:o?[['write_o','O 写回']]:ret?[['send_p','P → AIC']]:[['recv_p','接收 P'],['pv','P × V → O'],['write_o','O 写回']];
    const nodes=sequence.map(([op,label])=>'<button type="button" class="ctl '+(state.object===op||soft&&state.object==='softmax'?'on':'')+'" data-qk-object="'+op+'">'+label+'</button>').join('<span>→</span>');
    const input=soft?'分数 32×128 FP32':output?'局部 m、l':o?'局部 O 32×512 FP32':ret?'exp 权重 P 32×128 BF16':'P 32×128 BF16 × KV 128×512 BF16';
    const result=soft?'exp 权重 P 32×128 BF16；行统计 m、l':output?'GM · sparse_blk_mi / sparse_blk_li':o?'GM · sparse_blk_oi':ret?'AIC · Mat 输入，供 PV 消费':'局部 O 32×512 FP32';
    if(state.step===1||state.step===2)return selectedEvent(p)+(state.leaf==='qk_pv_aic'?pairOrder():'')+'<div class="qk-stage-analysis"><h3>'+info.label+' · 第 '+(state.iter+1)+' 轮</h3><div class="qk-stage-sequence">'+nodes+'</div><dl><div><dt>输入</dt><dd>'+input+'</dd></div><div><dt>输出</dt><dd>'+result+'</dd></div></dl><a href="../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/25_after_SplitVectorKernel.py" target="_blank" rel="noopener">查看计算与数据传递 ↗</a></div>';
    return '<div class="qk-stage-analysis"><h3>'+info.label+' · '+(state.step===3?'检查项':'待验证项')+'</h3><ul>'+(soft?'<li>核对缩放、bias、row_max、exp、row_sum 和 BF16 转换位置</li><li>P 为局部 exp 权重，最终归一化在后续合并阶段完成</li>':o||output?'<li>核对写回张量、分块行索引与所属计算侧</li><li>检查写回完成与后继读取依赖</li>':ret?'<li>检查 P 的 push / pop 顺序及管道槽位释放</li>':'<li>核对 M=32、N=512、K=128</li><li>检查 P 接收完成后再执行 PV</li><li>检查 P / KV 的 L0 提取与布局</li>')+'<li>采集实际执行、搬运和等待事件</li></ul></div>';
  }
  function memoryEvidence(p){const aic=state.leaf==='qk_pv_aic';if(!aic)return'<div class="rc-inspector"><h4>改动落点：qk_pv_aic 的固定 Q 加载</h4><p>本侧的向量计算与 PV 处理不套用同一 load 外提。AIV 的执行数据保持原值；可以切换到 AIC，观察单个长 block 的搬运与优化。</p><button type="button" class="ctl" data-qk-leaf="qk_pv_aic" data-qk-open-pass>查看 AIC 的 Pass 前后对照 →</button></div>';
      const facts={q:'同一 query / head 的 qk_q_tile：从 KV 循环内移至循环前；不同 head_row 不能共用这次加载。',kv:'qk_kv 随 KV 分片改变，继续逐轮加载。',l0:'Left / L0 的提取留在循环内，不把 L0 生命周期一并外提。',mm:'QK 数学计算保持一致。',loop:'固定 query / head，循环遍历 KV 分片。'};
      return '<div class="qk-pass-change"><h3>加载从循环内移到循环外</h3><button type="button" class="ctl qk-replay" data-qk-replay aria-label="重播 Q 加载外提">↻ 重播外提</button></div>'+loopDiagram()+'<div class="qk-operation-note">'+(facts[state.object]||'Q 驻留不改变 PV 计算与跨核数据依赖。')+'</div><details class="qk-code-detail"><summary>实现与代码依据</summary><div class="rc-two">'+code(beforeCode,'Pass 前 · 循环内加载')+code(afterCode,'候选 Pass 后 · 循环外加载')+'</div><a target="_blank" rel="noopener" href="'+reference.pr+'">#2080 参考来源 ↗</a></details>';}

  function evidence(p){
    if(state.step===2&&state.leaf==='qk_pv_aic')return selectedEvent(p)+window.QK_PIPELINE_PASSES.navigation(state.pipelinePass)+(state.pipelinePass==='memory'?memoryEvidence(p):window.QK_PIPELINE_PASSES.render({iter:state.iter,object:state.object,pass:state.pipelinePass}));
    if(state.stage!=='qk')return stageEvidence(p);
    const aic=state.leaf==='qk_pv_aic';
    if(state.step===0)return '<div class="rc-finding"><div><small>'+ctx.rank.rank+' · Task #'+id+'</small><h3>'+e(state.leaf)+'</h3></div></div><div class="rc-two"><article class="rc-card"><h4>所选 block</h4><strong>'+us(p.b.computeUs)+'</strong><p>Core '+p.b.core+' · Setup '+us(p.b.setupUs)+'</p></article><article class="rc-card"><h4>当前任务</h4><strong>'+us(p.original.spanUs)+'</strong><p>'+p.original.blocks.n+' blocks</p></article></div>'+next(aic?1:2,aic?'定位重复 Q 搬运':'查看 AIC 生产者');
    if(state.step===1){if(!aic)return'<div class="rc-inspector"><h4>AIV 不重复执行这条 query Mat 加载</h4><p>Q 加载位于 qk_pv_aic。</p><button type="button" class="ctl" data-qk-leaf="qk_pv_aic">选中生产者 qk_pv_aic →</button></div>';
      return selectedEvent(p)+pairOrder()+'<div class="qk-stage-analysis"><h3>QK · Q × Kᵀ</h3><div class="qk-stage-sequence"><button type="button" class="ctl" data-qk-object="q">Q 加载</button>→<button type="button" class="ctl" data-qk-object="kv">KV 加载</button>→<button type="button" class="ctl" data-qk-object="l0">L0 提取</button>→<button type="button" class="ctl" data-qk-object="mm">QK 计算</button></div><dl><div><dt>输出</dt><dd>32 × 128 · FP32</dd></div></dl></div><div class="rc-verdict">固定 Q 面板：32 × 512 × BF16 = <b>32 KiB</b><br>KV 循环：32 轮 · Q 加载：32 次 → 1 次</div><div class="rc-two">'+[false,true].map(after=>'<article class="rc-card"><h4>'+(after?'候选结构：一次加载，其余复用':'待分析：每轮从 GM 加载')+'</h4><div class="rc-cells">'+Array.from({length:32},(_,i)=>'<button type="button" class="'+(after?'resident':'stationary')+(i===state.iter?' selected':'')+'" data-qk-iter="'+i+'" aria-pressed="'+(i===state.iter)+'">'+(i+1)+'<small>'+(after?(i===0?'加载':'复用'):'加载')+'</small></button>').join('')+'</div></article>').join('')+'</div>'+next(2,'联动这条 qk_q_tile 加载的 Pass 位置');}
    if(state.step===3)return'<div class="rc-gate-options">'+[['safe','固定 query/head · 规则通过'],['head','head_row 发生变化'],['alias','别名或可写源未知']].map(([id,label])=>'<button type="button" class="ctl '+(state.gate===id?'on':'')+'" data-qk-gate="'+id+'">'+label+'</button>').join('')+'</div><div class="rc-decision '+(state.gate==='safe'?'':'decline')+'"><h3>'+(state.gate==='safe'?'允许外提':'拒绝外提')+'</h3><p>'+(state.gate==='safe'?'只有在固定 query/head、源只读且可证明不别名、无顺序边界并满足 Mat 容量时，才把 Q 加载外提。':'head_row 变化或存储安全性不能证明时，保留原始加载；不能跨不同头组复用 Q。')+'</p></div>'+next(4,'查看尚需验证的项目');
    return '<div class="rc-validation"><article class="rc-card"><small>已知锚点</small><h4>任务身份、block 时长与 Q 形状</h4><p>Q 为 32×512 BF16，32 KiB。</p></article><article class="rc-card"><small>需要验证</small><h4>该 Q 是否真的循环不变</h4><p>核对 query/head、只读和别名关系，检查 Mat 容量及实际 Pass dump。</p></article><article class="rc-card"><small>需要编译</small><h4>外提是否生效</h4><p>对比加载的循环归属，确认 KV 与 L0 仍逐轮执行。</p></article><article class="rc-card"><small>需要上板</small><h4>耗时与数值对照</h4><p>核对数值、MTE2 字节及执行时长。</p></article></div><details class="rc-sources"><summary>#2080 参考案例</summary><a href="'+reference.pr+'" target="_blank" rel="noopener">PR #2080 ↗</a></details>';
  }
  function analysisJourney(p){
    if(state.leaf!=='qk_pv_aic')return '<div class="qk-journey">'+stageEvidence(p)+'</div>';
    return '<div class="qk-journey"><div class="qk-context-label">当前事件</div>'+selectedEvent(p)+window.QK_PASS_EVOLUTION.render({iter:state.iter,object:state.object,pass:state.pipelinePass})+'<details class="qk-journey-checks" data-qk-section="checks"><summary>检查条件与待验证项</summary><ul><li>P19：确认固定 query/head、只读、别名和 Mat 容量条件</li><li>后续阶段：同一分片保持 QK → AIV → P → PV，检查槽位释放</li><li>实际搬运、重叠及等待长度需要编译与设备验证</li></ul></details></div>';
  }
  function render(){if(!ctx)return;stopHoist();state.step=state.leaf==='qk_pv_aic'?2:1;const openSections=[...header.querySelectorAll('[data-qk-section][open]'),...panel.querySelectorAll('[data-qk-section][open]')].map(n=>n.dataset.qkSection);const codeOpen=panel.querySelector('.qk-code-detail')?.hasAttribute('open'),detailsOpen=header.querySelector('.qk-summary-details')?.hasAttribute('open'),p=pair();
    header.innerHTML='<div class="qk-compact-title"><strong>Block 分析</strong><button type="button" class="ctl" data-qk-dismiss>返回</button></div>'+(p?blockSummary(p):'');
    if(detailsOpen)header.querySelector('.qk-summary-details')?.setAttribute('open','');
    if(state.step===0)state.step=1;
    panel.innerHTML=p?analysisJourney(p):'<div class="rc-empty"><h3>选择一个 block</h3></div>';
    for(const key of openSections){header.querySelector('[data-qk-section="'+key+'"]')?.setAttribute('open','');panel.querySelector('[data-qk-section="'+key+'"]')?.setAttribute('open','');}
    if(codeOpen)panel.querySelector('.qk-code-detail')?.setAttribute('open','');
    if(p&&state.leaf==='qk_pv_aic')window.QK_PASS_EVOLUTION.mount(panel,{iter:state.iter,object:state.object,pass:state.pipelinePass},id=>{state.pipelinePass=id;});
  }
  function update(raw,rank,selected){
    ctx={raw,rank};if(isTarget(selected)){if(state.leaf!==selected.leaf){state.leaf=selected.leaf;state.core=null;state.stage=selected.leaf==='qk_pv_aic'?'qk':'softmax';state.object=selected.leaf==='qk_pv_aic'?'q':'softmax';state.pipelinePass=passFor(state.object);state.step=selected.leaf==='qk_pv_aic'?2:1;}}else state.leaf=null;
    if(pendingCore!==null){state.core=pendingCore;pendingCore=null;}render();
  }
  function click(event){const b=event.target.closest('[data-qk-skew-replay],[data-qk-pipeline-pass],[data-qk-stage],[data-qk-replay],[data-qk-reveal],[data-qk-leaf],[data-qk-step],[data-qk-iter],[data-qk-object],[data-qk-gate],[data-qk-locate],[data-qk-dismiss]');if(!b)return;const d=b.dataset,scroll=panel.scrollTop,host=header.contains(b)?header:panel;
    const selector=[...b.attributes].filter(a=>a.name.startsWith('data-qk-')||a.name==='data-pass-side').map(a=>'['+a.name+'="'+a.value+'"]').join('');
    const refocus=()=>host.querySelector(selector)?.focus({preventScroll:true});
    if(b.hasAttribute('data-qk-skew-replay')){window.QK_PIPELINE_PASSES.play(panel,{iter:state.iter});return;}
    if(d.qkPipelinePass){if(d.qkPipelinePass==='skew')lastSkewKey='';if(d.qkPipelinePass==='memory'){state.object='q';state.stage='qk';}state.pipelinePass=d.qkPipelinePass;state.step=2;callbacks.redraw();if(state.pipelinePass==='memory'&&state.object==='q')playHoist();panel.scrollTop=scroll;refocus();return;}
    if(d.qkStage){chooseStage(d.qkStage);refocus();return;}
    if(b.hasAttribute('data-qk-replay')){state.object='q';state.stage='qk';state.pipelinePass='memory';state.step=2;callbacks.redraw();playHoist();refocus();return;}
    if(b.hasAttribute('data-qk-reveal')){callbacks.showMicro?.();const op=state.object==='softmax'?'scale':state.object;document.querySelector('#lane [data-qk-version][data-qk-inline-event="'+op+'"][data-qk-inline-iter="'+state.iter+'"]')?.focus({preventScroll:true});return;}
    if(b.hasAttribute('data-qk-dismiss')){callbacks.dismiss();return;}
    if(d.qkLeaf){callbacks.select(d.qkLeaf);if(b.hasAttribute('data-qk-open-pass')){state.step=2;render();}refocus();return;}if(b.hasAttribute('data-qk-locate')){callbacks.locate();return;}
    if(d.qkStep!==undefined){state.step=Number(d.qkStep);if(d.qkObject==='q'&&state.step===2)state.pipelinePass='memory';if(state.step===1)callbacks.showMicro?.();}if(d.qkIter!==undefined){state.iter=Number(d.qkIter);state.object=d.qkObject||'q';state.stage=stageFor(state.object);}if(d.qkObject){state.object=d.qkObject;state.stage=stageFor(state.object);}if(d.qkGate)state.gate=d.qkGate;
    if(d.qkObject!==undefined||d.qkIter!==undefined)callbacks.redraw();else render();panel.scrollTop=d.qkStep!==undefined?0:scroll;refocus();
  }
  function init(h,p,ref,cb){
    header=h;panel=p;reference=ref;callbacks=cb;window.addEventListener('resize',stopHoist);
    for(const node of [h,p]){node.addEventListener('click',click);node.addEventListener('change',ev=>{if(ev.target.matches('[data-qk-core]'))callbacks.core(Number(ev.target.value));});}
    panel.addEventListener('keydown',ev=>{
      const node=ev.target.closest('.qk-pass-node');if(node&&['Enter',' '].includes(ev.key)){ev.preventDefault();click({target:node});return;}
      const tab=ev.target.closest('[role="tab"]');if(!tab||!['ArrowLeft','ArrowRight','Home','End'].includes(ev.key))return;ev.preventDefault();
      const tabs=[...panel.querySelectorAll('[role="tab"]')],at=tabs.indexOf(tab),next=ev.key==='Home'?0:ev.key==='End'?tabs.length-1:(at+(ev.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length,id=tabs[next].id;
      click({target:tabs[next]});panel.querySelector('#'+id)?.focus({preventScroll:true});
    });
  }
  // Bind the shared inline renderer without mounting the original page's inspector.
  function bindInline(rank,task,core,redraw){
    ctx={raw:rank,rank};panel=document.createElement('div');callbacks={redraw};
    if(state.leaf!==task.leaf){state.iter=0;state.object=task.leaf==='qk_pv_aic'?'q':'softmax';state.stage=task.leaf==='qk_pv_aic'?'qk':'softmax';}
    state.leaf=task.leaf;state.core=core;
  }
  function selectCore(core){pendingCore=core;}
  function inlineEvents(original,leaf='qk_pv_aic') {
    const out=[];let cursor=original.startUs+original.setupUs;
    const add=(op,stage,iter,duration)=>{const ev={op,stage,iter,startUs:cursor,endUs:cursor+duration,leaf};out.push(ev);cursor+=duration;return ev;};
    if(leaf==='qk_pv_aiv'){
      for(let iter=0;iter<32;iter++)for(const [op,stage,share] of [['scale','softmax',.15],['exp','softmax',.4],['reduce','softmax',.25],['send_p','returnp',.1],['write_ml','output',.1]])add(op,stage,iter,original.computeUs/32*share);
    }else{
      // The final code issues QK0, QK1, then consumes P0/P1 for PV0/PV1.
      // Relative durations and overlap below are illustrative, not trace events.
      const base=cursor;
      for(let group=0;group<16;group++){
        const local=[],ready=new Map();let c=0,v=0;
        const put=(op,stage,iter,startUs,endUs,related=false)=>{const ev={op,stage,iter,startUs,endUs,leaf:related?'qk_pv_aiv':leaf,related,group};local.push(ev);return ev;};
        const own=(op,stage,iter,dt)=>{const ev=put(op,stage,iter,c,c+dt);c+=dt;return ev;};
        // Both QK fragments are issued before the first P is consumed.
        for(let slot=0;slot<2;slot++){
          const iter=group*2+slot;
          for(const [op,dt] of [['q',.18],['kv',.2],['l0',.1],['mm',.6],['send_score',.04]])own(op,op==='send_score'?'dependency':'qk',iter,dt);
          v=Math.max(v,c);
          for(const [op,dt] of [['scale',.18],['exp',.5],['reduce',.4],['send_p',.17]]){put(op,'dependency',iter,v,v+dt,true);v+=dt;}
          ready.set(iter,v);
          put('write_ml','dependency',iter,v,v+.08,true);v+=.08;
        }
        for(let slot=0;slot<2;slot++){
          const iter=group*2+slot;
          own('wait_p','dependency',iter,Math.max(0,ready.get(iter)-c));
          own('recv_p','dependency',iter,.04);
          own('pv','pv',iter,.55);
          own('write_o','ooutput',iter,.08);
        }
        const scale=original.computeUs/16/Math.max(c,v),offset=base+original.computeUs/16*group;
        for(const ev of local)out.push({...ev,startUs:offset+ev.startUs*scale,endUs:offset+ev.endUs*scale});
      }
    }

    return out.sort((a,b)=>a.startUs-b.startUs);
  }
  function renderInline(svg,{y,height,W,LEFT,xOf,viewStart,viewEnd,original,leaf='qk_pv_aic'}) {
    const doc=svg.ownerDocument,ns='http://www.w3.org/2000/svg';
    const make=(tag,attrs={},parent)=>{const node=doc.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))node.setAttribute(k,v);if(parent)parent.append(node);return node;};
    const g=make('g',{'class':'qk-inline-pipeline','data-inline-core':original.core,'aria-label':'Core '+original.core+' 待分析核内流水，事件为模拟'},svg);
    const text=(value,x,ty,color='#b0b0b0',size=11,parent=g)=>{const t=make('text',{x,y:ty,fill:color,'font-family':'var(--mono)','font-size':size},parent);t.textContent=value;return t;};
    const inset=LEFT-95,branch=LEFT-113,aic=leaf==='qk_pv_aic';g.setAttribute('data-stage-focus',state.stage);g.setAttribute('data-inline-leaf',leaf);
    // Same neutral surfaces and typography as the parent lane. Hierarchy comes
    // from the branch rail, row spacing and shared grid, rather than a card frame.
    make('rect',{class:'qk-panel-surface',x:8,y:y+6,width:W-24,height:height-20,fill:'#111111'},g);
    for(const ty of [y+6,y+height-14])make('line',{x1:16,x2:W-16,y1:ty,y2:ty,stroke:'#383838'},g);
    const middle=(value,x,ty,color,size,parent=g)=>{const node=text(value,x,ty,color,size,parent);node.setAttribute('dominant-baseline','middle');return node;};
    middle('↳',branch-4,y+28,'#8a8a8a',16);
    middle('Core '+original.core+' · 核内流水',inset,y+28,'#e8e8e8',13);
    make('path',{d:'M'+branch+' '+(y+37)+' V'+(y+90+(leaf==='qk_pv_aic'?10:5)*26),fill:'none',stroke:'#383838'},g);
    const control=(label,attrs,x,width=64)=>{const segment=attrs['data-qk-nav'],disabled=attrs['aria-disabled']==='true';const b=make('g',{'role':'button',tabindex:disabled?-1:0,'aria-label':label,...attrs,class:'qk-inline-control'},g);make('rect',{x,y:y+9,width,height:38,rx:19,fill:attrs['data-qk-inline-pass']?'#63e6b3':segment?'transparent':'#303030',stroke:'none'},b);const t=middle(label,x+width/2,y+28,attrs['data-qk-inline-pass']?'#111':disabled?'#666666':'#e0e0e0',segment?18:12,b);t.setAttribute('text-anchor','middle');return b;};
    const navX=W-272;
    control('Pass 分析',{'data-qk-inline-pass':'true'},navX-94,82);
    make('rect',{class:'qk-round-control',x:navX,y:y+9,width:160,height:38,rx:19,fill:'#292929',stroke:'none'},g);
    for(const x of [navX+34,navX+126])make('line',{x1:x,x2:x,y1:y+20,y2:y+36,stroke:'#303030'},g);
    control('‹',{'data-qk-nav':'previous','data-qk-inline-iter':Math.max(0,state.iter-1),'aria-label':'上一轮','aria-disabled':String(state.iter===0)},navX,28);
    const roundText=middle('轮次 '+String(state.iter+1).padStart(2,'0')+' / 32',navX+80,y+28,'#e8e8e8',12);roundText.setAttribute('text-anchor','middle');roundText.setAttribute('class','qk-round-label');
    control('›',{'data-qk-nav':'next','data-qk-inline-iter':Math.min(31,state.iter+1),'aria-label':'下一轮','aria-disabled':String(state.iter===31)},navX+132,28);
    control('收起',{'data-qk-collapse':'true'},W-96,64);
    if(aic){
      const problem=make('g',{role:'button',tabindex:0,class:'qk-pipe-event qk-repeat-summary','data-qk-inline-event':'q','data-qk-inline-iter':Math.max(1,state.iter),'aria-label':'31 次重复 Q 搬运，查看 InferTileMemorySpace 前后变化'},g);
      make('rect',{x:inset,y:y+50,width:400,height:24,rx:3,fill:'#221817'},problem);
      middle('同一 Q 搬运 32 次 · 其中 31 次重复 → 查看 Pass',inset+10,y+62,'#f28b80',12,problem);
    }else middle('AIV · 缩放 → exp → 归约 → 返回 P · 写回 m、l',inset,y+62,'#b0b0b0',11);
    const start=original.startUs+original.setupUs,before=inlineEvents(original,leaf);
    const rows=aic?[['q','MTE2 · 固定 Q','#b0b0b0'],['kv','MTE2 · KV','#8e8e8e'],['l0','MTE1 · L0','#7a7a7a'],['mm','CUBE · QK/PV','#b0b0b0'],['send_score','AIC → 分数','#a0a0a0'],['wait_p','AIC · 等待 P','#444444'],['related_vec','关联 AIV · VEC','#9a9a9a'],['recv_p','AIC ← P','#b0b0b0'],['write_o','AIC · O 写回','#9a9a9a'],['write_ml','关联 AIV · m、l','#7a7a7a']]:[['scale','VEC · 缩放','#8e8e8e'],['exp','VEC · exp','#b0b0b0'],['reduce','VEC · 归约','#8e8e8e'],['send_p','AIV → P','#a0a0a0'],['write_ml','AIV · m、l 写回','#7a7a7a']];
    const plotBottom=y+92+rows.length*26;
    const clipId='qk-inline-clip-'+original.core;
    const clip=make('clipPath',{id:clipId},make('defs',{},g));make('rect',{x:LEFT,y:y+90,width:W-LEFT-24,height:rows.length*26+4},clip);
    const pattern=make('pattern',{id:'qk-wait-hatch-'+original.core,width:5,height:5,patternUnits:'userSpaceOnUse'},g.querySelector('defs'));make('path',{d:'M-1 1 L1 -1 M0 5 L5 0 M4 6 L6 4',stroke:'#666666','stroke-width':1},pattern);
    rowsBackground();
    function rowsBackground(){for(let i=0;i<rows.length;i++){make('rect',{class:'qk-engine-row',x:LEFT,y:y+92+i*26,width:W-LEFT-24,height:24,fill:i%2?'#111111':'#171717'},g);make('line',{x1:branch,x2:branch+12,y1:y+104+i*26,y2:y+104+i*26,stroke:'#383838'},g);}for(let i=0;i<=8;i++){const x=LEFT+(W-LEFT-12)*i/8;make('line',{x1:x,x2:x,y1:y+90,y2:plotBottom,stroke:'#262626'},g);}}
    const plot=make('g',{'data-lane-camera':'qk','clip-path':'url(#'+clipId+')'},g);
    if(aic){const group=before.filter(ev=>Math.floor(ev.iter/2)===Math.floor(state.iter/2)),from=Math.min(...group.map(ev=>ev.startUs)),to=Math.max(...group.map(ev=>ev.endUs));make('rect',{class:'qk-pair-window',x:xOf(from),y:y+90,width:xOf(to)-xOf(from),height:rows.length*26+4,fill:'#ffffff','fill-opacity':.035,'pointer-events':'none','data-pair-group':Math.floor(state.iter/2)},plot);}

    const sections=[{version:'before',label:'执行引擎',base:82,events:before,end:original.endUs}];
    sections.forEach(({version,label,base,events,end})=>{
      const color='#8a8a8a';
      text(label,inset,y+base,color,11);
      rows.forEach(([op,rowLabel,normal],index)=>{
        const rowY=y+96+index*26;
        const rowText=text(rowLabel,LEFT-10,rowY+12,'#c4c4c4',12);rowText.setAttribute('text-anchor','end');
        for(const ev of events.filter(ev=>ev.op===op||op==='mm'&&ev.op==='pv'||op==='related_vec'&&ev.related&&['scale','exp','reduce','send_p'].includes(ev.op))){
          const from=Math.max(viewStart,ev.startUs),to=Math.min(viewEnd,ev.endUs);if(to<=from)continue;
          const paired=aic&&Math.floor(ev.iter/2)===Math.floor(state.iter/2);
          const linked=ev.iter===state.iter&&['dependency','pv'].includes(state.stage)&&['mm','send_score','wait_p','scale','exp','reduce','send_p','recv_p','pv','write_ml'].includes(ev.op);
          const repeat=ev.op==='q'&&ev.iter>0,selected=(state.object===ev.op||state.object==='softmax'&&ev.stage==='softmax')&&ev.iter===state.iter;
          const b=make('g',{'class':'qk-pipe-event'+(ev.stage===state.stage?' stage-related':'')+(paired?' pair-related':'')+(linked?' dependency-related':'')+(repeat?' qk-repeat-event':'')+(selected?' selected':''),role:'button',tabindex:0,'data-qk-version':version,'data-qk-inline-event':ev.op,'data-qk-stage':ev.stage,'data-fragment':ev.iter,'data-pair-group':Math.floor(ev.iter/2),'data-event-leaf':ev.leaf,'data-related-aiv':String(!!ev.related),'data-qk-inline-iter':ev.iter,'data-event-start':ev.startUs,'data-event-end':ev.endUs,'aria-pressed':selected,'aria-label':'第 '+(ev.iter+1)+' 轮 '+(ev.op==='send_p'?'AIV 返回 P':ev.op==='pv'?'CUBE PV':rowLabel)+(repeat?'，重复搬运，查看 Pass':'')},plot);
          const ex=xOf(from),ew=Math.max(.8,xOf(to)-ex);
          make('rect',{x:ex,y:rowY,width:ew,height:17,rx:1,fill:repeat?'#fd5644':ev.op==='wait_p'?'url(#qk-wait-hatch-'+original.core+')':ev.op==='pv'?'#777777':normal,stroke:selected?'#f0f0f0':linked?'#9a9a9a':'none','stroke-width':selected?2:linked?1:0},b);
          if(['mm','pv'].includes(ev.op)&&ew>=26){const caption=middle((ev.op==='mm'?'QK':'PV')+subscript(ev.iter),ex+ew/2,rowY+8.5,'#151515',9,b);caption.setAttribute('text-anchor','middle');caption.setAttribute('pointer-events','none');}
          make('rect',{x:ex-2,y:rowY-2,width:Math.max(7,ew+4),height:21,fill:'transparent'},b);
          if(op==='q')make('rect',{x:ex-3,y:rowY-2,width:Math.max(8,ew+6),height:21,fill:'transparent'},b);
          const title=make('title',{},b);title.textContent='分片 '+ev.iter+' · 第 '+(ev.iter+1)+' 轮 · '+(ev.op==='pv'?'CUBE · PV':ev.op==='send_p'?'AIV → AIC · 返回 P':ev.op==='scale'?'AIV · 缩放 + bias':ev.op==='exp'?'AIV · 减最大值 / exp':ev.op==='reduce'?'AIV · 行求和 / BF16':op==='mm'?'CUBE · QK':rowLabel)+' · '+us(ev.endUs-ev.startUs)+(op==='q'?' · 同一 Q 32 KiB'+(repeat?' · 重复加载 → InferTileMemorySpace 外提':' · 首次加载'):'');
        }
      });
      for(const t of [start,end])make('line',{x1:xOf(t),x2:xOf(t),y1:y+90,y2:plotBottom,stroke:'#858585','stroke-dasharray':'3 4','stroke-opacity':.5,'pointer-events':'none'},plot);
    });
    if(aic){
      const first=Math.floor(state.iter/2)*2,pairX=inset+186;
      const label=middle('组 '+String(first/2+1).padStart(2,'0'),pairX,y+28,'#777777',9);label.setAttribute('class','qk-pair-caption');
      [['mm',first],['mm',first+1],['pv',first],['pv',first+1]].forEach(([op,iter],i)=>{
        const x=pairX+44+i*66,active=state.object===op&&state.iter===iter;
        const button=make('g',{class:'qk-inline-control qk-pair-control',role:'button',tabindex:0,'data-qk-inline-event':op,'data-qk-inline-iter':iter,'aria-label':(op==='mm'?'QK':'PV')+' 分片 '+iter,'aria-pressed':active},g);
        make('rect',{x,y:y+14,width:52,height:28,rx:14,fill:active?'#e0e0e0':'#303030',stroke:'none'},button);
        const caption=middle((op==='mm'?'QK':'PV')+subscript(iter),x+26,y+28,active?'#111111':'#d0d0d0',12,button);caption.setAttribute('text-anchor','middle');
        if(i<3)middle('→',x+56,y+28,'#666666',9);
      });
      const rowOf=op=>op==='mm'||op==='pv'?3:['scale','exp','reduce','send_p'].includes(op)?6:rows.findIndex(row=>row[0]===op);
      const marker=make('marker',{id:'qk-dep-arrow-'+original.core,viewBox:'0 0 6 6',refX:5,refY:3,markerWidth:5,markerHeight:5,orient:'auto'},g.querySelector('defs'));make('path',{d:'M0 0 L6 3 L0 6',fill:'#bdbdbd'},marker);
      for(const iter of [first,first+1])for(const [from,to] of [['mm','send_score'],['send_score','scale'],['send_p','recv_p'],['recv_p','pv']]){
        const a=before.find(ev=>ev.iter===iter&&ev.op===from),b=before.find(ev=>ev.iter===iter&&ev.op===to);if(!a||!b)continue;
        make('path',{class:'qk-dependency-link',d:'M'+xOf(a.endUs)+' '+(y+104+rowOf(from)*26)+' L'+xOf(b.startUs)+' '+(y+104+rowOf(to)*26),fill:'none',stroke:iter===state.iter?'#bdbdbd':'#555555','stroke-width':1,'stroke-dasharray':'3 3','marker-end':'url(#qk-dep-arrow-'+original.core+')','pointer-events':'none','data-dependency-from':from,'data-dependency-to':to,'data-dependency-iter':iter},plot);
      }
    }
    return g;
  }
  function chooseInline(op,iter){if(op){state.object=op;state.stage=stageFor(op);state.pipelinePass=passFor(op);state.step=state.leaf==='qk_pv_aic'?2:1;}state.iter=Math.max(0,Math.min(31,Number(iter)));callbacks.redraw(!!op);if(op)panel.scrollTop=0;if(op==='q'&&state.iter>0&&state.step===2&&state.pipelinePass==='memory')playHoist();}
  function openPass(){state.step=2;state.pipelinePass=passFor(state.object);callbacks.redraw();if(state.pipelinePass==='memory'&&state.object==='q')playHoist();}
  return{init,update,bindInline,isTarget,selectCore,hero,withLongBlock,inlineEvents,renderInline,chooseInline,stopHoist,openPass,selectionCore:()=>pendingCore??state.core};
})();
