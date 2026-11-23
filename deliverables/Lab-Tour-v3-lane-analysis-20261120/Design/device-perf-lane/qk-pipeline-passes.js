// Pass structure follows archived IR; event widths are schematic under the demo's repeated-Q premise.
window.QK_PIPELINE_PASSES=(()=>{
  const root='../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/';
  const stages=[
    {id:'split',label:'P23 / P25',name:'拆分计算侧与双向传递',before:'P22 · 混合计算',after:'P23 / P25 · AIC / AIV 协作',left:'mixed',right:'serial',change:'QK / PV 留在 AIC，向量计算拆到 AIV；以分数和 P 的 push / pop 保持依赖。',sources:[['23_after_ExpandMixedKernel.py','P23 · 双向 pipe / push / pop'],['25_after_SplitVectorKernel.py','P25 · subblock 分支 / valid_shape']]},
    {id:'skew',label:'P28',name:'跨分片交错',before:'P25 · 逐分片生产 / 消费',after:'P28 · 提前生产下一分片',left:'serial',right:'skew',change:'QK₁ 提前到 PV₀ 之前；分数₀ 发出后，AIV₀ 可与 QK₁ 交叠。每个 PV 仍等待自己的 P。',sources:[['25_after_SplitVectorKernel.py','Pass 前 · 分片内 push → pop → PV'],['28_after_SkewCrossCorePipeline.py','Pass 后 · 序言 / 错位循环 / 尾部']]},
    {id:'lower',label:'P30',name:'流水降级与 L0 展开',before:'P28 · 已交错的流水结构',after:'P30 · 降级后的执行结构',left:'skew',right:'lower',change:'QK₀ → QK₁ → PV₀ → PV₁ 的主顺序保持；经 P29 槽位化后，P30 将内部流水循环降为显式 range / 展开计算。',sources:[['28_after_SkewCrossCorePipeline.py','P28 · 高层流水'],['29_after_LowerPipelineToSlots.py','P29 · 槽位化桥接'],['30_after_LowerPipelineLoops.py','P30 · range / pipeline_membership']]}
  ];
  const sub=n=>String(n).replace(/\d/g,d=>'₀₁₂₃₄₅₆₇₈₉'[+d]);
  const labels={q:'Q',mm:'QK',pv:'PV',send_score:'分数 →',wait_p:'等 P',recv_p:'← P',scale:'缩放',exp:'exp',reduce:'归约',send_p:'P →',write_o:'O',write_ml:'m/l'};
  function model(kind,first){
    const events=[];let c=0,v=0;const ready=new Map();
    const add=(op,iter,row,start,end)=>{const ev={op,iter,row,start,end};events.push(ev);return ev;};
    const own=(op,iter,row,dt)=>{const ev=add(op,iter,row,c,c+dt);c+=dt;return ev;};
    const produce=iter=>{own('q',iter,0,.4);own('mm',iter,1,1);own('send_score',iter,2,.18);v=Math.max(v,c);for(const [op,dt] of [['scale',.35],['exp',.65],['reduce',.45],['send_p',.2]]){add(op,iter,3,v,v+dt);v+=dt;}ready.set(iter,v);add('write_ml',iter,6,v,v+.25);v+=.25;};
    const consume=iter=>{own('wait_p',iter,4,Math.max(.05,ready.get(iter)-c));own('recv_p',iter,4,.18);own('pv',iter,1,.9);own('write_o',iter,5,.25);};
    if(kind==='serial'||kind==='mixed'){for(const iter of [first,first+1]){produce(iter);consume(iter);}}else{produce(first);produce(first+1);consume(first);consume(first+1);}
    return {events,end:Math.max(c,v)};
  }
  function diagram(kind,side,state){
    const first=Math.floor(state.iter/2)*2,{events,end}=model(kind,first),x=t=>100+t/end*440,rows=['Q · 重复加载','AIC · CUBE','AIC → 分数','AIV · VEC / P','AIC · 等待 / 收 P','AIC · O 写回','AIV · m/l 写回'];
    let svg='<svg class="qk-pass-graph" viewBox="0 0 560 230" role="group" aria-label="'+side+'的执行顺序与分片依赖"><defs><marker id="qk-pass-arrow-'+side+'" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#999"/></marker></defs>';
    rows.forEach((label,row)=>{const y=30+row*25;svg+='<text x="94" y="'+(y+12)+'" text-anchor="end" fill="#999" font-size="9">'+label+'</text><rect x="100" y="'+y+'" width="440" height="20" fill="'+(row%2?'#111':'#191919')+'"/>';});
    if(kind==='mixed')svg+='<rect x="96" y="25" width="448" height="178" rx="3" fill="none" stroke="#555" stroke-dasharray="4 4"/><text x="102" y="15" fill="#999" font-size="9">同一混合计算体 · 按计算侧归类</text>';
    else svg+='<text x="102" y="15" fill="#999" font-size="9">分数 / P · 双向 pipe · 2 个槽位</text>';
    for(const ev of events){const active=state.object===ev.op&&state.iter===ev.iter,related=state.iter===ev.iter,w=Math.max(4,x(ev.end)-x(ev.start)),y=30+ev.row*25;
      svg+='<g class="qk-pass-node '+(active?'selected':'')+' '+(related?'fragment-related':'')+'" data-qk-object="'+ev.op+'" data-qk-iter="'+ev.iter+'" data-pass-side="'+side+'" data-pass-op="'+ev.op+'" role="button" tabindex="0" aria-pressed="'+active+'" aria-label="'+labels[ev.op]+' 分片 '+ev.iter+'"><rect x="'+x(ev.start)+'" y="'+y+'" width="'+w+'" height="18" rx="2" fill="'+(ev.op==='q'?'#763a32':ev.op==='wait_p'?'#292929':ev.iter===first?'#929292':'#5d5d5d')+'" stroke="'+(active?'#fff':related?'#aaa':'#333')+'" stroke-width="'+(active?2:1)+'"/>';
      if(w>20)svg+='<text x="'+(x(ev.start)+w/2)+'" y="'+(y+12)+'" text-anchor="middle" fill="'+(ev.op==='q'||ev.op==='wait_p'?'#eee':'#101010')+'" font-size="8" pointer-events="none">'+labels[ev.op]+(ev.op==='mm'||ev.op==='pv'?sub(ev.iter):'')+'</text>';
      if(kind==='lower'&&['mm','pv'].includes(ev.op))for(let k=1;k<4;k++)svg+='<line x1="'+(x(ev.start)+w*k/4)+'" x2="'+(x(ev.start)+w*k/4)+'" y1="'+(y+1)+'" y2="'+(y+17)+'" stroke="#252525"/>';
      svg+='</g>';
    }
    for(const iter of [first,first+1])for(const [from,to] of [['mm','send_score'],['send_score','scale'],['send_p','recv_p'],['recv_p','pv']]){const a=events.find(e=>e.iter===iter&&e.op===from),b=events.find(e=>e.iter===iter&&e.op===to);svg+='<path class="qk-pass-dependency '+(iter===state.iter?'on':'')+'" data-pass-fragment="'+iter+'" data-dependency-from="'+from+'" data-dependency-to="'+to+'" d="M'+x(a.end)+' '+(39+a.row*25)+' L'+x(b.start)+' '+(39+b.row*25)+'" fill="none" stroke="'+(iter===state.iter?'#ccc':'#555')+'" stroke-dasharray="3 2" marker-end="url(#qk-pass-arrow-'+side+')" pointer-events="none"/>';}
    svg+='<text x="100" y="222" fill="#888" font-size="9">'+(kind==='lower'?'主顺序保持 · CUBE 内细线表示 L0 计算展开':'分片 '+first+' / '+(first+1)+' · '+(kind==='skew'?'QK₁ 提前，分片依赖不变':'每个分片等待自己的 P'))+'</text></svg>';
    const order=events.filter(e=>e.op==='mm'||e.op==='pv').sort((a,b)=>a.start-b.start).map(e=>'<button type="button" data-qk-object="'+e.op+'" data-qk-iter="'+e.iter+'" class="'+(e.op===state.object&&e.iter===state.iter?'on':'')+'">'+labels[e.op]+sub(e.iter)+'</button>').join('<span>→</span>');
    return svg+'<div class="qk-pass-order">'+order+'</div>';
  }

  let motionRun=null;
  function stop(){
    const run=motionRun;if(!run)return;motionRun=null;
    if(run.frame)cancelAnimationFrame(run.frame);
    for(const animation of run.animations){animation.onfinish=null;animation.cancel();}
    for(const node of run.linked)node.classList.remove('qk-skew-linked');
    for(const ghost of run.ghosts||[])ghost.remove();
    run.host.removeAttribute('aria-busy');run.host.classList.remove('is-playing');
  }
  function play(panel,state){
    stop();const host=panel.querySelector('[data-pass-version="after"].qk-skew-motion'),before=panel.querySelector('[data-pass-version="before"]');if(!host||!before)return;
    const first=Math.floor(state.iter/2)*2,second=first+1,linked=[];
    const pairs=[['mm',first],['send_score',first],['mm',second],['scale',first],['exp',first],['reduce',first],['send_p',first],['recv_p',first],['pv',first]];
    for(const [op,iter] of pairs)for(const node of document.querySelectorAll('#lane [data-qk-version][data-qk-inline-event="'+op+'"][data-qk-inline-iter="'+iter+'"],.qk-pass-node[data-qk-object="'+op+'"][data-qk-iter="'+iter+'"]')){node.classList.add('qk-skew-linked');linked.push(node);}
    for(const path of document.querySelectorAll('.qk-pass-dependency[data-pass-fragment="'+first+'"],#lane .qk-dependency-link[data-dependency-iter="'+first+'"]')){path.classList.add('qk-skew-linked');linked.push(path);}
    const run={host,linked,animations:[],ghosts:[],frame:0};motionRun=run;
    const status=host.querySelector('.qk-skew-status'),qk=host.querySelector('.qk-pass-node[data-qk-object="mm"][data-qk-iter="'+second+'"]');
    const complete=()=>{if(motionRun!==run)return;host.removeAttribute('aria-busy');host.classList.remove('is-playing');status.textContent='QK'+sub(second)+' 已提前；PV'+sub(first)+' 仍依赖 P'+sub(first);};
    if(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches||typeof qk?.animate!=='function'){complete();return;}
    host.setAttribute('aria-busy','true');host.classList.add('is-playing');
    run.frame=requestAnimationFrame(()=>{
      run.frame=0;if(motionRun!==run)return;
      const options={duration:1800,easing:'linear',fill:'both'},animate=(node,frames)=>{if(!node)return null;const animation=node.animate(frames,options);run.animations.push(animation);return animation;};
      let flight=null;
      // Both SVGs share one viewBox. Start each after-node at its before coordinates.
      // Rect/text geometry stays intact while transform maps the full event width.
      for(const target of host.querySelectorAll('.qk-pass-node')){
        const op=target.dataset.qkObject,iter=Number(target.dataset.qkIter),source=before.querySelector('.qk-pass-node[data-qk-object="'+op+'"][data-qk-iter="'+iter+'"]');if(!source)continue;
        const from=source.querySelector('rect'),to=target.querySelector('rect'),sx=Number(from.getAttribute('width'))/Number(to.getAttribute('width')),dx=Number(from.getAttribute('x'))-sx*Number(to.getAttribute('x')),dy=Number(from.getAttribute('y'))-Number(to.getAttribute('y'));
        const start='matrix('+sx+',0,0,1,'+dx+','+dy+')',arc=op==='mm'&&iter===second?-15:op==='pv'&&iter===first?12:0,middle='matrix('+((sx+1)/2)+',0,0,1,'+(dx/2)+','+(dy/2+arc)+')';
        const animation=animate(target,[{transform:start,offset:0},{transform:start,offset:.2,easing:'ease-in-out'},{transform:middle,offset:.48,easing:'ease-out'},{transform:'matrix(1,0,0,1,0,0)',offset:.76},{transform:'matrix(1,0,0,1,0,0)',offset:1}]);
        if(op==='mm'&&iter===second)flight=animation;
      }
      const graph=host.querySelector('.qk-pass-graph');
      // Keep the old edges while events are still in their original positions.
      // Fade them during travel; reveal final edges only after all endpoints land.
      for(const path of before.querySelectorAll('.qk-pass-dependency')){const ghost=path.cloneNode(true);ghost.classList.add('qk-skew-old-dependency');ghost.setAttribute('aria-hidden','true');ghost.setAttribute('marker-end','url(#qk-pass-arrow-after)');graph.append(ghost);run.ghosts.push(ghost);animate(ghost,[{opacity:1,offset:0},{opacity:1,offset:.2},{opacity:0,offset:.56},{opacity:0,offset:1}]);}
      for(const path of host.querySelectorAll('.qk-pass-dependency:not(.qk-skew-old-dependency)'))animate(path,[{opacity:0,offset:0},{opacity:0,offset:.76},{opacity:1,offset:.94},{opacity:1,offset:1}]);
      // The compact execution-order strip belongs to this same after chart.
      for(const target of host.querySelectorAll('.qk-pass-order button')){const source=before.querySelector('.qk-pass-order [data-qk-object="'+target.dataset.qkObject+'"][data-qk-iter="'+target.dataset.qkIter+'"]');if(!source)continue;const dx=source.getBoundingClientRect().left-target.getBoundingClientRect().left;animate(target,[{transform:'translateX('+dx+'px)',offset:0},{transform:'translateX('+dx+'px)',offset:.2},{transform:'translateX(0)',offset:.76},{transform:'translateX(0)',offset:1}]);}
      for(const node of linked)animate(node,[{opacity:.5},{opacity:1,offset:.35},{opacity:1}]);
      if(flight)flight.onfinish=()=>{if(motionRun!==run)return;for(const animation of run.animations){animation.onfinish=null;animation.cancel();}run.animations=[];for(const ghost of run.ghosts)ghost.remove();run.ghosts=[];complete();};else complete();
    });
  }
  function navigation(active){return '<nav class="qk-pass-chain" aria-label="Pass 分析阶段">'+[{id:'memory',label:'P19',name:'Q 驻留'},...stages].map((item,i)=>'<button type="button" data-qk-pipeline-pass="'+item.id+'" aria-pressed="'+(item.id===active)+'"><b>'+item.label+'</b><small>'+item.name+'</small></button>'+(i<3?'<span>→</span>':'')).join('')+'</nav>';}
  function render(state){const current=stages.find(s=>s.id===state.pass)||stages[1];
    return '<div class="qk-pipeline-passes'+(state.compact?' compact':'')+'">'+(state.compact?'':'<div class="qk-pass-change-caption"><strong>'+current.name+'</strong><span>结构证据 · 模拟排布</span></div>')+'<div class="qk-pass-comparison">'+[['before',current.before,current.left],['after',current.after,current.right]].map(([side,title,kind])=>'<article class="qk-pass-frame '+(current.id==='skew'&&side==='after'?'qk-skew-motion':'')+'" data-pass-version="'+side+'"><h4>'+title+(current.id==='skew'&&side==='after'?'<button type="button" class="ctl" data-qk-skew-replay>↻ 重播重排</button>':'')+'</h4>'+diagram(kind,side,state)+(current.id==='skew'&&side==='after'?'<span class="qk-skew-status" role="status" aria-live="polite"></span>':'')+(current.id==='split'&&side==='after'?'<div class="qk-vector-split"><span>AIC<b>QK / PV</b></span><span>AIV · subblock 0<b>有效 VEC 计算</b></span><span class="masked">AIV · subblock 1<b>valid_shape = 0</b></span></div>':'')+'</article>').join('')+'</div>'+(state.compact?'':'<p class="qk-pass-delta">'+current.change+'</p>')+'<div class="qk-pass-premise"> <span>固定 Q 分片</span><b>重复加载保持</b><span>→</span><button type="button" data-qk-step="2" data-qk-object="q">P19 · 驻留分析 ↗</button></div><details class="rc-sources"><summary>Pass 证据</summary>'+current.sources.map(([file,label])=>'<a href="'+root+file+'" target="_blank" rel="noopener">'+label+' ↗</a>').join('<br>')+'</details></div>';
  }
  return {render,model,play,stop,navigation};
})();
