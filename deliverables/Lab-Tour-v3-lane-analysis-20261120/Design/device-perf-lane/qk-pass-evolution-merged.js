// Continuous candidate compilation story. Geometry expresses structure, never device time.
window.QK_PASS_EVOLUTION=(()=>{
  const phases=[{id:'memory',label:'P19',title:'Q 驻留候选',note:'将重复 Q 加载移到循环外，分片复用同一面板。',file:'19_after_InferTileMemorySpace.py'}, {id:'exchange',label:'P23',title:'拆分 AIC / AIV',note:'显现分数和 P 的双向传递；QK / PV 与向量计算分到不同计算侧。',file:'23_after_ExpandMixedKernel.py'}, {id:'split',label:'P25',title:'明确 AIV 子块',note:'展开 subblock 分支与有效形状，保留已有数据依赖。',file:'25_after_SplitVectorKernel.py'}, {id:'skew',label:'P28',title:'形成跨分片流水',note:'QK₁ 前移到 PV₀ 前，与 AIV₀ 协作；PV₀ 仍等待 P₀。',file:'28_after_SkewCrossCorePipeline.py'}, {id:'lower',label:'P30',title:'展开内部执行结构',note:'保持 QK / PV 主顺序，显现 L0 计算展开。',file:'30_after_LowerPipelineLoops.py'}];
  const sub=n=>String(n).replace(/\d/g,d=>'₀₁₂₃₄₅₆₇₈₉'[+d]),names={q:'Q',mm:'QK',pv:'PV',send_score:'分数',wait_p:'等 P',recv_p:'收 P',scale:'缩放',exp:'exp',reduce:'归约',send_p:'P',write_o:'O',write_ml:'m/l'};
  let run=null;
  const phaseAt=value=>Math.max(0,Math.min(4,Math.ceil(value)-1)),valueFor=id=>Math.max(1,phases.findIndex(p=>p.id===id)+1);
  function geometry(step,first){
    const kind=step>=4?'skew':'serial',model=window.QK_PIPELINE_PASSES.model(kind,first),x=t=>118+t/model.end*418;
    return model.events.map(ev=>{let row=ev.row>=4?ev.row+1:ev.row;
      if(step<2&&['scale','exp','reduce'].includes(ev.op))row=1;
      let px=x(ev.start),y=36+row*29,w=Math.max(7,x(ev.end)-px),opacity=1;
      if(step<2&&['send_score','send_p','recv_p','wait_p'].includes(ev.op))opacity=.08;
      if(ev.op==='q'&&step>=1){if(ev.iter===first){px=118;y=7;w=95;}else{px=222;y=7;w=78;opacity=.65;}}
      return{...ev,x:px,y,w,opacity};
    });
  }
  function canvas(state,prefix='live'){
    const first=Math.floor(state.iter/2)*2,nodes=geometry(0,first),rows=['Q 加载 / 驻留','混合计算 / CUBE','分数 → AIV','AIV · VEC','AIV · 子块 1','AIC · 收 P','AIC · O 写回','AIV · m/l'];
    return '<svg class="qk-evolution-canvas" viewBox="0 0 560 292" role="group" aria-label="同一组分片的连续编译演变"><defs><marker id="evo-arrow-'+prefix+'" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#aaa"/></marker></defs><text x="310" y="20" class="evo-residency-caption" fill="#aaa" font-size="9">循环外 · 同一 Q 分片候选驻留</text>'+rows.map((label,i)=>'<text data-evo-row="'+i+'" x="109" y="'+(48+i*29)+'" text-anchor="end" fill="#999" font-size="9">'+label+'</text><rect x="118" y="'+(35+i*29)+'" width="418" height="23" fill="'+(i%2?'#111':'#181818')+'"/>').join('')+'<g class="evo-subblock"><rect x="118" y="152" width="418" height="20" fill="none" stroke="#555" stroke-dasharray="3 3"/><text x="326" y="165" text-anchor="middle" fill="#777" font-size="9">subblock 1 · valid_shape = 0</text></g>'+nodes.map(ev=>'<g class="qk-pass-node evo-node" data-evo-key="'+ev.op+'-'+ev.iter+'" data-qk-object="'+ev.op+'" data-qk-iter="'+ev.iter+'" tabindex="0" role="button" aria-label="'+names[ev.op]+' 分片 '+ev.iter+'"><rect rx="2"/><text text-anchor="middle" font-size="8" pointer-events="none"></text>'+(['mm','pv'].includes(ev.op)?'<g class="evo-l0-lines">'+[1,2,3].map(i=>'<line data-part="'+i+'" stroke="#333"/>').join('')+'</g>':'')+'</g>').join('')+'<g class="evo-dependencies">'+[first,first+1].flatMap(iter=>[['q','mm'],['mm','send_score'],['send_score','scale'],['send_p','recv_p'],['recv_p','pv']].map(([from,to])=>'<path data-evo-from="'+from+'-'+iter+'" data-evo-to="'+to+'-'+iter+'" fill="none" stroke="#888" stroke-dasharray="3 2" marker-end="url(#evo-arrow-'+prefix+')"/>')).join('')+'</g><text x="118" y="285" fill="#888" font-size="9">分片 '+first+' / '+(first+1)+' · 结构顺序（非设备计时）</text></svg>';
  }
  function emphasis(phase,op,iter,first){
    if(phase===0)return op==='q'?'changed':'context';
    if(phase===1)return ['scale','exp','reduce','send_score','send_p','recv_p','wait_p'].includes(op)?'changed':'context';
    if(phase===2)return ['scale','exp','reduce'].includes(op)?'changed':'context';
    if(phase===3){if(op==='mm'&&iter===first+1||op==='pv'&&iter===first)return'changed';return iter===first&&['send_score','scale','exp','reduce','send_p','recv_p','wait_p'].includes(op)?'related':'context';}
    return ['mm','pv'].includes(op)?'changed':'context';
  }
  const resultFor=(phase,first)=>['Q 加载已移到循环外，后续分片改为复用。','向量计算已拆到 AIV，分数与 P 的跨核依赖显现。','子块有效形状已明确，计算顺序没有改变。','QK'+sub(first+1)+' 已前移；PV'+sub(first)+' 仍等待自己的 P'+sub(first)+'。','内部计算已展开，QK/PV 主顺序保持。'][phase];
  function draw(svg,position,state){
    const first=Math.floor(state.iter/2)*2,lo=Math.floor(position),hi=Math.min(5,lo+1),t=position-lo,a=geometry(lo,first),b=geometry(hi,first),map=new Map(),phase=phaseAt(position),moving=lo===3&&t>0&&t<1;
    for(let i=0;i<a.length;i++){const from=a[i],to=b.find(e=>e.op===from.op&&e.iter===from.iter),n=svg.querySelector('[data-evo-key="'+from.op+'-'+from.iter+'"]'),mix=k=>from[k]+(to[k]-from[k])*t;let x=mix('x'),y=mix('y'),w=mix('w');if(moving&&from.op==='mm'&&from.iter===first+1)y-=Math.sin(t*Math.PI)*15;if(moving&&from.op==='pv'&&from.iter===first)y+=Math.sin(t*Math.PI)*10;
      const chosen=state.object===from.op&&state.iter===from.iter,related=state.iter===from.iter,rect=n.querySelector('rect'),text=n.querySelector('text');
      for(const[k,v]of Object.entries({x,y,width:w,height:18,fill:from.op==='q'?(position>=1?'#686868':'#6d3830'):from.op==='wait_p'?'#292929':from.iter===first?'#939393':'#626262',stroke:chosen?'#fff':related?'#aaa':'#444','stroke-width':chosen?2:1}))rect.setAttribute(k,v);
      const level=emphasis(phase,from.op,from.iter,first);n.setAttribute('data-evo-emphasis',level);n.setAttribute('opacity',mix('opacity')*(level==='changed'?1:level==='related'?.78:.22));rect.setAttribute('stroke',level==='changed'?'#f0f0f0':level==='related'?'#aaa':chosen?'#999':'#444');rect.setAttribute('stroke-width',level==='changed'?2:1);n.setAttribute('aria-pressed',String(chosen));text.setAttribute('x',x+w/2);text.setAttribute('y',y+12);text.setAttribute('fill',from.op==='q'||from.op==='wait_p'?'#ddd':'#111');text.textContent=w>20?(from.op==='q'&&position>=1?(from.iter===first?'Q 驻留':'复用 Q'):names[from.op]+(['mm','pv'].includes(from.op)?sub(from.iter):'')):'';
      const lines=n.querySelector('.evo-l0-lines');if(lines){lines.setAttribute('opacity',Math.max(0,position-4));for(const line of lines.children)line.setAttribute('stroke',phase===4?'#ededed':'#333');for(const line of lines.children){const px=x+w*Number(line.dataset.part)/4;for(const[k,v]of Object.entries({x1:px,x2:px,y1:y+1,y2:y+17}))line.setAttribute(k,v);}}
      map.set(from.op+'-'+from.iter,{x,y,w});
    }
    svg.querySelector('[data-evo-row="1"]').textContent=position>=2?'AIC · CUBE':'混合计算体';
    svg.querySelector('[data-evo-row="6"]').textContent=position>=2?'AIC · O 写回':'O 写回';
    svg.querySelector('[data-evo-row="7"]').textContent=position>=2?'AIV · m/l 写回':'m/l 写回';
    svg.querySelector('[data-evo-row="3"]').textContent=position>=3?'AIV0 · 有效 VEC':'AIV · VEC';
    svg.querySelector('.evo-subblock').setAttribute('opacity',Math.max(0,Math.min(1,position-2))*(phase===2?1:.23));svg.querySelector('.evo-subblock').setAttribute('data-evo-emphasis',phase===2?'changed':'context');svg.querySelector('.evo-subblock rect').setAttribute('stroke',phase===2?'#eee':'#555');svg.querySelector('.evo-subblock text').setAttribute('fill',phase===2?'#ddd':'#777');svg.querySelector('.evo-residency-caption').setAttribute('opacity',Math.min(1,position));
    for(const path of svg.querySelectorAll('[data-evo-from]')){
      const from=map.get(path.dataset.evoFrom),to=map.get(path.dataset.evoTo),qEdge=path.dataset.evoFrom.startsWith('q-'),tracked=phase===0&&qEdge||phase===3&&(path.dataset.evoTo==='mm-'+(first+1)||path.dataset.evoTo==='pv-'+first),added=phase===1&&!qEdge,related=phase===3&&[first].some(i=>path.dataset.evoFrom==='send_score-'+i||path.dataset.evoFrom==='send_p-'+i),level=added?'added':tracked?'tracked':related?'related':'context';
      path.setAttribute('d','M'+(from.x+from.w)+' '+(from.y+9)+' L'+to.x+' '+(to.y+9));path.setAttribute('data-evo-emphasis',level);path.setAttribute('stroke',added||tracked?'#eee':related?'#aaa':'#555');path.setAttribute('stroke-width',added||tracked?1.5:1);path.setAttribute('stroke-dasharray',added?'none':'3 2');path.setAttribute('opacity',(qEdge?1:Math.max(0,Math.min(1,position-1)))*(added||tracked?1:related?.75:.17));
    }

  }
  function render(state){const value=valueFor(state.pass);return '<section class="qk-pass-evolution" data-initial-stage="'+state.pass+'"><div class="evo-transport"><button type="button" class="ctl" data-evo-play>播放全过程</button><button type="button" class="ctl" data-evo-replay>↻ 本阶段</button><span class="evo-position-label"></span></div><input class="evo-scrubber" type="range" min="0" max="5" step="0.01" value="'+value+'" aria-label="拖动定位编译阶段"/><nav class="evo-stages" aria-label="连续 Pass 阶段">'+phases.map((p,i)=>'<button type="button" data-evo-stage="'+(i+1)+'" aria-pressed="'+(p.id===state.pass)+'"><b>'+p.label+'</b><small>'+p.title+'</small></button>').join('')+'</nav><div class="evo-stage-description"><strong></strong><span></span><em class="evo-playback-state"></em><p></p></div>'+canvas(state)+'<div class="evo-focus-legend"><span>白边：本步改动</span><span>实线：新增依赖</span><span>亮虚线：依赖跟随，关系不变</span></div><details class="evo-comparison" data-qk-section="comparison"><summary>展开本阶段前后对照</summary><div class="evo-comparison-body"></div></details><details class="evo-source"><summary>本阶段代码依据</summary><a target="_blank" rel="noopener"></a></details><span class="evo-status" role="status" aria-live="polite"></span></section>';}
  function stop(){const r=run;if(!r)return;run=null;r.playing=false;r.last=-1;r.host.removeAttribute('data-holding');const button=r.host.querySelector('[data-evo-play]');if(button)button.textContent=r.position>=5?'重新播放':'继续播放';if(r.raf)cancelAnimationFrame(r.raf);for(const n of r.linked)n.classList.remove('qk-skew-linked','evo-primary-link');r.host.removeAttribute('aria-busy');}
  function mount(panel,state,onStage){
    stop();const host=panel.querySelector('.qk-pass-evolution');if(!host)return;const r={host,position:valueFor(state.pass),playing:false,hasRun:false,raf:0,linked:[],last:-1,limit:5,holdRemaining:0};run=r;
    const svg=host.querySelector('.qk-evolution-canvas'),slider=host.querySelector('.evo-scrubber'),button=host.querySelector('[data-evo-play]');
    const update=()=>{
      if(run!==r)return;if(r.lastDraw!==r.position){draw(svg,r.position,state);r.lastDraw=r.position;}slider.value=String(r.position);const index=phaseAt(r.position),p=phases[index];slider.setAttribute('aria-valuetext',r.position===0?'Pass 前':p.label+' '+p.title);host.querySelector('.evo-position-label').textContent=r.position===0?'Pass 前':p.label;host.toggleAttribute('data-holding',r.holdRemaining>0);host.querySelector('.evo-playback-state').textContent=r.holdRemaining>0?(r.playing?'阶段停留':'已暂停 · 阶段停留'):r.playing?'演变中':'';
      for(const n of host.querySelectorAll('[data-evo-stage]'))n.setAttribute('aria-pressed',String(Number(n.dataset.evoStage)===index+1));
      if(index!==r.last){r.last=index;onStage(p.id);host.querySelector('.evo-stage-description strong').textContent=p.title;host.querySelector('.evo-stage-description span').textContent=index===0?'候选假设':'后续结构参考';host.querySelector('.evo-stage-description p').textContent=p.note;host.querySelector('.evo-status').textContent=p.label+' '+p.title;
        for(const n of r.linked)n.classList.remove('qk-skew-linked');r.linked=[];const first=Math.floor(state.iter/2)*2,ops=index===0?['q']:index===1?['send_score','recv_p','send_p','wait_p','scale','exp','reduce']:index===2?['scale','exp','reduce']:index===3?['mm','pv','scale','exp','reduce','send_p','recv_p']:['mm','pv','l0'];for(const op of ops)for(const iter of [first,first+1])for(const n of document.querySelectorAll('#lane [data-qk-version][data-qk-inline-event="'+op+'"][data-qk-inline-iter="'+iter+'"]')){n.classList.add('qk-skew-linked');if(emphasis(index,op,iter,first)==='changed')n.classList.add('evo-primary-link');r.linked.push(n);}
        const comparison=host.querySelector('.evo-comparison-body');comparison.innerHTML='<article><h4>'+p.label+' 前</h4>'+canvas(state,'compare-before')+'</article><article><h4>'+p.label+' 后</h4>'+canvas(state,'compare-after')+'</article>';const charts=comparison.querySelectorAll('svg');draw(charts[0],index,state);draw(charts[1],index+1,state);const source=host.querySelector('.evo-source a');source.href='../../Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/'+p.file;source.textContent=p.file+' ↗';
      }
      const firstFragment=Math.floor(state.iter/2)*2;host.querySelector('.evo-stage-description p').textContent=r.holdRemaining>0?resultFor(index,firstFragment):p.note.replace(/QK₁/g,'QK'+sub(firstFragment+1)).replace(/PV₀/g,'PV'+sub(firstFragment)).replace(/AIV₀/g,'AIV'+sub(firstFragment)).replace(/P₀/g,'P'+sub(firstFragment));
      button.textContent=r.playing?'暂停':r.hasRun&&r.position<r.limit?'继续播放':r.position>=5?'重新播放':'播放全过程';host.toggleAttribute('aria-busy',r.playing);
    };
    const pause=()=>{r.playing=false;if(r.raf)cancelAnimationFrame(r.raf);r.raf=0;update();};
    const start=limit=>{
      r.limit=limit;if(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches){r.position=limit;r.playing=false;r.hasRun=true;r.holdRemaining=0;update();return;}
      r.playing=true;r.hasRun=true;let last=null;update();
      const tick=now=>{
        if(run!==r||!r.playing)return;let dt=last===null?0:Math.max(0,now-last);last=now;
        for(let guard=0;dt>0&&guard<12;guard++){
          if(r.holdRemaining>0){const used=Math.min(dt,r.holdRemaining);r.holdRemaining-=used;dt-=used;if(r.holdRemaining>0)break;if(r.position>=r.limit){r.playing=false;break;}}
          if(r.position>=r.limit){r.playing=false;break;}
          const boundary=Math.min(r.limit,Math.floor(r.position)+1),needed=(boundary-r.position)*1400;
          if(dt<needed){r.position+=dt/1400;dt=0;}else{r.position=boundary;dt-=needed;r.holdRemaining=1200;}
        }
        update();if(!r.playing){r.raf=0;return;}r.raf=requestAnimationFrame(tick);
      };r.raf=requestAnimationFrame(tick);
    };
    host.addEventListener('click',ev=>{const control=ev.target.closest('[data-evo-stage],[data-evo-play],[data-evo-replay]');if(!control)return;ev.stopPropagation();if(run!==r){stop();run=r;}if(control.dataset.evoStage){pause();r.position=Number(control.dataset.evoStage);r.holdRemaining=0;r.hasRun=false;update();return;}if(control.hasAttribute('data-evo-replay')){pause();const target=phaseAt(r.position)+1;r.position=target-1;r.holdRemaining=0;start(target);return;}if(r.playing){pause();return;}if(!r.hasRun||r.position>=r.limit&&r.holdRemaining===0){r.position=0;r.holdRemaining=0;}start(5);});
    slider.addEventListener('input',ev=>{ev.stopPropagation();if(run!==r){stop();run=r;}const value=Number(slider.value);pause();r.position=value;r.holdRemaining=0;r.hasRun=true;r.limit=5;update();});
    update();
  }
  return{render,mount,stop,geometry,phases};
})();
