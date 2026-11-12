/* Static loop semantics from paged_attention_pypto.py. Ticks are logical
 * iterations, never measured durations. Slot ribbons are modulo reuse, not data dependencies. */
(function(global){
  'use strict';
  const stages={qk:'QK',softmax:'Softmax',pv:'PV',accumulate:'Context Accumulate'};
  const buffers={qk:{write:'score_transfer'},softmax:{read:'score_transfer',write:'probability_transfer'},pv:{read:'probability_transfer',write:'pv_transfer'},accumulate:{read:'pv_transfer'}};
  const events={QK_READY_EVENT:['qk','softmax'],SOFTMAX_READY_EVENT:['softmax','pv'],PV_READY_EVENT:['pv','accumulate']};
  const transfers=[{buffer:'score_transfer',producer:'qk',consumer:'softmax',event:'QK_READY_EVENT'},
    {buffer:'probability_transfer',producer:'softmax',consumer:'pv',event:'SOFTMAX_READY_EVENT'},
    {buffer:'pv_transfer',producer:'pv',consumer:'accumulate',event:'PV_READY_EVENT'}];
  const sourceVersion='sha256:4087a49cd0b6c6396d4fef859576a02f8c6b096e9b94f5b55560e1f4b238e843';
  const sourceRanges={qk:[[279,357],[357,362]],softmax:[[762,817],[817,823]],pv:[[364,503],[504,718],[719,724]],accumulate:[[827,845],[846,874]]};
  const sourceRef=range=>({file:'paged_attention_pypto.py',ranges:Array.isArray(range[0])?range:[range],version:sourceVersion,origin:'PTO_PAGED_ATTENTION_PYPTO_SOURCE'});
  function createModel({anatomy,preLaunch}){
    return {...anatomy,preLaunch,tasks:anatomy.tasks.map(task=>{
      const transferBase=task.block*anatomy.transferSlots*anatomy.rowTile;
      const stacks=task.stacks.map(stack=>({...stack,produceTick:stack.index,consumeTick:stack.index+preLaunch,
        slotEvidence:{level:'model-derived',rule:'stack.index % TRANSFER_SLOTS',inputs:{stackIndex:stack.index,transferSlots:anatomy.transferSlots},source:sourceRef([31,32])},
        produceRow:transferBase+stack.slot*anatomy.rowTile,consumeRow:transferBase+stack.slot*anatomy.rowTile}));
      const ticks=Array.from({length:stacks.length+preLaunch},(_,index)=>({index,evidence:{level:'model-derived',rule:'pl.range(stack_count + PRE_LAUNCH)',inputs:{stackCount:stacks.length,preLaunch},source:sourceRef([758,833])},
        produce:index<stacks.length?index:null,consume:index>=preLaunch?index-preLaunch:null}));
      return {...task,stacks,ticks,transferBase,dependencies:stacks.flatMap(s=>transfers.map(t=>({...t,kind:'data',synchronizationKind:'event',stack:s.index,slot:s.slot,evidence:{level:'source-confirmed',source:sourceRef([270,874])}}))),
        reuse:stacks.filter(s=>s.index>=anatomy.transferSlots).flatMap(s=>transfers.map(t=>({...t,kind:'resource-reuse',slot:s.slot,from:s.index-anatomy.transferSlots,to:s.index,evidence:{level:'model-derived',rule:'stack.index % TRANSFER_SLOTS',inputs:{stackIndex:s.index,transferSlots:anatomy.transferSlots},source:sourceRef([31,32])}}))),
        operations:stacks.flatMap(s=>Object.keys(stages).map(stage=>{const tick=['qk','softmax'].includes(stage)?s.produceTick:s.consumeTick,operationId=`paged/task/${task.id}/stack/${s.index}/${stage}`,bufferObject=name=>({name,kind:'buffer'});return {stack:s.index,stage,tick,slot:s.slot,...buffers[stage],computeUnitId:`paged/task/${task.id}`,iterationId:`paged/task/${task.id}/stack/${s.index}`,operationId,operationType:stage,logicalPosition:{kind:'tick',index:tick},inputs:buffers[stage].read?[bufferObject(buffers[stage].read)]:[],outputs:buffers[stage].write?[bufferObject(buffers[stage].write)]:[],dependencies:transfers.filter(t=>t.consumer===stage).flatMap(t=>[{kind:'data',from:`paged/task/${task.id}/stack/${s.index}/${t.producer}`,to:operationId,buffer:t.buffer},{kind:'synchronization',from:`paged/task/${task.id}/stack/${s.index}/${t.producer}`,to:operationId,event:t.event}]),resourceRefs:[`paged/task/${task.id}/slot/${s.slot}`],evidence:{operation:{level:'source-confirmed',source:sourceRef(sourceRanges[stage])},logicalPosition:{level:'model-derived',rule:['qk','softmax'].includes(stage)?'stack.index':'stack.index + PRE_LAUNCH',inputs:{stackIndex:s.index,preLaunch},source:sourceRef([758,833])},visualSpacing:{level:'presentation-only'},runtimeDuration:{level:'unverified'}}};}))};
    })};
  }
  function mount(host,{model,scenario,onTask,onFocus,onCamera,onInspector,onNavigate,onExit,presentation=false,onOpenSource}){
    const media=global.matchMedia('(prefers-reduced-motion: reduce)'),listeners=new AbortController();
    const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const listen=(element,type,fn,options={})=>element.addEventListener(type,fn,{...options,signal:listeners.signal});
    host.classList.add('viz2-anatomy','viz2-intra','viz2-compute','viz2-pipeline-detail');
    window.PtoPhotoNarrative.html(host,`${global.PtoComputeDetailControls.identity()}<div class="compute-parameter-control" data-compute-toolbar></div><button class="btn btn-ghost btn-sm" data-intra-back>← 返回 Block 间</button>
      <div class="viz2-anatomy-body"><div class="viz2-anatomy-stage"><div class="viz2-anatomy-workspace">
        <div class="viz2-anatomy-viewport" id="viz2-intra-carousel" role="region" tabindex="0" aria-label="当前 Task 逻辑流水线，画布内可平移浏览 Logical Tick"><div class="viz2-anatomy-track" data-intra-track></div></div>
        <div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div>
        <footer class="viz2-anatomy-footer"><span class="viz2-intra-legend"><i class="viz2-intra-flow-key"></i>数据依赖<i class="viz2-intra-event-key"></i>READY 事件<i class="viz2-intra-reuse-key"></i>Slot 复用<small>Logical Tick 为离散进度，不代表耗时</small><small data-intra-dependency-summary aria-live="polite"></small></span><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" data-intra-zoom="0.8" aria-label="缩小逻辑流水线">−</button><output data-intra-scale></output><button class="btn btn-ghost btn-sm" data-intra-zoom="1.25" aria-label="放大逻辑流水线">＋</button><button class="btn btn-ghost btn-sm" data-intra-fit>适应</button></div></footer></div>
        <aside class="pto-ide-frame__pane viz2-anatomy-inspector" aria-label="流水线对象详情" hidden><header class="pto-ide-frame__pane-header"><div><b data-intra-title></b><small data-intra-context></small></div><button class="btn btn-ghost btn-sm" data-intra-close aria-label="关闭流水线详情">×</button></header><div class="pto-ide-frame__pane-body" data-intra-detail></div></aside></div>`);
    const viewport=host.querySelector('.viz2-anatomy-viewport'),track=host.querySelector('[data-intra-track]'),workspace=host.querySelector('.viz2-anatomy-workspace'),inspector=host.querySelector('.viz2-anatomy-inspector'),tooltip=host.querySelector('[role="tooltip"]');
    let task=null,focus=null,active=false,block=null,tasks=[],cards=[],position=0,target=0,velocity=0,frame=0,drag=null,suppress=false,scale=1,pitch=440,pendingFocus=null,navButtons=[],navTask=null,inspectorOpen=false;
    let sourceTask=null,isCustom=false;
    const controls=global.PtoComputeDetailControls,toolbar=host.querySelector('[data-compute-toolbar]');
    window.PtoPhotoNarrative.html(toolbar,controls.parameter('Stacks','Pages'));
    const syncScenario=controls.connect(toolbar,scenario);
    window.PtoPhotoNarrative.html(host.querySelector('[data-compute-identity]'),controls.picker(model,'Pipeline Task'));
    const stackInput=toolbar.querySelector('[data-compute-pages]'),stackRange=toolbar.querySelector('[data-page-range]');
    let maxStacks=1;
    function configureStacks(value){
      const result=scenario.edit('stacks',value),error=toolbar.querySelector('[data-page-error]');
      window.PtoPhotoNarrative.text(error,result.error||'');if(result.error)stackInput.setAttribute('aria-invalid','true');else stackInput.removeAttribute('aria-invalid');
    }
    listen(stackInput,'input',()=>configureStacks(stackInput.value));
    listen(stackRange,'input',()=>configureStacks(stackRange.value));
    listen(toolbar,'click',event=>{const button=event.target.closest('[data-page-step]');if(button)configureStacks(task.stacks.length+Number(button.dataset.pageStep));});
    listen(host.querySelector('[data-compute-task]'),'change',event=>onTask(Number(event.target.value)));
    function identity(){
      syncScenario();
      const picker=host.querySelector('[data-compute-task]');picker.value=sourceTask.id;window.PtoPhotoNarrative.attr(picker,'data-display-label',`Task ${task.id}`);
      window.PtoPhotoNarrative.text(host.querySelector('[data-compute-owner]'),`Block ${String(task.block).padStart(2,'0')} / KV Head ${task.kvHead}`);
      window.PtoPhotoNarrative.text(host.querySelector('[data-compute-facts]'),`Batch ${task.batch}\n${task.seqLen} tokens`);
      stackInput.value=stackRange.value=task.stacks.length;
      window.PtoPhotoNarrative.text(toolbar.querySelector('[data-stack-count]'),task.pages.length);
      toolbar.querySelector('[data-page-step="-1"]').disabled=task.stacks.length<=1;
      toolbar.querySelector('[data-page-step="1"]').disabled=task.stacks.length>=maxStacks;
      window.PtoPhotoNarrative.text(toolbar.querySelector('[data-page-summary]'),isCustom?'Custom · 模型推导调度':'');
    }
    const scrollStates=new Map(),scales=new Map();let pan={x:0,y:0};
    const clamp=index=>Math.max(0,Math.min(tasks.length-1,index));
    const paging=global.PtoVisualization2Anatomy.createPagingInput({isAnimating:()=>!!frame,onPage:direction=>{
      const next=clamp(target+direction);if(next!==target){pendingFocus=null;tooltip.hidden=true;snap(next,true);}
    }});
    // Both operators use horizontal logical progress and fixed vertical stage rows.
    const LEFT=118, TOP=presentation?74:82, STEP=presentation?145:174, ROW=presentation?64:104, W=presentation?124:148, H=56;
    const stageRows={qk:0,softmax:1,pv:2,accumulate:3}, outputWidth=164;
    const diagramSize=t=>({width:LEFT+t.ticks.length*STEP+outputWidth+28,height:TOP+4*ROW+(presentation?44:105)});
    const viewportSize=t=>presentation ? diagramSize({ticks:{length:presentation.maxTicks}}) : diagramSize(t);
    let automaticFit=false;
    const fitScale=(scroller,size)=>Math.max(.05,Math.min(1,
      (scroller.clientWidth-(presentation?12:64))/size.width,
      (scroller.clientHeight-(presentation?12:48))/size.height));
    const stagePosition=(t,s,stage)=>({x:LEFT+t.operations.find(op=>op.stack===s.index&&op.stage===stage).tick*STEP,y:TOP+stageRows[stage]*ROW});
    const text=(x,y,s,cls='')=>`<text x="${x}" y="${y}" class="${cls}">${esc(s)}</text>`;
    const rect=(x,y,w,h,cls,extra='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" class="${cls}" ${extra}/>`;
    const object=(kind,index,content,attrs='')=>`<g tabindex="0" role="button" data-intra-kind="${kind}" data-intra-index="${index}" ${attrs} aria-pressed="false">${content}</g>`;
    const stroke=slot=>`style="--slot-color:var(--intra-slot-${slot%3})"`;
    const currentCard=()=>cards.find(c=>Number(c.dataset.intraTask)===task?.id);
    const currentScene=()=>currentCard()?.querySelector('[data-intra-scene]');
    const syncFor=f=>global.PTO_PYPA_SEMANTIC_GRAPH.synchronizations.find(item=>item.name+'_EVENT'===f?.event||item.id===f?.syncId);
    const eventEdge=f=>global.PtoPagedAttentionStackDependencyModel.build(global.PtoPagedAttentionIntro.model.pipelineFor(task.seqLen),global.PTO_PYPA_SEMANTIC_GRAPH.synchronizations).edges.find(item=>item.id===`sync:s${f.index}:${syncFor(f)?.id}`);
    function dependencyGeometry(t,link){
      const a=stagePosition(t,t.stacks[link.stack],link.producer),b=stagePosition(t,t.stacks[link.stack],link.consumer);
      const x=a.x+W/2,y=a.y+H,tx=b.x+W/2,ty=b.y,mid=(y+ty)/2+(link.producer==='softmax'?(link.stack%model.transferSlots)*7:0);
      return {d:link.producer==='softmax' ? `M ${x} ${y} C ${x} ${y+24}, ${tx} ${ty-24}, ${tx} ${ty}` : `M ${x} ${y} V ${ty}`,event:{x:(x+tx)/2,y:(y+ty)/2}};
    }
    function reuseGeometry(t,link){
      const a=stagePosition(t,t.stacks[link.from],link.consumer),b=stagePosition(t,t.stacks[link.to],link.producer);
      const band=link.buffer==='score_transfer'?TOP-22-link.slot*5:link.buffer==='probability_transfer'?TOP+ROW*2-18-link.slot*5:TOP+ROW*4-18+link.slot*5;
      return `M ${a.x+W} ${a.y+H/2} H ${a.x+W+12} V ${band} H ${b.x-12} V ${b.y+H/2} H ${b.x}`;
    }
    function card(t){
      const {width,height}=diagramSize(t),outputX=LEFT+t.ticks.length*STEP+10,outputY=TOP+stageRows.accumulate*ROW;
      const arrow=`intra-dependency-arrow-${host.id||'pipeline'}-${t.id}`;
      const last=t.stacks.at(-1),lastPosition=last?stagePosition(t,last,'accumulate'):null;
      const diagram=`<defs><marker id="${arrow}" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>`+
        rect(0,0,width,height,'viz2-intra-frame','data-intra-frame')+
        text(LEFT,24,'Logical Tick →','viz2-intra-axis')+
        Object.entries(stageRows).map(([stage,index])=>text(10,TOP+index*ROW+H/2+5,stage==='accumulate'?'Accumulate':stages[stage],'viz2-intra-row-title')).join('')+
        t.ticks.map(tick=>object('tick',tick.index,rect(LEFT+tick.index*STEP-10,48,STEP,4*ROW+12,'viz2-intra-tick-hit')+text(LEFT+tick.index*STEP+W/2,66,`T${tick.index}`,'viz2-intra-tick-label'),`aria-label="逻辑 Tick ${tick.index}"`)).join('')+
        t.reuse.map(link=>`<path tabindex="0" role="button" aria-pressed="false" aria-label="${link.buffer} Slot ${link.slot}，Stack ${link.from} 到 ${link.to} 复用" data-intra-kind="slot" data-intra-index="${link.slot}" data-intra-buffer="${link.buffer}" data-intra-from="${link.from}" data-intra-to="${link.to}" data-intra-reuse="${link.buffer}:${link.from}:${link.to}" class="viz2-intra-reuse" ${stroke(link.slot)} d="${reuseGeometry(t,link)}"/>`).join('')+
        t.stacks.map(stack=>`<g data-intra-stack-path="${stack.index}" ${stroke(stack.slot)}>`+
          t.dependencies.filter(link=>link.stack===stack.index).map(link=>{const g=dependencyGeometry(t,link);return `<path class="viz2-intra-flow" data-intra-dependency="${stack.index}:${link.buffer}" d="${g.d}" marker-end="url(#${arrow})"/>`+
            object('event',stack.index,`<circle cx="${g.event.x}" cy="${g.event.y}" r="13" class="viz2-intra-event-hit"/><circle cx="${g.event.x}" cy="${g.event.y}" r="5" class="viz2-intra-event"/>`,`data-intra-event="${link.event}" data-intra-buffer="${link.buffer}" aria-label="Stack ${stack.index} ${link.event}"`);}).join('')+
          Object.keys(stages).map(stage=>{const n=stagePosition(t,stack,stage);
            const label=presentation&&stage==='accumulate'?'Accum.':stages[stage];
            return object('stage',stack.index,
            rect(n.x,n.y,W,H,'viz2-intra-operation',`data-intra-operation="${stack.index}:${stage}"`)+(global.PtoAIInfraPresentation?.operationGlyph ? global.PtoAIInfraPresentation.operationGlyph(`${t.id}:${stack.index}:${stage}`, `S${stack.index} · ${label}`, n.x,n.y,W,H,stack.slot) : text(n.x+10,n.y+19,`S${stack.index}`)+text(n.x+10,n.y+36,`Slot ${stack.slot}`,'viz2-intra-node-slot')),
            `data-intra-stage="${stage}" aria-label="Stack ${stack.index} ${stages[stage]}，Slot ${stack.slot}"`);}).join('')+`</g>`).join('')+
        // A compact Stack navigator gives Stack selection a distinct hit target.
        t.stacks.map(stack=>{const n=stagePosition(t,stack,'qk');return object('stack',stack.index,rect(n.x,TOP+4*ROW+10,W,24,'viz2-intra-stack-hit')+text(n.x+8,TOP+4*ROW+27,`Stack ${stack.index}`,'viz2-intra-small'),`aria-label="检查 Stack ${stack.index} 的完整路径"`);}).join('')+
        (lastPosition?`<path class="viz2-intra-output-link" d="M ${lastPosition.x+W} ${lastPosition.y+H/2} H ${outputX}" marker-end="url(#${arrow})"/>`:'')+
        object('output',0,rect(outputX,outputY,outputWidth,H,'viz2-intra-output-hit','data-intra-output')+rect(outputX,outputY,outputWidth,H,'viz2-intra-output')+text(outputX+10,outputY+19,'Normalize / Store','viz2-intra-small')+text(outputX+10,outputY+38,'全部 Stack 完成','viz2-intra-small'),`aria-label="全部 Stack 完成后的 Output Normalize 与 BF16 Store"`);
      return `<article class="viz2-anatomy-card viz2-intra-card" data-intra-task="${t.id}" role="group" aria-roledescription="幻灯片" aria-label="${isCustom?`Custom · 模型推导调度`:`Task ${t.id}，Batch ${t.batch}，${t.seqLen} tokens`}"><header><h3 class="viz2-intra-heading">Task ${t.id}</h3><p>Batch ${t.batch} · KV Head ${t.kvHead} · ${t.seqLen} tokens · ${t.stacks.length} Stacks</p></header><div class="viz2-anatomy-cache-scroll viz2-intra-tick-scroll" tabindex="0" aria-label="Task ${t.id} 的横向 Logical Tick"><svg class="viz2-intra-svg" viewBox="0 0 ${width} ${height}" data-intra-width="${width}" data-intra-height="${height}" aria-label="横向 Logical Tick × 纵向计算阶段"><g data-intra-scene>${diagram}</g></svg></div></article>`;
    }
    function saveScroll(){cards.forEach(card=>{const scroller=card.querySelector('.viz2-intra-tick-scroll');scrollStates.set(Number(card.dataset.intraTask),{top:scroller.scrollTop,left:scroller.scrollLeft});});}
    function buildBlock(){
      block=task.block;tasks=[task];
      window.PtoPhotoNarrative.html(track,tasks.map(card).join(''));cards=[...track.children];
      cards.forEach(card=>{const saved=scrollStates.get(Number(card.dataset.intraTask));if(saved){const s=card.querySelector('.viz2-intra-tick-scroll');s.scrollTop=saved.top;s.scrollLeft=saved.left;}});
      position=target=tasks.findIndex(t=>t.id===task.id);velocity=0;
    }
    function paint(){
      if(!active||!task||!viewport.clientWidth)return;
      const width=viewport.clientWidth;pitch=width;
      track.style.transform='none';
      cards.forEach((element,index)=>{
        const current=Number(element.dataset.intraTask)===task.id;
        element.style.width=`${width}px`;element.style.left='0px';element.hidden=!current;
        element.style.opacity='1';
        element.classList.toggle('is-current',current);element.classList.remove('is-before');element.setAttribute('aria-current',String(current));
        const svg=element.querySelector('svg[data-intra-height]');
        const scroller=element.querySelector('.viz2-intra-tick-scroll');
        const size=viewportSize(tasks[index]);
        if(automaticFit&&current)scale=fitScale(scroller,size);
        const localScale=current?scale:scales.get(tasks[index].id)||1;
        const vw=scroller.clientWidth,vh=scroller.clientHeight,dw=size.width,dh=size.height;svg.style.width='100%';svg.style.height='100%';const cp=current?pan:{x:0,y:0};svg.setAttribute('viewBox',[(dw-vw/localScale)/2+cp.x,(dh-vh/localScale)/2+cp.y,vw/localScale,vh/localScale].join(' '));
      });
      window.PtoPhotoNarrative.text(host.querySelector('[data-intra-scale]'),`${Math.round(scale*100)}%`);
    }
    function stop(){cancelAnimationFrame(frame);frame=0;paging.reset();if(drag&&viewport.hasPointerCapture(drag.id))viewport.releasePointerCapture(drag.id);drag=null;pendingFocus=null;tooltip.hidden=true;}
    function commit(){
      const next=tasks[target];if(!next)return;
      if(next.id!==task.id)onTask(next.id);
      if(pendingFocus){const nextFocus=pendingFocus;pendingFocus=null;onFocus(nextFocus);}
    }
    function snap(index,fromWheel=false){
      if(!fromWheel)paging.reset();cancelAnimationFrame(frame);target=clamp(Math.round(index));velocity=0;tooltip.hidden=true;
      position=target;frame=0;paint();commit();paging.flush();
    }
    const resistance=value=>{const bound=clamp(value);return bound+(value-bound)*.22;};
    function objectFocus(el){return el?{kind:el.dataset.intraKind,index:Number(el.dataset.intraIndex),...(el.dataset.intraStage?{stage:el.dataset.intraStage}:{}),...(el.dataset.intraEvent?{event:el.dataset.intraEvent}:{}),...(el.dataset.intraBuffer?{buffer:el.dataset.intraBuffer}:{}),...(el.dataset.intraFrom!=null?{from:Number(el.dataset.intraFrom),to:Number(el.dataset.intraTo)}:{})}:null;}
    function effective(f){if(f?.kind==='page')return {kind:'stack',index:task.pages[f.index]?.stack};return f;}
    function validFocus(f){
      if(!f)return false;const e=effective(f);
      if(e.kind==='output')return true;
      if(e.kind==='tick')return !!task.ticks[e.index];
      if(e.kind==='slot')return !!task.slots[e.index]&&(!e.buffer||transfers.some(t=>t.buffer===e.buffer))&&(e.from==null||task.reuse.some(r=>r.buffer===e.buffer&&r.from===e.from&&r.to===e.to));
      return !!task.stacks[e.index]&&(['stack'].includes(e.kind)||e.kind==='stage'&&!!stages[e.stage]||e.kind==='event'&&!!events[e.event]);
    }
    function paintFocus(current=focus){
      const scene=currentScene();if(!scene)return;
      const f=effective(current),fixed=effective(focus),chosen=f&&['stack','stage','event'].includes(f.kind)?task.stacks[f.index]:null,slot=f?.kind==='slot'?f.index:chosen?.slot;
      scene.classList.toggle('has-selection',!!f);
      const relevantBuffers=f?.buffer?[f.buffer]:f?.stage?Object.values(buffers[f.stage]):transfers.map(t=>t.buffer);
      track.querySelectorAll('.is-selected,.is-related,.is-shared,.is-highlighted,.is-upstream,.is-downstream').forEach(el=>{el.classList.remove('is-selected','is-related','is-shared','is-highlighted','is-upstream','is-downstream');if(el.hasAttribute('aria-pressed'))el.setAttribute('aria-pressed','false');});
      scene.querySelectorAll('[data-intra-reuse]').forEach(el=>el.classList.toggle('is-highlighted',Number(el.dataset.intraIndex)===slot&&relevantBuffers.includes(el.dataset.intraBuffer)));
      scene.querySelectorAll('[data-intra-stack-path]').forEach(el=>{const s=task.stacks[Number(el.dataset.intraStackPath)];el.classList.toggle('is-related',f?.kind==='stack'&&s.index===chosen?.index);el.classList.toggle('is-shared',slot!=null&&s.slot===slot&&s.index!==chosen?.index);});
      scene.querySelectorAll('[data-intra-dependency]').forEach(el=>{const [stack,buffer]=el.dataset.intraDependency.split(':');const link=transfers.find(item=>item.buffer===buffer);el.classList.toggle('is-related',Number(stack)===f?.index&&(f?.kind==='stage'&&(link?.producer===f.stage||link?.consumer===f.stage)||f?.kind==='event'&&link?.event===f.event));});
      scene.querySelectorAll('[data-intra-kind]').forEach(el=>{
        const item=objectFocus(el),s=task.stacks[item.index];let related=false;
        if(f?.kind==='tick')related=item.kind==='tick'?item.index===f.index:item.kind==='stage'&&(['qk','softmax'].includes(item.stage)?s.produceTick:s.consumeTick)===f.index;
        if(f?.kind==='stack')related=['stage','stack','event'].includes(item.kind)&&item.index===f.index;
        if(f?.kind==='event')related=item.kind==='stage'&&item.index===f.index&&events[f.event]?.includes(item.stage);
        if(f?.kind==='stage')related=item.kind==='stage'&&item.index===f.index&&transfers.some(link=>(link.producer===f.stage&&link.consumer===item.stage)||(link.consumer===f.stage&&link.producer===item.stage))||item.kind==='event'&&item.index===f.index&&transfers.some(link=>link.event===item.event&&(link.producer===f.stage||link.consumer===f.stage));
        if(f?.kind==='slot')related=['stack','stage'].includes(item.kind)&&s?.slot===f.index;
        const selected=!!fixed&&item.kind===fixed.kind&&item.index===fixed.index&&item.stage===fixed.stage&&item.event===fixed.event&&item.buffer===fixed.buffer&&(item.kind!=='slot'||item.from===fixed.from);
        el.classList.toggle('is-selected',selected||fixed?.kind==='stack'&&item.kind==='stage'&&item.index===fixed.index);el.classList.toggle('is-upstream',related&&item.kind==='stage'&&f?.kind==='stage'&&transfers.some(link=>link.consumer===f.stage&&link.producer===item.stage));el.classList.toggle('is-downstream',related&&item.kind==='stage'&&f?.kind==='stage'&&transfers.some(link=>link.producer===f.stage&&link.consumer===item.stage));el.classList.toggle('is-related',related);el.setAttribute('aria-pressed',String(selected));
        if(item.kind==='slot')el.classList.toggle('is-related',item.index===slot&&relevantBuffers.includes(item.buffer));
      });
      const summary=host.querySelector('[data-intra-dependency-summary]');if(summary){const link=transfers.filter(t=>f?.kind==='stage'&&(t.producer===f.stage||t.consumer===f.stage));window.PtoPhotoNarrative.text(summary,f?.kind==='stage'?`S${f.index} · ${stages[f.stage]} | ↑ ${link.filter(t=>t.consumer===f.stage).map(t=>stages[t.producer]).join(', ')||'输入'} | ↓ ${link.filter(t=>t.producer===f.stage).map(t=>stages[t.consumer]).join(', ')||'输出'}`:f?.kind==='event'?`S${f.index} · ${f.event} | ${events[f.event]?.map(stage=>stages[stage]).join(' → ')}`:f?.kind==='slot'?`Slot ${f.index} · ${f.buffer||'Transfer buffers'} | S${f.from??task.stacks.find(s=>s.slot===f.index)?.index} → S${f.to??task.stacks.filter(s=>s.slot===f.index).at(-1)?.index}`:f?.kind==='stack'?`Stack ${f.index} · Slot ${task.stacks[f.index].slot}`:'');}
    }
    function revealFocus(){if(global.PtoAIInfraPresentation)return;
      if(!validFocus(focus))return;
      const scene=currentScene(),f=effective(focus);if(!scene)return;
      const el=[...scene.querySelectorAll('[data-intra-kind]')].find(el=>{const item=objectFocus(el);return item.kind===f.kind&&item.index===f.index&&(!f.stage||item.stage===f.stage)&&(!f.event||item.event===f.event)&&(!f.buffer||item.buffer===f.buffer)&&(f.from==null||item.from===f.from);});
      if(!el)return;const scroller=currentCard().querySelector('.viz2-intra-tick-scroll'),r=el.getBoundingClientRect(),v=scroller.getBoundingClientRect();
      if(r.top<v.top||r.bottom>v.bottom)scroller.scrollTop+=r.top-v.top-(v.height-Math.min(r.height,v.height))/2;
      if(r.left<v.left||r.right>v.right)scroller.scrollLeft+=r.left-v.left-(v.width-Math.min(r.width,v.width))/2;
    }
    function describe(f){
      f=effective(f);if(!f)return {title:`Task ${task.id}`,facts:[]};
      const s=task.stacks[f.index],slot=f.kind==='slot'?f.index:s?.slot;
      const reuse=task.stacks.filter(item=>item.slot===slot).map(item=>item.index);
      if(f.kind==='tick'){const t=task.ticks[f.index];return {title:`Tick ${t.index}`,facts:[['produce_stack',t.produce??'无'],['consume_stack',t.consume??'无'],['活跃阶段',[t.produce!=null?`Stack ${t.produce} · QK / Softmax`:'',t.consume!=null?`Stack ${t.consume} · PV / Context Accumulate`:''].filter(Boolean).join('；')]]};}
      if(f.kind==='output')return {title:'Attention Output',facts:[['开始条件',`全部 ${task.stacks.length} Stacks 完成累加`],['Normalize','ctx = BF16(o / l_sum)'],['Store',`out2d rows ${task.output.rowStart}–${task.output.rowEnd}`],['Shape',task.output.shape.join(' × ')]]};
      if(f.kind==='slot'){
        const transfer=transfers.find(t=>t.buffer===f.buffer);
        return {title:`${f.buffer||'Transfer buffers'} · Slot ${slot}`,facts:[['Slot ID',slot],['前次 Stack / 后次 Stack',f.from!=null?`S${f.from} → S${f.to}`:reuse.map(i=>`S${i}`).join(' → ')],['Buffer',f.buffer||transfers.map(t=>t.buffer).join(' / ')],['Producer / Consumer',transfer?`${stages[transfer.producer]} / ${stages[transfer.consumer]}`:'按各 Buffer 分别映射'],['READY Event',transfer?.event||'按 Buffer 区分'],['复用端点',transfer&&f.from!=null?`S${f.from} ${stages[transfer.consumer]} → S${f.to} ${stages[transfer.producer]}`:'前次消费 → 后次写入'],['produce_row / consume_row',`${task.transferBase+slot*model.rowTile} / ${task.transferBase+slot*model.rowTile}`]]};
      }
      const pairs=f.kind==='event'?events[f.event]:null,stage=f.stage,ready=f.kind==='event'?syncFor(f):null;
      const producerTick=pairs?task.operations.find(item=>item.stack===s.index&&item.stage===pairs[0])?.tick:null;
      const consumerTick=pairs?task.operations.find(item=>item.stack===s.index&&item.stage===pairs[1])?.tick:null;
      const consumerName=ready?.consumer.boundary?`${ready.consumer.boundary}（${ready.tensor} 加载边界）`:pairs?stages[pairs[1]]:null;
      const involved=transfers.filter(t=>f.kind==='event'?t.event===f.event:f.buffer?t.buffer===f.buffer:stage?(t.producer===stage||t.consumer===stage):true);
      return {title:f.kind==='event'?`Stack ${s.index} · ${f.event}`:f.kind==='stage'?`Stack ${s.index} · ${stages[stage]}`:`Stack ${s.index}`,facts:[
        ['stack index',s.index],['逻辑 Pages',`P${s.pages[0]}–P${s.pages.at(-1)}`],['valid tokens',`${s.tokens} / ${task.stackTokens}`],['Slot ID',s.slot],['阶段',stage?stages[stage]:pairs?`${stages[pairs[0]]} → ${consumerName}`:'完整 Stack 路径'],['Buffer',involved.map(t=>t.buffer).join(' / ')],['Producer / Consumer',involved.map(t=>`${t.buffer}: ${stages[t.producer]} / ${ready?.consumer.boundary||stages[t.consumer]}`).join('；')],['READY Event',involved.map(t=>t.event).join(' / ')],['前后复用 Stack',`S${s.index-model.transferSlots>=0?s.index-model.transferSlots:'—'} ← S${s.index} → S${s.index+model.transferSlots<task.stacks.length?s.index+model.transferSlots:'—'}`],
        [pairs?'SET tick / WAIT tick':'produce tick / consume tick',pairs?`${producerTick} / ${consumerTick} · 静态逻辑顺序`:`${s.produceTick} / ${s.consumeTick}`],['produce_row / consume_row',`${s.produceRow} / ${s.consumeRow}`],
        ...(pairs?[['生产者 / 消费者',`${stages[pairs[0]]} → ${consumerName}`],['Transfer buffer',buffers[pairs[0]].write]]:stage?[['写入',buffers[stage].write||'—'],['读取',buffers[stage].read||'—']]:[['QK → Softmax','写入 / 读取 score_transfer'],['Softmax → PV','写入 / 读取 probability_transfer'],['PV → Accumulate','写入 / 读取 pv_transfer']]),
        ['row 区间',`${s.produceRow}–${s.produceRow+model.rowTile-1}（AIV 按 lane_row 切分）`]]};
    }
    function inspect(){
      if(!inspector)return;
      inspector.hidden=!inspectorOpen||!focus;if(inspector.hidden)return;
      const f=effective(focus),d=describe(focus),sync=f.kind==='event'?syncFor(f):null,operation=f.kind==='stage'?task.operations.find(op=>op.stack===f.index&&op.stage===f.stage):null;
      window.PtoPhotoNarrative.text(host.querySelector('[data-intra-title]'),d.title);
      window.PtoPhotoNarrative.text(host.querySelector('[data-intra-context]'),`Task ${task.id}${f.index!=null?' · Stack '+f.index:''}${operation?' · Tick '+operation.tick:''}`);
      const section=(title,rows)=>rows.length?`<section><h3>${title}</h3><dl>${rows.map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join('')}</dl></section>`:'';
      const jump=(kind,index,stage,label)=>`<button class="btn btn-ghost btn-sm viz2-intra-relation" data-intra-jump-task="${task.id}" data-intra-jump-kind="${kind}" data-intra-jump-index="${index}" ${stage?`data-intra-jump-stage="${stage}"`:''}>${esc(label)} ↗</button>`;
      let upstream=[],downstream=[],resource=[];
      if(f.kind==='stage'){
        upstream=transfers.filter(t=>t.consumer===f.stage).map(t=>jump('stage',f.index,t.producer,stages[t.producer]+' · '+t.buffer));
        downstream=transfers.filter(t=>t.producer===f.stage).map(t=>jump('stage',f.index,t.consumer,stages[t.consumer]+' · '+t.buffer));
        resource=[`Slot ${task.stacks[f.index].slot} · ${[buffers[f.stage].read,buffers[f.stage].write].filter(Boolean).join(' / ')}`];
      }else if(f.kind==='event'){
        const pair=events[f.event];upstream=[jump('stage',f.index,pair[0],stages[pair[0]]+' · SET')];downstream=[jump('stage',f.index,pair[1],stages[pair[1]]+' · WAIT')];resource=[f.buffer+' · 静态同步关系，WAIT 不代表实测 Stall'];
      }else if(f.kind==='slot'){
        upstream=f.from!=null?[jump('stack',f.from,null,`Stack ${f.from} · 前次使用`)]:[];downstream=f.to!=null?[jump('stack',f.to,null,`Stack ${f.to} · 后次使用`)]:[];resource=[`Slot ${f.index} · ${f.buffer||'Transfer buffers'} · 复用不是同一数据值的传递`];
      }else if(f.kind==='stack'){
        downstream=['qk','softmax','pv','accumulate'].map(stage=>jump('stage',f.index,stage,stages[stage]));resource=[`Slot ${task.stacks[f.index].slot}`];
      }else if(f.kind==='output')upstream=[jump('stage',task.stacks.at(-1).index,'accumulate','最后一个 Stack · Accumulate')];
      const relation=(title,items)=>items.length?`<div class="viz2-intra-relation-group"><b>${title}</b>${items.join('')}</div>`:'';
      const detail=sync?global.PTO_COMPUTATION_PRESENTATION.synchronization(sync,global.PTO_PYPA_SEMANTIC_GRAPH):null,edge=sync?eventEdge(f):null,waitRanges=edge?.consumerSource?.ranges||sync?.consumer.source.ranges||[];
      const positionEvidence=f.kind==='tick'?task.ticks[f.index]?.evidence:operation?.evidence.logicalPosition;
      if(global.PtoPhotoNarrative.enabled){
        const batch=f.kind==='slot'?null:task.stacks[f.index];
        const actions={qk:'前段处理：按顾客选择的主题匹配本批照片。',softmax:'前段处理：筛选入选照片并去重，整理打印清单。',pv:'后段加工：按清单准备照片并完成打印。',accumulate:'后段加工：待全部批次打印完成后，按工单整理并装帧。'};
        const description=f.kind==='stage'?actions[f.stage]:f.kind==='event'?'上道工序发出交接就绪，下道工序等待对应资料后继续。':f.kind==='slot'?`周转台位 ${f.index} 循环服务处理批次 ${task.slots[f.index].stacks.map(i=>i+1).join('、')}；复用表示共用位置，不表示传递同一批照片。`:f.kind==='output'?'全部处理批次完成后，装帧并保存成品。':'沿本批次路径查看前段处理、后段加工与交接关系。';
        window.PtoPhotoNarrative.text(host.querySelector('[data-intra-title]'),f.kind==='slot'?`周转台位 ${f.index}`:f.kind==='tick'?`流水轮次 ${f.index}`:f.kind==='output'?'装帧':`第 ${f.index+1} 批${f.stage?' · '+stages[f.stage]:''}`);
        window.PtoPhotoNarrative.html(host.querySelector('[data-intra-detail]'),`<section><h3>当前工单</h3><p>${global.PtoPhotoNarrative.taskSummary(task)}</p></section><section><h3>处理内容</h3><p>${description}</p>${batch?`<p>第 ${batch.index+1} 批 · ${batch.tokens} / ${task.stackTokens} 张照片 · ${batch.pages.length} 个标准照片包 · 周转台位 ${batch.slot}</p>`:''}</section><section><h3>工序交接</h3>${relation('上道工序',upstream)}${relation('下道工序',downstream)}</section><section><h3>逻辑编排</h3><p>前段先处理并交接资料，后段按逻辑轮次接续。轮次横向间距不表示实际时间；就绪等待不表示实际发生停顿；不据此推断提速或真实并发。</p></section>`);
        return;
      }
      window.PtoPhotoNarrative.html(host.querySelector('[data-intra-detail]'),section('Inputs',operation?.read?[['Buffer',operation.read]]:f.kind==='event'?[['Producer',stages[events[f.event][0]]]]:[])+
        section('Computation',[['对象',f.kind==='stage'?stages[f.stage]:d.title]])+
        section('Outputs',operation?.write?[['Buffer',operation.write]]:f.kind==='event'?[['Consumer',stages[events[f.event][1]]]]:[])+
        `<section><h3>Dependencies</h3>${relation('Upstream',upstream)}${relation('Downstream',downstream)}${resource.length?`<p>Resource / Synchronization · ${esc(resource.join('；'))}</p>`:''}</section>`+
        (positionEvidence?`<p>Logical Tick · Model-derived：${esc(positionEvidence.rule)}；输入 ${esc(JSON.stringify(positionEvidence.inputs))}。同 Tick 不代表真实并发或耗时。</p>`:'')+
        (f.kind==='slot'?`<p>Slot · Model-derived：stack.index % TRANSFER_SLOTS；复用不表示同一数据值。</p>`:'')+
        (operation?`<section><h3>Source Evidence</h3>${operation.evidence.operation.source.ranges.map(range=>`<button class="btn btn-ghost btn-sm" data-intra-operation-source="${range[0]}" data-intra-operation-end="${range[1]}">paged_attention_pypto.py · L${range.join('–')} ↗</button>`).join('')}<p>计算关系：Source-confirmed（页面内源码快照）；操作位置：Model-derived。</p></section>`:'')+
        (detail?`<section><h3>Source Evidence</h3><dl>${detail.rows.map(row=>`<dt>${esc(row.label)}</dt><dd>${esc(row.value)}</dd>`).join('')}</dl><div class="viz2-ready-source-links">${[...sync.producer.source.ranges.map((range,index)=>({end:'producer',index,range})),...waitRanges.map((range,index)=>({end:'consumer',index,range}))].map(item=>`<button class="btn btn-ghost btn-sm" data-intra-sync-source="${item.end}" data-intra-sync-range="${item.index}">${item.end==='producer'?'SET':'WAIT'} · :${item.range.start}–${item.range.end}</button>`).join('')}</div><button class="btn btn-sm" data-intra-graph-ready>在 Graph 中查看</button><pre data-intra-source-preview hidden></pre></section>`:'')+
        '<p>静态逻辑映射；Tick 不表示实测时长，READY WAIT 不表示已发生 Stall。</p>');
    }

    function update(id,nextFocus,nextCamera,options={}){
      const entering=!!options.active&&!active;active=!!options.active;
      if(!active){saveScroll();stop();host.classList.remove('viz2-intra-masked');return;}
      const source=model.tasks[id];if(!source)return;
      sourceTask=source;
      const next=scenario.task('pipeline',id)||source,rebuild=task!==next;
      isCustom=!!scenario.get()?.isCustom;
      maxStacks=scenario.limits().stacks;stackInput.max=stackRange.max=maxStacks;
      if(rebuild){if(!presentation||task?.id!==id){nextFocus=null;nextCamera=null;scales.delete(id);inspectorOpen=false;}window.PtoPhotoNarrative.text(toolbar.querySelector('[data-page-error]'),'');stackInput.removeAttribute('aria-invalid');}
      const changed=task?.id!==id,focusChanged=JSON.stringify(focus)!==JSON.stringify(nextFocus);task=next;
      focus=validFocus(nextFocus)?nextFocus:null;inspectorOpen=!!options.inspectorOpen&&!!focus;identity();
      automaticFit=!!presentation&&!nextCamera;
      const needsInitialFit=!presentation&&!nextCamera&&!scales.has(id);
      pan={x:nextCamera?.x||0,y:nextCamera?.y||0};scale=Math.max(.05,Math.min(3,nextCamera?.scale||scales.get(id)||1));scales.set(id,scale);
      if(rebuild||block!==task.block){saveScroll();stop();buildBlock();}
      else if(entering){stop();position=target=tasks.findIndex(t=>t.id===id);velocity=0;}
      else if(changed){const index=tasks.findIndex(t=>t.id===id);if(Math.abs(position-index)>.001)snap(index);else target=index;}
      inspect();paint();paintFocus();if(entering||changed||focusChanged)revealFocus();
      if(needsInitialFit){const scroller=currentCard()?.querySelector('.viz2-intra-tick-scroll'),size=diagramSize(task);if(scroller?.clientWidth&&scroller?.clientHeight){const fit=fitScale(scroller,size);onCamera(task.id,{scale:fit,x:0,y:0});}}
    }
    function zoom(factor,point){
      paging.reset();const next=Math.max(.05,Math.min(3,scale*factor)),r=currentCard()?.querySelector('.viz2-intra-tick-scroll')?.getBoundingClientRect();const dx=point&&r?point.x-r.left-r.width/2:0,dy=point&&r?point.y-r.top-r.height/2:0;
      onCamera(task.id,{scale:next,x:pan.x+dx*(1/scale-1/next),y:pan.y+dy*(1/scale-1/next)});
    }
    function manualPage(index,nextFocus=null){pendingFocus=nextFocus;snap(index);}
    listen(host,'click',event=>{
      if(suppress){suppress=false;event.preventDefault();return;}tooltip.hidden=true;
      if(event.target.closest('[data-intra-graph-ready]')){stop();onNavigate?.();return;}
      const operationSource=event.target.closest('[data-intra-operation-source]');if(operationSource){(onOpenSource??global.PtoComputeIDE?.openSource)?.({file:'paged_attention_pypto.py',start:Number(operationSource.dataset.intraOperationSource),end:Number(operationSource.dataset.intraOperationEnd)});return;}
      const jump=event.target.closest('[data-intra-jump-task]');if(jump){if(Number(jump.dataset.intraJumpTask)!==task.id)return;onFocus({kind:jump.dataset.intraJumpKind,index:Number(jump.dataset.intraJumpIndex),...(jump.dataset.intraJumpStage?{stage:jump.dataset.intraJumpStage}:{})});return;}
      const sourceButton=event.target.closest('[data-intra-sync-source]');
      if(sourceButton){const sync=syncFor(focus),edge=eventEdge(focus);if(!sync)return;const ranges=sourceButton.dataset.intraSyncSource==='producer'?sync.producer.source.ranges:(edge?.consumerSource?.ranges||sync.consumer.source.ranges),line=ranges[Number(sourceButton.dataset.intraSyncRange)]?.start,preview=host.querySelector('[data-intra-source-preview]');if(line&&preview){preview.hidden=false;window.PtoPhotoNarrative.text(preview,global.PTO_PAGED_ATTENTION_PYPTO_SOURCE.split('\n').slice(Math.max(0,line-3),line+12).map((text,i)=>`${Math.max(1,line-2)+i}  ${text}`).join('\n'));}return;}
      const taskButton=event.target.closest('[data-intra-task-go]');if(taskButton){manualPage(tasks.findIndex(t=>t.id===Number(taskButton.dataset.intraTaskGo)));return;}
      if(event.target.closest('[data-intra-back]')){stop();onExit?.();return;}
      if(event.target.closest('[data-intra-close]')){paging.reset();onFocus(null);onInspector(false);return;}
      if(event.target.closest('[data-intra-fit]')){paging.reset();const scroller=currentCard()?.querySelector('.viz2-intra-tick-scroll'),size=viewportSize(task),fit=scroller?fitScale(scroller,size):1;onCamera(task.id,{scale:fit,x:0,y:0});if(scroller){scroller.scrollLeft=0;scroller.scrollTop=0;}return;}
      const z=event.target.closest('[data-intra-zoom]');if(z){zoom(Number(z.dataset.intraZoom));return;}
      if(event.target.closest('[data-intra-prev]')){manualPage(target-1);return;}
      if(event.target.closest('[data-intra-next]')){manualPage(target+1);return;}
      const card=event.target.closest('[data-intra-task]'),el=event.target.closest('[data-intra-kind]'),nextFocus=objectFocus(el);
      if(card&&Number(card.dataset.intraTask)!==task.id){manualPage(tasks.findIndex(t=>t.id===Number(card.dataset.intraTask)),nextFocus);return;}
      if(nextFocus){paging.reset();onFocus(nextFocus);}
      else if(card&&Number(card.dataset.intraTask)===task.id&&focus){onFocus(null);onInspector(false);}
    });
    listen(viewport,'wheel',event=>{
      if(event.target.closest('button,input,select,.explorer-dropdown-trigger'))return;
      event.preventDefault();tooltip.hidden=true;const unit=event.deltaMode===1?16:event.deltaMode===2?viewport.clientHeight:1;
      zoom(Math.exp(Math.max(-.25,Math.min(.25,-event.deltaY*unit*.002))),{x:event.clientX,y:event.clientY});
    },{passive:false});
    listen(viewport,'pointerdown',event=>{
      if(event.button!==0||event.target.closest('button,input,select'))return;
      stop();suppress=false;drag={id:event.pointerId,x:event.clientX,y:event.clientY,moved:false,canvas:true,pan:{...pan}};viewport.setPointerCapture(event.pointerId);
    });
    listen(viewport,'pointermove',event=>{
      if(!drag)return;if(!(event.buttons&1)){endDrag(true);return;}
      const dx=event.clientX-drag.x,dy=event.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>4)drag.moved=true;
      if(drag.moved){event.preventDefault();tooltip.hidden=true;onCamera(task.id,{scale,x:drag.pan.x-dx/scale,y:drag.pan.y-dy/scale});}
    });
    function endDrag(cancelled=false){
      if(!drag)return;const current=drag;drag=null;suppress=current.moved;
      if(viewport.hasPointerCapture(current.id))viewport.releasePointerCapture(current.id);
      if(current.scroller||current.canvas)return;
      if(current.moved){if(performance.now()-current.lastTime>100)velocity=0;const offset=position-current.index,direction=Math.abs(offset)>.18?Math.sign(offset):Math.abs(velocity)>.55?Math.sign(velocity):0;snap(current.index+(cancelled?0:direction));}
      else if(Math.abs(position-current.index)>.001)snap(current.index);
    }
    listen(global,'pointerup',()=>endDrag());listen(viewport,'pointercancel',()=>endDrag(true));listen(viewport,'lostpointercapture',()=>endDrag(true));
    listen(host,'pointermove',event=>{
      if(!active||drag||frame){tooltip.hidden=true;return;}
      const el=event.target.closest('[data-intra-kind]'),card=el?.closest('[data-intra-task]');
      if(!el||Number(card?.dataset.intraTask)!==task.id){tooltip.hidden=true;paintFocus();return;}
      const f=objectFocus(el);
      paintFocus(['stage','stack'].includes(f.kind)?{kind:'stack',index:f.index}:f);
      tooltip.hidden=true;
    });
    listen(host,'pointerleave',()=>{tooltip.hidden=true;paintFocus();});
    listen(viewport,'focusin',event=>{const card=event.target.closest('[data-intra-task]');if(card&&Number(card.dataset.intraTask)!==task.id)manualPage(tasks.findIndex(t=>t.id===Number(card.dataset.intraTask)));});
    listen(host,'keydown',event=>{
      if(!active||event.target.closest('input,select,button,.explorer-dropdown'))return;const button=event.target.closest('[data-intra-task-go]');
      if(button&&['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const index=navButtons.indexOf(button),next=event.key==='Home'?0:event.key==='End'?tasks.length-1:clamp(index+(event.key==='ArrowLeft'?-1:1));navButtons[next].focus();manualPage(next);return;}
      if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();manualPage(target+(event.key==='ArrowLeft'?-1:1));}
      else if(event.key==='Escape'){tooltip.hidden=true;paging.reset();onInspector(false);}
      else if(['+','=','-','0'].includes(event.key)){event.preventDefault();if(event.key==='0')host.querySelector('[data-intra-fit]').click();else zoom(event.key==='-'?.8:1.25);}
      else if(['Enter',' '].includes(event.key)&&event.target.closest('[data-intra-kind]')){event.preventDefault();event.target.closest('[data-intra-kind]').dispatchEvent(new MouseEvent('click',{bubbles:true}));}
    });
    listen(viewport,'scroll',()=>{tooltip.hidden=true;saveScroll();},{capture:true});
    const resizeObserver=new ResizeObserver(()=>{if(active){if(frame||drag){stop();position=target=tasks.findIndex(t=>t.id===task.id);}paint();}});resizeObserver.observe(viewport);
    listen(media,'change',()=>{paging.reset();if(frame){const index=target;stop();position=target=index;paint();commit();}});
    const screenRect=element=>{const r=element.getBoundingClientRect();return {x:r.left,y:r.top,width:r.width,height:r.height};};
    function getTransitionSnapshot(){
      if(!active||!task)return null;const card=currentCard(),scene=currentScene();if(!card||!scene)return null;
      const scroller=card.querySelector('.viz2-intra-tick-scroll'),clip=screenRect(scroller);
      // Offscreen rows collapse onto the nearest visible edge during outer morph.
      const visibleRect=element=>{const r=screenRect(element),x=Math.max(clip.x,Math.min(r.x,clip.x+clip.width-1)),y=Math.max(clip.y,Math.min(r.y,clip.y+clip.height-1));return {x,y,width:Math.max(1,Math.min(r.width,clip.x+clip.width-x)),height:Math.max(1,Math.min(r.height,clip.y+clip.height-y))};};
      return {task:task.id,model:task,clip,frame:screenRect(card),title:screenRect(host.querySelector('[data-compute-identity]')),output:visibleRect(scene.querySelector('[data-intra-output]')),stacks:task.stacks.map(s=>({index:s.index,rect:visibleRect(scene.querySelector(`[data-intra-operation="${s.index}:qk"]`))}))};
    }
    return {update,model,validFocus,getOperationDescriptor(taskId,stack,stage){return (task?.id===taskId?task:model.tasks[taskId])?.operations.find(op=>op.stack===stack&&op.stage===stage)||null;},getTransitionSnapshot,hideTooltip(){tooltip.hidden=true;},setTransitionMasked(value){host.classList.toggle('viz2-intra-masked',value);},
      destroy(){active=false;saveScroll();stop();resizeObserver.disconnect();listeners.abort();host.classList.remove('viz2-intra-masked');host.replaceChildren();}};
  }
  global.PtoVisualization2PipelineIntra=Object.freeze({createModel,mount});
})(window);
