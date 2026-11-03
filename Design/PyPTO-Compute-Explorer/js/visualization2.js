/* Independent visualization workspace. No legacy renderer or selection state is mounted here. */
(function () {
  'use strict';
  const tabs = ['assignment', 'pipeline', 'graph'];
  let graphIntraController, graphIntraModel, graphZoomJob, graphOuterSnapshot;
  let tabTransition, semanticZoom, anatomyController, anatomyModel, pipelineIntraController, pipelineIntraModel, pipelineZoom;
  function createStore() {
    let state = { open: false, activeTab: 'assignment', selection: null,
      views: Object.fromEntries(tabs.map(id => [id, { subview: null, camera: null, ...(id === 'assignment' ? { phase: 'attention', phase0Camera: null } : {}) }])) };
    const listeners = new Set();
    const snapshot = () => structuredClone(state);
    return {
      getState: snapshot,
      dispatch(action) {
        // A wheel gesture emits several events after crossing the semantic-zoom
        // threshold. Those follow-up camera updates must not cancel the morph.
        if(action.type==='view'&&action.id==='assignment'&&semanticZoom?.animating)return;
        if(['tab','close','select','pipeline-intra-task','pipeline-intra-focus'].includes(action.type)||(action.type==='view'&&pipelineZoom?.animating))pipelineZoom?.cancel();
        if(['tab','close','select','graph-intra-focus','graph-intra-task'].includes(action.type)||(action.type==='view'&&graphZoomJob?.running))cancelGraphZoom();
        if(action.type==='tab')semanticZoom?.cancel();
        if(action.type==='tab' && state.activeTab==='assignment' && state.views.assignment.phase==='attention') assignmentMorph?.settle(action.id==='pipeline'?1:undefined);
        if(action.type==='tab' && state.activeTab==='pipeline' && action.id==='assignment') assignmentMorph?.settle(1);
        const transition = action.type === 'tab' ? tabTransition?.prepare(state,action.id) : null;
        if (['tab','close','phase','select','anatomy-focus','anatomy-task'].includes(action.type)) semanticZoom?.cancel();
        if (['close','phase','select'].includes(action.type) || (action.type === 'view' && tabTransition?.animating)) tabTransition?.cancel();
        if (action.type === 'open') state = { ...state, open: true, activeTab: 'assignment',views:{...state.views,assignment:{...state.views.assignment,subview:null}} };
        else if (action.type === 'close') state = { ...state, open: false };
        else if (action.type === 'tab' && tabs.includes(action.id)) {
          // Scope follows the current workspace context, just like Task identity.
          // Each view keeps its own camera and maps shared subobjects on entry.
          const target=state.views[action.id];
          const inner=state.views[state.activeTab].subview==='block'&&state.selection?.kind==='task';
          const task=state.selection?.task;
          const focus=inner?window.PtoVisualization2IntraMorph.mapFocus(state.activeTab,action.id,state.selection.subobject,anatomyModel.tasks[task]):state.selection?.subobject;
          if(action.id==='graph'&&inner&&target.subview!=='block')
            graphOuterSnapshot={camera:target.camera,nodeId:target.nodeId,inspectorOpen:target.inspectorOpen};
          const context=inner?(action.id==='assignment'?{phase:'attention',anatomyTask:task,
            anatomyFocus:{...target.anatomyFocus,[task]:focus},anatomyInspectorOpen:target.anatomyInspectorOpen??!!focus}:
            action.id==='pipeline'?{intraTask:task,intraFocus:{...target.intraFocus,[task]:focus},intraInspectorOpen:target.intraInspectorOpen??!!focus}:
            {intraTask:task,intraFocus:focus,intraInspectorOpen:target.intraInspectorOpen??!!focus}):{};
          state={...state,activeTab:action.id,selection:inner?{...state.selection,subobject:focus}:state.selection,views:{...state.views,[action.id]:{...target,...context,subview:inner?'block':null}}};
        }
        else if (action.type === 'phase' && ['phase0', 'attention'].includes(action.id)) state = { ...state, views: { ...state.views, assignment: { ...state.views.assignment, phase: action.id } } };
        else if (action.type === 'anatomy-enter' || action.type === 'anatomy-task') {
          const task=anatomyModel.tasks[action.id];if(!task)return;
          const assignment=state.views.assignment,shared=state.selection?.task===task.id?state.selection.subobject:null;
          const incoming=['stage','event'].includes(shared?.kind)?{kind:'stack',index:shared.index}:Number.isInteger(shared?.stackIndex)?{kind:'stack',index:shared.stackIndex}:shared;
          const focus=['stack','page','slot','output'].includes(incoming?.kind)?incoming:assignment.anatomyFocus?.[task.id]||null;
          state={...state,selection:{kind:'task',task:task.id,block:task.block,batch:task.batch,kvHead:task.kvHead,subobject:focus},views:{...state.views,assignment:{...assignment,subview:'block',phase:'attention',anatomyTask:task.id,anatomyFocus:{...assignment.anatomyFocus,[task.id]:focus},anatomyInspectorOpen:!!focus}}};
        }
        else if (action.type === 'anatomy-exit') state={...state,views:{...state.views,assignment:{...state.views.assignment,subview:null}}};
        // Subobjects are scoped by the root Task identity; outer views can retain
        // this context while their renderers continue to select the owning Task.
        else if (action.type === 'anatomy-focus') {
          const assignment=state.views.assignment;
          state={...state,selection:{...state.selection,subobject:action.focus},views:{...state.views,assignment:{...assignment,anatomyFocus:{...assignment.anatomyFocus,[assignment.anatomyTask]:action.focus},anatomyInspectorOpen:true}}};
        }
        else if (action.type === 'pipeline-intra-enter' || action.type === 'pipeline-intra-task') {
          const task=pipelineIntraModel.tasks[action.id];if(!task)return;
          const view=state.views.pipeline,shared=state.selection?.task===task.id?state.selection.subobject:null,incoming=Number.isInteger(shared?.stackIndex)&&!['stage','event','page','stack'].includes(shared?.kind)?{kind:'stack',index:shared.stackIndex}:shared;
          const focus=['stack','page','slot','event','stage','tick','output'].includes(incoming?.kind)?incoming:view.intraFocus?.[task.id]||null;
          state={...state,selection:{kind:'task',task:task.id,block:task.block,batch:task.batch,kvHead:task.kvHead,subobject:focus},views:{...state.views,pipeline:{...view,subview:'block',intraTask:task.id,intraFocus:{...view.intraFocus,[task.id]:focus},intraInspectorOpen:!!focus}}};
        }
        else if(action.type==='pipeline-intra-exit')state={...state,views:{...state.views,pipeline:{...state.views.pipeline,subview:null}}};
        else if(action.type==='pipeline-intra-focus'){
          const view=state.views.pipeline;
          state={...state,selection:{...state.selection,subobject:action.focus},views:{...state.views,pipeline:{...view,intraFocus:{...view.intraFocus,[view.intraTask]:action.focus},intraInspectorOpen:true}}};
        }
        else if(action.type==='graph-intra-enter'||action.type==='graph-intra-task'){
          const task=graphIntraModel.tasks[action.id];if(!task)return;
          const v=state.views.graph,focus=state.selection?.task===task.id?state.selection.subobject:null;
          state={...state,selection:{kind:'task',task:task.id,block:task.block,batch:task.batch,kvHead:task.kvHead,subobject:focus},views:{...state.views,graph:{...v,subview:'block',intraTask:task.id,intraFocus:focus,intraInspectorOpen:!!focus}}};
        }
        else if(action.type==='graph-intra-exit')state={...state,views:{...state.views,graph:{...state.views.graph,...graphOuterSnapshot,subview:null}}};
        else if(action.type==='graph-intra-focus')state={...state,selection:{...state.selection,subobject:action.focus},views:{...state.views,graph:{...state.views.graph,intraFocus:action.focus,intraInspectorOpen:true}}};
        else if (action.type === 'select') state = { ...state, selection: action.selection };
        else if (action.type === 'view' && tabs.includes(action.id)) state = { ...state,
          views: { ...state.views, [action.id]: { ...state.views[action.id], ...action.patch } } };
        else return;
        listeners.forEach(fn => fn(snapshot(), action));
        if (transition) tabTransition.play(transition);
      },
      subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    };
  }
  const store = createStore();
  // The same embedded Python source shown by the Studio editor owns this configuration.
  const source = globalThis.PTO_PAGED_ATTENTION_PYPTO_SOURCE;
  const sourceNumber = name => {
    const match = source?.match(new RegExp(`^${name} = (\\d+)`, 'm'));
    if (!match) throw new Error(`Missing source configuration: ${name}`);
    return Number(match[1]);
  };
  const constants = Object.freeze({ blocks: sourceNumber('ATTN_SPMD_BLOCKS'), kvHeads: sourceNumber('NUM_KV_HEADS'),
    qPerKv: sourceNumber('NUM_HEADS') / sourceNumber('NUM_KV_HEADS'), ropeCores: sourceNumber('ROPE_CORES'), batch: sourceNumber('BATCH'),
    lanesPerBlock: Number(source.match(/for aiv_id in pl\.split_aiv\((\d+)/)[1]) });
  const seqLens = [1025, 512, 2048, 128, 1536, 3584, 3072, 256, 769, 1024, 2049, 640, 2560, 129, 1792, 3328];
  const taskCount = constants.batch * constants.kvHeads;
  const taskInfo = id => ({ id, batch: Math.floor(id / constants.kvHeads), kv: id % constants.kvHeads, block: id % constants.blocks,
    qStart: id % constants.kvHeads * constants.qPerKv, qEnd: (id % constants.kvHeads + 1) * constants.qPerKv - 1 });
  const tasksForBlock = block => Array.from({length: taskCount}, (_, id) => id).filter(id => id % constants.blocks === block);
  const taskDescription = id => { const t = taskInfo(id); return `Task ${id} · Batch ${t.batch} · KV ${t.kv} · Q${t.qStart}–Q${t.qEnd} · Block ${t.block} · Seq Len ${seqLens[t.batch]} tokens`; };
  let root, viewport, scene, tooltip, trigger, hoverTask = null, drag = null, suppressClick = false;
  let assignmentMorph, previousInert = [], phase0Controller, phaseAnimation, pipelineController, tabAnimation, graphController;
  const phase0Model = window.PtoVisualization2Phase0.createModel({ ...constants, activeBatch: seqLens.length, itemsPerCore: Math.ceil(constants.kvHeads * constants.batch / constants.ropeCores) });
  function mount() {
    root = document.createElement('section');
    root.id = 'visualization2'; root.className = 'pto-ide-frame__pane viz2'; root.hidden = true;
    root.setAttribute('aria-label', 'Compute Explorer');
    root.innerHTML = `<header class="pto-ide-frame__pane-header viz2-header">
      <button type="button" class="btn btn-ghost btn-sm" data-viz2-back>← 返回代码</button>
      <span class="pto-ide-frame__pane-title">Compute Explorer</span>
      <span class="pto-ide-frame__pane-meta">paged_attention_pypto.py</span>
    </header>
    <div class="viz2-content">
      <div class="pto-ide-frame__tabstrip viz2-tabs"><div class="tab-control" role="tablist" aria-label="可视化视图">${tabs.map(id => `<button type="button" class="tab-control-item" role="tab" id="viz2-tab-${id}" aria-controls="viz2-${id}" data-viz2-tab="${id}"><span class="viz2-tab-icon viz2-tab-icon--${id}" aria-hidden="true"></span>${id[0].toUpperCase()+id.slice(1)}</button>`).join('')}</div></div>
      ${tabs.map(id => `<section class="viz2-canvas" id="viz2-${id}" role="tabpanel" aria-labelledby="viz2-tab-${id}" tabindex="0" hidden></section>`).join('')}
    </div>`;
    document.querySelector('#ideWorkarea').append(root);
    const host = root.querySelector('#viz2-assignment');
    host.innerHTML = `<div class="viz2-phase-toolbar viz2-assignment-hud"><div data-viz2-phases class="segmented-control segmented-control-muted" role="group" aria-label="Assignment 阶段"><button class="btn btn-sm" type="button" data-viz2-phase="phase0" aria-pressed="false" aria-controls="viz2-phase0">Phase0</button><button class="btn btn-sm is-selected" type="button" data-viz2-phase="attention" aria-pressed="true" aria-controls="viz2-attention">Attention+Output</button></div><div class="segmented-control segmented-control-muted" role="group" aria-label="Assignment 范围"><button class="btn btn-sm is-selected" data-viz2-scope="inter" aria-label="Block 间" title="Block 间" aria-pressed="true"><span class="viz2-block-scope-icon viz2-block-scope-icon--inter" aria-hidden="true"></span></button><button class="btn btn-sm" data-viz2-scope="intra" aria-label="Block 内" title="Block 内" aria-pressed="false" disabled><span class="viz2-block-scope-icon" aria-hidden="true"></span></button></div><nav aria-label="当前对象" data-viz2-breadcrumb></nav><button class="btn btn-ghost btn-sm" data-viz2-matrix-back hidden>← 返回矩阵</button></div>
      <div class="viz2-phase-surface" id="viz2-anatomy" aria-label="Block 内 Task 数据结构" hidden></div>
      <div class="viz2-phase-surface" id="viz2-phase0" aria-label="Phase0 工作分配" hidden></div>
      <div class="viz2-phase-surface" id="viz2-attention" aria-label="Attention+Output 工作分配"></div>`;
    assignmentMorph = window.PtoVisualization2AssignmentMorph.mount(host.querySelector('#viz2-attention'), {config: constants, seqLens, onEnter: enterAnatomy});
    viewport = assignmentMorph.viewport; scene = assignmentMorph.scene; tooltip = assignmentMorph.tooltip;
    anatomyModel=window.PtoVisualization2Anatomy.createModel({seqLens,blocks:constants.blocks,numHeads:sourceNumber('NUM_HEADS'),kvHeads:constants.kvHeads,
      headDim:sourceNumber('HEAD_DIM'),rowTile:sourceNumber('ROW_TILE'),pageTokens:sourceNumber('BLOCK_SIZE'),pagesPerStack:sourceNumber('STACK_PAGES'),transferSlots:sourceNumber('PRE_LAUNCH')+1});
    graphIntraModel=window.PtoVisualization2GraphIntra.createModel({anatomy:anatomyModel,source,preLaunch:sourceNumber('PRE_LAUNCH')});
    pipelineIntraModel=window.PtoVisualization2PipelineIntra.createModel({anatomy:anatomyModel,preLaunch:sourceNumber('PRE_LAUNCH')});
    pipelineZoom=window.PtoVisualization2PipelineZoom.create({root,getOuter:()=>pipelineController,getInner:()=>pipelineIntraController});
    semanticZoom=window.PtoVisualization2SemanticZoom.create({root,getMatrix:matrixSnapshot,getAnatomy:()=>anatomyController?.getTransitionSnapshot()});
    phase0Controller = window.PtoVisualization2Phase0.mount(root.querySelector('#viz2-phase0'), {
      model: phase0Model, seqLens,
      onSelect: selection => store.dispatch({type:'select',selection}),
      onCamera: camera => store.dispatch({type:'view',id:'assignment',patch:{phase0Camera:camera}}),
    });
    root.querySelector('#viz2-pipeline').innerHTML=`<div class="viz2-phase-toolbar viz2-assignment-hud"><div class="segmented-control segmented-control-muted" role="group" aria-label="Pipeline 范围"><button class="btn btn-sm is-selected" data-pipeline-scope="inter" aria-label="Block 间" title="Block 间" aria-pressed="true"><span class="viz2-block-scope-icon viz2-block-scope-icon--inter" aria-hidden="true"></span></button><button class="btn btn-sm" data-pipeline-scope="intra" aria-label="Block 内" title="Block 内" aria-pressed="false" disabled><span class="viz2-block-scope-icon" aria-hidden="true"></span></button></div><nav aria-label="Pipeline 当前对象" data-pipeline-breadcrumb>Pipeline</nav><button class="btn btn-ghost btn-sm" data-pipeline-back hidden>← 返回 Block 间</button></div><div class="viz2-phase-surface" id="viz2-pipeline-inter"></div><div class="viz2-phase-surface" id="viz2-pipeline-intra" hidden></div>`;
    root.querySelector('#viz2-graph').innerHTML=`<div class="viz2-phase-toolbar viz2-assignment-hud"><div class="segmented-control segmented-control-muted" role="group" aria-label="Graph 范围"><button class="btn btn-sm is-selected" data-graph-scope="inter" aria-label="Block 间" title="Block 间" aria-pressed="true"><span class="viz2-block-scope-icon viz2-block-scope-icon--inter" aria-hidden="true"></span></button><button class="btn btn-sm" data-graph-scope="intra" aria-label="Block 内" title="Block 内" aria-pressed="false"><span class="viz2-block-scope-icon" aria-hidden="true"></span></button></div><nav data-graph-breadcrumb aria-label="Graph 当前对象">Graph</nav><button class="btn btn-ghost btn-sm" data-graph-back hidden>← 返回 Block 间</button></div><div class="viz2-phase-surface" id="viz2-graph-inter"></div><div class="viz2-phase-surface" id="viz2-graph-intra" hidden></div>`;
    root.addEventListener('click', event => {
      if (event.target.closest('[data-viz2-back]')) { close(); return; }
      const tab = event.target.closest('[data-viz2-tab]');
      if (tab) { store.dispatch({type:'tab', id:tab.dataset.viz2Tab}); return; }
      const gs=event.target.closest('[data-graph-scope]');if(gs){gs.dataset.graphScope==='intra'?enterGraphIntra():exitGraphIntra();return;}
      if(event.target.closest('[data-graph-back]')){exitGraphIntra();return;}
      if(event.target.closest('[data-pipeline-back]')){exitPipelineIntra();return;}
      const pipelineScope=event.target.closest('[data-pipeline-scope]');if(pipelineScope){if(pipelineScope.dataset.pipelineScope==='inter')exitPipelineIntra();else enterPipelineIntra(store.getState().selection?.task);return;}
      if(event.target.closest('[data-viz2-matrix-back]')){exitAnatomy();return;}
      const scope=event.target.closest('[data-viz2-scope]');if(scope){if(scope.dataset.viz2Scope==='inter')exitAnatomy();else enterAnatomy(store.getState().selection?.task);return;}
      const phase = event.target.closest('[data-viz2-phase]');
      if (phase) { store.dispatch({type:'phase',id:phase.dataset.viz2Phase}); return; }
      if (suppressClick) { suppressClick = false; return; }
      const block = event.target.closest('[data-pa-block],[data-am-block-hit]');
      if (block) { assignmentMorph.pause(); const id = Number(block.dataset.paBlock ?? block.dataset.amBlockHit); store.dispatch({type:'select',selection: store.getState().selection?.kind === 'block' && store.getState().selection.block === id ? null : {kind:'block', block:id, task:null}}); }
      const task = event.target.closest('[data-pa-task]');
      if (task) {assignmentMorph.pause(); selectTask(Number(task.dataset.paTask));}
      if(event.target.closest('[data-am-enter]'))enterAnatomy(store.getState().selection?.task);
      const zoom = event.target.closest('[data-viz2-zoom]');
      if (zoom) zoomBy(Number(zoom.dataset.viz2Zoom));
      if (event.target.closest('[data-viz2-fit]')) fit();
    });
    root.querySelector('[role="tablist"]').addEventListener('keydown', event => {
      if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault(); const current = tabs.indexOf(store.getState().activeTab);
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (current + (event.key === 'ArrowRight' ? 1 : 2)) % 3;
      store.dispatch({type:'tab', id:tabs[index]}); root.querySelector(`[data-viz2-tab="${tabs[index]}"]`).focus();
    });
    viewport.addEventListener('pointermove', event => {
      const cell=event.target.closest('[data-pa-task]'); hoverTask=cell?Number(cell.dataset.paTask):null; paintSelection();
      tooltip.hidden=hoverTask===null;
      if(hoverTask!==null){tooltip.textContent=taskDescription(hoverTask); const r=viewport.getBoundingClientRect(); tooltip.style.left=`${Math.min(r.width-240,event.clientX-r.left+12)}px`; tooltip.style.top=`${Math.min(r.height-40,event.clientY-r.top+12)}px`;}
    });
    viewport.addEventListener('pointerleave',()=>{hoverTask=null;tooltip.hidden=true;paintSelection();});
    viewport.addEventListener('keydown', event => {
      const stride=assignmentMorph.progress>.5?constants.blocks:constants.kvHeads;
      const delta={ArrowLeft:-1,ArrowRight:1,ArrowUp:-stride,ArrowDown:stride}[event.key];
      if (delta !== undefined) { event.preventDefault(); const id=Math.max(0,Math.min(taskCount-1,(store.getState().selection?.task ?? 0)+delta)); selectTask(id); root.querySelector(`[data-pa-task="${id}"]`).focus(); }
      else if (['Enter',' '].includes(event.key) && event.target.matches('[data-pa-task],[data-am-block-hit]')) { event.preventDefault(); event.target.click(); }
      else if (event.key === 'Escape') { store.dispatch({type:'select',selection:null}); viewport.focus(); }
    });
    viewport.addEventListener('dblclick', event => { const target=event.target.closest('[data-pa-task]');if(target)enterAnatomy(Number(target.dataset.paTask));else fit(); });
    tabTransition = window.PtoVisualization2Transition.create({root,getPipeline:()=>pipelineController,getGraph:()=>graphController,getIntraTask:id=>anatomyModel?.tasks[id]});
    store.subscribe(render);
  }
  function matrixSnapshot(id) {
    const task=anatomyModel.tasks[id],element=scene.querySelector(`[data-pa-task="${id}"]`);
    const strip=scene.querySelector(`[data-pa-column="${task.batch}"] .kf-pa-map__sequence`);
    const rect=element=>{const r=element.getBoundingClientRect();return {x:r.left,y:r.top,width:r.width,height:r.height};};
    return {task:id,model:task,frame:rect(element.querySelector('.kf-pa-map__cell')),title:rect(element.querySelector('text')),
      qRows:[...element.querySelectorAll('.kf-pa-map__q')].map(rect),segments:[...strip.querySelectorAll('.kf-pa-map__seq-track')].map(rect)};
  }
  function enterAnatomy(id) {
    if(!Number.isInteger(id)||!anatomyModel.tasks[id]||store.getState().views.assignment.subview==='block')return;
    tooltip.hidden=true;hoverTask=null;if(store.getState().selection?.task!==id)selectTask(id);tabTransition?.cancel();
    assignmentMorph.pause();const transition=semanticZoom.prepare('in',id);store.dispatch({type:'anatomy-enter',id});semanticZoom.play(transition);
  }
  function exitAnatomy() {
    const state=store.getState();if(state.views.assignment.subview!=='block')return;
    anatomyController?.hideTooltip();
    const transition=semanticZoom.prepare('out',state.views.assignment.anatomyTask);
    store.dispatch({type:'anatomy-exit'});semanticZoom.play(transition);
  }
  function enterPipelineIntra(id){
    const state=store.getState();if(!Number.isInteger(id)||!pipelineIntraModel.tasks[id]||state.views.pipeline.subview==='block')return;
    pipelineController.hideTooltip();tabTransition?.cancel();
    const transition=pipelineZoom.prepare('in',id);store.dispatch({type:'pipeline-intra-enter',id});pipelineZoom.play(transition);
  }
  function exitPipelineIntra(){
    const state=store.getState();if(state.views.pipeline.subview!=='block')return;
    pipelineIntraController?.hideTooltip();const transition=pipelineZoom.prepare('out',state.views.pipeline.intraTask);
    store.dispatch({type:'pipeline-intra-exit'});pipelineZoom.play(transition);
  }
  function cancelGraphZoom(){if(!graphZoomJob)return;graphZoomJob.animations.forEach(a=>a.cancel());graphZoomJob.overlay.remove();graphZoomJob.hold.remove();graphZoomJob=null;}
  function graphZoomPrepare(entering){
    cancelGraphZoom();if(matchMedia('(prefers-reduced-motion: reduce)').matches)return null;
    const pane=root.querySelector(entering?'#viz2-graph-inter':'#viz2-graph-intra'),r=pane.getBoundingClientRect(),o=root.getBoundingClientRect(),hold=pane.cloneNode(true);
    hold.removeAttribute('id');hold.inert=true;Object.assign(hold.style,{position:'absolute',left:`${r.left-o.left}px`,top:`${r.top-o.top}px`,width:`${r.width}px`,height:`${r.height}px`,zIndex:12,pointerEvents:'none'});root.append(hold);
    const source=entering?graphController?.getAttentionRect():graphIntraController?.getTransitionRect();
    graphZoomJob={hold,source,entering,animations:[],overlay:document.createElementNS('http://www.w3.org/2000/svg','svg')};return graphZoomJob;
  }
  function graphZoomPlay(job){if(!job)return;requestAnimationFrame(()=>requestAnimationFrame(()=>{if(job!==graphZoomJob)return;
    const target=job.entering?graphIntraController?.getTransitionRect():graphController?.getAttentionRect();if(!target||!job.source){cancelGraphZoom();return;}
    job.running=true;const o=root.getBoundingClientRect(),overlay=job.overlay;overlay.classList.add('viz2-semantic-overlay');overlay.setAttribute('width',root.clientWidth);overlay.setAttribute('height',root.clientHeight);overlay.dataset.graphSemanticTask=store.getState().selection?.task;root.append(overlay);
    const rect=document.createElementNS(overlay.namespaceURI,'rect');rect.setAttribute('fill','none');rect.setAttribute('stroke','var(--primary)');rect.setAttribute('rx',8);overlay.append(rect);
    const destination=root.querySelector(job.entering?'#viz2-graph-intra':'#viz2-graph-inter');job.animations.push(job.hold.animate([{opacity:1},{opacity:0}],{duration:600,fill:'both'}),destination.animate([{opacity:0},{opacity:1}],{duration:700,fill:'both'}));
    // Interpolate source graph region to the destination Task canvas in screen space.
    const from=job.source;const localX=from.left-o.left,localY=from.top-o.top;
    job.hold.style.transformOrigin=`${localX-parseFloat(job.hold.style.left)}px ${localY-parseFloat(job.hold.style.top)}px`;
    job.animations.push(job.hold.animate([{transform:'translate(0,0) scale(1,1)',opacity:1},{transform:`translate(${target.left-from.left}px,${target.top-from.top}px) scale(${target.width/from.width},${target.height/from.height})`,opacity:0}],{duration:720,easing:'cubic-bezier(.2,.8,.2,1)',fill:'both'}));
    const start=performance.now();function frame(now){if(job!==graphZoomJob)return;const p=Math.min(1,(now-start)/720),t=1-(1-p)**3;for(const [key,a,b] of [['x',from.left-o.left,target.left-o.left],['y',from.top-o.top,target.top-o.top],['width',from.width,target.width],['height',from.height,target.height]])rect.setAttribute(key,a+(b-a)*t);rect.setAttribute('opacity',Math.sin(Math.PI*p));if(p<1)requestAnimationFrame(frame);else cancelGraphZoom();}frame(start);
  }));}
  function enterGraphIntra(previous,taskId){const s=store.getState();if(s.views.graph.subview==='block')return;const id=taskId??s.selection?.task??85;if(!graphIntraModel.tasks[id])return;graphOuterSnapshot=previous||{camera:s.views.graph.camera,nodeId:s.views.graph.nodeId,inspectorOpen:s.views.graph.inspectorOpen};const job=graphZoomPrepare(true);store.dispatch({type:'graph-intra-enter',id});graphZoomPlay(job);}
  function exitGraphIntra(){if(store.getState().views.graph.subview!=='block')return;const job=graphZoomPrepare(false);store.dispatch({type:'graph-intra-exit'});graphZoomPlay(job);}
  addEventListener('resize',cancelGraphZoom);matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change',cancelGraphZoom);
  function selectTask(id) { const t=taskInfo(id); store.dispatch({type:'select',selection:{kind:'task',task:id,block:t.block,batch:t.batch,kvHead:t.kv}}); }
  function fit() { assignmentMorph.fit(); }
  function zoomBy(factor, point, pointedTask=null) { if(factor>1 && Number.isInteger(pointedTask))enterAnatomy(pointedTask); }
  function paintSelection() {
    const selection=store.getState().selection;
    root.querySelectorAll('[data-pa-block]').forEach(el => { const id=Number(el.dataset.paBlock); el.classList.toggle('is-selected',id===selection?.block); el.classList.toggle('is-hovered',hoverTask!==null && taskInfo(hoverTask).block===id); el.setAttribute('aria-pressed',String(id===selection?.block)); });
    root.querySelectorAll('[data-pa-task]').forEach(el => { const id=Number(el.dataset.paTask); el.classList.toggle('is-assigned',id%constants.blocks===selection?.block); el.classList.toggle('is-selected',id===selection?.task); el.classList.toggle('is-hovered',id===hoverTask); el.setAttribute('aria-pressed',String(id===selection?.task)); el.setAttribute('tabindex',id===(selection?.task ?? 0)?'0':'-1'); });
    assignmentMorph.update(selection, store.getState().open && store.getState().activeTab==='assignment' && store.getState().views.assignment.phase==='attention' && store.getState().views.assignment.subview!=='block');
    root.querySelector('[data-viz2-selection]').textContent=selection?.task != null ? taskDescription(selection.task) : selection?.kind==='block' ? `Block ${selection.block} → ${tasksForBlock(selection.block).join(', ')} · Task ID % ${constants.blocks}` : `每个 Task 对应 ${constants.qPerKv} 个 Q Head`;
  }
  function render(state, action) {
    root.hidden=!state.open;
    phase0Controller?.update(state.selection, state.views.assignment.phase0Camera, {
      active:state.open && state.activeTab==='assignment' && state.views.assignment.phase==='phase0' && state.views.assignment.subview!=='block',
      selectionChanged:action.type!=='view',
    });
    if (!state.open) {
      assignmentMorph?.update(state.selection,false);
      pipelineIntraController?.update(null,null,null,{active:false});
      return;
    }
    tabs.forEach(id => { const active=id===state.activeTab, tab=root.querySelector(`[data-viz2-tab="${id}"]`); tab.classList.toggle('is-selected',active); tab.setAttribute('aria-selected',String(active)); tab.tabIndex=active?0:-1; root.querySelector(`#viz2-${id}`).hidden=!active; });
    if (action.type === 'tab') {
      tooltip.hidden = true; phase0Controller.hideTooltip(); pipelineController?.hideTooltip(); graphController?.hideTooltip();
      tabAnimation?.cancel();
      if (!tabTransition?.active && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        tabAnimation = root.querySelector(`#viz2-${state.activeTab}`).animate(
          [{opacity:0,transform:'translateY(6px)'},{opacity:1,transform:'translateY(0)'}],
          {duration:220,easing:'cubic-bezier(.2,.8,.2,1)'});
      }
    }
    if (state.activeTab === 'pipeline' && !pipelineController) {
      const model = window.PtoVisualization2Pipeline.createModel({ ...constants, seqLens,
        pageTokens: sourceNumber('BLOCK_SIZE'), pagesPerStack: sourceNumber('STACK_PAGES'), phase0Model });
      pipelineController = window.PtoVisualization2Pipeline.mount(root.querySelector('#viz2-pipeline-inter'), {
        model,
        onSelect: selection => store.dispatch({type:'select',selection}),
        onCamera: camera => store.dispatch({type:'view',id:'pipeline',patch:{camera}}),
        onEnter: enterPipelineIntra,
      });
    }
    const pipeline=state.views.pipeline,pipelineInner=pipeline.subview==='block';
    if(state.activeTab==='pipeline'&&pipelineInner&&action.type==='tab'){
      if(state.selection?.kind!=='task'){store.dispatch({type:'pipeline-intra-exit'});return;}
      if(state.selection.task!==pipeline.intraTask||JSON.stringify(state.selection.subobject??null)!==JSON.stringify(pipeline.intraFocus?.[pipeline.intraTask]??null)){store.dispatch({type:'pipeline-intra-task',id:state.selection.task});return;}
    }
    root.querySelector('#viz2-pipeline-inter').hidden=pipelineInner;
    root.querySelector('#viz2-pipeline-intra').hidden=!pipelineInner;
    root.querySelector('[data-pipeline-back]').hidden=!pipelineInner;
    root.querySelectorAll('[data-pipeline-scope]').forEach(button=>{const selected=(button.dataset.pipelineScope==='intra')===pipelineInner;button.classList.toggle('is-selected',selected);button.setAttribute('aria-pressed',String(selected));if(button.dataset.pipelineScope==='intra')button.disabled=!pipelineInner&&state.selection?.kind!=='task';});
    const pf=state.selection?.subobject;
    root.querySelector('[data-pipeline-breadcrumb]').textContent=state.selection?.task!=null?`Pipeline / Block ${String(state.selection.block).padStart(2,'0')} / Task ${state.selection.task}${pipelineInner&&pf?` / ${pf.event||pf.stage||({stack:'Stack',page:'Page',slot:'Slot',tick:'Tick',output:'Output'}[pf.kind]||pf.kind)} ${pf.index}`:''}`:'Pipeline';
    pipelineController?.update(state.selection,pipeline.camera,
      {active:state.activeTab==='pipeline'&&!pipelineInner,enter:action.type==='tab'&&state.activeTab==='pipeline'});
    if(state.activeTab==='pipeline'&&pipelineInner&&!pipelineIntraController){
      pipelineIntraController=window.PtoVisualization2PipelineIntra.mount(root.querySelector('#viz2-pipeline-intra'),{model:pipelineIntraModel,
        onTask:id=>store.dispatch({type:'pipeline-intra-task',id}),onFocus:focus=>store.dispatch({type:'pipeline-intra-focus',focus}),
        onExit:exitPipelineIntra,
        onNavigate:()=>{store.dispatch({type:'tab',id:'graph'});store.dispatch({type:'view',id:'graph',patch:{intraInspectorOpen:true}});},
        onCamera:(id,camera)=>store.dispatch({type:'view',id:'pipeline',patch:{intraCameras:{...store.getState().views.pipeline.intraCameras,[id]:camera}}}),
        onInspector:open=>store.dispatch({type:'view',id:'pipeline',patch:{intraInspectorOpen:open}})});
    }
    pipelineIntraController?.update(pipeline.intraTask,state.selection?.subobject||null,pipeline.intraCameras?.[pipeline.intraTask]||null,{active:state.activeTab==='pipeline'&&pipelineInner,inspectorOpen:pipeline.intraInspectorOpen});
    const gv=state.views.graph,gi=gv.subview==='block';
    if(state.activeTab==='graph'&&gi&&action.type==='tab'&&state.selection?.kind==='task'&&(gv.intraTask!==state.selection.task||JSON.stringify(gv.intraFocus)!==JSON.stringify(state.selection.subobject))){store.dispatch({type:'graph-intra-task',id:state.selection.task});return;}
    root.querySelector('#viz2-graph-inter').hidden=gi;root.querySelector('#viz2-graph-intra').hidden=!gi;root.querySelector('[data-graph-back]').hidden=!gi;
    root.querySelectorAll('[data-graph-scope]').forEach(b=>{const yes=(b.dataset.graphScope==='intra')===gi;b.classList.toggle('is-selected',yes);b.setAttribute('aria-pressed',String(yes));});
    root.querySelector('[data-graph-breadcrumb]').textContent=state.selection?.task!=null?`Graph / Block ${state.selection.block} / Task ${state.selection.task}${gi&&state.selection.subobject?` / ${state.selection.subobject.node||state.selection.subobject.stage||state.selection.subobject.kind} ${state.selection.subobject.index??''}`:''}`:'Graph';
    if(state.activeTab==='graph'&&!graphController)graphController=window.PtoVisualization2Graph.mount(root.querySelector('#viz2-graph-inter'),{
      execution:{blocks:constants.blocks,taskCount:seqLens.length*constants.kvHeads,phase0Model,seqLens},onSelect:selection=>store.dispatch({type:'select',selection}),onView:patch=>store.dispatch({type:'view',id:'graph',patch}),onEnter:enterGraphIntra,
      onBlockEnter:block=>{const selected=store.getState().selection;const first=graphIntraModel.tasks.find(task=>task.block===block);if(first)enterGraphIntra(null,selected?.kind==='task'&&selected.block===block?selected.task:first.id);}});
    graphController?.update(state.selection,gv,{active:state.activeTab==='graph'&&!gi,enter:action.type==='tab'&&state.activeTab==='graph'&&!gi,restore:action.type==='graph-intra-exit'});
    if(state.activeTab==='graph'&&gi&&!graphIntraController)graphIntraController=window.PtoVisualization2GraphIntra.mount(root.querySelector('#viz2-graph-intra'),{model:graphIntraModel,onTask:id=>store.dispatch({type:'graph-intra-task',id}),onFocus:focus=>store.dispatch({type:'graph-intra-focus',focus}),onView:patch=>store.dispatch({type:'view',id:'graph',patch}),onNavigate:()=>{store.dispatch({type:'tab',id:'pipeline'});store.dispatch({type:'view',id:'pipeline',patch:{intraInspectorOpen:true}});}});
    graphIntraController?.update(gv.intraTask,state.selection?.subobject,gv,{active:state.activeTab==='graph'&&gi});
    if (state.activeTab !== 'assignment') {assignmentMorph?.update(state.selection,false);anatomyController?.update(null,null,null,{active:false});return;}
    const assignment=state.views.assignment,inner=assignment.subview==='block';
    if(inner&&action.type==='tab'){
      if(state.selection?.kind!=='task'){store.dispatch({type:'anatomy-exit'});return;}
      if(state.selection.task!==assignment.anatomyTask||JSON.stringify(state.selection.subobject??null)!==JSON.stringify(assignment.anatomyFocus?.[assignment.anatomyTask]??null)){store.dispatch({type:'anatomy-task',id:state.selection.task});return;}
    }
    root.querySelector('#viz2-anatomy').hidden=!inner;
    root.querySelector('[data-viz2-phases]').hidden=inner;
    root.querySelector('[data-viz2-matrix-back]').hidden=!inner;
    root.querySelectorAll('[data-viz2-scope]').forEach(button=>{const selected=(button.dataset.viz2Scope==='intra')===inner;button.classList.toggle('is-selected',selected);button.setAttribute('aria-pressed',String(selected));if(button.dataset.viz2Scope==='intra')button.disabled=!inner&&(assignment.phase!=='attention'||state.selection?.kind!=='task');});
    const detailFocus=inner?assignment.anatomyFocus?.[assignment.anatomyTask]:null;
    const detailName=detailFocus?({q:'Q Tile',output:'Attention Output'}[detailFocus.kind]||`${{page:'Page',stack:'Stack',slot:'Slot'}[detailFocus.kind]} ${detailFocus.index}`):'';
    const breadcrumb=root.querySelector('[data-viz2-breadcrumb]');
    breadcrumb.hidden=!!state.selection;
    breadcrumb.textContent=state.selection?.task!=null?`Assignment / Block ${String(state.selection.block).padStart(2,'0')} / Task ${state.selection.task}${detailName?` / ${detailName}`:''}`:'Assignment';
    if(assignment.phase==='phase0'&&!inner){
      const s=state.selection,item=s?.kind==='task'?phase0Model.itemFor(s.batch,s.kvHead):null;
      breadcrumb.textContent=item?`Assignment / Phase0 / Block ${String(item.block).padStart(2,'0')} / AIV${item.lane} / G${item.gIndex}`:s?`Assignment / Phase0 / Block ${String(s.block).padStart(2,'0')}${s.lane==null?'':` / AIV${s.lane}`}`:'Assignment / Phase0';
    }
    if(inner){
      assignmentMorph?.update(state.selection,false);
      root.querySelector('#viz2-phase0').hidden=true;root.querySelector('#viz2-attention').hidden=true;
      if(!anatomyController)anatomyController=window.PtoVisualization2Anatomy.mount(root.querySelector('#viz2-anatomy'),{model:anatomyModel,
        onTask:id=>store.dispatch({type:'anatomy-task',id}),onFocus:focus=>store.dispatch({type:'anatomy-focus',focus}),
        onExit:exitAnatomy,
        onCamera:(id,camera)=>store.dispatch({type:'view',id:'assignment',patch:{anatomyCameras:{...store.getState().views.assignment.anatomyCameras,[id]:camera}}}),
        onInspector:open=>store.dispatch({type:'view',id:'assignment',patch:{anatomyInspectorOpen:open}})});
      anatomyController.update(assignment.anatomyTask,assignment.anatomyFocus?.[assignment.anatomyTask]||null,assignment.anatomyCameras?.[assignment.anatomyTask]||null,{active:true,inspectorOpen:assignment.anatomyInspectorOpen});return;
    }
    anatomyController?.update(null,null,null,{active:false});
    const phase = state.views.assignment.phase;
    root.querySelectorAll('[data-viz2-phase]').forEach(button => {
      const selected = button.dataset.viz2Phase === phase;
      button.classList.toggle('is-selected', selected); button.setAttribute('aria-pressed', String(selected));
    });
    const phase0 = root.querySelector('#viz2-phase0'), attention = root.querySelector('#viz2-attention');
    phase0.hidden = phase !== 'phase0'; attention.hidden = phase !== 'attention';
    if (action.type === 'phase') {
      tooltip.hidden = true; hoverTask = null; phase0Controller.hideTooltip();
      phaseAnimation?.cancel();
      if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        phaseAnimation = (phase === 'phase0' ? phase0 : attention).animate(
          [{opacity:0, transform:'translateY(6px)'},{opacity:1, transform:'translateY(0)'}],
          {duration:220, easing:'cubic-bezier(.2,.8,.2,1)'});
      }
    }
    if (phase === 'phase0') {
      assignmentMorph?.update(state.selection,false);
      return;
    }
    assignmentMorph.update(state.selection, state.open && state.activeTab==='assignment' && phase==='attention');
    if (action.type !== 'view') paintSelection();
  }
  function open(button) {
    if (!root) mount();
    if (store.getState().open) return;
    trigger=button;
    const workarea=document.querySelector('#ideWorkarea'); workarea.dataset.visualization2='open';
    previousInert=[...workarea.children].filter(el=>el!==root).map(el=>[el,el.inert]); previousInert.forEach(([el])=>{el.inert=true;});
    store.dispatch({type:'open'}); root.querySelector('[data-viz2-tab="assignment"]').focus();
  }
  function close() {
    store.dispatch({type:'close'}); delete document.querySelector('#ideWorkarea').dataset.visualization2;
    previousInert.forEach(([el,inert])=>{el.inert=inert;}); previousInert=[];
    if (trigger?.isConnected) trigger.focus();
  }
  document.addEventListener('click',event=>{ const button=event.target.closest('[data-visualization2-open]'); if(button && window.PtoComputeIDE?.getScene() !== 'csa') open(button); });
  // Read-only snapshots + explicit actions keep future subviews and shared selection extensible.
  window.PtoVisualization2=Object.freeze({getState:store.getState,subscribe:store.subscribe,dispatch:store.dispatch,close});
})();
