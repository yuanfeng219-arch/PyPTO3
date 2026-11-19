(() => {
  'use strict';
  const $ = selector => document.querySelector(selector);
  const source = window.PTO_PAGED_ATTENTION_PYPTO_SOURCE;
  const {constants} = window.PTO_PYPA_SEMANTIC_GRAPH;
  const value = name => constants[name].value;
  const scenarioAPI = window.PtoComputeScenario;
  const anatomyAPI = window.PtoVisualization2Anatomy;
  const graphAPI = window.PtoVisualization2GraphIntra;
  const pipelineAPI = window.PtoVisualization2PipelineIntra;
  const preLaunch = value('PRE_LAUNCH');
  const anatomy = anatomyAPI.createModel({
    seqLens:window.PtoPagedAttentionIntro.model.defaultSeqLens, blocks:value('ATTN_SPMD_BLOCKS'),
    numHeads:value('NUM_HEADS'), kvHeads:value('NUM_KV_HEADS'), headDim:value('HEAD_DIM'),
    rowTile:value('ROW_TILE'), pageTokens:value('BLOCK_SIZE'), pagesPerStack:value('STACK_PAGES'),
    transferSlots:value('TRANSFER_SLOTS'),
  });
  const task = anatomy.tasks[48], id = task.id;
  const original = scenarioAPI.create(task);
  const example = scenarioAPI.edit(original,'sequence','1153').scenario;
  const project = config => scenarioAPI.project(config,anatomy,source,preLaunch);
  const layoutConfig = scenarioAPI.edit(original,'sequence',String(scenarioAPI.limits(original).sequence)).scenario;
  const layout = project(layoutConfig);
  let config = original, comparison = example, projections = project(config);
  let graphView = {}, graphFocus = null, assignmentFocus = null, pipelineFocus = null;
  let assignmentCamera = null, pipelineCamera = null, assignmentInspector = false, pipelineInspector = false;
  let beat = 0, active = true, rendering = false, renderRequested = false;
  const noop = () => {};
  const scenario = {
    get: () => config,
    limits: () => scenarioAPI.limits(config),
    task: (kind,taskId) => taskId === id ? projections[kind] : null,
    edit(kind,raw) {
      const result = scenarioAPI.edit(config,kind,raw);
      if (result.error) return result;
      if (result.scenario.isCustom) comparison = result.scenario;
      beat = result.scenario.isCustom ? 1 : 0;
      apply(result.scenario);
      return result;
    },
    reset,
  };
  const assignmentHost = $('#p14-assignment'), graphHost = $('#p14-graph'), pipelineHost = $('#p14-pipeline');
  const assignment = anatomyAPI.mount(assignmentHost,{
    model:anatomy, scenario, toolbarParent:$('#shared-controls'), stackColumns:2, onTask:noop, onExit:noop,
    onFocus(focus) { assignmentFocus = focus; assignmentInspector = !!focus; render(); },
    onCamera(_id,camera) { assignmentCamera = camera; render(); },
    onInspector(open) { assignmentInspector = open; render(); },
  });
  const graph = graphAPI.mount(graphHost,{
    model:graphAPI.createModel({anatomy,source,preLaunch}), scenario, presentation:{layoutTask:layout.graph}, onTask:noop,
    onFocus(focus) { graphFocus = focus; graphView.intraInspectorOpen = !!focus; render(); },
    onView(patch) { Object.assign(graphView,patch); if (patch.intraInspectorOpen === false) graphFocus = null; render(); },
    onNavigate() { navigateReady('pipeline',graphFocus); },
  });
  const pipeline = pipelineAPI.mount(pipelineHost,{
    model:pipelineAPI.createModel({anatomy,preLaunch}), scenario, presentation:{maxTicks:layout.pipeline.ticks.length}, onTask:noop, onExit:noop,
    onFocus(focus) { pipelineFocus = focus; pipelineInspector = !!focus; render(); },
    onCamera(_id,camera) { pipelineCamera = camera; render(); },
    onInspector(open) { pipelineInspector = open; render(); },
    onNavigate() { navigateReady('graph',pipelineFocus); },
    onOpenSource({file,start,end}) {
      $('#source-title').textContent = `${file} · L${start}–${end}`;
      $('#source-lines').textContent = source.split('\n').slice(start-1,end).map((line,i)=>`${start+i}  ${line}`).join('\n');
      $('#source-dialog').showModal();
    },
  });
  for (const host of [graphHost,pipelineHost]) host.querySelector('[data-compute-toolbar]').remove();
  for (const host of [graphHost,assignmentHost,pipelineHost]) {
    const chrome = host.querySelectorAll(':scope > .compute-identity, :scope > [data-anatomy-back], :scope > [data-intra-back]');
    chrome.forEach(el => { el.hidden = true; el.inert = true; });
  }
  $('#task').textContent = `Task ${id} / Batch ${task.batch} / KV Head ${task.kvHead} / Block ${String(task.block).padStart(2,'0')}`;
  const input = $('#shared-controls [data-compute-pages]'), error = $('#shared-controls [data-page-error]');
  error.id = 'sequence-error';
  input.setAttribute('aria-describedby',error.id);

  function navigateReady(destination,focus) {
    if (!focus?.event) return;
    const index = focus.stackIndex ?? focus.index;
    if (destination === 'pipeline') {
      const link = projections.pipeline.dependencies.find(link=>link.stack===index&&link.event===focus.event);
      if (!link) return;
      pipelineFocus = {kind:'event',index,event:link.event,buffer:link.buffer};
      pipelineInspector = true;
      pipelineHost.querySelector('.viz2-anatomy-viewport').focus({preventScroll:true});
    } else {
      const sync = projections.graph.graph.synchronizations.find(sync=>sync.name+'_EVENT'===focus.event);
      if (!sync) return;
      graphFocus = {kind:'event',index,stackIndex:index,syncId:sync.id,event:focus.event};
      graphView.intraInspectorOpen = true;
      graphHost.querySelector('.viz2-graph-viewport').focus({preventScroll:true});
    }
    render();
  }

  function render() {
    if (!active) return;
    if (rendering) { renderRequested = true; return; }
    rendering = true;
    try {
      const assignmentScroll = assignmentHost.querySelector('.viz2-anatomy-cache-scroll');
      const position = assignmentScroll ? [assignmentScroll.scrollLeft,assignmentScroll.scrollTop] : null;
      const viewport = assignmentHost.querySelector('.viz2-anatomy-viewport');
      const scale = Math.max(.25,Math.min(1.15,
        (viewport.clientWidth-10)/assignment.stackViewportWidth,
        (viewport.clientHeight-4)/assignment.stackViewportHeight(task)));
      assignment.update(id,assignmentFocus,assignmentCamera ?? {scale},{active:true,inspectorOpen:assignmentInspector});
      if (position) assignmentHost.querySelector('.viz2-anatomy-cache-scroll').scrollTo(...position);
      graph.update(id,graphFocus,graphView,{active:true});
      pipeline.update(id,pipelineFocus,pipelineCamera,{active:true,inspectorOpen:pipelineInspector});
      if (pipelineFocus && !pipeline.validFocus(pipelineFocus)) {
        pipelineFocus = null;
        pipelineInspector = false;
      }
      resetButton.hidden = false;
      const tail = projections.assignment.pages.at(-1);
      $('#tail').textContent = `尾页 P${tail.index} · ${tail.tokens} / ${projections.assignment.pageTokens} tokens`;
      $('#ticks').textContent = `${projections.pipeline.ticks.length} Logical Ticks · 逻辑轮次，非硬件耗时`;
      for (const [selector,label,preset] of [['#baseline','原配置',original],['#comparison','对比配置',comparison]]) {
        const button = $(selector);
        button.textContent = `${label} ${preset.effectiveSeqLen}`;
        button.setAttribute('aria-pressed',String(config.effectiveSeqLen===preset.effectiveSeqLen));
      }
    } finally {
      rendering = false;
      if (renderRequested) { renderRequested = false; queueMicrotask(render); }
    }
  }

  function clearRemovedFocus() {
    const data = projections.assignment;
    if ((assignmentFocus?.kind==='page' && !data.pages[assignmentFocus.index]) ||
        (assignmentFocus?.kind==='stack' && !data.stacks[assignmentFocus.index])) {
      assignmentFocus = null;
      assignmentInspector = false;
    }
    if (graphFocus?.instancePage != null && !data.pages[graphFocus.instancePage]) {
      const {instancePage,stackIndex,index,...nodeFocus} = graphFocus;
      graphFocus = nodeFocus.node ? {...nodeFocus,index:0,stackIndex:null} : null;
    }
    const stack = graphFocus?.stackIndex ?? (graphFocus?.kind==='stack' ? graphFocus.index : null);
    if (stack != null && !data.stacks[stack]) {
      graphFocus = graphFocus.node ? {kind:'graph',node:graphFocus.node,index:0,stackIndex:null} : null;
    }
    if (graphFocus?.kind==='page' && !data.pages[graphFocus.index]) graphFocus = null;
    if (!graphFocus) graphView.intraInspectorOpen = false;
  }

  function apply(next,{quiet=false}={}) {
    if (next.effectiveSeqLen !== config.effectiveSeqLen) {
      const updated = project(next);
      config = next;
      projections = updated;
      clearRemovedFocus();
    }
    error.textContent = '';
    input.removeAttribute('aria-invalid');
    render();
    if (!quiet) window.dispatchEvent(new CustomEvent('p14-change',{detail:{beat,context:controller.context}}));
  }

  function reset() {
    beat = 0;
    comparison = example;
    graphFocus = assignmentFocus = pipelineFocus = null;
    assignmentInspector = pipelineInspector = false;
    graphView = {};
    pipelineCamera = assignmentCamera = null;
    apply(original);
    assignmentHost.querySelector('.viz2-anatomy-cache-scroll').scrollTo(0,0);

  }

  function set(index,{context,quiet=false}={}) {
    if (!Number.isInteger(index) || index<0 || index>2) return false;
    beat = index;
    if (context) {
      comparison = scenarioAPI.edit(original,'sequence',String(context.comparison)).scenario ?? comparison;
      const restored = scenarioAPI.edit(original,'sequence',String(context.sequence)).scenario ?? original;
      apply(restored,{quiet});
    } else {
      apply(index===1 ? comparison : original,{quiet});
    }
    return true;
  }
  const controller = {
    start:0, last:2, set,
    get beat() { return beat; },
    get context() { return {task:id,sequence:config.effectiveSeqLen,comparison:comparison.effectiveSeqLen}; },
    enter(index=0,context) { active=true; return set(index,{context,quiet:true}); },
    advance(direction) { return set(beat+direction); },
    leave() {
      active=false;
      assignment.update(id,null,null,{active:false});
      graph.hideTooltip();
      pipeline.update(id,null,null,{active:false});
    },
  };
  window.ptoWhatIfStory = controller;
  $('#baseline').onclick = () => { beat=0; apply(original); };
  $('#comparison').onclick = () => { beat=1; apply(comparison); };
  // Keep the native reset connected to Scenario, including at baseline.
  const resetButton = $('#shared-controls [data-scenario-reset]');
  resetButton.id = 'reset';
  controller.enter();
  const observer = new ResizeObserver(render);
  observer.observe(assignmentHost.querySelector('.viz2-anatomy-viewport'));
})();
