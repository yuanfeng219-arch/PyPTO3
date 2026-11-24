/* Agent 版 Graph 的 L1 / L2 源码语义下钻。 */
(function (global) {
  'use strict';
  const base = global.PTO_PYPA_SEMANTIC_GRAPH;
  const stages = base.nodes.filter(node => node.type === 'composite' && node.composite?.subgraph);
  const stageMap = new Map(stages.map(node => [node.id, node]));
  let lastStage = 'qk', level = 1, controller = null, view = {}, panel, surface, select, l1, l2, caption;
  const stageFor = id => {
    const root = String(id || '').split('/')[0];
    if (stageMap.has(root)) return root;
    const producers = base.edges.filter(edge => edge.t === root && stageMap.has(edge.s));
    return producers.length === 1 ? producers[0].s : null;
  };
  function graphModel(stage) {
    const subgraph = stage.composite.subgraph;
    return { schemaVersion:2, id:'source-detail-' + stage.id, sourceFiles:base.sourceFiles,
      constants:base.constants, runtimeDims:base.runtimeDims, nodes:subgraph.nodes, edges:subgraph.edges, groups:[] };
  }
  function syncSource(stage) {
    if (stage.source?.file !== 'paged_attention_pypto.py' || !stage.source.ranges?.length) return;
    global.parent.postMessage({type:'pypto-agent:source',file:stage.source.file,ranges:stage.source.ranges,
      primaryRange:stage.source.primaryRange || 0,title:stage.name}, '*');
  }
  function paintLevel() {
    panel.hidden = level !== 2;
    l1.setAttribute('aria-pressed', String(level === 1));
    l2.setAttribute('aria-pressed', String(level === 2));
    select.hidden = level !== 2;
    caption.hidden = level !== 2;
  }
  function showL1() {
    level = 1; paintLevel();
    if (controller) { controller.destroy(); controller = null; surface.replaceChildren(); }
    l1.focus({preventScroll:true});
  }
  function showL2(id) {
    const stage = stageMap.get(id) || stageMap.get('qk');
    if (!stage) return;
    lastStage = stage.id; level = 2; select.value = stage.id;
    caption.textContent = 'L2 · ' + stage.name + ' · 源码静态子图，非编译 Kernel 清单';
    if (controller) controller.destroy();
    surface.replaceChildren(); paintLevel();
    const model = graphModel(stage);
    view = { camera:null, nodeId:null, inspectorOpen:false, overlays:{execution:false} };
    const adapter = {
      getModel:() => model,
      executionData:() => ({nodes:{},edges:{}}),
      resolveFocus:(graph, selection, state) => ({nodeId:state.nodeId || null,related:state.nodeId ? [state.nodeId] : [],kind:state.nodeId ? 'node' : 'none'}),
      onNode:id => setView({nodeId:id,inspectorOpen:false}),
      onEdge:() => {}, onSync:() => {}, onHover:() => {},
      tooltip:id => model.nodes.find(node => node.id === id)?.name || '',
      inspector:() => ({title:'',html:''}),
      context:() => '源码静态子图',
      initialMinScale:.25,
      fitPadding:{top:28,right:28,bottom:28,left:28}
    };
    function setView(patch) { view = {...view,...patch,inspectorOpen:false}; controller.update(null,view,{active:true}); }
    controller = global.PtoVisualization2Graph.mount(surface, {adapter,onView:setView,onSelect:()=>{}});
    controller.update(null, view, {active:true});
    requestAnimationFrame(() => { if (level === 2) surface.querySelector('[data-viz2-graph-fit]')?.click(); });
    syncSource(stage);
  }
  function install() {
    const graph = document.getElementById('viz2-graph');
    if (!graph || !graph.querySelector('#viz2-graph-inter') || graph.querySelector('.agent-kernel-toolbar')) return;
    const toolbar = document.createElement('div'); toolbar.className = 'agent-kernel-toolbar';
    toolbar.innerHTML = '<div class="agent-kernel-segment" role="group" aria-label="计算图层级"><button type="button" data-agent-level="1" aria-pressed="true"><b>L1</b> 整体</button><button type="button" data-agent-level="2" aria-pressed="false"><b>L2</b> Kernel</button></div><select aria-label="选择计算模块" hidden></select><span class="agent-kernel-caption" hidden></span><span class="agent-kernel-hint">双击模块进入 L2</span>';
    l1=toolbar.querySelector('[data-agent-level="1"]'); l2=toolbar.querySelector('[data-agent-level="2"]');
    select=toolbar.querySelector('select'); caption=toolbar.querySelector('.agent-kernel-caption');
    stages.forEach(stage => { const option=document.createElement('option'); option.value=stage.id; option.textContent=stage.name; select.append(option); });
    l1.addEventListener('click', () => { if (level===2) showL1(); });
    l2.addEventListener('click', () => { if (level===1) showL2(lastStage); });
    select.addEventListener('change', () => showL2(select.value));
    panel=document.createElement('section'); panel.className='agent-kernel-panel'; panel.setAttribute('aria-label','L2 Kernel 源码子图'); panel.hidden=true;
    surface=document.createElement('div'); surface.className='agent-kernel-surface'; panel.append(surface);
    graph.append(panel,toolbar);
    graph.addEventListener('click',event => {
      if (level !== 1 || !event.target.closest('#viz2-graph-inter')) return;
      const el=event.target.closest('[data-pypa-node],[data-viz2-graph-label]');
      const id=stageFor(el?.dataset.pypaNode || el?.dataset.viz2GraphLabel);
      if (id) lastStage=id;
    },true);
    graph.addEventListener('dblclick',event => {
      if (level !== 1 || !event.target.closest('#viz2-graph-inter') || event.target.closest('[data-pypa-expand-toggle]')) return;
      const el=event.target.closest('[data-pypa-node],[data-viz2-graph-label]');
      const id=stageFor(el?.dataset.pypaNode || el?.dataset.viz2GraphLabel);
      if (!id) return;
      event.preventDefault(); event.stopImmediatePropagation(); showL2(id);
    },true);
    graph.addEventListener('keydown',event => {
      if (level === 2 && event.key === 'Escape') { event.preventDefault(); showL1(); }
    });
  }
  const observer=new MutationObserver(install);
  observer.observe(document.getElementById('ideWorkarea'),{childList:true,subtree:true});
  install();
})(window);
