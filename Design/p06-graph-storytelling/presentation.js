(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const slide = $('#slide');
  const graphHost = $('#graph');
  const viewport = $('#ideWorkarea');
  const resize = new ResizeObserver(([entry]) => {
    slide.style.setProperty('--slide-scale', entry.contentRect.width / 1600);
  });
  resize.observe(viewport);

  try {
    mount();
  } catch (error) {
    $('#error').hidden = false;
    console.error(error);
  }

  function mount() {
    const model = window.PtoPagedAttentionIntro.model;
    const semantic = window.PTO_PYPA_SEMANTIC_GRAPH;
    const constant = name => semantic.constants[name].value;
    const anatomy = window.PtoVisualization2Anatomy.createModel({
      seqLens: model.defaultSeqLens,
      blocks: constant('ATTN_SPMD_BLOCKS'),
      numHeads: constant('NUM_HEADS'),
      kvHeads: constant('NUM_KV_HEADS'),
      headDim: constant('HEAD_DIM'),
      rowTile: constant('ROW_TILE'),
      pageTokens: constant('BLOCK_SIZE'),
      pagesPerStack: constant('STACK_PAGES'),
      transferSlots: constant('TRANSFER_SLOTS'),
    });
    const graphAPI = window.PtoVisualization2GraphIntra;
    const task = graphAPI.createModel({
      anatomy,
      source: window.PTO_PAGED_ATTENTION_PYPTO_SOURCE,
      preLaunch: constant('PRE_LAUNCH'),
    }).tasks[48];
    const executionAPI = window.PtoPagedAttentionExecution;
    const execution = executionAPI.derive(task.graph, model, {task: task.id, seqLen: task.seqLen});
    const graphLayout = window.PTO_SEMANTIC_COMPUTATION_GRAPH;
    const options = {flowLayout: true, overviewLayout: false, compactGlyphs: true, rootLayerGap: 64, childLayerGap: 56};
    const baseGraph = graphLayout.materialize(task.graph, new Set(), options);
    const executionFootprints = Object.fromEntries(baseGraph.nodes.flatMap(node => {
      const count = executionAPI.decoration(node, execution)?.count;
      return count ? [[node.id, executionAPI.deckOffset(count)]] : [];
    }));
    const nativeGraph = graphLayout.materialize(task.graph, new Set(), {...options, executionFootprints});
    const renderer = window.PtoStudioComputationGraph;
    const rendered = renderer.renderScene(graphHost, nativeGraph, {
      view: 'compute', allowExpand: false, synchronizations: true,
      markerPrefix: 'p06-story', cardHtml: node => renderer.cardHtml(node, task.graph, {}),
    });
    renderer.applyExecution(graphHost, rendered.placedNodes, nativeGraph, execution);
    const attentionRegion = graphHost.querySelector('[data-pypa-region="task-compute"]');
    attentionRegion.classList.add('attention-region');
    attentionRegion.querySelector('span').textContent = '';
    task.stacks.slice(1).reverse().forEach(({index}) => {
      const layer = attentionRegion.cloneNode(false);
      layer.removeAttribute('data-pypa-region');
      layer.classList.replace('attention-region', 'attention-layer');
      layer.style.transform = `translate(${index * 10}px, ${index * 8}px)`;
      layer.setAttribute('aria-hidden', 'true');
      attentionRegion.before(layer);
    });
    graphHost.querySelectorAll('[data-pypa-node]').forEach(node => { node.disabled = true; });
    graphHost.querySelectorAll('[data-pypa-execution-stack],[data-pypa-execution-count]').forEach(layer => {
      layer.removeAttribute('role');
      layer.removeAttribute('tabindex');
      layer.setAttribute('aria-hidden', 'true');
      if ((layer.dataset.pypaExecutionStack ?? layer.dataset.pypaExecutionCount) !== 'qk') layer.hidden = true;
    });
    const surface = graphHost.parentElement;
    const camera = Math.min(surface.clientWidth / rendered.sceneWidth, surface.clientHeight / rendered.sceneHeight);
    graphHost.style.transform = `translate(${(surface.clientWidth - rendered.sceneWidth * camera) / 2}px, 0) scale(${camera})`;

    $('#task-context').textContent = `Task ${task.id} · Batch ${task.batch} · KV Head ${task.kvHead}`;
    $('#module-count').textContent = `×${task.stackCount}`;
    $('#qk-count').textContent = `×${execution.nodes.qk.count}`;
    $('#story-note').textContent = `${task.seqLen.toLocaleString('zh-CN')} tokens · ${task.stackCount} Stacks · ${task.pageCount} Pages · 源码推导的逻辑重复`;
    $('#assignment-context').textContent = `Block ${String(task.block).padStart(2, '0')} · Task ${task.id}`;

    // Use the same Block 内 renderer as Compute Explorer and place its data
    // surfaces in the slide. The temporary controller owns the product chrome.
    const assignmentTask = anatomy.tasks[task.id];
    const scenario = window.PtoComputeScenario.create(assignmentTask);
    const source = document.createElement('div');
    const assignment = window.PtoVisualization2Anatomy.mount(source, {
      model: anatomy,
      scenario: {get: () => scenario, task: () => assignmentTask},
    });
    assignment.update(task.id, null, null, {active: true});
    $('#assignment-stacks').append(source.querySelector('[data-anatomy-track] svg'));
    $('#assignment-query').append(...source.querySelector('[data-anatomy-query]').children);
    assignment.destroy();
    document.querySelectorAll('.assignment [data-anatomy-kind]').forEach(object => {
      object.removeAttribute('tabindex');
      object.removeAttribute('aria-pressed');
      object.setAttribute('role', 'img');
    });
  }
})();
