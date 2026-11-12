(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const slide = $('#slide');
  slide.innerHTML = `
<header class="mast">
  <p>PyPTO 3.0 <span> / </span> UX 设计 · 算法可感知</p>
  <p><b id="task-context"></b><span> / </span> P06</p>
</header>
<h1>循环执行规模直观可见</h1>
<div class="stage" aria-label="Task 的数据分块与计算规模">
  <header class="scene-heading">
    <h2>数据分块决定重复计算规模</h2>
    <p>Stack 决定模块重复几次，Page 决定 QK Matmul 重复几次。</p>
  </header>
  <div id="overview">
    <section class="viz2 graph-view" aria-label="Task Graph 特写">
      <div class="graph-heading">
        <h3>Graph · QK Matmul 特写</h3>
        <div class="scale-summary" aria-label="逻辑重复规模">
          <span>Attention 计算模块 <strong id="module-count"></strong></span>
          <span>QK Matmul <strong id="qk-count"></strong></span>
        </div>
      </div>
      <div class="graph-surface"><div class="graph-crop"><div id="graph" class="kf-pypa-graph__scene"></div></div></div>
    </section>
    <section class="assignment viz2" aria-labelledby="assignment-title">
      <h3 id="assignment-title">Assignment · Block 内</h3>
      <div class="task-controls">
        <div id="task-list" role="group" aria-label="选择 Task"></div>
        <div class="playback" role="group" aria-label="Task 轮播控制">
          <button class="btn btn-ghost" id="previous-task" type="button">上一个</button>
          <button class="btn btn-ghost" id="next-task" type="button">下一个</button>
          <button class="btn btn-ghost" id="toggle-play" type="button">暂停轮播</button>
          <button class="btn btn-ghost" id="reset-task" type="button">重置</button>
        </div>
      </div>
      <p id="assignment-context" aria-live="polite"></p>
      <div class="assignment-data">
        <section><h4>KV Cache</h4><div id="assignment-stacks"></div></section>
        <section id="assignment-query" aria-label="Q Tile"></section>
      </div>
    </section>
  </div>
  <p id="story-note"></p>
  <p id="error" role="alert" hidden>Task 视图未能加载。请保留本页与 Compute Explorer 的目录关系后重新打开。</p>
</div>
`;
  const graphHost = $('#graph');
  const viewport = $('#ideWorkarea');
  const embedded = document.body.hasAttribute('data-p06-deck');
  if (embedded) {
    document.body.classList.add('p06-embedded');
    const header = slide.querySelector('.mast');
    header.setAttribute('data-deck-title', '');
    header.append(slide.querySelector('h1'));
    header.querySelector('h1').id = 'title';
    $('#assignment-title').append($('#task-context'));
  }
  const resize = new ResizeObserver(([entry]) => {
    slide.style.setProperty('--slide-scale', entry.contentRect.width / 1600);
  });
  if (!embedded) resize.observe(viewport);

  function start() {
    try {
      mount();
    } catch (error) {
      $('#error').hidden = false;
      console.error(error);
    }
  }

  if (embedded) window.mountP06GraphStory = start;
  else start();

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
    const {tasks} = graphAPI.createModel({
      anatomy,
      source: window.PTO_PAGED_ATTENTION_PYPTO_SOURCE,
      preLaunch: constant('PRE_LAUNCH'),
    });
    const executionAPI = window.PtoPagedAttentionExecution;
    const derive = task => executionAPI.derive(task.graph, model, {task: task.id, seqLen: task.seqLen});
    // Keep the existing Task 48 geometry and prepare the deepest selected stack once.
    const task = tasks[48];
    const execution = derive(task);
    const candidates = [24, 65, 18, 43].map(id => ({task: tasks[id], execution: derive(tasks[id])}));
    const largest = candidates.reduce((max, entry) => entry.task.pageCount > max.task.pageCount ? entry : max);
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
    renderer.applyExecution(graphHost, rendered.placedNodes, nativeGraph, largest.execution);
    const attentionRegion = graphHost.querySelector('[data-pypa-region="task-compute"]');
    attentionRegion.classList.add('attention-region');
    attentionRegion.querySelector('span').textContent = '';
    const attentionLayers = largest.task.stacks.slice(1).reverse().map(({index}) => {
      const layer = attentionRegion.cloneNode(false);
      layer.removeAttribute('data-pypa-region');
      layer.classList.replace('attention-region', 'attention-layer');
      layer.style.transform = `translate(${index * 10}px, ${index * 8}px)`;
      layer.setAttribute('aria-hidden', 'true');
      attentionRegion.before(layer);
      return {layer, index};
    });
    graphHost.querySelectorAll('[data-pypa-node]').forEach(node => { node.disabled = true; });
    graphHost.querySelectorAll('[data-pypa-execution-stack],[data-pypa-execution-count]').forEach(layer => {
      layer.removeAttribute('role');
      layer.removeAttribute('tabindex');
      layer.setAttribute('aria-hidden', 'true');
      if ((layer.dataset.pypaExecutionStack ?? layer.dataset.pypaExecutionCount) !== 'qk') layer.hidden = true;
    });
    const qkBadge = graphHost.querySelector('[data-pypa-execution-count="qk"]');
    const qkLayers = [...graphHost.querySelectorAll('[data-pypa-execution-stack="qk"] [data-pypa-execution-instance]')];
    const edgesById = new Map(nativeGraph.edges.map(edge => [edge.id, edge]));
    const edgePaths = [...graphHost.querySelectorAll('[data-pypa-edge]')];
    const edgeCounts = [...graphHost.querySelectorAll('[data-pypa-edge-label] tspan')];
    const edgeEntries = (id, counts) => {
      const edge = edgesById.get(id);
      return (edge.sourceEdgeIds ?? [id]).map(sourceId => counts.edges[sourceId]).filter(Boolean);
    };
    qkBadge.removeAttribute('title');
    qkBadge.removeAttribute('data-ai-tip');
    const graphCrop = graphHost.parentElement;
    const surface = graphCrop.parentElement;
    // Measure the native layout before transforming, including the largest
    // possible QK silhouette and badge. Reserve that same camera for every Task.
    const origin = graphHost.getBoundingClientRect();
    const unit = slide.getBoundingClientRect().width / slide.offsetWidth;
    const cropNodeIds = ['qk', 'scores', 'scale-mask', 'softmax', 'probability', 'm', 'l', 'rescale', 'pv'];
    const bounds = [
      graphHost.querySelector('[data-pypa-region="task-inputs"]'),
      ...cropNodeIds.map(id => graphHost.querySelector(`[data-pypa-node="${id}"]`)),
      qkBadge, ...qkLayers,
    ].map(element => element.getBoundingClientRect());
    const nextNode = graphHost.querySelector('[data-pypa-node="partial"]').getBoundingClientRect();
    const left = (Math.min(...bounds.map(box => box.left)) - origin.left) / unit - 28;
    const top = (Math.min(...bounds.map(box => box.top)) - origin.top) / unit - 20;
    const right = (Math.max(...bounds.map(box => box.right)) - origin.left) / unit + 28;
    const contentBottom = (Math.max(...bounds.map(box => box.bottom)) - origin.top) / unit;
    const bottom = Math.min(contentBottom + 50, (nextNode.top - origin.top) / unit - 10);
    const scale = 0.65;
    graphCrop.style.width = `${Math.min((right - left) * scale, surface.clientWidth)}px`;
    graphCrop.style.height = `${Math.min((bottom - top) * scale, surface.clientHeight)}px`;
    graphHost.style.transform = `translate(${-left * scale}px, ${-top * scale}px) scale(${scale})`;

    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const assignmentData = $('.assignment-data');
    let current = 0;
    let playing = true;
    let slideActive = !embedded;
    let inView = true;
    let pageActive = true;
    let timer;
    let transition;

    // Render each native Block 内 data surface from the same task used for QK.
    // Product chrome and its event handlers stay on the detached controller.
    candidates.forEach(entry => {
      const assignmentTask = anatomy.tasks[entry.task.id];
      const scenario = window.PtoComputeScenario.create(assignmentTask);
      const source = document.createElement('div');
      const controller = window.PtoVisualization2Anatomy.mount(source, {
        model: anatomy,
        scenario: {get: () => scenario, task: () => assignmentTask},
      });
      controller.update(entry.task.id, null, null, {active: true});
      entry.stacks = source.querySelector('[data-anatomy-track] svg');
      entry.query = [...source.querySelector('[data-anatomy-query]').children];
      entry.stacks.remove();
      entry.query.forEach(node => node.remove());
      controller.destroy();
      [entry.stacks, ...entry.query].forEach(root => {
        root.querySelectorAll('[data-anatomy-kind]').forEach(object => {
          object.removeAttribute('tabindex');
          object.removeAttribute('aria-pressed');
          object.setAttribute('role', 'img');
        });
      });
    });

    const buttons = candidates.map(({task: item}, index) => {
      const button = document.createElement('button');
      button.className = 'btn btn-ghost';
      button.type = 'button';
      button.textContent = `Task ${item.id}`;
      button.onclick = () => select(index);
      $('#task-list').append(button);
      return button;
    });

    $('#story-note').textContent = '源码推导的逻辑重复 · QK Matmul 按每 AIC、每 Task 计数';

    function render(index) {
      current = index;
      const {task: item, execution: counts, stacks, query} = candidates[index];
      $('#task-context').textContent = `Task ${item.id} · Batch ${item.batch} · KV Head ${item.kvHead}`;
      const stackLabel = item.stackCount === 1 ? 'Stack' : 'Stacks';
      const pageLabel = item.pageCount === 1 ? 'Page' : 'Pages';
      $('#assignment-context').textContent = `Block ${String(item.block).padStart(2, '0')} · ${item.seqLen.toLocaleString('zh-CN')} tokens · ${item.stackCount} ${stackLabel} · ${item.pageCount} ${pageLabel}`;
      $('#assignment-stacks').replaceChildren(stacks);
      $('#assignment-query').replaceChildren(...query);
      const qkCount = counts.nodes.qk.count;
      $('#module-count').textContent = `×${item.stackCount}`;
      $('#qk-count').textContent = qkBadge.textContent = `×${qkCount}`;
      attentionLayers.forEach(({layer, index}) => layer.classList.toggle('is-inactive', index >= item.stackCount));
      qkLayers.forEach(layer => layer.classList.toggle('is-inactive', Number(layer.dataset.pypaExecutionInstance) > qkCount));
      edgePaths.forEach(path => {
        const entries = edgeEntries(path.dataset.pypaEdge, counts);
        path.style.strokeWidth = entries.length ? String(Math.max(...entries.map(entry => entry.width))) : '';
      });
      edgeCounts.forEach(label => {
        const entries = edgeEntries(label.parentElement.dataset.pypaEdgeLabel, counts);
        label.textContent = [...new Set(entries.map(entry => entry.label))].join(' → ');
      });
      buttons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
      if (embedded) window.dispatchEvent(new Event('p06-task-change'));
    }

    function cancel() {
      clearTimeout(timer);
      timer = undefined;
      transition?.cancel();
      transition = undefined;
    }

    function schedule() {
      cancel();
      $('#toggle-play').textContent = playing ? '暂停轮播' : '继续轮播';
      if (playing && slideActive && pageActive && inView && !document.hidden) timer = setTimeout(advance, 4600);
    }

    async function advance() {
      timer = undefined;
      if (!motion.matches) {
        const exit = assignmentData.animate([{opacity: 1}, {opacity: .2}], {duration: 150, fill: 'forwards'});
        transition = exit;
        try { await exit.finished; } catch { return; }
        if (transition !== exit) return;
      }
      render((current + 1) % candidates.length);
      schedule();
      if (!motion.matches) transition = assignmentData.animate([{opacity: .2}, {opacity: 1}], {duration: 250});
    }

    function select(index) {
      playing = false;
      cancel();
      render((index + candidates.length) % candidates.length);
      schedule();
    }

    $('#previous-task').onclick = () => select(current - 1);
    $('#next-task').onclick = () => select(current + 1);
    $('#reset-task').onclick = () => select(0);
    $('#toggle-play').onclick = () => {
      playing = !playing;
      schedule();
    };
    document.addEventListener('visibilitychange', schedule);
    window.addEventListener('pagehide', () => { pageActive = false; schedule(); });
    window.addEventListener('pageshow', () => { pageActive = true; schedule(); });
    new IntersectionObserver(([entry]) => {
      if (inView === entry.isIntersecting) return;
      inView = entry.isIntersecting;
      schedule();
    }).observe(slide);

    render(0);
    if (embedded) window.p06GraphStory = {
      get taskId() { return candidates[current].task.id; },
      setTask(id) {
        const index = candidates.findIndex(({task}) => task.id === id);
        if (index !== -1 && index !== current) render(index);
      },
      setActive(active) {
        if (slideActive === active) return;
        slideActive = active;
        schedule();
      },
    };
    schedule();
  }
})();
