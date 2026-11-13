(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const slide = $('#slide');
  slide.innerHTML = `
<header class="mast">
  <p>PyPTO 3.0 <span> / </span> UX 设计 · 算法可感知</p>
  <p>P06</p>
</header>
<h1>循环执行规模直观可见</h1>
<div class="stage" aria-label="KV Cache 分块与计算规模">
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
      <div class="cache-heading">
        <h3 id="assignment-title">KV Cache · Stack / Page</h3>
        <fieldset class="stack-controls">
          <legend class="stack-control-label">Stack Count</legend>
          <div id="stack-options"></div>
        </fieldset>
        <p id="cache-tokens" aria-live="polite"></p>
      </div>
      <div id="assignment-stacks"></div>
    </section>
  </div>
  <p id="error" role="alert" hidden>计算视图未能加载。请保留本页与 Compute Explorer 的目录关系后重新打开。</p>
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
    // Task 48 supplies the existing node coordinates. The control varies
    // sequence length while keeping the Task and its compute topology fixed.
    const task = tasks[48];
    const derive = seqLen => executionAPI.derive(task.graph, model, {task: task.id, seqLen});
    const execution = derive(task.seqLen);
    const candidates = [128, 769, 1536, 2048].map(derive);
    const largest = candidates.at(-1);
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
    renderer.applyExecution(graphHost, rendered.placedNodes, nativeGraph, largest);
    const attentionRegion = graphHost.querySelector('[data-pypa-region="task-compute"]');
    const attentionBadge = attentionRegion.querySelector('span');
    attentionBadge.className = 'attention-count';
    attentionBadge.setAttribute('aria-hidden', 'true');
    const attentionLayers = largest.stacksDetail.slice(1).reverse().map(({id: index}) => {
      const layer = attentionRegion.cloneNode(false);
      layer.removeAttribute('data-pypa-region');
      layer.classList.add('attention-layer');
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
    // possible QK silhouette and badge. Use the same camera for every Stack Count.
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

    const stackHost = $('#assignment-stacks');
    const tokenFormat = new Intl.NumberFormat('zh-CN');
    stackHost.style.setProperty('--pages-per-stack', largest.pagesPerStack);
    const stackRows = largest.stacksDetail.map(({id, pages}) => {
      const row = document.createElement('div');
      row.className = 'cache-stack-row';
      row.innerHTML = `
        <div class="cache-stack-clip">
          <section class="cache-stack" aria-label="Stack ${id}">
            <header><strong>Stack ${id}</strong><span></span></header>
            <div class="cache-pages">${pages.map(page => `
              <div class="cache-page"><strong>Page ${page.id}</strong><span></span></div>
            `).join('')}</div>
          </section>
        </div>`;
      stackHost.append(row);
      return {
        row,
        tokens: row.querySelector('header span'),
        pages: [...row.querySelectorAll('.cache-page')],
      };
    });

    const buttons = candidates.map(({stacks}) => {
      const label = document.createElement('label');
      label.innerHTML = `
        <input type="radio" name="stack-count" value="${stacks}">
        <span>${stacks} ${stacks === 1 ? 'Stack' : 'Stacks'}</span>`;
      const input = label.querySelector('input');
      input.addEventListener('change', () => {
        render(stacks);
        if (embedded) window.dispatchEvent(new Event('p06-stack-change'));
      });
      $('#stack-options').append(label);
      return input;
    });

    let current;
    function render(stackCount) {
      const counts = candidates.find(candidate => candidate.stacks === stackCount);
      if (!counts || current === stackCount) return;
      current = stackCount;
      $('#cache-tokens').textContent = `${tokenFormat.format(counts.seqLen)} 有效 tokens`;
      stackRows.forEach(({row, tokens, pages}, index) => {
        const stack = counts.stacksDetail[index];
        row.classList.toggle('is-active', Boolean(stack));
        row.inert = !stack;
        if (!stack) return;

        const validTokens = stack.tokenEnd - stack.tokenStart + 1;
        tokens.textContent = `${validTokens} / ${counts.pageTokens * counts.pagesPerStack} tokens`;
        pages.forEach((cell, offset) => {
          const page = stack.pages[offset];
          cell.classList.toggle('is-empty', !page);
          cell.inert = !page;
          if (!page) return;
          const ratio = page.validTokens / counts.pageTokens;
          cell.style.setProperty('--valid-page', `${ratio * 100}%`);
          cell.lastElementChild.textContent = page.validTokens === counts.pageTokens
            ? `${page.validTokens} tokens`
            : `${page.validTokens} / ${counts.pageTokens} tokens`;
        });
      });
      const qkCount = counts.nodes.qk.count;
      $('#module-count').textContent = attentionBadge.textContent = `×${stackCount}`;
      $('#qk-count').textContent = qkBadge.textContent = `×${qkCount}`;
      attentionLayers.forEach(({layer, index}) => layer.classList.toggle('is-inactive', index >= stackCount));
      qkLayers.forEach(layer => layer.classList.toggle('is-inactive', Number(layer.dataset.pypaExecutionInstance) > qkCount));
      edgePaths.forEach(path => {
        const entries = edgeEntries(path.dataset.pypaEdge, counts);
        path.style.strokeWidth = entries.length ? String(Math.max(...entries.map(entry => entry.width))) : '';
      });
      edgeCounts.forEach(label => {
        const entries = edgeEntries(label.parentElement.dataset.pypaEdgeLabel, counts);
        label.textContent = [...new Set(entries.map(entry => entry.label))].join(' → ');
      });
      buttons.forEach(button => { button.checked = Number(button.value) === stackCount; });
    }

    render(candidates[0].stacks);
    slide.classList.add('is-ready');
    if (embedded) window.p06GraphStory = {
      get stackCount() { return current; },
      setStackCount: render,
    };
  }
})();
