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
      </div>
      <div class="graph-surface">
        <div class="graph-crop"><div id="graph" class="kf-pypa-graph__scene"></div></div>
        <div id="photo-graph-crop" class="graph-crop" hidden><div id="photo-graph" class="kf-pypa-graph__scene"></div></div>
      </div>
    </section>
    <section class="assignment viz2" aria-labelledby="assignment-title">
      <div class="p06-modes" role="group" aria-label="P6 展示模式">
        <button type="button" data-p06-photo="true" aria-pressed="false">冲印店</button>
        <button type="button" data-p06-photo="false" aria-pressed="true">算子</button>
      </div>
      <div class="cache-heading">
        <h3 id="assignment-title"><span id="block-label">Block 01</span><span id="task-kind">Tasks</span></h3>
        <fieldset class="task-controls">
          <legend id="task-legend"></legend>
          <div id="task-options"></div>
        </fieldset>
      </div>
      <div class="allocation-metrics" aria-live="polite">
        <div class="token-metric"><strong id="cache-tokens"></strong><span id="tokens-label">有效 tokens</span></div>
        <div><strong id="stack-count"></strong><span id="stacks-label">Stacks</span></div>
        <div><strong id="page-count"></strong><span id="pages-label">Pages</span></div>
        <div class="repeat-metric"><strong id="module-count"></strong><span id="module-label">Attention 计算模块</span></div>
        <div class="repeat-metric"><strong id="qk-count"></strong><span id="qk-label">QK Matmul</span></div>
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
  slide.querySelector('.mast').append(slide.querySelector('.p06-modes'));
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
    const block = 1;
    const blockLabel = String(block).padStart(2, '0');
    const blockTasks = model.tasksForBlock(block).map(id => tasks[id]);
    const candidates = blockTasks.map(({id, seqLen, graph}) => executionAPI.derive(graph, model, {task: id, seqLen}));
    const largest = candidates.reduce((max, candidate) => candidate.pages > max.pages ? candidate : max);
    // Keep the camera and dependency layout stable as Tasks change.
    const {graph} = tasks[largest.task];
    const graphLayout = window.PTO_SEMANTIC_COMPUTATION_GRAPH;
    const options = {flowLayout: true, alignSpineFront: true, overviewLayout: false, compactGlyphs: true, rootLayerGap: 64, childLayerGap: 56};
    const baseGraph = graphLayout.materialize(graph, new Set(), options);
    const executionFootprints = Object.fromEntries(baseGraph.nodes.flatMap(node => {
      const count = executionAPI.decoration(node, largest)?.count;
      return count ? [[node.id, executionAPI.deckOffset(count)]] : [];
    }));
    const nativeGraph = graphLayout.materialize(graph, new Set(), {...options, executionFootprints});
    const renderer = window.PtoStudioComputationGraph;
    const rendered = renderer.renderScene(graphHost, nativeGraph, {
      view: 'compute', allowExpand: false, synchronizations: true,
      markerPrefix: 'p06-story', cardHtml: node => renderer.cardHtml(node, graph, {}),
    });
    renderer.applyExecution(graphHost, rendered.placedNodes, nativeGraph, largest);
    const attentionRegion = graphHost.querySelector('[data-pypa-region="task-compute"]');
    const attentionBadge = attentionRegion.querySelector('span');
    attentionBadge.className = 'attention-count';
    attentionBadge.setAttribute('aria-hidden', 'true');
    function createBatchLayers(region) {
      return largest.stacksDetail.slice(1).reverse().map(({id: index}) => {
        const layer = region.cloneNode(false);
        layer.removeAttribute('data-pypa-region');
        layer.classList.add('attention-layer');
        layer.style.transform = `translate(${index * 10}px, ${index * 8}px)`;
        layer.setAttribute('aria-hidden', 'true');
        region.before(layer);
        return {layer, index};
      });
    }
    const attentionLayers = createBatchLayers(attentionRegion);
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
    // Keep input tensors and the computation visible at a stable scale across Tasks.
    const focus = rendered.placedNodes.find(node => node.id === 'qk');
    const focusX = focus.x - 560;
    const focusY = focus.y - 230;
    const scale = .82;
    graphCrop.style.width = '100%';
    graphCrop.style.height = '100%';
    graphHost.style.transform = `translate(${-focusX * scale + 60}px, ${-focusY * scale}px) scale(${scale})`;

    const photoHost = $('#photo-graph');
    const photoCrop = $('#photo-graph-crop');
    const photoAPI = window.PtoPhotoNarrativeGraph;
    let photoGraph, photoLayout, photoBadge, photoLayers;
    function renderPhotoGraph(task, stackCount) {
      if (!photoLayout) {
        const projection = photoAPI.project(graph, {task: tasks[largest.task]});
        photoGraph = graphLayout.materialize(projection, new Set(), {
          ...options, rootLayerGap: 60,
          executionFootprints: {qk: executionAPI.deckOffset(largest.pages)},
        });
        photoLayout = renderer.renderScene(photoHost, photoGraph, {
          view: 'compute', allowExpand: false, synchronizations: false, compactNodes: true, markerPrefix: 'p06-photo',
        });
        const photoRegion = photoHost.querySelector('[data-pypa-region="task-compute"]');
        photoLayers = createBatchLayers(photoRegion);
        photoBadge = photoRegion.querySelector('span');
        photoBadge.className = 'attention-count';
        photoBadge.setAttribute('aria-hidden', 'true');
        const focus = photoLayout.placedNodes.find(node => node.id === 'qk');
        photoCrop.style.width = '100%';
        photoCrop.style.height = '100%';
        photoHost.style.transform = `translate(${-(focus.x - 560) * scale + 60}px, ${-(focus.y - 230) * scale}px) scale(${scale})`;
        photoHost.querySelectorAll('[data-pypa-node]').forEach(node => { node.disabled = true; });
      }
      photoBadge.textContent = `×${stackCount}`;
      photoLayers.forEach(({layer, index}) => layer.classList.toggle('is-inactive', index >= stackCount));
      renderer.applyExecution(photoHost, photoLayout.placedNodes, photoGraph, photoAPI.executionData(task));
      photoHost.querySelectorAll('[data-pypa-execution-stack],[data-pypa-execution-count]').forEach(layer => {
        layer.removeAttribute('role');
        layer.removeAttribute('tabindex');
        layer.removeAttribute('title');
        layer.setAttribute('aria-hidden', 'true');
      });
    }

    const stackHost = $('#assignment-stacks');
    stackHost.style.setProperty('--pages-per-stack', largest.pagesPerStack);
    const stackRows = largest.stacksDetail.map(({id}) => {
      const row = document.createElement('div');
      row.className = 'cache-stack-row';
      row.innerHTML = `
        <div class="cache-stack-clip">
          <section class="cache-stack" aria-label="Stack ${id}">
            <header><strong>Stack ${id}</strong><span></span></header>
            <div class="cache-pages">${'<div class="cache-page"><strong></strong><span></span></div>'.repeat(largest.pagesPerStack)}</div>
          </section>
        </div>`;
      stackHost.append(row);
      return {
        row,
        tokens: row.querySelector('header span'),
        pages: [...row.querySelectorAll('.cache-page')],
      };
    });

    const buttons = blockTasks.map(({id}) => {
      const label = document.createElement('label');
      label.innerHTML = `
        <input type="radio" name="p06-task" value="${id}">
        <span>T${String(id).padStart(2, '0')}</span>`;
      const input = label.querySelector('input');
      input.addEventListener('change', () => select(id));
      $('#task-options').append(label);
      return input;
    });

    const modeButtons = [...slide.querySelectorAll('[data-p06-photo]')];
    modeButtons.forEach(button => {
      button.onclick = () => select(current, button.dataset.p06Photo === 'true');
    });
    function select(taskId, photo = photoMode) {
      if (render(taskId, photo) && embedded) window.dispatchEvent(new Event('p06-state-change'));
    }

    let current;
    let photoMode = false;
    function render(taskId, photo = photoMode) {
      const counts = candidates.find(candidate => candidate.task === taskId);
      if (!counts || (current === taskId && photoMode === photo)) return false;
      current = taskId;
      photoMode = photo;
      const {stacks: stackCount, pages: pageCount} = counts;
      const unitLabel = photo ? '张照片' : 'tokens';
      $('#task-legend').textContent = photo ? `工作小组 ${blockLabel} 工单选择` : `Block ${blockLabel} Task 选择`;
      $('#block-label').textContent = photo ? `小组 ${blockLabel}` : `Block ${blockLabel}`;
      $('#task-kind').textContent = photo ? '工单' : 'Tasks';
      $('.graph-heading h3').textContent = photo ? '冲印店 · 照片加工流程' : 'Graph · QK Matmul 特写';
      $('#module-label').textContent = photo ? '处理批次' : 'Attention 计算模块';
      $('#qk-label').textContent = photo ? '筛选去重' : 'QK Matmul';
      modeButtons.forEach(button => button.setAttribute('aria-pressed', String((button.dataset.p06Photo === 'true') === photo)));
      graphCrop.hidden = photo;
      photoCrop.hidden = !photo;
      if (photo) renderPhotoGraph(tasks[taskId], stackCount);
      $('#cache-tokens').textContent = String(counts.seqLen);
      $('#stack-count').textContent = stackCount;
      $('#page-count').textContent = pageCount;
      $('#tokens-label').textContent = photo ? '张照片' : 'tokens';
      $('.assignment').classList.toggle('is-dense', stackCount > 6);
      $('#stacks-label').textContent = photo ? '处理批次' : 'stacks';
      $('#pages-label').textContent = photo ? '标准照片包' : 'pages';
      stackRows.forEach(({row, tokens, pages}, index) => {
        const stack = counts.stacksDetail[index];
        row.classList.toggle('is-active', Boolean(stack));
        row.inert = !stack;
        if (!stack) return;

        const validTokens = stack.tokenEnd - stack.tokenStart + 1;
        const stackLabel = photo ? `第${stack.id + 1}批` : `Stack${stack.id}`;
        row.querySelector('header strong').textContent = stackLabel;
        row.querySelector('section').setAttribute('aria-label', stackLabel);
        tokens.textContent = `${validTokens} / ${counts.pageTokens * counts.pagesPerStack}`;
        pages.forEach((cell, offset) => {
          const page = stack.pages[offset];
          cell.classList.toggle('is-empty', !page);
          cell.inert = !page;
          cell.classList.toggle('is-partial', Boolean(page && page.validTokens < counts.pageTokens));
          if (!page) {
            cell.firstElementChild.textContent = '';
            cell.lastElementChild.textContent = '';
            cell.removeAttribute('aria-label');
            cell.style.setProperty('--valid-page', '0%');
            return;
          }
          const ratio = page.validTokens / counts.pageTokens;
          cell.style.setProperty('--valid-page', `${ratio * 100}%`);
          cell.firstElementChild.textContent = photo ? '包' : 'P';
          cell.lastElementChild.textContent = '';
          cell.setAttribute('aria-label', `${photo ? '照片包' : 'Page'} ${photo ? page.id + 1 : page.id}，${page.validTokens} / ${counts.pageTokens} ${unitLabel}`);
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
      buttons.forEach(button => {
        button.checked = Number(button.value) === taskId;
        button.nextElementSibling.textContent = `T${button.value.padStart(2, '0')}`;
        button.setAttribute('aria-label', `${photo ? '工单' : 'Task'} ${button.value}`);
      });
      if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const activeBadge = photo ? photoHost.querySelector('[data-pypa-execution-count="qk"]') : qkBadge;
        activeBadge?.animate([{opacity: .45}, {opacity: 1}], {duration: 240, easing: 'ease-out'});
        $('.allocation-metrics').animate([{opacity: .7}, {opacity: 1}], {duration: 180, easing: 'ease-out'});
      }
      return true;
    }

    render(candidates[0].task);
    slide.classList.add('is-ready');
    if (embedded) window.p06GraphStory = {
      get taskId() { return current; },
      get photoMode() { return photoMode; },
      setState: render,
      advance(direction) {
        const index = candidates.findIndex(candidate => candidate.task === current);
        const next = candidates[index + direction];
        if (!next) return false;
        select(next.task);
        return true;
      },
    };
  }
})();
