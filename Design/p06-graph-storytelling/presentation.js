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
          <span><span id="module-label">Attention 计算模块</span> <strong id="module-count"></strong></span>
          <span><span id="qk-label">QK Matmul</span> <strong id="qk-count"></strong></span>
        </div>
      </div>
      <div class="graph-surface">
        <div class="graph-crop"><div id="graph" class="kf-pypa-graph__scene"></div></div>
        <div id="photo-graph-crop" class="graph-crop" hidden><div id="photo-graph" class="kf-pypa-graph__scene"></div></div>
      </div>
    </section>
    <section class="assignment viz2" aria-labelledby="assignment-title">
      <div class="cache-heading">
        <div class="cache-title">
          <h3 id="assignment-title">KV Cache · Stack / Page</h3>
          <div class="p06-modes" role="group" aria-label="P6 展示模式">
            <button type="button" data-p06-photo="true" aria-pressed="false">冲印店</button>
            <button type="button" data-p06-photo="false" aria-pressed="true">算子</button>
          </div>
        </div>
        <fieldset class="task-controls">
          <legend id="task-legend"></legend>
          <div id="task-options"></div>
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
    const block = 1;
    const blockLabel = String(block).padStart(2, '0');
    const blockTasks = model.tasksForBlock(block).map(id => tasks[id]);
    const candidates = blockTasks.map(({id, seqLen, graph}) => executionAPI.derive(graph, model, {task: id, seqLen}));
    const largest = candidates.reduce((max, candidate) => candidate.pages > max.pages ? candidate : max);
    // Keep the camera and dependency layout stable as Tasks change.
    const {graph} = tasks[largest.task];
    const graphLayout = window.PTO_SEMANTIC_COMPUTATION_GRAPH;
    const options = {flowLayout: true, overviewLayout: false, compactGlyphs: true, rootLayerGap: 64, childLayerGap: 56};
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
    const surface = graphCrop.parentElement;
    // Measure the native layout before transforming, including the largest
    // possible QK silhouette and badge. Use the same camera for every Task.
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
    const scale = Math.min(0.65, surface.clientWidth / (right - left), surface.clientHeight / (bottom - top));
    graphCrop.style.width = `${(right - left) * scale}px`;
    graphCrop.style.height = `${(bottom - top) * scale}px`;
    graphHost.style.transform = `translate(${-left * scale}px, ${-top * scale}px) scale(${scale})`;

    const photoHost = $('#photo-graph');
    const photoCrop = $('#photo-graph-crop');
    const photoAPI = window.PtoPhotoNarrativeGraph;
    let photoGraph, photoLayout, photoBadge, photoLayers;
    function renderPhotoGraph(task, stackCount) {
      if (!photoLayout) {
        const projection = photoAPI.project(graph, {task: tasks[largest.task]});
        photoGraph = graphLayout.materialize(projection, new Set(), {
          ...options, rootLayerGap: 60, alignSpineFront: true,
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
        const {sceneWidth, sceneHeight} = photoLayout;
        const depth = largest.stacks - 1;
        const width = Math.max(sceneWidth, photoRegion.offsetLeft + photoRegion.offsetWidth + depth * 10 + 1);
        const height = Math.max(sceneHeight, photoRegion.offsetTop + photoRegion.offsetHeight + depth * 8 + 1);
        const fit = Math.min(1, surface.clientWidth / width, surface.clientHeight / height);
        photoCrop.style.width = `${width * fit}px`;
        photoCrop.style.height = `${height * fit}px`;
        photoHost.style.transform = `scale(${fit})`;
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

    const buttons = blockTasks.map(({id}) => {
      const label = document.createElement('label');
      label.innerHTML = `
        <input type="radio" name="p06-task" value="${id}">
        <span>Task ${id}</span>`;
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
      const quantity = tokenFormat.format(counts.seqLen);
      const unitLabel = photo ? '张照片' : 'tokens';
      $('#task-legend').textContent = photo ? `工作小组 ${blockLabel} 认领的工单` : `Block ${blockLabel} 认领的 Task`;
      $('#assignment-title').textContent = photo ? '顾客照片库 · 批次 / 照片包' : 'KV Cache · Stack / Page';
      $('.graph-heading h3').textContent = photo ? '冲印店 · 照片加工流程' : 'Graph · QK Matmul 特写';
      $('#module-label').textContent = photo ? '处理批次' : 'Attention 计算模块';
      $('#qk-label').textContent = photo ? '筛选去重' : 'QK Matmul';
      modeButtons.forEach(button => button.setAttribute('aria-pressed', String((button.dataset.p06Photo === 'true') === photo)));
      graphCrop.hidden = photo;
      photoCrop.hidden = !photo;
      if (photo) renderPhotoGraph(tasks[taskId], stackCount);
      $('#cache-tokens').textContent = photo
        ? `${stackCount} 个处理批次 · ${pageCount} 个标准照片包 · ${quantity} 张照片`
        : `${stackCount} ${stackCount === 1 ? 'Stack' : 'Stacks'} · ${pageCount} ${pageCount === 1 ? 'Page' : 'Pages'} · ${quantity} 有效 tokens`;
      stackRows.forEach(({row, tokens, pages}, index) => {
        const stack = counts.stacksDetail[index];
        row.classList.toggle('is-active', Boolean(stack));
        row.inert = !stack;
        if (!stack) return;

        const validTokens = stack.tokenEnd - stack.tokenStart + 1;
        const stackLabel = photo ? `第 ${stack.id + 1} 批` : `Stack ${stack.id}`;
        row.querySelector('header strong').textContent = stackLabel;
        row.querySelector('section').setAttribute('aria-label', stackLabel);
        tokens.textContent = `${validTokens} / ${counts.pageTokens * counts.pagesPerStack} ${unitLabel}`;
        pages.forEach((cell, offset) => {
          const page = stack.pages[offset];
          cell.classList.toggle('is-empty', !page);
          cell.inert = !page;
          if (!page) return;
          const ratio = page.validTokens / counts.pageTokens;
          cell.style.setProperty('--valid-page', `${ratio * 100}%`);
          cell.firstElementChild.textContent = photo ? `照片包 ${page.id + 1}` : `Page ${page.id}`;
          cell.lastElementChild.textContent = page.validTokens === counts.pageTokens
            ? `${page.validTokens} ${unitLabel}`
            : `${page.validTokens} / ${counts.pageTokens} ${unitLabel}`;
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
        button.nextElementSibling.textContent = photo ? `工单 ${button.value}` : `Task ${button.value}`;
      });
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
