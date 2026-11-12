(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const slide = $('#slide');
  const graphHost = $('#graph');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
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
    const nativeGraph = window.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(
      task.graph, new Set(), {flowLayout: true, compactGlyphs: true}
    );
    const renderer = window.PtoStudioComputationGraph;
    const overviewPositions = {
      q: [16, 86, 148], 'k-cache': [16, 187, 148], qk: [199, 129, 158],
      scores: [384, 139, 125], 'scale-mask': [532, 129, 173],
      softmax: [725, 129, 186], probability: [936, 139, 142],
      'v-cache': [1132, 106, 155], pv: [1138, 242, 165], partial: [1051, 373, 219],
      'context-acc': [803, 383, 225], 'output-norm': [422, 508, 212], output: [166, 498, 198],
      m: [576, 244, 132], l: [755, 244, 111], rescale: [911, 274, 145], o: [660, 372, 126],
    };
    const stages = [
      {label: '1 个 Stack', title: '从 Stack 0 开始', caption: '跟着 Task 48 的第一个 Stack，看数据怎样进入计算。'},
      {label: '4 个 Page', title: '一个 Stack 装入四个 Page', caption: '依次展开 P0–P3。每个 Page 给 PV 提供一份 Probability 和 V Cache 数据。'},
      {label: '逐 Page 计算', title: '四个 Page，四次矩阵乘贡献', caption: '首个 Page 用 Matmul。后续三个用 Matmul Accumulate，每次把新贡献加到前一个部分结果。'},
      {label: '累加结果', title: '四份贡献汇成一个 Stack 结果', caption: '最后一份部分结果是 Stack 0 的 Partial Context，再交给 Context Accumulate。'},
      {label: '2 个 Stack', title: '换一组 Page，重复同一条计算链', caption: 'Stack 1 使用 P4–P7。前一 Stack 的 Context 重标定后，再与这次的 Partial Context 合并。'},
      {label: '展开 Graph', title: '沿着已看过的计算，长成完整 Graph', caption: '先找回 PV 和 Context Accumulate，再展开 Task 的其余 Stack、上游计算和跨 Stack 状态。'},
      {label: '完整回看', title: '数据分块决定重复计算规模', caption: '同一条 PV 规则覆盖全部 24 个 Page，每个 Stack 汇出一份 Partial Context。'},
    ];
    const captions = {
      pv: '保留刚才的 PV → Partial Context → Context Accumulate，把两个 Stack 收进同一张图。',
      stacks: '其余 Stack 沿用同一套规则。依次加入，每组仍有自己的 Page。',
      inputs: '沿 Probability 向前补齐 QK、Scaling + Mask 和 Online Softmax。',
      output: '补齐跨 Stack 的状态传递。所有 Stack 处理完后，才做 Output Normalize。',
    };
    const steps = [];
    let step = 0;
    let frame = 0;
    let timer;
    let playback = false;
    let paused = false;
    let movingNode;

    const instances = graphAPI.pageInstances(task, 'pv');
    const tracks = instances.slice(0, 2).map(({index, pages}) => {
      const section = document.createElement('section');
      section.className = 'track';
      section.setAttribute('aria-label', `Stack ${index} 的 PV 计算`);
      section.innerHTML = `
        <div class="track-label reveal"><strong>Stack ${index}</strong><small>${pages.length} Pages · ${task.stacks[index].tokens} tokens</small></div>
        <div class="page-flow">${pages.map((page, ordinal) => `
          <div class="page-work">
            <div class="page-tile reveal"><b>P${page.index}</b><small>${page.tokens} tokens</small></div>
            <i class="feed reveal" aria-hidden="true"></i>
            <div class="operation reveal">${page.api === 'pl.matmul' ? 'Matmul' : 'Matmul Accumulate'}</div>
            <div class="partial reveal">${ordinal === pages.length - 1 ? `Partial Context S${index}` : `部分结果 ${ordinal + 1}`}</div>
          </div>`).join('')}</div>
        <div class="result reveal">
          <span>Partial Context S${index} →</span>
          <strong>Context Accumulate</strong>
          <small>${index === 0 ? '累加初值为 0，得到 Context S0' : '重标定 Context S0 + Partial Context S1'}</small>
        </div>`;
      $('#process').append(section);
      return {
        section,
        label: section.querySelector('.track-label'),
        pages: [...section.querySelectorAll('.page-tile')],
        operations: [...section.querySelectorAll('.operation')],
        results: [...section.querySelectorAll('.partial')],
        feeds: [...section.querySelectorAll('.feed')],
        context: section.querySelector('.result'),
      };
    });
    const stackChips = instances.map(({index, pages}) => {
      const chip = document.createElement('div');
      chip.className = 'stack-chip reveal';
      chip.innerHTML = `<b>Stack ${index}</b><span>P${pages[0].index}–P${pages.at(-1).index}</span>`;
      $('#stack-strip').append(chip);
      return chip;
    });

    const storyGraph = {
      ...nativeGraph,
      nodes: nativeGraph.nodes.map(node => {
        const [x, y, w] = overviewPositions[node.id];
        return {...node, x, y: 12 + y * .78, w};
      }),
      groups: [{id: 'task-compute', label: '循环内计算与跨 Stack 状态', x: 178, y: 53, w: 1190, h: 334}],
      edgeRoutes: null,
      width: 1480,
      height: 525,
    };
    // The native execution silhouettes read the laid-out node dimensions.
    $('#overview').hidden = false;
    const rendered = renderer.renderScene(graphHost, storyGraph, {
      view: 'compute', allowExpand: false, synchronizations: false,
      compactNodes: true, markerPrefix: 'p06-story',
    });
    renderer.applyExecution(graphHost, rendered.placedNodes, storyGraph, execution);
    const anchor = graphHost.querySelector('[data-pypa-node="pv"]');
    const pvPath = new Set(['pv', 'partial', 'context-acc', 'probability', 'v-cache']);
    const statePath = new Set(['m', 'l', 'rescale', 'o', 'output-norm', 'output']);
    const part = id => pvPath.has(id) ? 1 : statePath.has(id) ? 3 : 2;
    graphHost.querySelectorAll('[data-pypa-node]').forEach(node => {
      node.disabled = true;
      node.dataset.graphPart = part(node.dataset.pypaNode);
    });
    graphHost.querySelectorAll('[data-pypa-execution-stack],[data-pypa-execution-count]').forEach(layer => {
      layer.removeAttribute('role');
      layer.removeAttribute('tabindex');
      layer.setAttribute('aria-hidden', 'true');
      if (layer.dataset.pypaExecutionStack) layer.dataset.graphPart = part(layer.dataset.pypaExecutionStack);
    });
    graphHost.querySelectorAll('[data-pypa-edge]').forEach(edge => {
      edge.dataset.graphPart = Math.max(part(edge.dataset.edgeSource), part(edge.dataset.edgeTarget));
    });
    graphHost.querySelector('.kf-pypa-graph__group').dataset.graphPart = '3';
    const graphParts = [...graphHost.querySelectorAll('[data-graph-part]')];
    const scopeCounts = ['pv', 'context-acc'].map(id => ({
      id,
      badge: graphHost.querySelector(`[data-pypa-execution-count="${id}"]`),
      layers: [...graphHost.querySelectorAll(`[data-pypa-execution-stack="${id}"] [data-pypa-execution-instance]`)],
    }));
    const pageTotals = instances.map((_, index) =>
      instances.slice(0, index + 1).reduce((total, stack) => total + stack.pages.length, 0)
    );

    // Cumulative frames let forward, back, pause and replay use the same state.
    const state = (first, second = [0, 0, 0], graph = 0, stacks = 0, caption = '') =>
      ({tracks: [first, second], graph, stacks, caption});
    const empty = [0, 0, 0];
    const first = [4, 4, 1];
    steps.push(
      [state(empty)],
      Array.from({length: 4}, (_, i) => state([i + 1, 0, 0])),
      Array.from({length: 4}, (_, i) => state([4, i + 1, 0])),
      [state(first)],
      [
        state(first, empty),
        ...Array.from({length: 4}, (_, i) => state(first, [i + 1, 0, 0])),
        ...Array.from({length: 4}, (_, i) => state(first, [4, i + 1, 0])),
        state(first, first),
      ],
      [
        state(first, first),
        state(first, first, 1, 2, captions.pv),
        ...instances.slice(2).map((_, i) => state(first, first, 1, i + 3, captions.stacks)),
        state(first, first, 2, task.stackCount, captions.inputs),
        state(first, first, 3, task.stackCount, captions.output),
      ],
      [state(first, first, 3, task.stackCount)],
    );

    $('#task-context').textContent = `Task ${task.id} · Batch ${task.batch} · KV Head ${task.kvHead}`;
    const sceneButtons = stages.map(({label}, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'btn btn-ghost';
      button.textContent = label;
      button.onclick = () => navigate(index);
      $('#scenes').append(button);
      return button;
    });

    function stop() {
      clearTimeout(timer);
      timer = undefined;
    }

    function navigate(index, settled = false) {
      stop();
      playback = paused = false;
      movingNode?.cancel();
      step = Math.max(0, Math.min(steps.length - 1, index));
      frame = settled ? steps[step].length - 1 : 0;
      render();
      schedule();
    }

    function play() {
      if (timer !== undefined) {
        stop();
        paused = true;
      } else if (paused) {
        paused = false;
        schedule();
      } else {
        navigate(0);
        playback = true;
        schedule();
      }
      updateControls();
    }

    function schedule() {
      stop();
      const hasFrame = frame < steps[step].length - 1;
      if (paused || (!hasFrame && (!playback || step === steps.length - 1))) {
        if (!hasFrame) playback = false;
        updateControls();
        return;
      }
      const wait = step === 5 ? (frame === 0 ? 1600 : 1450)
        : step === 4 ? (hasFrame ? 940 : 1900)
        : hasFrame ? 980 : 1800;
      timer = setTimeout(() => {
        timer = undefined;
        if (hasFrame) frame++;
        else {
          step++;
          frame = 0;
        }
        render();
        schedule();
      }, wait);
      updateControls();
    }

    function updateControls() {
      $('#play').textContent = timer !== undefined ? '暂停' : paused ? '继续' : '完整慢放';
      $('#previous').disabled = step === 0;
      $('#next').disabled = step === steps.length - 1;
      sceneButtons.forEach((button, index) => {
        if (index === step) button.setAttribute('aria-current', 'step');
        else button.removeAttribute('aria-current');
      });
    }

    function render() {
      const current = steps[step][frame];
      const overview = $('#overview');
      const wasLocal = overview.hidden;
      const localAnchor = wasLocal && current.graph && step === 5 && !motion.matches
        ? tracks[0].operations.at(-1).getBoundingClientRect() : null;
      $('#process').hidden = Boolean(current.graph);
      overview.hidden = !current.graph;
      $('#scene-title').textContent = stages[step].title;
      $('#scene-caption').textContent = current.caption || stages[step].caption;
      tracks.forEach((track, index) => {
        const [pageCount, operations, context] = current.tracks[index];
        track.section.hidden = index === 1 && step < 4;
        track.label.classList.toggle('visible', !track.section.hidden);
        track.pages.forEach((page, i) => page.classList.toggle('visible', i < pageCount));
        [track.operations, track.results, track.feeds].forEach(items => {
          items.forEach((item, i) => item.classList.toggle('visible', i < operations));
        });
        track.context.classList.toggle('visible', Boolean(context));
      });
      stackChips.forEach((chip, index) => chip.classList.toggle('visible', index < current.stacks));
      graphParts.forEach(element => element.classList.toggle('graph-waiting', Number(element.dataset.graphPart) > current.graph));
      if (current.graph) {
        scopeCounts.forEach(({id, badge, layers}) => {
          const count = id === 'pv' ? pageTotals[current.stacks - 1] : current.stacks;
          badge.textContent = `×${count}`;
          layers.forEach(layer => { layer.hidden = Number(layer.dataset.pypaExecutionInstance) > count; });
        });
      }
      anchor.classList.toggle('graph-anchor', step === 5);
      if (localAnchor?.width) {
        const box = anchor.getBoundingClientRect();
        const scale = slide.getBoundingClientRect().width / slide.offsetWidth;
        movingNode = anchor.animate([
          {translate: `${(localAnchor.left - box.left) / scale}px ${(localAnchor.top - box.top) / scale}px`},
          {translate: '0 0'},
        ], {duration: 1050, easing: 'cubic-bezier(.2,.75,.3,1)'});
      }
      $('#story-note').textContent = step === 6
        ? `${task.seqLen.toLocaleString('zh-CN')} tokens · ${task.stackCount} Stacks · ${task.pageCount} Pages · PV ${execution.pvMatmul} 次 Matmul + ${execution.pvMatmulAcc} 次 Matmul Accumulate / AIC / Task`
        : '源码推导的逻辑计算 · 动画节奏用于讲述';
      slide.dataset.step = step;
      slide.dataset.frame = frame;
      updateControls();
    }

    $('#previous').onclick = () => navigate(step - 1, true);
    $('#next').onclick = () => navigate(step + 1);
    $('#reset').onclick = () => navigate(0);
    $('#play').onclick = play;
    document.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) return;
      if (event.key === 'ArrowLeft' && step > 0) navigate(step - 1, true);
      else if (event.key === 'ArrowRight' && step < steps.length - 1) navigate(step + 1);
      else return;
      event.preventDefault();
    });
    navigate(0);
  }
})();
