(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const svgNS = 'http://www.w3.org/2000/svg';
  const clamp = value => Math.max(0, Math.min(1, value));
  const mix = (a, b, progress) => a + (b - a) * progress;
  const ramp = (value, from, to) => clamp((value - from) / (to - from));
  const smooth = value => value * value * (3 - 2 * value);
  const motion = matchMedia('(prefers-reduced-motion: reduce)');

  const screen = $('#screen');
  const story = $('#story');
  new ResizeObserver(() => {
    const scale = Math.min(screen.clientWidth / 1600, screen.clientHeight / 900);
    story.style.setProperty('--story-scale', scale);
    story.style.left = (screen.clientWidth - 1600 * scale) / 2 + 'px';
    story.style.top = (screen.clientHeight - 900 * scale) / 2 + 'px';
  }).observe(screen);

  try {
    mount();
  } catch (error) {
    $('#error').hidden = false;
    story.querySelectorAll('footer button').forEach(button => { button.disabled = true; });
    console.error(error);
  }

  function mount() {
    const id = 48;
    const model = window.PtoPagedAttentionIntro.model;
    const sourceGraph = window.PTO_PYPA_SEMANTIC_GRAPH;
    const constant = key => sourceGraph.constants[key].value;
    const anatomy = window.PtoVisualization2Anatomy.createModel({
      seqLens: model.defaultSeqLens, blocks: constant('ATTN_SPMD_BLOCKS'),
      numHeads: constant('NUM_HEADS'), kvHeads: constant('NUM_KV_HEADS'),
      headDim: constant('HEAD_DIM'), rowTile: constant('ROW_TILE'),
      pageTokens: constant('BLOCK_SIZE'), pagesPerStack: constant('STACK_PAGES'),
      transferSlots: constant('TRANSFER_SLOTS'),
    });
    const preLaunch = constant('PRE_LAUNCH');
    const task = window.PtoVisualization2GraphIntra.createModel({
      anatomy, source: window.PTO_PAGED_ATTENTION_PYPTO_SOURCE, preLaunch,
    }).tasks[id];
    const pipelineAPI = window.PtoVisualization2PipelineIntra;
    const pipeline = pipelineAPI.createModel({anatomy, preLaunch});
    const timeline = pipeline.tasks[id];
    const assignmentTask = model.taskInfo(id);
    const selection = {kind: 'task', task: id, block: assignmentTask.block};
    const tokens = new Intl.NumberFormat('zh-CN').format(task.seqLen);
    const stack = timeline.stacks[0];

    const assignmentHost = $('#assignment');
    const assignment = window.PtoAssignmentFocus.mount(assignmentHost, {
      model, title: false,
      onSelect: selection => { if (selection.task === id && !aim) go(1); },
    });
    assignment.update();
    const matrix = assignmentHost.querySelector('[data-focus-fragment="matrix"]');
    const cell = matrix.querySelector('[data-pa-task="48"]');
    matrix.querySelectorAll('[data-pa-task]:not([data-pa-task="48"])').forEach(node => {
      node.removeAttribute('role');
      node.removeAttribute('tabindex');
    });
    const matrixCell = localRect(cell.querySelector('.kf-pa-map__cell'));
    const blockDetail = $('#block-detail');
    buildAssignment(anatomy, id);
    const detailRect = localRect(blockDetail);
    const localCell = localRect($('#local-task'));
    const graphParts = buildGraph(task, model);
    const pipelineParts = buildPipeline(pipelineAPI, pipeline, timeline);
    const objectMorphs = collectMorphs(task, localRect, graphParts, pipelineParts);
    const anchor = $('#anchor');
    const anchorRect = anchor.querySelector('rect');
    const anchorLabel = anchor.querySelector('text');
    const ghosts = Array.from({length: 4}, () => {
      const node = document.createElementNS(svgNS, 'rect');
      node.setAttribute('rx', '6');
      $('#ghosts').append(node);
      return node;
    });

    const taskContext = `Task ${id}  /  Batch ${task.batch}  /  KV Head ${task.kvHead}  /  Block ${String(task.block).padStart(2, '0')}`;
    const partition = `${tokens} tokens · ${task.pageCount} Pages · ${task.stackCount} Stacks`;
    const beats = [
      {at: 0, title: '在 Assignment 中找到 Task 48',
        caption: `${model.defaultSeqLens.length} 个 Batch × ${anatomy.kvHeads} 个 KV Head。点选矩阵中的 48，或按下一步。`},
      {at: 0, local: true, title: '进入 Block 内的 Task 48',
        caption: `Q Tile 与 KV Cache · ${partition}`},
      {at: -.76, title: '向左展开计算依赖',
        caption: 'Q Tile 对齐计算图的 Q 输入。KV Cache 中每个逻辑 Page 对应一次 QK Matmul。'},
      {at: -1, title: '展开 Page 与 Stack 的重复层',
        caption: `QK 与 PV 各覆盖 ${task.pageCount} 个 Page；Online Softmax 和 Context Accumulate 在每条 AIV Lane 各执行 ${task.stackCount} 个 Stack。`},
      {at: 0, local: true, title: '回到 Assignment 的 Block 内',
        caption: `Batch ${task.batch} × KV Head ${task.kvHead} · ${partition}`},
      {at: .76, title: '向右展开逻辑执行顺序',
        caption: `每个 Stack 对齐自己的 QK 起点。Stack 0 从 T${stack.produceTick} 开始生产，在 T${stack.consumeTick} 进入 PV 与累加。`},
      {at: 1, title: '看见阶段交接与资源复用',
        caption: `Stack 0 的 PV 等待 SOFTMAX_READY_EVENT；每 ${anatomy.transferSlots} 个 Stack 循环复用 Transfer Slot。`},
      {at: 0, title: 'Task 48 回到整个任务空间',
        caption: '同一个 Task，沿任务坐标找到它，沿计算依赖理解它，沿逻辑轮次跟踪它。'},
    ];
    let step = 0;
    let camera = {branch: 0, zoom: 0};
    let aim = 0;
    let frame = 0;
    let selected = false;

    function localRect(node) {
      const rect = node.getBoundingClientRect();
      const origin = $('#stage').getBoundingClientRect();
      const scale = story.getBoundingClientRect().width / 1600;
      return {
        x: (rect.left - origin.left) / scale - 1, y: (rect.top - origin.top) / scale - 1,
        width: rect.width / scale, height: rect.height / scale,
      };
    }

    function setSelection(next, overview = false) {
      const focus = next && !overview ? 'task' : 'overall';
      if (next === selected && assignmentHost.dataset.focus === focus) return;
      selected = next;
      assignment.update({selection: next ? selection : null, focus});
      matrix.querySelectorAll('[data-pa-task]:not([data-pa-task="48"])').forEach(node => {
        node.setAttribute('tabindex', '-1');
      });
      cell.setAttribute('tabindex', '0');
    }
    cell.setAttribute('tabindex', '0');

    function identityBox(progress, direction) {
      const travel = smooth(ramp(progress, 0, .61));
      return {
        x: mix(localCell.x, direction < 0 ? 28 : 1300, travel),
        y: mix(localCell.y, 20, travel) - 27 * Math.sin(Math.PI * travel),
        width: mix(localCell.width, 150, travel),
        height: mix(localCell.height, 40, travel),
      };
    }

    function box(element, geometry) {
      Object.entries(geometry).forEach(([key, value]) => element.setAttribute(key, value));
    }

    function paint(currentCamera, moving) {
      camera = currentCamera;
      const {branch: value, zoom} = camera;
      const progress = Math.abs(value);
      const direction = Math.sign(value);
      const stage = $('#stage');
      const view = progress < .40 ? 'assignment' : direction < 0 ? 'graph' : 'pipeline';
      stage.dataset.view = view;
      const matrixExit = smooth(ramp(progress, 0, .56));
      matrix.style.transform = `translateX(${-direction * 160 * matrixExit}px)`;
      matrix.style.opacity = (1 - zoom) * (1 - ramp(progress, .02, .48));
      // Open the native Block detail around the selected matrix cell.
      const scale = mix(.38, 1, zoom);
      const dx = mix(matrixCell.x - localCell.x, 0, zoom);
      const dy = mix(matrixCell.y - localCell.y, 0, zoom);
      blockDetail.style.transformOrigin = `${localCell.x - detailRect.x}px ${localCell.y - detailRect.y}px`;
      blockDetail.style.transform = `translate(${dx}px, ${dy}px) scale(${scale})`;
      blockDetail.style.opacity = ramp(zoom, .12, .86) * (1 - ramp(progress, .10, .55));
      blockDetail.hidden = zoom === 0 || progress >= .55;
      assignmentHost.inert = zoom !== 0 || aim !== 0;
      assignmentHost.hidden = zoom === 1;

      graphParts.host.hidden = value >= -.22;
      if (!graphParts.host.hidden) paintGraph(graphParts, progress);
      pipelineParts.host.hidden = value <= .22;
      if (!pipelineParts.host.hidden) paintPipeline(pipelineParts, progress);

      paintObjectMorphs(objectMorphs, direction, progress, moving);
      const current = identityBox(progress, direction);
      box(anchorRect, current);
      anchorLabel.setAttribute('x', current.x + current.width / 2);
      anchorLabel.setAttribute('y', current.y + current.height / 2);
      // At the branch origin the native crop supplies the identity.
      anchor.style.opacity = ramp(progress, .12, .34);
      $('#local-task').style.opacity = 1 - ramp(progress, .01, .21);
      anchorLabel.style.opacity = ramp(progress, .10, .25);
      ghosts.forEach((ghost, index) => {
        const offsets = [-.18, -.09, .09, .18];
        box(ghost, identityBox(clamp(progress + offsets[index]), direction));
        ghost.style.opacity = moving ? Math.sin(Math.PI * ramp(progress, 0, .67)) *
          [ .26, .49, .38, .22 ][index] : 0;
      });
      const label = $('#view');
      const viewName = {
        assignment: zoom > .55 ? 'Assignment · Block 内' : 'Assignment · Batch × KV Head',
        graph: 'Graph · 计算依赖',
        pipeline: 'Pipeline · 逻辑执行顺序',
      }[view];
      if (label.textContent !== viewName) label.textContent = viewName;
      label.style.opacity = view === 'assignment' ? 1 - ramp(progress, .08, .23) : ramp(progress, .46, .66);
      $('#graph-key').hidden = !(view === 'graph' && progress > .85);
      $('#tick-key').hidden = !(view === 'pipeline' && progress > .72);
      $('#relation-key').hidden = !(view === 'pipeline' && progress > .87);
    }

    function describe() {
      $('#beat-title').textContent = beats[step].title;
      $('#beat-caption').textContent = beats[step].caption;
      $('#context').textContent = step ? taskContext : 'Paged Attention';
      $('#step').textContent = `${step + 1} / ${beats.length}`;
      $('#prev').disabled = step === 0;
      $('#next').disabled = step === beats.length - 1;
      const view = aim < 0 ? 'graph' : aim > 0 ? 'pipeline' : 'assignment';
      story.querySelectorAll('button[data-view]').forEach(button => {
        button.setAttribute('aria-pressed', String(button.dataset.view === view));
      });
    }

    function go(next) {
      if (!beats[next]) return;
      cancelAnimationFrame(frame);
      step = next;
      aim = beats[step].at;
      describe();
      if (aim && assignmentHost.contains(document.activeElement)) story.focus({preventScroll: true});
      if (step) setSelection(true);
      const target = {branch: aim, zoom: aim || beats[step].local ? 1 : 0};
      const start = camera;
      if (motion.matches || (target.branch === start.branch && target.zoom === start.zoom)) {
        finish(target);
        return;
      }
      const local = {branch: 0, zoom: 1};
      // Side switches and moves between the overview and a branch go through
      // the same Task crop, in either direction.
      const viaLocal = start.branch * aim < 0 || (aim && start.zoom < 1) ||
        (start.branch && !target.zoom);
      const waypoints = viaLocal ? [local, target] : [target];

      function segment(from, to, hold = 0) {
        const distance = Math.abs(from.branch - to.branch);
        const duration = distance ? 280 + distance * 1040 : 440;
        const started = performance.now() + hold;
        function tick(now) {
          const elapsed = clamp((now - started) / duration);
          const time = smooth(elapsed);
          paint({branch: mix(from.branch, to.branch, time), zoom: mix(from.zoom, to.zoom, time)}, elapsed < 1);
          if (elapsed < 1) frame = requestAnimationFrame(tick);
          else if (waypoints.length) segment(to, waypoints.shift(), 220);
          else finish(target);
        }
        frame = requestAnimationFrame(tick);
      }
      segment(start, waypoints.shift());
    }

    function finish(target = {branch: aim, zoom: aim || beats[step].local ? 1 : 0}) {
      frame = 0;
      setSelection(step > 0, step === beats.length - 1);
      paint(target, false);
    }

    const nav = {graph: 3, assignment: 4, pipeline: 6};
    story.querySelectorAll('button[data-view]').forEach(button => {
      button.onclick = () => go(button.dataset.view === 'assignment' && !step ? 0 : nav[button.dataset.view]);
    });
    $('#prev').onclick = () => go(step - 1);
    $('#next').onclick = () => go(step + 1);
    $('#reset').onclick = () => go(0);
    $('#fullscreen').hidden = !document.fullscreenEnabled;
    $('#fullscreen').onclick = () => document.fullscreenElement ? document.exitFullscreen() : screen.requestFullscreen();
    motion.addEventListener('change', () => {
      if (!frame || !motion.matches) return;
      cancelAnimationFrame(frame);
      finish();
    });
    // Keep presentation keys scoped to the focused surface.
    story.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const destination = {ArrowLeft: step - 1, ArrowRight: step + 1, Home: 0}[event.key];
      if (destination === undefined) return;
      event.preventDefault();
      event.stopPropagation();
      go(destination);
    });
    finish();
    describe();
  }

  function fixedScenario(task) {
    const api = window.PtoComputeScenario;
    const state = api.create(task);
    return {get: () => state, limits: () => api.limits(state), task: () => null};
  }

  function buildAssignment(model, id) {
    const host = $('#block-render');
    const task = model.tasks[id];
    const native = window.PtoVisualization2Anatomy.mount(host, {
      model, scenario: fixedScenario(task),
    });
    native.update(id, null, null, {active: true});
    const query = host.querySelector('.viz2-anatomy-query');
    const cache = host.querySelector('[data-anatomy-task] svg');
    // Reflow the native Stack groups into two columns so all Pages are visible.
    // Stack and Page IDs, ordering, fill and labels come from the renderer.
    const rows = Math.ceil(task.stacks.length / 2);
    const rowHeight = Number(cache.getAttribute('viewBox').split(' ').at(-1)) / task.stacks.length;
    const columnWidth = 400;
    cache.setAttribute('viewBox', `0 0 ${columnWidth + 368} ${rows * rowHeight}`);
    cache.querySelectorAll('[data-anatomy-stack-container]').forEach((group, index) => {
      if (index >= rows) group.setAttribute('transform', `translate(${columnWidth} ${-rows * rowHeight})`);
    });
    cache.removeAttribute('style');
    native.destroy();
    const cacheGroup = document.createElement('section');
    const cacheTitle = document.createElement('h3');
    cacheTitle.textContent = 'KV Cache';
    cacheGroup.append(cacheTitle, cache);
    host.append(query, cacheGroup);
    host.querySelectorAll('[data-anatomy-kind]').forEach(node => {
      node.removeAttribute('tabindex');
      node.removeAttribute('aria-pressed');
      node.setAttribute('role', 'img');
    });
  }

  function collectMorphs(task, rect, graph, pipeline) {
    // Use the same correspondences as the Explorer's Block-local transition:
    // Page → QK instance and Stack → its own first Pipeline stage.
    graph.host.style.transform = 'scale(.88)';
    const overlay = $('#object-morphs');
    function pair(source, destination, emphasize = false) {
      const shapes = Array.from({length: emphasize ? 4 : 1}, () => {
        const shape = document.createElementNS(svgNS, 'rect');
        shape.setAttribute('rx', '4');
        overlay.append(shape);
        return shape;
      });
      return {from: rect(source), to: rect(destination), emphasize, shapes};
    }
    const q = pair($('#block-render .viz2-anatomy-q-boundary'),
      graph.host.querySelector('[data-pypa-node="q"]'), true);
    const pages = task.pages.map(({index, stack}) => {
      const slot = task.stacks[stack].pages.indexOf(index);
      const node = index ? `[data-pypa-execution-stack="qk"] [data-pypa-execution-instance="${index + 1}"]`
        : '[data-pypa-node="qk"]';
      return pair($('#block-render [data-anatomy-cell="' + stack + ':' + slot + '"]'),
        graph.host.querySelector(node), index === 0);
    });
    const stacks = task.stacks.map(({index}) =>
      pair($('#block-render [data-anatomy-kind="stack"][data-anatomy-index="' + index + '"] rect'),
        pipeline.host.querySelector(`[data-intra-kind="stage"][data-intra-index="${index}"][data-intra-stage="qk"] .ai-operation-glyph [data-kind="op"] > rect`),
        index === 0));
    return {'-1': [q, ...pages], 1: stacks};
  }

  function paintObjectMorphs(morphs, direction, progress, moving) {
    for (const side of [-1, 1]) {
      const active = side === direction && moving;
      for (const {from, to, emphasize, shapes} of morphs[side]) {
        const end = side < 0 ? .76 : .71;
        const opacity = active ? ramp(progress, .04, .23) * (1 - ramp(progress, end - .12, end)) : 0;
        shapes.forEach((shape, index) => {
          const time = smooth(ramp(progress + [0, -.08, .08, .16][index], .10, end));
          for (const key of ['x', 'y', 'width', 'height']) {
            shape.setAttribute(key, mix(from[key], to[key], time));
          }
          shape.style.opacity = opacity * (emphasize ? [1, .38, .31, .19][index] : .44);
        });
      }
    }
  }

  function buildGraph(task, model) {
    const host = $('#graph');
    const semantic = window.PTO_SEMANTIC_COMPUTATION_GRAPH.materialize(
      task.graph, new Set(), {flowLayout: true, compactGlyphs: true},
    );
    // Positions follow P06's overview. The edges and loop counts retain their
    // native IDs and are routed by the shared renderer after positioning.
    const positions = {
      q: [16, 86, 148], 'k-cache': [16, 187, 148], qk: [199, 129, 158],
      scores: [384, 139, 125], 'scale-mask': [532, 129, 173],
      softmax: [725, 129, 186], probability: [936, 139, 142],
      'v-cache': [1132, 106, 155], pv: [1138, 242, 165], partial: [1051, 373, 219],
      'context-acc': [803, 383, 225], 'output-norm': [422, 508, 212], output: [166, 498, 198],
      m: [576, 244, 132], l: [755, 244, 111], rescale: [911, 274, 145], o: [660, 372, 126],
    };
    const graph = {
      ...semantic, nodes: semantic.nodes.map(node => {
        const [x, y, w] = positions[node.id];
        return {...node, x, y, w};
      }),
      groups: [{id: 'loop', label: '循环内计算与跨 Stack 状态', x: 177, y: 62, w: 1180, h: 406}],
      edgeRoutes: null, width: 1480, height: 605,
    };
    const renderer = window.PtoStudioComputationGraph;
    const result = renderer.renderScene(host, graph, {
      view: 'compute', allowExpand: false, synchronizations: false,
      compactNodes: true, markerPrefix: 'p09-crossview',
    });
    const counts = window.PtoPagedAttentionExecution.derive(task.graph, model, {task: task.id, seqLen: task.seqLen});
    renderer.applyExecution(host, result.placedNodes, graph, counts);
    host.querySelectorAll('button,[tabindex]').forEach(node => { node.tabIndex = -1; });
    host.querySelectorAll('[data-pypa-execution-instance]').forEach(layer => {
      const offset = Number(layer.dataset.pypaExecutionInstance) - 1;
      layer.style.setProperty('--execution-offset', `${offset * 4}px`);
      layer.style.setProperty('--execution-offset-y', `${offset * 2.8}px`);
    });
    return {
      host, edges: host.querySelector('.kf-pypa-graph__edges'),
      nodes: result.placedNodes.map(node => ({
        ...node, element: host.querySelector(`[data-pypa-node="${node.id}"]`),
        delay: node.kind === 'data' ? .03 : 0,
      })),
      scope: host.querySelector('.kf-pypa-graph__group'),
      repetitions: [...host.querySelectorAll('[data-pypa-execution-count],[data-pypa-execution-instance]')],
    };
  }

  function paintGraph(parts, progress) {
    const open = smooth(ramp(progress, .22, .63));
    parts.host.style.transform = `translateX(${95 * (1 - open)}px) scale(.88)`;
    parts.nodes.forEach(({element, x, y, delay}) => {
      const local = smooth(ramp(progress, .28 + delay, .62 + delay));
      element.style.opacity = ramp(local, 0, .55);
      element.style.transform = `translate(${(690 - x) * .53 * (1 - local)}px, ${(246 - y) * .60 * (1 - local)}px) scale(${mix(.50, 1, local)})`;
    });
    parts.edges.style.opacity = ramp(progress, .55, .72);
    parts.scope.style.opacity = ramp(progress, .72, .91);
    parts.scope.style.transformOrigin = '50% 50%';
    parts.scope.style.transform = `scale(${mix(.88, 1, ramp(progress, .71, .93))})`;
    parts.repetitions.forEach(element => {
      const ordinal = Number(element.dataset.pypaExecutionInstance || 1);
      const revealed = ramp(progress, .73 + Math.min(ordinal, 24) * .006, .86 + Math.min(ordinal, 24) * .0058);
      element.style.opacity = element.closest('[data-pypa-execution-stack="qk"]')
        ? ramp(progress, .68, .76) : revealed;
    });
  }

  function buildPipeline(api, model, task) {
    const source = document.createElement('div');
    source.id = 'crossview-native-pipeline';
    source.hidden = true;
    document.body.append(source);
    const native = api.mount(source, {model, scenario: fixedScenario(task), onCamera() {}});
    native.update(task.id, null, null, {active: true});
    const svg = source.querySelector('.viz2-intra-svg');
    const host = $('#pipeline');
    const [, , width, height] = svg.getAttribute('viewBox').split(' ').map(Number);
    // Trim the padding below the native Stack strip.
    const stackHit = svg.querySelector('.viz2-intra-stack-hit');
    const contentBottom = Number(stackHit.getAttribute('y')) + Number(stackHit.getAttribute('height'));
    svg.setAttribute('viewBox', `0 0 ${width} ${Math.min(height, contentBottom + 18)}`);
    host.append(svg);
    native.destroy();
    source.remove();
    svg.querySelectorAll('[tabindex]').forEach(node => node.removeAttribute('tabindex'));
    const operations = task.operations.map(operation => ({
      ...operation,
      element: svg.querySelector(`[data-intra-operation="${operation.stack}:${operation.stage}"]`).parentElement,
    }));
    const relations = [...svg.querySelectorAll('[data-intra-dependency],[data-intra-event],[data-intra-reuse]')];
    const ticks = [...svg.querySelectorAll('[data-intra-kind="tick"]')];
    const final = [...svg.querySelectorAll('[data-intra-kind="output"],.viz2-intra-output-link')];
    const stacks = [...svg.querySelectorAll('[data-intra-kind="stack"]')];
    return {host, svg, operations, relations, ticks, final, stacks};
  }

  function paintPipeline(parts, progress) {
    parts.host.style.transform = `translateX(${-94 * (1 - smooth(ramp(progress, .23, .66)))}px)`;
    parts.svg.style.opacity = ramp(progress, .22, .35);
    parts.ticks.forEach((element, index) => {
      element.style.opacity = ramp(progress, .28 + index * .035, .40 + index * .035);
    });
    parts.operations.forEach(({element, tick}) => {
      const open = smooth(ramp(progress, .34 + tick * .034, .49 + tick * .037));
      element.style.opacity = open;
      element.setAttribute('transform', `translate(${-48 * (1 - open)} ${17 * (1 - open)})`);
    });
    parts.stacks.forEach((element, index) => {
      element.style.opacity = ramp(progress, .47 + index * .035, .55 + index * .035);
    });
    parts.final.forEach(element => { element.style.opacity = ramp(progress, .65, .76); });
    parts.relations.forEach(element => {
      const reuse = element.dataset.intraReuse;
      const handoff = element.dataset.intraDependency === '0:probability_transfer' ||
        (element.dataset.intraEvent === 'SOFTMAX_READY_EVENT' && element.dataset.intraIndex === '0');
      const resource = reuse === 'probability_transfer:0:3';
      const start = reuse ? .89 : .77;
      const emphasis = handoff ? 1 : resource ? .64 : reuse ? .18 : .48;
      element.style.opacity = ramp(progress, start, start + .105) * emphasis;
      element.classList.toggle('is-related', handoff);
    });
  }
})();
