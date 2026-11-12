(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const screen = $('#screen');
  const story = $('#story');
  const camera = $('#camera');
  const views = [...story.querySelectorAll('[data-view]')];
  const shots = [...camera.querySelectorAll('[data-shot]')];
  const nativeMorph = window.PtoVisualization2IntraMorph;
  const blockViews = Object.fromEntries(shots.map(shot => [shot.dataset.shot, {subview: 'block'}]));

  const beats = [
    {view: 'assignment', caption: '在 Assignment 查看 Task 48 的 Q Tile 与 KV Cache。'},
    {view: 'graph', caption: '同一个 Task 在 Graph 中呈现计算依赖与循环层级。'},
    {view: 'assignment', caption: '回到同一个 Task 的数据组织。'},
    {view: 'pipeline', caption: '沿 Logical Tick 查看同一个 Task 的执行阶段、同步与 Slot 复用。'},
    {view: 'assignment', caption: '同一个 Task，从数据组织、计算依赖和逻辑执行三个角度观察。'},
  ];

  let step = 0;

  new ResizeObserver(() => {
    const scale = Math.min(screen.clientWidth / 1600, screen.clientHeight / 900);
    story.style.setProperty('--story-scale', scale);
    story.style.left = (screen.clientWidth - 1600 * scale) / 2 + 'px';
    story.style.top = (screen.clientHeight - 900 * scale) / 2 + 'px';
  }).observe(screen);

  try {
    const {task, fitViews, getFocus, setFocus} = mountNativeViews();
    const transition = nativeMorph.create({
      root: camera,
      getTask: () => task,
      getClip: view => camera.querySelector(`[data-shot="${view}"] .crop`),
    });
    fitCrops();
    new ResizeObserver(() => {
      transition.cancel();
      fitCrops();
    }).observe(camera);
    describe();
    show(beats[step].view);

    function go(next, {reset = false} = {}) {
      if (!beats[next]) return;
      const from = beats[step].view;
      const to = beats[next].view;
      if (reset) {
        transition.cancel();
        fitViews();
      }
      const focus = getFocus(from);
      const animation = transition.prepare({
        open: true, activeTab: from, views: blockViews,
        selection: {kind: 'task', task: task.id, subobject: focus},
      }, to);
      if (from !== to) setFocus(to, nativeMorph.mapFocus(from, to, focus, task));
      step = next;
      describe();
      show(to);
      transition.play(animation);
    }

    const direct = {graph: 1, assignment: 2, pipeline: 3};
    views.forEach(button => {
      button.onclick = () => {
        if (beats[step].view !== button.dataset.view) go(direct[button.dataset.view]);
      };
    });
    $('#prev').onclick = () => go(step - 1);
    $('#next').onclick = () => go(step + 1);
    $('#reset').onclick = () => go(0, {reset: true});
    $('#fullscreen').hidden = !document.fullscreenEnabled;
    $('#fullscreen').onclick = () => document.fullscreenElement
      ? document.exitFullscreen() : screen.requestFullscreen();
    story.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey ||
          event.target.closest('.native-frame')) return;
      const next = {ArrowLeft: step - 1, ArrowRight: step + 1, Home: 0}[event.key];
      if (next === undefined) return;
      event.preventDefault();
      event.stopPropagation();
      go(next, {reset: event.key === 'Home'});
    });
  } catch (error) {
    $('#error').hidden = false;
    story.querySelectorAll('button').forEach(button => { button.disabled = true; });
    console.error(error);
  }

  function mountNativeViews() {
    const id = 48;
    const source = window.PTO_PAGED_ATTENTION_PYPTO_SOURCE;
    const semantic = window.PTO_PYPA_SEMANTIC_GRAPH;
    const value = name => semantic.constants[name].value;
    const anatomyAPI = window.PtoVisualization2Anatomy;
    const graphAPI = window.PtoVisualization2GraphIntra;
    const pipelineAPI = window.PtoVisualization2PipelineIntra;
    const anatomy = anatomyAPI.createModel({
      seqLens: window.PtoPagedAttentionIntro.model.defaultSeqLens,
      blocks: value('ATTN_SPMD_BLOCKS'), numHeads: value('NUM_HEADS'),
      kvHeads: value('NUM_KV_HEADS'), headDim: value('HEAD_DIM'), rowTile: value('ROW_TILE'),
      pageTokens: value('BLOCK_SIZE'), pagesPerStack: value('STACK_PAGES'),
      transferSlots: value('TRANSFER_SLOTS'),
    });
    const preLaunch = value('PRE_LAUNCH');
    const task = anatomy.tasks[id];
    const scenarioAPI = window.PtoComputeScenario;
    const state = scenarioAPI.create(task);
    const projected = scenarioAPI.project(state, anatomy, source, preLaunch);
    const scenario = {
      get: () => state,
      limits: () => scenarioAPI.limits(state),
      task: (kind, taskId) => taskId === id ? projected[kind] : null,
    };
    const keepTask = () => {};
    const assignment = anatomyAPI.mount($('#viz2-anatomy'), {
      model: anatomy, scenario, onTask: keepTask, onExit: keepTask,
      onFocus: focus => { assignmentFocus = focus; renderAssignment(); },
      onCamera: (_, camera) => { assignmentCamera = camera; renderAssignment(); },
      onInspector: keepTask,
    });
    const graph = graphAPI.mount($('#viz2-graph-intra'), {
      model: graphAPI.createModel({anatomy, source, preLaunch}), scenario, onTask: keepTask,
      onFocus: focus => { graphFocus = focus; renderGraph(); },
      onView: patch => { graphView = {...graphView, ...patch}; renderGraph(); },
    });
    const pipeline = pipelineAPI.mount($('#viz2-pipeline-intra'), {
      model: pipelineAPI.createModel({anatomy, preLaunch}), scenario, onTask: keepTask, onExit: keepTask,
      onFocus: focus => { pipelineFocus = focus; renderPipeline(); },
      onCamera: (_, camera) => { pipelineCamera = camera; renderPipeline(); },
      onInspector: keepTask,
    });

    let assignmentFocus = null, assignmentCamera = null;
    let graphFocus = null, graphView = {};
    let pipelineFocus = null, pipelineCamera = null;
    const renderAssignment = () => assignment.update(id, assignmentFocus, assignmentCamera, {active: true});
    const renderGraph = () => graph.update(id, graphFocus, graphView, {active: true});
    const renderPipeline = () => pipeline.update(id, pipelineFocus, pipelineCamera, {active: true});

    $('#context').textContent = `Batch ${task.batch} · KV Head ${task.kvHead} · Block ${String(task.block).padStart(2, '0')}`;
    renderAssignment();
    renderGraph();
    renderPipeline();
    window.PtoAIInfraWorkspace?.flush();

    function fitViews() {
      assignmentFocus = graphFocus = pipelineFocus = null;
      assignmentCamera = null;
      graphView = {intraCamera: graphView.intraCamera};
      pipelineCamera = null;
      renderAssignment();
      renderGraph();
      renderPipeline();
      $('#viz2-graph-intra [data-viz2-graph-fit]').click();
      $('#viz2-pipeline-intra [data-intra-fit]').click();
      $('#viz2-anatomy .viz2-anatomy-cache-scroll').scrollTop = 0;
    }
    fitViews();
    return {
      task, fitViews,
      getFocus: view => ({assignment: assignmentFocus, graph: graphFocus, pipeline: pipelineFocus})[view],
      setFocus(view, focus) {
        if (view === 'assignment') { assignmentFocus = focus; renderAssignment(); }
        if (view === 'graph') { graphFocus = focus; renderGraph(); }
        if (view === 'pipeline') { pipelineFocus = focus; renderPipeline(); }
      },
    };
  }

  function fitCrops() {
    // Match the original 1600 × 900 Explorer viewport. Crop away its
    // workbench chrome and fit each remaining region as a single surface.
    const regions = {
      assignment: {width: 1600, height: 650, top: 110},
      graph: {width: 1268, height: 590, top: 160},
      pipeline: {width: 1268, height: 470, top: 195},
    };
    shots.forEach(shot => {
      const {width, height, top} = regions[shot.dataset.shot];
      const scale = Math.min(camera.clientWidth / width, camera.clientHeight / height);
      const crop = shot.querySelector('.crop');
      crop.style.setProperty('--native-top', -top + 'px');
      Object.assign(crop.style, {
        width: width + 'px', height: height + 'px',
        left: (camera.clientWidth - width * scale) / 2 + 'px',
        top: (camera.clientHeight - height * scale) / 2 + 'px',
        transform: `scale(${scale})`,
      });
    });
  }

  function show(view) {
    for (const shot of shots) {
      const active = shot.dataset.shot === view;
      shot.style.visibility = active ? 'visible' : 'hidden';
      shot.style.opacity = active ? '1' : '0';
      shot.inert = !active;
    }
  }

  function describe() {
    const {view, caption} = beats[step];
    views.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
    $('#caption').textContent = caption;
    $('#prev').disabled = step === 0;
    $('#next').disabled = step === beats.length - 1;
  }

})();
