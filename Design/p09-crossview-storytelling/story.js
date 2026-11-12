(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const screen = $('#screen');
  const story = $('#story');
  const camera = $('#camera');
  const embedded = window.parent !== window && new URLSearchParams(location.search).get('deck') === 'P9';
  const deckOrigin = location.protocol === 'file:' ? '*' : location.origin;
  const sendToDeck = (type, data = {}) => parent.postMessage({pyptoCrossview: 1, type, ...data}, deckOrigin);
  if (embedded) {
    document.title = 'P9 · 跨视角对象定位';
    $('.mast').textContent = 'PyPTO Compute Explorer / P9';
    story.setAttribute('aria-label', 'P9 跨视角对象定位 · 16:9');
  }
  const views = [...story.querySelectorAll('[data-view]')];
  const shots = [...camera.querySelectorAll('[data-shot]')];
  const nativeMorph = window.PtoVisualization2IntraMorph;
  const blockViews = Object.fromEntries(shots.map(shot => [shot.dataset.shot, {subview: 'block'}]));

  const branches = {
    graph: {beat: 1, caption: '同一个 Task 在 Graph 中呈现计算依赖与循环层级。'},
    pipeline: {beat: 3, caption: '沿 Logical Tick 查看同一个 Task 的执行阶段、同步与 Slot 复用。'},
  };
  const storyViews = ['assignment', 'graph', 'assignment', 'pipeline', 'assignment'];
  const clamp = value => Math.max(0, Math.min(1, value));

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
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const tracks = [...story.querySelectorAll('.morph-track')].map(track => ({
      track, path: track.parentElement, branch: track.parentElement.dataset.path,
      tooltip: track.querySelector('.morph-value'),
    }));
    const previous = $('#prev');
    const next = $('#next');
    let branch = 'assignment';
    let progress = 0;
    let storyPosition = 0;
    let leg = 0;
    let animation = null;
    let frame = 0;
    let playing = false;
    let drag = null;
    let reportFrame = 0;
    let lastReport = '';
    const getState = () => ({branch, progress, playing, storyPosition, dragging: !!drag});

    function reportState() {
      if (!embedded || reportFrame) return;
      reportFrame = requestAnimationFrame(() => {
        reportFrame = 0;
        const state = getState();
        const key = JSON.stringify(state);
        if (key === lastReport) return;
        lastReport = key;
        sendToDeck('state', {state});
      });
    }

    function describe() {
      const view = progress === 1 ? branch : progress === 0 ? 'assignment' : null;
      views.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
      previous.setAttribute('aria-disabled', String(playing || storyPosition === 0));
      next.setAttribute('aria-disabled', String(playing || storyPosition === storyViews.length - 1));
      const caption = branches[branch]?.caption ?? '在 Assignment 查看 Task 48 的 Q Tile 与 KV Cache。';
      if ($('#caption').textContent !== caption) $('#caption').textContent = caption;
      for (const entry of tracks) {
        const value = entry.branch === branch ? progress * 100 : 0;
        const percentage = Math.round(value * 100) / 100;
        entry.path.style.setProperty('--progress', value + '%');
        entry.path.toggleAttribute('data-active', value > 0);
        entry.track.setAttribute('aria-valuenow', String(percentage));
        entry.track.setAttribute('aria-valuetext', value === 0 ? 'Assignment' : percentage + '%');
        entry.tooltip.textContent = Math.round(value) + '%';
      }
      reportState();
    }

    function releaseAnimation() {
      transition.cancel();
      animation = null;
    }

    function prepareAnimation(view) {
      show('assignment', false);
      const focus = getFocus('assignment');
      animation = transition.prepare({
        open: true, activeTab: 'assignment', views: blockViews,
        selection: {kind: 'task', task: task.id, subobject: focus},
      }, view, {manual: true});
      setFocus(view, nativeMorph.mapFocus('assignment', view, focus, task));
      show(view, false);
      return transition.seek(animation, 0);
    }

    function paint(view, value) {
      if (branch !== view && progress > 0) paint(branch, 0);
      if (animation && animation.to !== view) releaseAnimation();
      if (value === 0 || value === 1) {
        releaseAnimation();
        show(value === 0 ? 'assignment' : view);
      } else if ((!animation && !prepareAnimation(view)) || !transition.seek(animation, value)) {
        releaseAnimation();
        show(progress === 1 ? branch : 'assignment');
        return false;
      }
      branch = value === 0 ? 'assignment' : view;
      progress = value;
      describe();
      return true;
    }

    function stopPlayback() {
      cancelAnimationFrame(frame);
      frame = 0;
      playing = false;
    }

    function updateStoryPosition(view, value) {
      const beat = branches[view].beat;
      if (leg !== beat) leg = beat - 1;
      storyPosition = leg === beat ? beat + 1 - value : beat - 1 + value;
    }

    function scrub(view, value) {
      stopPlayback();
      if (!paint(view, value)) return false;
      updateStoryPosition(view, value);
      describe();
      return true;
    }

    function playTo(view, target, done) {
      const from = progress;
      if (target === from) { done(); return; }
      if (motion.matches) {
        if (paint(view, target)) done();
        else { stopPlayback(); describe(); }
        return;
      }
      if (!animation && !prepareAnimation(view)) {
        releaseAnimation();
        show(from === 0 ? 'assignment' : view);
        stopPlayback();
        describe();
        return;
      }
      const duration = Math.abs(target - from) * animation.duration;
      let start;
      function tick(now) {
        start ??= now;
        const time = motion.matches ? 1 : clamp((now - start) / duration);
        const value = from + (target - from) * time;
        if (time === 0) { frame = requestAnimationFrame(tick); return; }
        if (!paint(view, value)) { stopPlayback(); describe(); return; }
        updateStoryPosition(view, value);
        if (time < 1) frame = requestAnimationFrame(tick);
        else done();
      }
      frame = requestAnimationFrame(tick);
    }

    function navigate(view, beat) {
      if (playing || drag || view === branch && (progress === 0 || progress === 1)) return false;
      playing = true;
      describe();
      const finish = () => {
        stopPlayback();
        storyPosition = beat;
        describe();
      };
      const forward = () => {
        leg = branches[view].beat - 1;
        playTo(view, 1, finish);
      };
      if (view === 'assignment') playTo(branch, 0, finish);
      else if (branch === 'assignment' || view === branch) forward();
      else playTo(branch, 0, forward);
      return true;
    }

    function step(direction) {
      const beat = direction > 0 ? Math.floor(storyPosition) + 1 : Math.ceil(storyPosition) - 1;
      const view = storyViews[beat];
      if (!view) return false;
      if (!playing && !drag) leg = Math.min(beat, Math.floor(storyPosition));
      return navigate(view, beat);
    }

    function switchBranch(view) {
      if (!Object.hasOwn(blockViews, view)) return false;
      const beat = branches[view]?.beat ?? ((branches[branch]?.beat ?? -1) + 1);
      return navigate(view, beat);
    }

    function endDrag() {
      if (!drag) return;
      const {entry, id} = drag;
      drag = null;
      entry.path.removeAttribute('data-dragging');
      if (entry.track.hasPointerCapture(id)) entry.track.releasePointerCapture(id);
      reportState();
    }

    function reset() {
      endDrag();
      stopPlayback();
      storyPosition = 0;
      leg = 0;
      releaseAnimation();
      fitViews();
      paint('assignment', 0);
    }

    fitCrops();
    new ResizeObserver(() => {
      fitCrops();
      if (!animation) return;
      stopPlayback();
      releaseAnimation();
      if (!paint(branch, progress)) reset();
    }).observe(camera);
    paint('assignment', 0);

    const stepForward = () => step(1);
    const stepBackward = () => step(-1);
    window.PtoCrossviewStory = Object.freeze({
      stepForward, stepBackward, switchBranch, reset,
      seek(view, value) {
        return !drag && Object.hasOwn(branches, view) && Number.isFinite(value) && scrub(view, clamp(value));
      },
      enter: reset, leave: reset,
      getState,
    });

    addEventListener('pagehide', reset);
    views.forEach(button => { button.onclick = () => switchBranch(button.dataset.view); });
    for (const entry of tracks) {
      const {track, path, branch: view} = entry;
      function seekPointer(event) {
        const bounds = track.getBoundingClientRect();
        const position = clamp((event.clientX - bounds.left) / bounds.width);
        scrub(view, view === 'graph' ? 1 - position : position);
      }
      track.addEventListener('pointerdown', event => {
        if (drag || event.button !== 0) return;
        event.preventDefault();
        track.focus({preventScroll: true});
        track.setPointerCapture(event.pointerId);
        drag = {entry, id: event.pointerId};
        path.setAttribute('data-dragging', '');
        seekPointer(event);
      });
      track.addEventListener('pointermove', event => {
        if (drag?.id === event.pointerId) seekPointer(event);
      });
      track.addEventListener('pointerup', event => {
        if (drag?.id !== event.pointerId) return;
        seekPointer(event);
        endDrag();
      });
      for (const eventName of ['pointercancel', 'lostpointercapture']) {
        track.addEventListener(eventName, event => { if (drag?.id === event.pointerId) endDrag(); });
      }
      track.addEventListener('keydown', event => {
        if (drag || event.altKey || event.ctrlKey || event.metaKey) return;
        const horizontal = view === 'graph' ? -.01 : .01;
        const delta = {ArrowLeft: -horizontal, ArrowRight: horizontal, ArrowUp: .01, ArrowDown: -.01}[event.key];
        if (delta === undefined && event.key !== 'Home' && event.key !== 'End') return;
        event.preventDefault();
        event.stopPropagation();
        const current = branch === view ? progress : 0;
        const target = delta === undefined ? Number(event.key === 'End') : clamp(current + delta);
        if (target !== current) scrub(view, target);
      });
    }
    previous.onclick = stepBackward;
    next.onclick = stepForward;
    $('#reset').onclick = reset;
    $('#fullscreen').hidden = !document.fullscreenEnabled;
    $('#fullscreen').onclick = () => document.fullscreenElement
      ? document.exitFullscreen() : screen.requestFullscreen();
    story.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey ||
          !event.target.closest('.local-navigation, .transport')) return;
      const action = {ArrowLeft: stepBackward, ArrowRight: stepForward, Home: reset}[event.key];
      if (!action || (embedded && !playing && !drag &&
          (event.key === 'ArrowLeft' && storyPosition === 0 ||
           event.key === 'ArrowRight' && storyPosition === storyViews.length - 1))) return;
      event.preventDefault();
      event.stopPropagation();
      if (!event.repeat) action();
    });

    if (embedded) {
      addEventListener('message', event => {
        const message = event.data;
        if (event.source !== parent || (deckOrigin !== '*' && event.origin !== deckOrigin) ||
            message?.pyptoCrossview !== 1) return;
        if (message.type === 'reset') reset();
        if (message.type === 'pause') { stopPlayback(); describe(); }
        if (message.type === 'state') {
          const state = message.state;
          if (!state || !Object.hasOwn(blockViews, state.branch) ||
              !Number.isFinite(state.progress) || !Number.isFinite(state.storyPosition)) return;
          endDrag();
          stopPlayback();
          storyPosition = Math.max(0, Math.min(storyViews.length - 1, state.storyPosition));
          leg = Math.floor(storyPosition);
          paint(state.branch, state.branch === 'assignment' ? 0 : clamp(state.progress));
        }
        if (message.type === 'advance' && !playing && !drag &&
            (message.direction === 1 || message.direction === -1) && !step(message.direction)) {
          sendToDeck('page', {direction: message.direction});
        }
      });
      document.addEventListener('keydown', event => {
        if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey ||
            (event.code === 'Space' && event.target.closest('button')) ||
            !['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','Space','KeyF','KeyP','Escape'].includes(event.code)) return;
        event.preventDefault();
        sendToDeck('key', {code: event.code, key: event.key, repeat: event.repeat, shiftKey: event.shiftKey});
      });
      sendToDeck('ready');
    }
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
      const graphCamera = graphView.intraCamera;
      if (graphCamera) {
        const viewport = $('#viz2-graph-intra .viz2-graph-viewport');
        const x = viewport.clientWidth / 2;
        const y = viewport.clientHeight / 2;
        graphView.intraCamera = {
          scale: graphCamera.scale * 1.5,
          x: x + (graphCamera.x - x) * 1.5,
          y: y + (graphCamera.y - y) * 1.5,
        };
        renderGraph();
      }
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

  function show(view, interactive = true) {
    for (const shot of shots) {
      const active = shot.dataset.shot === view;
      shot.style.visibility = active ? 'visible' : 'hidden';
      shot.style.opacity = active ? '1' : '0';
      shot.inert = !active || !interactive;
    }
  }

})();
