(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const stage = $('#story');
  const view = $('#view');
  const canvas = $('#canvas');
  const inDeck = document.body.hasAttribute('data-p10-deck');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const axes = [...document.querySelectorAll('[data-p09-level]')];
  const captions = [
    '从 Device 看执行全貌，在 AI Core 中定位 {block}。',
    '聚焦 {block}，在块内定位 Task {task}。',
    '在 Task {task} 内沿逻辑轮次查看生产、消费与阶段交接。',
  ];
  const blockName = id => 'Block ' + String(id).padStart(2, '0');
  const baseCamera = {scale:1, x:0};
  let beat = 0;
  let taskId = 48;
  let deviceTask = taskId;
  let focus = null;
  let outerCamera = baseCamera;
  let innerCamera;
  let scenarioState;
  let primedTask;
  let drawing = false;
  let blockMotion;
  let focusSequence;
  let blockStrength = 1;
  let taskLocated = false;
  let demoTask;
  let replay = 0;
  let outer, inner, execution, pipeline, anatomy;

  const outerHost = document.createElement('div');
  const innerHost = document.createElement('div');
  outerHost.id = 'viz2-pipeline-inter';
  innerHost.id = 'viz2-pipeline-intra';
  outerHost.className = innerHost.className = 'renderer';
  const insideTask = () => beat >= 2;
  const task = () => pipeline.tasks[taskId];
  const row = id => outerHost.querySelector(`[data-bem-block="${id}"]`);
  const surface = () => outerHost.querySelector('.viz2-pipeline-surface');

  function fitStage() {
    const scale = $('#frame').clientWidth / stage.offsetWidth;
    stage.style.transform = `scale(${scale})`;
  }

  function select(id) {
    if (id === taskId && scenarioState) return;
    taskId = id;
    focus = null;
    scenarioState = PtoComputeScenario.create(anatomy.tasks[id]);
    innerCamera = undefined;
    outerCamera = baseCamera;
  }

  function positionBlock() {
    const source = row(task().block);
    const map = surface();
    if (!source || insideTask()) return;
    const onlyRow = beat === 1;
    const y = source.offsetTop;
    const height = source.offsetHeight;
    map.style.transform = onlyRow ? `translateY(${68 - y}px)` : 'none';
    map.style.clipPath = onlyRow ? `inset(${y}px 0 ${Math.max(0, map.clientHeight - y - height)}px 0)` : 'inset(0px)';
    map.querySelectorAll('button').forEach(button => {
      const id = button.hasAttribute('data-bem-task')
        ? execution.tasks[Number(button.dataset.bemTask)].block : Number(button.dataset.bemBlock);
      button.tabIndex = onlyRow && id !== task().block ? -1 : 0;
      button.inert = onlyRow && id !== task().block;
    });
  }

  function updateOuter() {
    const selection = beat === 1 && (!inDeck || taskLocated)
      ? {kind:'task', block:task().block, task:taskId} : {kind:'block', block:task().block};
    drawing = true;
    try {
      outer.update(selection, outerCamera, {
        active:!insideTask(), blockFocus:inDeck && beat === 1 ? blockStrength : undefined,
      });
      if (!inDeck) positionBlock();
    } finally {
      drawing = false;
    }
  }

  function updateInner() {
    drawing = true;
    try {
      if (primedTask !== taskId) {
        inner.update(taskId, null, innerCamera, {active:true});
        primedTask = taskId;
      }
      if (!innerCamera) {
        const svg = innerHost.querySelector('.viz2-intra-card.is-current svg[data-intra-width]');
        const viewport = svg.parentElement;
        innerCamera = {scale:Math.min(1, (viewport.clientWidth - 50) / Number(svg.dataset.intraWidth),
          (viewport.clientHeight - 42) / Number(svg.dataset.intraHeight)), x:0, y:0};
      }
      inner.update(taskId, focus, innerCamera, {active:true});
    } finally {
      drawing = false;
    }
  }

  function render() {
    const level = beat;
    const current = task();
    const name = blockName(current.block);
    const lane = execution.lanes[current.block];
    const operation = focus?.kind === 'stage'
      ? inner.getOperationDescriptor(taskId, focus.index, focus.stage) : null;
    const tick = current.ticks[operation?.tick ?? (focus?.kind === 'tick' ? focus.index : -1)];
    const target = insideTask() ? innerHost : outerHost;
    if (view.firstChild !== target) {
      if (insideTask()) outer.update(null, outerCamera, {active:false});
      else inner.update(taskId, null, null, {active:false});
      view.replaceChildren(target);
    }

    stage.dataset.level = axes[level].dataset.p09Level;
    stage.dataset.beat = beat;
    axes.forEach((button, index) => {
      button.setAttribute('aria-pressed', String(index === level));
      button.toggleAttribute('data-ancestor', index < level);
    });

    $('#device-context').textContent = execution.blocks + ' Blocks';
    $('#block-context').textContent = name;
    const showTaskContext = level === 2 || (level === 1 && (!inDeck || taskLocated));
    $('#iteration-context').textContent = !inDeck && level === 2 && tick
      ? 'Tick ' + tick.index : showTaskContext ? 'Task ' + taskId : '逻辑轮次';
    $('#block-detail').hidden = inDeck || beat !== 1;
    $('#block-facts').textContent = `${lane.phase0Groups} 组预处理 · SYNCALL · ${lane.tasks.length} Tasks`;
    const deckCaption = level === 1
      ? taskLocated ? `定位 ${name} 内的 Task ${taskId}，继续深入执行细节。`
        : `聚焦 ${name}，在整体任务分布中观察其执行组织。`
      : level === 2 ? `从 Task ${taskId} 进入 Iteration，观察更细粒度的执行过程。` : null;
    const nextBlock = inDeck && level === 0 ? blockName(execution.tasks[demoTask].block) : name;
    const caption = (inDeck && deckCaption) || captions[beat].replaceAll('{task}', taskId).replaceAll('{block}', nextBlock);
    const work = level === 2 && tick && [
      tick.produce === null ? '' : `生产 Stack ${tick.produce}（QK / Softmax）`,
      tick.consume === null ? '' : `消费 Stack ${tick.consume}（PV / Accumulate）`,
    ].filter(Boolean).join('，');
    $('#caption').textContent = work ? `Tick ${tick.index} · ${work}` : caption;
    $('#prev').disabled = beat === 0;
    $('#next').disabled = beat === captions.length - 1;
    $('#up').disabled = level === 0;
    $('#legend').innerHTML = level === 2
      ? '<i class="viz2-intra-flow-key"></i>数据依赖 <i class="viz2-intra-event-key"></i>READY 事件 <i class="viz2-intra-reuse-key"></i>Slot 复用 <span>· Logical Tick 表示源码循环进度</span>'
      : '';
    $('#legend').hidden = level !== 2;

    if (insideTask()) updateInner();
    else updateOuter();
  }

  // The native outer controller reports layout pixels relative to its canvas.
  // The Deck scales the whole slide, while the native inner snapshot uses screen pixels.
  const zoom = PtoVisualization2PipelineZoom.create({
    root:canvas,
    duration:inDeck ? 600 : undefined,
    coordinateScale:() => canvas.getBoundingClientRect().width / canvas.offsetWidth,
    getOuter:() => ({
      getTransitionSnapshot() {
        const snapshot = outer.getTransitionSnapshot();
        if (!snapshot) return null;
        const ratio = canvas.getBoundingClientRect().width / canvas.offsetWidth;
        const origin = outerHost.querySelector('canvas').getBoundingClientRect();
        const project = rect => ({
          x:origin.left + (rect.x - origin.left) * ratio,
          y:origin.top + (rect.y - origin.top) * ratio,
          width:rect.width * ratio, height:rect.height * ratio,
        });
        return {tasks:snapshot.tasks.map(entry => ({
          ...entry, rect:project(entry.rect), output:project(entry.output),
        }))};
      },
      setSemanticTaskMask:id => outer.setSemanticTaskMask(id),
    }),
    getInner:() => ({
      getTransitionSnapshot() {
        const snapshot = inner.getTransitionSnapshot();
        if (!snapshot) return null;
        const title = innerHost.querySelector('.viz2-intra-card.is-current .viz2-intra-axis').getBoundingClientRect();
        return {...snapshot, title:{x:title.x,y:title.y,width:title.width,height:title.height}};
      },
      setTransitionMasked:value => inner.setTransitionMasked(value),
    }),
  });

  function stopMotion() {
    blockMotion?.cancel();
    blockMotion = undefined;
    if (focusSequence) cancelAnimationFrame(focusSequence.frame);
    focusSequence = undefined;
    blockStrength = 1;
    zoom.cancel();
  }

  function publish() {
    window.dispatchEvent(new Event('p10-change'));
  }

  function playFocusSequence(advanceFocusLocally) {
    const sequence = {frame:0, continueToIteration:false};
    focusSequence = sequence;
    const start = performance.now();
    const frame = now => {
      if (focusSequence !== sequence) return;
      const progress = Math.min(1, (now - start) / 320);
      if (progress !== blockStrength) {
        blockStrength = progress;
        outer.setBlockFocusStrength(progress);
      }
      if (advanceFocusLocally && !taskLocated && now - start >= 320) {
        taskLocated = true;
        render();
        publish();
      }
      if (now - start < (advanceFocusLocally ? 640 : 320)) {
        sequence.frame = requestAnimationFrame(frame);
      } else {
        focusSequence = undefined;
        if (sequence.continueToIteration) change(2);
      }
    };
    sequence.frame = requestAnimationFrame(frame);
  }

  function playBlockEntry() {
    const map = surface();
    const before = {transform:map.style.transform, clipPath:map.style.clipPath};
    render();
    const animation = map.animate([before, {transform:map.style.transform, clipPath:map.style.clipPath}], {
      duration:520, easing:'cubic-bezier(.22,.78,.22,1)',
    });
    blockMotion = animation;
    animation.onfinish = () => {
      if (blockMotion !== animation) return;
      blockMotion = undefined;
    };
  }

  function change(next, id = taskId, withMotion = true, context, replayFocus = false) {
    if (!captions[next] || !pipeline.tasks[id]) return false;
    const oldBeat = beat;
    const fromView = view.contains(document.activeElement);
    stopMotion();
    if (inDeck && oldBeat === 0 && next > 0) deviceTask = taskId;
    if (inDeck && next === 0 && !context) id = deviceTask;
    select(id);
    beat = next;
    const animate = withMotion && !reducedMotion.matches;
    focus = context?.focus ?? null;
    taskLocated = next === 1 && (context?.taskLocated ?? !(animate && (oldBeat <= 1 || replayFocus)));
    if (inDeck && next === 1 && animate && !taskLocated) {
      blockStrength = 0;
      outerCamera = baseCamera;
      render();
      playFocusSequence(!context);
    } else if (!inDeck && next === 1 && animate && oldBeat <= 1) {
      beat = 0;
      render();
      beat = 1;
      playBlockEntry();
    } else if (next === 2 && animate) {
      beat = 1;
      taskLocated = true;
      render();
      const transition = zoom.prepare('in', taskId);
      beat = next;
      render();
      zoom.play(transition);
    } else {
      render();
    }
    if (fromView && oldBeat !== next) {
      axes[next].focus({preventScroll:true});
    }
    if (!context) publish();
    return true;
  }

  function reset(silent = false) {
    scenarioState = null;
    deviceTask = 48;
    select(48);
    change(0, 48, false, silent ? {} : undefined);
    outerHost.querySelector('.viz2-pipeline-viewport').scrollTop = 0;
  }

  function advance(direction) {
    if (direction > 0 && focusSequence) {
      focusSequence.continueToIteration = true;
      return true;
    }
    if (direction > 0 && (blockMotion || zoom.pending)) return true;
    const id = inDeck && beat === 0 && direction > 0 ? demoTask : taskId;
    return change(beat + direction, id, direction > 0);
  }

  function goUp() {
    if (beat > 0) advance(-1);
  }

  function init() {
    const constant = name => PTO_PYPA_SEMANTIC_GRAPH.constants[name].value;
    const config = {
      blocks:constant('ATTN_SPMD_BLOCKS'), kvHeads:constant('NUM_KV_HEADS'),
      seqLens:PtoPagedAttentionIntro.model.defaultSeqLens,
      pageTokens:constant('BLOCK_SIZE'), pagesPerStack:constant('STACK_PAGES'),
    };
    const phase0Model = PtoVisualization2Phase0.createModel({
      ...config, activeBatch:config.seqLens.length, ropeCores:constant('ROPE_CORES'),
      qPerKv:constant('NUM_HEADS') / config.kvHeads,
      lanesPerBlock:Number(PTO_PAGED_ATTENTION_PYPTO_SOURCE.match(/for aiv_id in pl\.split_aiv\((\d+)/)[1]),
    });
    execution = PtoVisualization2Pipeline.createModel({...config, phase0Model});
    anatomy = PtoVisualization2Anatomy.createModel({
      ...config, numHeads:constant('NUM_HEADS'), headDim:constant('HEAD_DIM'),
      rowTile:constant('ROW_TILE'), transferSlots:constant('TRANSFER_SLOTS'),
    });
    pipeline = PtoVisualization2PipelineIntra.createModel({anatomy, preLaunch:constant('PRE_LAUNCH')});
    if (inDeck) {
      demoTask = execution.lanes[7]?.tasks.find(({id}) => id === 55)?.id;
      if (demoTask === undefined || pipeline.tasks[demoTask]?.block !== 7) {
        throw new Error('P10 requires Task 55 to belong to Block 07 in both execution views.');
      }
    }

    outer = PtoVisualization2Pipeline.mount(outerHost, {
      model:execution,
      onSelect(selection) {
        if (!selection) return goUp();
        if (blockMotion || focusSequence) return;
        const defaultId = inDeck && selection.block === execution.tasks[demoTask].block
          ? demoTask : execution.lanes[selection.block]?.tasks[0]?.id;
        const id = selection.task ?? (task().block === selection.block ? taskId : defaultId);
        if (id == null) return;
        if (inDeck && beat === 1 && selection.kind === 'task' && (!taskLocated || id !== taskId)) {
          change(1, id, false);
        } else {
          change(beat === 0 || selection.kind === 'block' ? 1 : 2, id);
        }
      },
      onEnter(id) { if (beat <= 1 && !blockMotion && !focusSequence && !zoom.pending) change(beat + 1, id); },
      onCamera(camera) { outerCamera = camera; if (!drawing) updateOuter(); },
    });
    inner = PtoVisualization2PipelineIntra.mount(innerHost, {
      model:pipeline,
      scenario:{get:() => scenarioState, limits:() => PtoComputeScenario.limits(scenarioState), task:() => null},
      onTask(id) { change(beat, id, false); },
      onFocus(nextFocus) {
        stopMotion();
        focus = nextFocus;
        render();
        publish();
      },
      onCamera(id, camera) { innerCamera = camera; if (!drawing && insideTask()) updateInner(); },
      onInspector() {},
      onExit:goUp,
    });

    axes.forEach((button, index) => button.onclick = () => {
      if (inDeck && index > beat) {
        if (beat === 0) advance(1);
        if (index === 2) advance(1);
      } else if (index !== beat) change(index, taskId, index > beat);
    });
    $('#prev').onclick = () => advance(-1);
    $('#next').onclick = () => advance(1);
    $('#replay').onclick = () => {
      replay++;
      if (inDeck && beat === 2 && !reducedMotion.matches) {
        change(1, taskId, true, undefined, true);
        advance(1);
      } else {
        change(beat, taskId, true);
      }
    };
    $('#up').onclick = goUp;
    $('#reset').onclick = () => reset();
    if (inDeck) document.addEventListener('keydown', event => {
      if (event.key !== 'Escape' || !beat || event.target.closest?.('input,textarea,[contenteditable]')) return;
      event.preventDefault();
      event.stopPropagation();
      goUp();
    }, true);
    document.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      if (inDeck) return;
      if (event.key === 'Escape' && beat) { event.preventDefault(); goUp(); }
      if (event.target instanceof Element && event.target.closest('#view') && !['PageDown','PageUp'].includes(event.key)) return;
      if (['ArrowRight','PageDown','ArrowLeft','PageUp'].includes(event.key)) {
        event.preventDefault();
        advance(['ArrowRight','PageDown'].includes(event.key) ? 1 : -1);
      }
    });

    fitStage();
    reset(true);
    window.ptoDrilldownStory = {
      get beat() { return beat; },
      get last() { return captions.length - 1; },
      get context() { return {task:taskId, focus, replay, taskLocated}; },
      set(next, {context, animate = false} = {}) {
        if (!next && !context) { reset(true); return true; }
        const id = context?.task ?? 48;
        const sameStage = next === beat && id === taskId;
        const isReplay = context?.replay !== undefined && context.replay !== replay;
        const sameLocation = !inDeck || next !== 1 || (context?.taskLocated ?? true) === taskLocated;
        if (sameStage && sameLocation && !isReplay && JSON.stringify(context?.focus ?? null) === JSON.stringify(focus)) return true;
        replay = context?.replay ?? 0;
        return change(next, id, animate && (!sameStage || isReplay) && (next >= beat || isReplay), context ?? {});
      },
      advance,
      cancel() {
        stopMotion();
        if (stage.getBoundingClientRect().width) {
          render();
        } else {
          inner.update(taskId, null, null, {active:false});
          outer.update(null, outerCamera, {active:false});
        }
      },
    };
    new ResizeObserver(fitStage).observe($('#frame'));
    reducedMotion.addEventListener('change', () => { stopMotion(); render(); });
  }

  try {
    init();
  } catch (error) {
    $('#error').hidden = false;
    console.error(error);
  }
})();
