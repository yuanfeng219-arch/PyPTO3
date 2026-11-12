(() => {
  'use strict';

  const $ = selector => document.querySelector(selector);
  const stage = $('#story');
  const view = $('#view');
  const motion = $('#motion');
  const inDeck = document.body.hasAttribute('data-p10-deck');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const axes = [...document.querySelectorAll('[data-p09-level]')];
  const beats = [
    {depth:0, caption:'从设备范围看准备、同步、注意力任务和写回。'},
    {depth:0, caption:'选中 Task {task}，沿它所在的执行块深入。'},
    {depth:1, caption:'聚焦 {block}，Task {task} 仍位于原来的注意力任务序列中。'},
    {depth:2, caption:'展开 Task {task}，查看逻辑轮次里的生产、消费与阶段交接。'},
    {depth:1, caption:'返回 {block}，Task {task} 回到所属执行块。'},
    {depth:0, caption:'恢复设备全貌，仍能沿 {block} 找到 Task {task}。'},
  ];
  const blockName = id => 'Block ' + String(id).padStart(2, '0');
  const baseCamera = {scale:1, x:0};
  let scale = 1;
  let beat = 0;
  let taskId = 48;
  let focus = null;
  let outerCamera = baseCamera;
  let innerCamera;
  let scenarioState;
  let renderedStacks;
  let primedTask;
  let drawing = false;
  let playing = [];
  let shared = [];
  let outer, inner, execution, pipeline, anatomy;

  const outerHost = document.createElement('div');
  const innerHost = document.createElement('div');
  outerHost.id = 'execution-map';
  innerHost.id = 'task-pipeline';
  outerHost.className = innerHost.className = 'renderer';
  const depth = () => beats[beat].depth;
  const task = () => pipeline.tasks[taskId];
  const selected = () => beat !== 0;
  const row = id => outerHost.querySelector(`[data-bem-block="${id}"]`);
  const bar = id => outerHost.querySelector(`[data-bem-task="${id}"]`);
  const surface = () => outerHost.querySelector('.viz2-pipeline-surface');

  function fitStage() {
    scale = $('#frame').clientWidth / stage.offsetWidth;
    stage.style.transform = `scale(${scale})`;
  }

  function select(id) {
    if (id === taskId && scenarioState) return;
    taskId = id;
    const tick = task().ticks.find(item => item.produce !== null && item.consume !== null) ?? task().ticks[0];
    focus = {kind:'tick', index:tick.index};
    scenarioState = PtoComputeScenario.create(anatomy.tasks[id]);
    innerCamera = undefined;
    outerCamera = baseCamera;
  }

  function positionBlock() {
    const source = row(task().block);
    const map = surface();
    if (!source || depth() === 2) return;
    const onlyRow = depth() === 1;
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
    const selection = selected() ? {kind:'task', block:task().block, task:taskId} : null;
    drawing = true;
    try {
      outer.update(selection, outerCamera, {active:depth() !== 2});
      positionBlock();
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
    const level = depth();
    const current = task();
    const name = blockName(current.block);
    const lane = execution.lanes[current.block];
    const operation = focus?.kind === 'stage'
      ? inner.getOperationDescriptor(taskId, focus.index, focus.stage) : null;
    const tick = current.ticks[operation?.tick ?? (focus?.kind === 'tick' ? focus.index : -1)];
    const context = tick ? 'Logical Tick ' + tick.index
      : focus?.kind === 'event' ? focus.event
      : focus?.kind === 'slot' ? 'Slot ' + focus.index
      : focus?.kind === 'stack' ? 'Stack ' + focus.index : '';
    const target = level === 2 ? innerHost : outerHost;
    if (view.firstChild !== target) {
      if (level === 2) outer.update(null, outerCamera, {active:false});
      else inner.update(taskId, null, null, {active:false});
      view.replaceChildren(target);
    }

    stage.dataset.level = axes[level].dataset.p09Level;
    stage.dataset.beat = beat;
    axes.forEach((button, index) => {
      button.setAttribute('aria-pressed', String(index === level));
      button.toggleAttribute('data-ancestor', index < level);
    });

    $('#block').textContent = name;
    $('#block').hidden = !selected();
    $('#task').textContent = 'Task ' + taskId;
    $('#task').hidden = !selected();
    $('#tick').textContent = context;
    $('#tick').hidden = level !== 2 || !context;
    $('#device').setAttribute('aria-current', level === 0 ? 'location' : 'false');
    $('#block').setAttribute('aria-current', level === 1 ? 'location' : 'false');
    $('#task').setAttribute('aria-current', level === 2 ? 'location' : 'false');
    $('#device-context').textContent = execution.blocks + ' Blocks';
    $('#block-context').textContent = selected() ? name : '执行块';
    $('#iteration-context').textContent = level === 2
      ? tick ? 'Tick ' + tick.index : 'Task ' + taskId
      : selected() ? 'Task ' + taskId : '逻辑轮次';
    $('#block-detail').hidden = level !== 1;
    $('#block-facts').textContent = `${lane.phase0Groups} 组预处理 · SYNCALL · ${lane.tasks.length} Tasks`;
    $('#task-heading').textContent = 'Task ' + taskId;
    $('#task-facts').textContent = `Batch ${current.batch} · KV Head ${current.kvHead} · ${current.seqLen} tokens · ${current.stacks.length} Stacks`;
    if (level === 1 && renderedStacks !== current) {
      $('#stacks').innerHTML = current.stacks.map(stack =>
        `<button type="button" class="btn btn-ghost" data-tick="${stack.produceTick}">Stack ${stack.index}<small>Slot ${stack.slot} · ${stack.tokens} tokens</small></button>`
      ).join('');
      renderedStacks = current;
    }
    const caption = beats[beat].caption.replaceAll('{task}', taskId).replaceAll('{block}', name);
    const work = level === 2 && tick && [
      tick.produce === null ? '' : `生产 Stack ${tick.produce}（QK / Softmax）`,
      tick.consume === null ? '' : `消费 Stack ${tick.consume}（PV / Accumulate）`,
    ].filter(Boolean).join('，');
    $('#caption').textContent = work ? `Tick ${tick.index} · ${work}` : caption;
    $('#prev').disabled = beat === 0;
    $('#next').disabled = beat === beats.length - 1;
    $('#up').disabled = level === 0;
    $('#legend').innerHTML = level === 2
      ? '<i class="viz2-intra-flow-key"></i>数据依赖 <i class="viz2-intra-event-key"></i>READY 事件 <i class="viz2-intra-reuse-key"></i>Slot 复用 <span>· Logical Tick 表示源码循环进度</span>'
      : 'Attention 宽度对应 Stack 数 · N↓ 表示 Normalize 与 BF16 写回';

    if (level === 2) updateInner();
    else updateOuter();
  }

  function stopMotion() {
    playing.forEach(animation => animation.cancel());
    shared.forEach(element => element.remove());
    playing = [];
    shared = [];
  }

  function animate(element, frames, duration) {
    const animation = element.animate(frames, {duration, easing:'cubic-bezier(.22,.78,.22,1)'});
    playing.push(animation);
    return animation;
  }

  function geometry(rect) {
    const root = motion.getBoundingClientRect();
    return {
      left:`${(rect.x - root.x) / scale}px`, top:`${(rect.y - root.y) / scale}px`,
      width:`${rect.width / scale}px`, height:`${rect.height / scale}px`,
    };
  }

  function morph(from, to, label) {
    if (!from?.width || !to?.width) return;
    const element = document.createElement('div');
    element.className = label ? 'shared-task' : 'shared-stack';
    element.textContent = label;
    motion.append(element);
    shared.push(element);
    const animation = animate(element, [
      {...geometry(from), opacity:1}, {...geometry(to), opacity:1, offset:.87},
      {...geometry(to), opacity:0},
    ], 620);
    animation.onfinish = () => element.remove();
  }

  function mapRect(id) {
    const entry = outer.getTransitionSnapshot()?.tasks.find(item => item.id === id);
    const button = bar(id)?.getBoundingClientRect();
    if (!entry || !button?.width) return null;
    const canvasLeft = outerHost.querySelector('canvas').getBoundingClientRect().left;
    const nativeLeft = canvasLeft + Number(bar(id).dataset.attentionX) * scale;
    const left = Math.max(nativeLeft, button.left);
    const right = Math.min(nativeLeft + entry.rect.width * scale, button.right);
    return {x:left, y:button.y, width:Math.max(0, right - left), height:button.height};
  }

  function change(next, id = taskId, withMotion = true, context) {
    if (!beats[next] || !pipeline.tasks[id]) return false;
    stopMotion();
    const oldLevel = depth();
    const nextLevel = beats[next].depth;
    const sameTask = id === taskId;
    const mapBefore = nextLevel === 2 && oldLevel !== 2 ? mapRect(id) : null;
    const detailBefore = oldLevel === 2 && nextLevel !== 2 ? inner.getTransitionSnapshot() : null;
    const chipBefore = $('#task').getBoundingClientRect();
    const mapStyle = oldLevel === 2 ? null : {
      transform:surface().style.transform, clipPath:surface().style.clipPath,
    };
    const restoreFocus = view.contains(document.activeElement);

    select(id);
    if (context) focus = context.focus ?? null;
    beat = next;
    render();
    if (!context) window.dispatchEvent(new Event('p10-change'));
    if (restoreFocus && oldLevel !== nextLevel) {
      (selected() ? $('#task') : $('#device')).focus({preventScroll:true});
    }
    if (!withMotion || reducedMotion.matches || oldLevel === nextLevel) return true;

    if (oldLevel !== 2 && nextLevel !== 2) {
      const map = surface();
      animate(map, [mapStyle, {transform:map.style.transform, clipPath:map.style.clipPath}], 520);
      if (nextLevel === 1) {
        animate($('#block-detail'), [{opacity:0, offset:0}, {opacity:0, offset:.42}, {opacity:1}], 550);
      }
      return true;
    }

    const entering = nextLevel === 2;
    const map = entering ? mapBefore : mapRect(id);
    const detail = entering ? inner.getTransitionSnapshot() : detailBefore;
    const chip = entering ? $('#task').getBoundingClientRect() : chipBefore;
    if (!map || !detail || !sameTask) return true;

    morph(entering ? map : chip, entering ? chip : map, 'Task ' + id);
    // Each segment follows its Stack index to the QK operation in the native timeline.
    detail.stacks.forEach(stack => {
      const width = map.width / detail.stacks.length;
      const segment = {...map, x:map.x + stack.index * width, width};
      morph(entering ? segment : stack.rect, entering ? stack.rect : segment, '');
    });
    const bounds = view.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, (map.x + map.width / 2 - bounds.x) / bounds.width * 100));
    const y = Math.max(0, Math.min(100, (map.y + map.height / 2 - bounds.y) / bounds.height * 100));
    animate(view.firstChild, [
      {clipPath:`inset(${y}% ${100 - x}% ${100 - y}% ${x}% round 4px)`},
      {clipPath:'inset(0% 0% 0% 0% round 0px)'},
    ], 520);
    return true;
  }

  function reset(silent = false) {
    scenarioState = null;
    select(48);
    change(0, 48, false, silent ? {focus} : undefined);
    outerHost.querySelector('.viz2-pipeline-viewport').scrollTop = 0;
  }

  function goUp() {
    if (depth() === 2) change(4);
    else if (depth() === 1) change(beat === 4 ? 5 : 1);
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

    outer = PtoVisualization2Pipeline.mount(outerHost, {
      model:execution,
      onSelect(selection) {
        if (!selection) return goUp();
        const id = selection.task ?? (task().block === selection.block
          ? taskId : execution.lanes[selection.block]?.tasks[0]?.id);
        if (id == null) return;
        change(selection.kind === 'block' || depth() === 1 ? 2 : 1, id);
      },
      onEnter(id) { change(3, id); },
      onCamera(camera) { outerCamera = camera; if (!drawing) updateOuter(); },
    });
    inner = PtoVisualization2PipelineIntra.mount(innerHost, {
      model:pipeline,
      scenario:{get:() => scenarioState, limits:() => PtoComputeScenario.limits(scenarioState), task:() => null},
      onTask(id) { change(3, id); },
      onFocus(nextFocus) {
        focus = nextFocus;
        render();
        window.dispatchEvent(new Event('p10-change'));
      },
      onCamera(id, camera) { innerCamera = camera; if (!drawing && depth() === 2) updateInner(); },
      onInspector() {},
      onExit:goUp,
    });

    axes.forEach((button, index) => button.onclick = () => change([selected() ? 5 : 0, 2, 3][index]));
    $('#device').onclick = () => change(selected() ? 5 : 0);
    $('#block').onclick = () => change(depth() === 2 ? 4 : 2);
    $('#task').onclick = () => { if (depth() !== 2) change(3); else $('#task-pipeline .viz2-anatomy-viewport').focus(); };
    $('#open-task').onclick = () => change(3);
    $('#stacks').onclick = event => {
      const stack = event.target.closest('[data-tick]');
      if (!stack) return;
      focus = {kind:'tick', index:Number(stack.dataset.tick)};
      change(3);
    };
    $('#prev').onclick = () => change(beat - 1);
    $('#next').onclick = () => change(beat + 1);
    $('#up').onclick = goUp;
    $('#reset').onclick = () => reset();
    document.addEventListener('keydown', event => {
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      if (event.key === 'Escape' && depth()) { event.preventDefault(); goUp(); }
      if (inDeck) return;
      if (event.target instanceof Element && event.target.closest('#view') && !['PageDown','PageUp'].includes(event.key)) return;
      if (['ArrowRight','PageDown','ArrowLeft','PageUp'].includes(event.key)) {
        event.preventDefault();
        change(beat + (['ArrowRight','PageDown'].includes(event.key) ? 1 : -1));
      }
    });

    fitStage();
    reset(true);
    window.ptoDrilldownStory = {
      get beat() { return beat; },
      get last() { return beats.length - 1; },
      get context() { return {task:taskId, focus}; },
      set(next, {context, animate = false} = {}) {
        if (!next && !context) { reset(true); return true; }
        const id = context?.task ?? 48;
        const snapshot = context ?? {focus:taskId === id ? focus : undefined};
        if (!context && taskId !== id && pipeline.tasks[id]) {
          select(id);
          snapshot.focus = focus;
        }
        if (next === beat && id === taskId && JSON.stringify(snapshot.focus) === JSON.stringify(focus)) return true;
        return change(next, id, animate, snapshot);
      },
      advance: direction => change(beat + direction),
      cancel: stopMotion,
    };
    new ResizeObserver(() => { stopMotion(); fitStage(); }).observe($('#frame'));
    reducedMotion.addEventListener('change', stopMotion);
  }

  try {
    init();
  } catch (error) {
    $('#error').hidden = false;
    console.error(error);
  }
})();
