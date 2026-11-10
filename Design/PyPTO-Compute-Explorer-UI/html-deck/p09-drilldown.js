/* Presentation controller. The execution map and Task timeline are the
 * original Demo controllers already bundled in the Deck for P07/P08. */
(function (global) {
  'use strict';

  function mount(slide, {graph, seqLens, initialTask, onChange}) {
    const outerAPI = global.PtoVisualization2Pipeline;
    const innerAPI = global.PtoVisualization2PipelineIntra;
    const scenarioAPI = global.PtoComputeScenario;
    const value = name => graph.constants[name].value;
    const preLaunch = value('PRE_LAUNCH');
    const config = {
      blocks: value('ATTN_SPMD_BLOCKS'), kvHeads: value('NUM_KV_HEADS'),
      seqLens, pageTokens: value('BLOCK_SIZE'), pagesPerStack: value('STACK_PAGES'),
    };
    const phase0Model = global.PtoVisualization2Phase0.createModel({
      ...config, activeBatch: seqLens.length, ropeCores: value('ROPE_CORES'),
      qPerKv: value('NUM_HEADS') / config.kvHeads,
      lanesPerBlock: Number(global.PTO_PAGED_ATTENTION_PYPTO_SOURCE.match(/for aiv_id in pl\.split_aiv\((\d+)/)[1]),
    });
    const execution = outerAPI.createModel({...config, phase0Model});
    const anatomy = global.PtoVisualization2Anatomy.createModel({
      ...config, numHeads: value('NUM_HEADS'), headDim: value('HEAD_DIM'),
      rowTile: value('ROW_TILE'), transferSlots: value('TRANSFER_SLOTS'),
    });
    const pipeline = innerAPI.createModel({anatomy, preLaunch});

    const levels = ['device','device','ai-core','iteration','device'];
    const captions = [
      '从设备范围看整体：准备工作、同步边界、注意力任务与写回。',
      '在总览中定位同一个 Task，它所在的执行块就是下一步的来源。',
      '聚焦来源 Block，查看它的局部任务与资源分配。',
      '沿同一个 Task 进入一次逻辑迭代，看清本轮的阶段与交接。',
      '回到整体，仍能沿原来的 Block 与 Task 找到刚才的局部。',
    ];
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const $ = selector => slide.querySelector(selector);
    const axes = [...slide.querySelectorAll('[data-p09-level]')];
    const playButton = $('[data-p09-play]');
    const outerHost = $('[data-p09-outer]');
    const innerHost = $('[data-p09-inner]');
    const detailPane = $('.p09-detail');

    let active = false, canPlay = false, beat = 0, playMode = 'paused', timer;
    let selectedContext, nativeFocus, inspectorOpen, scenarioState;
    let outer, inner, outerCamera, innerCameras;
    let rendering = false, innerTask, needsFit;

    function selectTask(id) {
      const task = pipeline.tasks[id];
      if (!task) return false;
      if (selectedContext?.task === id) return true;
      const tick = task.ticks.find(item => item.produce !== null && item.consume !== null)
        ?? task.ticks[0];
      selectedContext = {block:task.block, task:id, tick:tick.index};
      scenarioState = scenarioAPI.create(anatomy.tasks[id]);
      nativeFocus = {kind:'tick', index:tick.index};
      inspectorOpen = false;
      needsFit = true;
      return true;
    }

    const scenario = {
      get: () => scenarioState,
      limits: () => scenarioAPI.limits(scenarioState),
      task: () => null,
    };

    function createViews() {
      outer = outerAPI.mount(outerHost, {
        model:execution,
        onSelect(selection) {
          if (!active) return;
          if (!selection) { set(0); return; }
          const id = selection.task ?? execution.lanes[selection.block]?.tasks[0]?.id;
          if (selectTask(id)) set(2);
        },
        onEnter(id) { if (active && selectTask(id)) set(3); },
        onCamera(camera) { outerCamera = camera; if (!rendering && active) render(); },
      });
      inner = innerAPI.mount(innerHost, {
        model:pipeline, scenario,
        onTask(id) { if (active && selectTask(id)) set(3); },
        onFocus(focus) {
          if (!active) return;
          nativeFocus = focus;
          if (focus?.kind === 'tick') selectedContext.tick = focus.index;
          inspectorOpen = !!focus;
          set(3, {preserveFocus:true});
        },
        onCamera(id,camera) {
          innerCameras.set(id,camera);
          if (!rendering && active) render();
        },
        onInspector(open) {
          inspectorOpen = open;
          if (active) render();
        },
        onExit: () => set(2),
      });
    }

    function render() {
      const {block,task,tick} = selectedContext;
      const level = levels[beat];
      slide.dataset.level = level;
      slide.dataset.beat = beat;
      detailPane.inert = level !== 'iteration';
      detailPane.setAttribute('aria-hidden', String(detailPane.inert));
      axes.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.p09Level === level)));
      $('.p09-caption').textContent = captions[beat];
      $('[data-p09-step]').textContent = 'S' + beat + ' / S4';
      const blockName = 'Block ' + String(block).padStart(2,'0');
      const iteration = pipeline.tasks[task].ticks[tick];
      $('.p09-tick-summary').textContent = level === 'iteration' && nativeFocus?.kind === 'tick'
        ? [iteration.produce == null ? null : '生产 Stack ' + iteration.produce + ' · QK / Softmax',
           iteration.consume == null ? null : '消费 Stack ' + iteration.consume + ' · PV / Accumulate'].filter(Boolean).join('　|　')
        : '';
      const base = 'Paged Attention · ' + execution.blocks + ' SPMD Blocks / ' + execution.tasks.length + ' Tasks';
      $('.p09-context').textContent = beat === 0 ? base
        : 'Paged Attention / ' + blockName + ' / Task ' + task
          + (level === 'iteration' && nativeFocus?.kind === 'tick' ? ' / Logical Tick ' + tick : '');
      rendering = true;
      try {
        const selection = beat === 0 ? null : beat === 2 ? {kind:'block',block} : {kind:'task',block,task};
        outer.update(selection, outerCamera, {active});
        outerHost.querySelectorAll('[data-bem-select]').forEach(button => {
          button.toggleAttribute('data-p09-anchor', Number(button.dataset.bemSelect) === task);
        });
        if (level !== 'iteration') {
          inner.update(task,null,null,{active:false});
          return;
        }
        // The native controller clears the old focus when it builds a new Task.
        // Prime that Task before applying the selected logical tick.
        if (innerTask !== task) {
          inner.update(task,null,innerCameras.get(task),{active:true});
          innerTask = task;
        }
        inner.update(task,nativeFocus,innerCameras.get(task),{active:true,inspectorOpen});
      } finally {
        rendering = false;
      }
    }

    function stop(mode = 'paused') {
      clearTimeout(timer);
      timer = undefined;
      playMode = mode;
      playButton.textContent = '继续播放';
    }

    function set(index, {automatic = false, preserveFocus = false, notify = true} = {}) {
      if (!active || !Number.isInteger(index) || !levels[index]) return false;
      if (!automatic) stop('manual');
      const detailWasOpen = beat === 3;
      beat = index;
      if (index === 3 && !preserveFocus) {
        nativeFocus = {kind:'tick',index:selectedContext.tick};
        inspectorOpen = false;
      }
      render();
      if (index === 3 && (reduced.matches || detailWasOpen)) fitDetail();
      if (notify) onChange();
      return true;
    }

    function fitDetail() {
      if (!active || beat !== 3 || !needsFit) return;
      needsFit = false;
      const tick = innerHost.querySelector('[data-intra-kind="tick"].is-selected');
      const svg = tick?.ownerSVGElement;
      const viewport = svg?.closest('.viz2-intra-tick-scroll');
      if (!svg || !viewport?.clientWidth || !viewport.clientHeight) return;
      const scale = Math.min(.68, (viewport.clientHeight - 8) / Number(svg.dataset.intraHeight));
      const width = Number(svg.dataset.intraWidth);
      const box = tick.getBBox();
      const halfPan = Math.max(0, (width - viewport.clientWidth / scale) / 2);
      const x = Math.max(-halfPan, Math.min(halfPan, box.x + box.width / 2 - width / 2));
      innerCameras.set(selectedContext.task,{scale,x,y:0});
      render();
    }

    function schedule() {
      clearTimeout(timer);
      if (!active || !canPlay || playMode !== 'auto') return;
      timer = setTimeout(() => {
        set(beat === 4 ? 1 : beat + 1,{automatic:true});
        schedule();
      }, beat === 3 ? 6600 : 4400);
    }

    function play() {
      if (!active || !canPlay) return;
      playMode = 'auto';
      playButton.textContent = '暂停';
      schedule();
    }

    function reset(autoplay = false) {
      selectedContext = null;
      outerCamera = {scale:1,x:0};
      innerCameras = new Map();
      selectTask(initialTask);
      set(0);
      if (autoplay && !reduced.matches) play();
    }

    function apply(index, context) {
      if (context && selectTask(context.task) && pipeline.tasks[context.task].ticks[context.tick]) {
        selectedContext.tick = context.tick;
      }
      if (context && index === 3) {
        nativeFocus = context.focus ?? null;
        inspectorOpen = context.inspectorOpen === true;
      }
      set(index,{notify:false,preserveFocus:!!context});
    }

    function enter({autoplay, step, context}) {
      active = true;
      canPlay = autoplay;
      if (!outer) createViews();
      reset(!Number.isInteger(step));
      if (Number.isInteger(step)) apply(step,context);
    }

    function leave() {
      active = false;
      stop();
      outer.hideTooltip();
      outer.update(null,outerCamera,{active:false});
      inner.update(selectedContext.task,null,null,{active:false});
    }

    axes.forEach((button,i) => { button.onclick = () => set([0,2,3][i]); });
    playButton.onclick = () => playMode === 'auto' ? stop() : play();
    $('[data-p09-reset]').onclick = () => reset();
    $('.p09-workspace').addEventListener('transitionend', event => {
      if (event.target === $('.p09-workspace') && event.propertyName === 'grid-template-columns') fitDetail();
    });
    $('.p09-workspace').addEventListener('pointerdown', () => { if (playMode === 'auto') stop('manual'); });
    $('.p09-workspace').addEventListener('keydown', () => { if (playMode === 'auto') stop('manual'); });
    reduced.addEventListener('change', () => { if (reduced.matches && playMode === 'auto') stop(); });
    return {
      enter, leave, apply,
      advance: direction => set(beat + direction),
      get beat() { return beat; },
      get context() { return {...selectedContext,focus:nativeFocus,inspectorOpen}; },
    };
  }

  global.PtoDeckDrilldown = {mount};
})(window);
