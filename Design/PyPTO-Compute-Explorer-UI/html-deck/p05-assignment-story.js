(function (global) {
  'use strict';

  const LAST = 4;
  const START = 0;
  const ASSIGNMENT = 1;
  const REFERENCE = 48;
  const PHOTO_CAPTION = '冲印店类比 · 按顾客和主题组织 128 张工单，再查看各工单归属的工作小组。';
  const CAPTIONS = [
    '每个 Batch × KV Head 交点生成一个 Task；seq_lens 决定 Task 内的 Page 和 Stack 划分。',
    '每个交点生成一个 Task。任务从代码中的循环和索引展开为空间关系。',
    '选中 Task 48，在 128 个工作项中定位它。',
    '查看 Task 48 的逻辑 Block 归属，以及同组的其他 Task。',
    '任务怎样生成、怎样组织、交给哪个 Block，现在可以直接观察和追踪。'
  ];

  function mount(slide,{model,onChange}) {
    const host = slide.querySelector('[data-p05-story]');
    const matrix = host.querySelector('[data-p05-matrix]');
    const narrative = global.PtoPhotoNarrative;
    const actions = slide.querySelector('[data-p05-actions]');
    const modes = [...slide.querySelectorAll('[data-p05-mode]')];
    const caption = slide.querySelector('[data-p05-caption]');
    const details = slide.querySelector('[data-p05-details]');
    const taskCount = model.constants.batch * model.constants.kvHeads;
    const taskSelection = task => ({kind:'task',task,block:model.taskInfo(task).block});
    const reference = taskSelection(REFERENCE);
    const presets = [
      {mode:'assignment',focus:'overall',selection:null},
      {mode:'assignment',focus:'overall',selection:null},
      {mode:'assignment',focus:'task',selection:reference},
      {mode:'assignment',focus:'block',selection:reference},
      {mode:'assignment',focus:'overall',selection:null}
    ];
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    const hardwareHost = host.querySelector('[data-p05-hardware]');
    const intro = global.PtoDeckHardwareStage.mount(hardwareHost,model,{reducedMotion});
    const nextButton = actions.querySelector('[data-p05-action="next"]');
    const hardwareButton = actions.querySelector('[data-p05-action="hardware"]');
    const operatorActions = actions.querySelector('[data-p05-operator-actions]');
    let restartAnimation;
    let view;
    let active = false;
    let beat = START;
    let state = presets[ASSIGNMENT];

    function addAssignmentSource() {
      const {blocks} = model.constants;
      const loopPrefix = '<span><b class="p05-py-keyword">for</b> task <b class="p05-py-keyword">in</b> pl.<b class="p05-py-call">range</b>(</span>';
      const excerpt = document.createElement('div');
      excerpt.className = 'p05-assignment-source';
      excerpt.innerHTML = `
        <section class="p05-algorithm-section">
          <h4>任务分配算法</h4>
          <div class="p05-code-surface p05-range-grid">
            <code>${loopPrefix}<span><span class="p05-range-arg">core</span>, </span><span><span class="p05-range-arg">num_tasks</span>, </span><span><span class="p05-range-arg p05-py-number">ATTN_SPMD_BLOCKS</span>):</span></code>
            <div class="p05-range-labels"><span>start</span><span>stop</span><span>step</span></div>
            <code>${loopPrefix}<span><b class="p05-py-number">0</b>, </span><span><b class="p05-py-number">${taskCount}</b>, </span><span><b class="p05-py-number">${blocks}</b>):</span></code>
          </div>
          <p>Core 0 从 0 开始，每次加 ${blocks}，只领取编号小于 ${taskCount} 的任务。</p>
        </section>`;
      matrix.querySelector('.assignment-focus-lens figcaption').after(excerpt);
    }

    function ensureViews() {
      if (view) return;
      view = global.PtoAssignmentFocus.mount(matrix,{model,title:false,onSelect(selection) {
        if (!active || beat === START || matrix.inert) return;
        state = {...state,selection,focus:state.focus === 'block' ? 'block' : 'task'};
        render();
        onChange(slide);
      }});
      addAssignmentSource();
    }

    function renderMode() {
      const isPhoto = state.mode === 'photo';
      narrative.setScope(matrix,isPhoto);
      matrix.setAttribute('aria-label',isPhoto ? '冲印店订单分配' : 'Assignment 任务矩阵');
      modes.forEach(button => button.setAttribute('aria-pressed',String(button.dataset.p05Mode === state.mode)));
      actions.querySelector('[data-p05-action="reference"]').textContent = narrative.translate('Task 48',isPhoto);
      const selection = state.selection;
      const task = selection?.task == null ? null : model.taskInfo(selection.task);
      const source = task
        ? `Task ${task.id} · Batch ${task.batch} / KV Head ${task.kv} · ${model.defaultSeqLens[task.batch]} tokens → Block ${String(task.block).padStart(2,'0')}`
        : selection ? `Block ${String(selection.block).padStart(2,'0')} · ${model.tasksForBlock(selection.block).length} Tasks` : '';
      details.textContent = isPhoto ? narrative.translate(source,true) : source.replace('Block ','逻辑 Block ');
      if (isPhoto) caption.textContent = state.focus === 'block'
        ? '工作小组表达工单的逻辑归属。'
        : PHOTO_CAPTION;
      else caption.textContent = state.focus === presets[beat].focus &&
        state.selection?.task === presets[beat].selection?.task &&
        state.selection?.kind === presets[beat].selection?.kind
        ? CAPTIONS[beat]
        : ({overall:CAPTIONS[ASSIGNMENT],task:'在整体任务空间中定位选中的 Task。',
            block:'在逻辑 Block 中查看归属与同组工作项。'})[state.focus];
    }

    function showStage(hardware) {
      slide.dataset.beat = String(beat);
      hardwareHost.hidden = !hardware;
      matrix.hidden = false;
      matrix.inert = hardware;
      nextButton.hidden = !hardware;
      operatorActions.hidden = hardware;
      hardwareButton.hidden = hardware;
    }

    function showIntro(mode) {
      restartAnimation?.cancel();
      intro.cancelFlight();
      ensureViews();
      view.pause();
      beat = START;
      state = {...presets[START],mode};
      showStage(true);
      view.update({selection:null,focus:'overall',visible:true});
      intro.setMode(mode);
      renderMode();
    }

    function render(animate = false) {
      restartAnimation?.cancel();
      ensureViews();
      showStage(false);
      view.update({selection:state.selection,focus:state.focus,animate,visible:active});
      const blockButton = actions.querySelector('[data-p05-action="block"]');
      blockButton.disabled = !state.selection;
      blockButton.textContent = state.focus === 'block' ? '任务分配' : '任务生成';
      renderMode();
    }

    function readContext(context, preset) {
      const selection = context.selection;
      const focus = context.focus === 'detail' ? 'task' : context.focus;
      if (context.scene !== 'p05-assignment' ||
          !['assignment','photo'].includes(context.mode) ||
          !['overall','task','block'].includes(focus)) return preset;
      let restored = null;
      if (selection?.kind === 'task' && Number.isInteger(selection.task) &&
          selection.task >= 0 && selection.task < taskCount) restored = taskSelection(selection.task);
      else if (selection?.kind === 'block' && Number.isInteger(selection.block) &&
          selection.block >= 0 && selection.block < model.constants.blocks) restored = {kind:'block',block:selection.block};
      return {mode:context.mode,selection:restored,focus:restored ? focus : 'overall'};
    }

    function restart(mode, animate = true) {
      intro.cancelFlight();
      beat = ASSIGNMENT;
      state = {...presets[ASSIGNMENT],mode};
      view?.pause();
      render();
      if (animate && !reducedMotion.matches) {
        restartAnimation = matrix.animate([{opacity:0},{opacity:1}],{duration:320,easing:'ease-out'});
      }
    }

    function set(index,{context,quiet = false,animate = !quiet} = {}) {
      if (!active || !Number.isInteger(index) || index < START || index > LAST) return false;
      if (index === START) {
        showIntro(context ? readContext(context,state).mode : state.mode);
      } else {
        const fromHardware = beat === START && !hardwareHost.hidden;
        intro.cancelFlight();
        const preset = {...presets[index],mode:state.mode};
        const next = context ? readContext(context,preset) : preset;
        if (fromHardware && index === ASSIGNMENT && animate && next.focus === 'overall') {
          intro.handoff(matrix,() => {
            beat = index;
            state = next;
            render();
          });
        } else if (index === ASSIGNMENT && next.mode !== state.mode && next.focus === 'overall') {
          restart(next.mode,animate);
        } else {
          beat = index;
          state = next;
          render(animate);
        }
      }
      if (!quiet) onChange(slide);
      return true;
    }

    modes.forEach(button => button.addEventListener('click',() => {
      const mode = button.dataset.p05Mode;
      if (!active || mode === state.mode) return;
      if (beat === START) showIntro(mode);
      else restart(mode);
      onChange(slide);
    }));

    actions.addEventListener('click',event => {
      const action = event.target.closest('[data-p05-action]')?.dataset.p05Action;
      if (!active || !action) return;
      if (action === 'reset' || action === 'hardware') {
        set(START);
        return;
      }
      if (action === 'next') {
        set(ASSIGNMENT);
        return;
      }
      if (beat !== START) {
        intro.cancelFlight();
        if (action === 'reference') state = {...state,selection:reference,focus:'task'};
        if (action === 'overall') state = {...state,selection:null,focus:'overall'};
        if (action === 'block' && state.selection) {
          if (state.focus === 'block') {
            const id = state.selection.task ?? model.tasksForBlock(state.selection.block)[0];
            state = {...state,selection:taskSelection(id),focus:'task'};
          } else state = {...state,focus:'block'};
        }
        render(true);
      }
      onChange(slide);
    });

    return {
      last:LAST,start:START,set,
      get beat() {return beat;},
      get context() {return {scene:'p05-assignment',...state};},
      enter(index = START,context) {
        active = true;
        state = presets[START];
        set(index,{context,quiet:true});
      },
      leave() {
        active = false;
        restartAnimation?.cancel();
        intro.cancelFlight();
        view?.pause();
        view?.update({visible:false});
      },
      pause() {
        restartAnimation?.cancel();
        intro.cancelFlight();
        view?.pause();
      }
    };
  }

  global.PtoDeckAssignmentStory = {mount};
})(window);
