(function (global) {
  'use strict';

  const LAST = 5;
  const START = 1;
  const REFERENCE = 48;
  const CAPTIONS = [
    '冲印店类比 · 按顾客和主题组织 128 张工单，再查看各工单归属的工作小组。',
    '每个交点生成一个 Task。任务从代码中的循环和索引展开为空间关系。',
    '选中 Task 48，在 128 个工作项中定位它。',
    '沿选中的 Task 回看 Batch、KV Head 和输入长度。',
    '查看 Task 48 的逻辑 Block 归属，以及同组的其他 Task。',
    '任务怎样生成、怎样组织、交给哪个 Block，现在可以直接观察和追踪。'
  ];

  function mount(slide,{model,onChange}) {
    const host = slide.querySelector('[data-p05-story]');
    const matrix = host.querySelector('[data-p05-matrix]');
    const narrative = global.PtoPhotoNarrative;
    const actions = slide.querySelector('[data-p05-actions]');
    const caption = slide.querySelector('[data-p05-caption]');
    const details = slide.querySelector('[data-p05-details]');
    const taskCount = model.constants.batch * model.constants.kvHeads;
    const taskSelection = task => ({kind:'task',task,block:model.taskInfo(task).block});
    const reference = taskSelection(REFERENCE);
    const presets = [
      {mode:'photo',focus:'overall',selection:null},
      {mode:'assignment',focus:'overall',selection:null},
      {mode:'assignment',focus:'task',selection:reference},
      {mode:'assignment',focus:'detail',selection:reference},
      {mode:'assignment',focus:'block',selection:reference},
      {mode:'assignment',focus:'overall',selection:null}
    ];
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    let labelTransitions = [];
    let view;
    let active = false;
    let beat = START;
    let state = presets[START];

    function ensureViews() {
      if (view) return;
      view = global.PtoAssignmentFocus.mount(matrix,{model,title:false,onSelect(selection) {
        if (!active) return;
        state = {...state,selection,focus:state.focus === 'block' ? 'block' : 'detail'};
        render();
        onChange(slide);
      }});
    }

    function renderMode() {
      const isPhoto = state.mode === 'photo';
      const changingMode = isPhoto !== matrix.hasAttribute('data-photo-narrative-scope');
      const labels = changingMode && !reducedMotion.matches
        ? [...matrix.querySelectorAll('text,figcaption,[data-focus-overview],[data-focus-source]'),caption,details]
          .map(label => [label,label.textContent])
        : [];
      narrative.setScope(matrix,isPhoto);
      matrix.setAttribute('aria-label',isPhoto ? '冲印店订单分配' : 'Assignment 任务矩阵');
      const toggle = actions.querySelector('[data-p05-action="mode"]');
      toggle.textContent = isPhoto ? '返回算子' : '冲印店类比';
      toggle.setAttribute('aria-pressed',String(isPhoto));
      actions.querySelector('[data-p05-action="reference"]').textContent = narrative.translate('Task 48',isPhoto);
      const selection = state.selection;
      const task = selection?.task == null ? null : model.taskInfo(selection.task);
      const source = task
        ? `Task ${task.id} · Batch ${task.batch} / KV Head ${task.kv} · ${model.defaultSeqLens[task.batch]} tokens → Block ${String(task.block).padStart(2,'0')}`
        : selection ? `Block ${String(selection.block).padStart(2,'0')} · ${model.tasksForBlock(selection.block).length} Tasks` : '';
      details.textContent = isPhoto ? narrative.translate(source,true) : source.replace('Block ','逻辑 Block ');
      if (isPhoto) caption.textContent = state.focus === 'block'
        ? '工作小组表达工单的逻辑归属。'
        : CAPTIONS[0];
      else caption.textContent = state.focus === presets[beat].focus &&
        state.selection?.task === presets[beat].selection?.task &&
        state.selection?.kind === presets[beat].selection?.kind
        ? CAPTIONS[beat === 0 ? START : beat]
        : ({overall:CAPTIONS[START],task:'在整体任务空间中定位选中的 Task。',
            detail:CAPTIONS[3],block:'在逻辑 Block 中查看归属与同组工作项。'})[state.focus];
      if (changingMode) {
        labelTransitions.forEach(transition => transition.cancel());
        labelTransitions = labels.filter(([label,before]) => label.textContent !== before)
          .map(([label]) => label.animate([{opacity:.2},{opacity:1}],{duration:380,easing:'ease-out'}));
      }
    }

    function render(animate = false) {
      slide.dataset.beat = String(beat);
      view.update({selection:state.selection,focus:state.focus,animate,visible:active});
      const blockButton = actions.querySelector('[data-p05-action="block"]');
      blockButton.disabled = !state.selection;
      blockButton.textContent = state.focus === 'block' ? '任务分配' : '任务生成';
      renderMode();
    }

    function readContext(context, preset) {
      const selection = context.selection;
      if (context.scene !== 'p05-assignment' ||
          !['assignment','photo'].includes(context.mode) ||
          !['overall','task','detail','block'].includes(context.focus)) return preset;
      let restored = null;
      if (selection?.kind === 'task' && Number.isInteger(selection.task) &&
          selection.task >= 0 && selection.task < taskCount) restored = taskSelection(selection.task);
      else if (selection?.kind === 'block' && Number.isInteger(selection.block) &&
          selection.block >= 0 && selection.block < model.constants.blocks) restored = {kind:'block',block:selection.block};
      return {mode:context.mode,selection:restored,focus:restored ? context.focus : 'overall'};
    }

    function set(index,{context,quiet = false,animate = !quiet} = {}) {
      if (!active || !Number.isInteger(index) || index < 0 || index > LAST) return false;
      beat = index;
      const next = context ? readContext(context,presets[index]) : presets[index];
      const modeOnly = context && state.focus === next.focus &&
        state.selection?.kind === next.selection?.kind &&
        state.selection?.task === next.selection?.task &&
        state.selection?.block === next.selection?.block;
      state = next;
      if (modeOnly) {
        slide.dataset.beat = String(beat);
        renderMode();
      } else render(animate);
      if (!quiet) onChange(slide);
      return true;
    }

    actions.addEventListener('click',event => {
      const action = event.target.closest('[data-p05-action]')?.dataset.p05Action;
      if (!active || !action) return;
      if (action === 'mode') {
        const isPhoto = state.mode === 'photo';
        if (isPhoto && beat === 0) {
          beat = START;
          slide.dataset.beat = String(beat);
        }
        state = {...state,mode:isPhoto ? 'assignment' : 'photo'};
        renderMode();
      } else if (action === 'reset') {
        set(START);
        return;
      } else {
        if (action === 'reference') state = {...state,selection:reference,focus:'detail'};
        if (action === 'overall') state = {...state,selection:null,focus:'overall'};
        if (action === 'block' && state.selection) {
          if (state.focus === 'block') {
            const id = state.selection.task ?? model.tasksForBlock(state.selection.block)[0];
            state = {...state,selection:taskSelection(id),focus:'detail'};
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
        ensureViews();
        active = true;
        set(index,{context,quiet:true});
        // An audience may enter on the same default state as its local copy.
        render();
      },
      leave() {
        active = false;
        labelTransitions.forEach(transition => transition.cancel());
        labelTransitions = [];
        view.pause();
        view.update({visible:false});
      },
      pause() {view?.pause();}
    };
  }

  global.PtoDeckAssignmentStory = {mount};
})(window);
