(function (global) {
  'use strict';

  function mount(slide, {anatomy, preLaunch, getTask, chooseTask, onChange}) {
    const source = global.PTO_PAGED_ATTENTION_PYPTO_SOURCE;
    const api = global.PtoComputeScenario;
    const graphAPI = global.PtoVisualization2GraphIntra;
    const graphModel = graphAPI.createModel({anatomy, source, preLaunch});
    const $ = selector => slide.querySelector(selector);
    const graphHost = $('[data-p13-graph]');
    const assignmentHost = $('[data-p13-assignment]');
    const captions = [
      '同一个 Task：Graph 展示计算规则，Assignment 展示具体工作。',
      '选中 QK，核对执行次数与对应的 Stack 集合。',
      '准备对比配置：改变输入规模，两个视图将同步更新。',
      '逐个核对 Stack 的增减，计算图保留同一套逻辑。',
      '来回切换两种配置，比较 Stack 与逻辑轮次。',
      '选中一个 Stack，沿同一个实例查看 Graph 属性和数据范围。',
      '从静态循环规则，到可观察、可推演的重复计算规模。',
    ];
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let graph, assignment, active = false, updating = false, beat = 0;
    let baseline, alternate, state, projection, graphFocus, assignmentFocus, view, camera;
    let animations = [], exitLayer;
    const notify = () => onChange(slide);
    const scenario = {
      get: () => state,
      limits: () => api.limits(state),
      task: (kind,id) => id === state.taskId ? projection?.[kind] : null,
      edit(kind,value) {
        const result = api.edit(state,kind,value);
        if (!result.error) apply(result.scenario);
        return result;
      },
      reset() { apply(baseline); },
    };

    function initTask() {
      baseline = api.create(anatomy.tasks[getTask()]);
      const max = api.limits(baseline).stacks;
      const count = baseline.stackCount > 2 ? baseline.stackCount - 2
        : Math.min(max,baseline.stackCount + 2);
      alternate = api.edit(baseline,'stacks',String(count)).scenario ?? baseline;
      graphFocus = assignmentFocus = null;
      view = {};
      camera = null;
    }

    function configure(id) {
      if (!active || !chooseTask(id)) return;
      initTask();
      apply(baseline);
    }

    function focusGraph(focus) {
      graphFocus = focus;
      view = {...view,intraInspectorOpen:!!focus};
      const index = focus?.stackIndex ?? (focus?.kind === 'stack' ? focus.index : null);
      assignmentFocus = Number.isInteger(index) ? {kind:'stack',index} : null;
      render();
      notify();
    }

    function focusAssignment(focus) {
      assignmentFocus = focus;
      const task = projection.assignment;
      const index = focus?.kind === 'stack' ? focus.index
        : focus?.kind === 'page' ? task.pages[focus.index]?.stack : null;
      graphFocus = Number.isInteger(index) ? {kind:'stack',index,stackIndex:index} : null;
      view = {...view,intraInspectorOpen:!!graphFocus};
      render();
      notify();
    }

    function mountViews() {
      graph = graphAPI.mount(graphHost,{
        model:graphModel,scenario,onTask:configure,onFocus:focusGraph,
        onView(patch) {
          view = {...view,...patch};
          if (active && !updating) render();
        },
      });
      assignment = global.PtoVisualization2Anatomy.mount(assignmentHost,{
        model:anatomy,scenario,onTask:configure,onFocus:focusAssignment,
        onCamera(id,next) { camera = next; if (!updating) render(); },
        onInspector(open) { if (!open && assignmentFocus) focusAssignment(null); },
        onExit: () => focusAssignment(null),
      });
    }

    function render() {
      updating = true;
      try {
        const id = state.taskId;
        graph.update(id,graphFocus,view,{active});
        const scale = Math.min(.82,360 / (state.stackCount * 104));
        assignment.update(id,assignmentFocus,camera ?? {scale},{active,inspectorOpen:!!assignmentFocus});
        const task = projection.assignment;
        $('.strategy-context').textContent = 'Task ' + id + ' · Batch ' + task.batch
          + ' · ' + state.effectiveSeqLen + ' tokens · ' + state.pageCount + ' Pages · '
          + state.stackCount + ' Stacks · ' + projection.graph.tickCount + ' Logical Ticks';
        $('[data-p13-baseline]').textContent = '原配置 · ' + baseline.stackCount + ' Stacks';
        $('[data-p13-alternate]').textContent = '对比 · ' + alternate.stackCount + ' Stacks';
        $('[data-p13-baseline]').setAttribute('aria-pressed',String(state.effectiveSeqLen === baseline.effectiveSeqLen));
        $('[data-p13-alternate]').setAttribute('aria-pressed',String(state.effectiveSeqLen === alternate.effectiveSeqLen));
      } finally {
        updating = false;
      }
    }

    function cancel() {
      animations.forEach(animation => animation.cancel());
      animations = [];
      exitLayer?.remove();
      exitLayer = null;
      $('[data-p13-pause]').textContent = '暂停过渡';
    }

    function groups() {
      return new Map([...assignmentHost.querySelectorAll('[data-anatomy-stack-container]')]
        .map(node => [Number(node.dataset.anatomyStackContainer),node]));
    }

    function captureExits(previous, count) {
      if (!previous.has(count)) return;
      const svg = previous.get(count).ownerSVGElement;
      const box = svg.getBoundingClientRect();
      const pane = assignmentHost.getBoundingClientRect();
      const scale = pane.width / assignmentHost.clientWidth;
      if (!scale) return;
      exitLayer = svg.cloneNode(false);
      exitLayer.setAttribute('aria-hidden','true');
      exitLayer.classList.add('p13-exits');
      Object.assign(exitLayer.style,{
        left:(box.left-pane.left)/scale+'px', top:(box.top-pane.top)/scale+'px',
        width:box.width/scale+'px', height:box.height/scale+'px',
      });
      previous.forEach((node,id) => { if (id >= count) exitLayer.append(node.cloneNode(true)); });
    }

    function apply(next,{animate=true,publish=true}={}) {
      cancel();
      const changed = state?.taskId !== next.taskId || state.effectiveSeqLen !== next.effectiveSeqLen;
      const motion = animate && !reduced.matches && state?.taskId === next.taskId && changed;
      const before = motion ? groups() : new Map();
      const boxes = new Map([...before].map(([id,node])=>[id,node.getBoundingClientRect()]));
      if (motion) captureExits(before,next.stackCount);
      if (changed) {
        state = next;
        projection = api.project(state,anatomy,source,preLaunch);
        camera = null;
        const limit = assignmentFocus?.kind === 'page' ? state.pageCount : state.stackCount;
        if (assignmentFocus?.index >= limit) assignmentFocus = null;
        if (graphFocus?.stackIndex >= state.stackCount || graphFocus?.kind === 'stack' && graphFocus.index >= state.stackCount) graphFocus = null;
        // The native Graph controller resets focus on a new projection.
        graph.update(state.taskId,null,view,{active:false});
      }
      render();
      if (motion) {
        groups().forEach((node,id) => {
          const old = boxes.get(id);
          const rect = node.getBoundingClientRect();
          const unit = node.getScreenCTM()?.a || 1;
          const frames = old
            ? [{translate:'0 '+(old.top-rect.top)/unit+'px'},{translate:'0 0'}]
            : [{opacity:0},{opacity:1}];
          animations.push(node.animate(frames,{duration:350,delay:old ? 0 : (id-before.size)*55,fill:'backwards'}));
        });
        if (exitLayer) {
          const layer = exitLayer;
          assignmentHost.append(layer);
          const animation = layer.animate([{opacity:1},{opacity:0}],{duration:280});
          animations.push(animation);
          animation.finished.then(()=>layer.remove(),()=>{});
        }
      }
      if (publish) notify();
    }

    const buttons = captions.map((_,i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = 'S'+i;
      button.onclick = () => set(i);
      $('.strategy-controls').append(button);
      return button;
    });

    function set(index,{context,quiet=false}={}) {
      if (!active || !Number.isInteger(index) || !captions[index]) return false;
      beat = index;
      if (context && context.task !== getTask() && chooseTask(context.task)) initTask();
      buttons.forEach((button,i) => {
        if (i===index) button.setAttribute('aria-current','step');
        else button.removeAttribute('aria-current');
      });
      $('.strategy-caption').textContent = captions[index];
      const instance = Math.min(2,alternate.stackCount-1);
      graphFocus = index === 1 ? {kind:'stage',stage:'qk',index:0}
        : index === 5 ? {kind:'stack',index:instance,stackIndex:instance} : null;
      assignmentFocus = index === 5 ? {kind:'stack',index:instance} : null;
      if (context) { graphFocus = context.graphFocus; assignmentFocus = context.assignmentFocus; }
      view = {...view,intraInspectorOpen:!!graphFocus};
      const next = context?.sequence
        ? api.edit(baseline,'sequence',String(context.sequence)).scenario
        : index < 3 || index === 6 ? baseline : alternate;
      apply(next ?? baseline,{animate:!quiet,publish:!quiet});
      return true;
    }

    $('[data-p13-baseline]').onclick = () => apply(baseline);
    $('[data-p13-alternate]').onclick = () => apply(alternate);
    $('[data-p13-pause]').onclick = event => {
      const running = animations.filter(animation => ['running','paused'].includes(animation.playState));
      if (!running.length) return;
      const paused = running.some(animation => animation.playState === 'paused');
      running.forEach(animation => {
        if (paused) animation.play();
        else animation.pause();
      });
      event.currentTarget.textContent = paused ? '暂停过渡' : '继续过渡';
    };
    $('[data-p13-reset]').onclick = () => { initTask(); set(0); };

    return {
      last:6,set,
      get beat() { return beat; },
      get context() { return {task:getTask(),sequence:state.effectiveSeqLen,graphFocus,assignmentFocus}; },
      enter(index=0,context) {
        active = true;
        initTask();
        if (!graph) mountViews();
        set(index,{context,quiet:true});
      },
      leave() {
        active = false;
        cancel();
        graph.update(state.taskId,null,{}, {active:false});
        assignment.update(state.taskId,null,null,{active:false});
      },
    };
  }
  global.PtoDeckStackScale = {mount};
})(window);
