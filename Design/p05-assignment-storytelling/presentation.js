(() => {
  'use strict';

  const slide = document.querySelector('#slide');
  const screen = document.querySelector('#screen');
  const prompt = document.querySelector('#prompt');
  const fullscreen = document.querySelector('#fullscreen');
  const requested = new URLSearchParams(location.search).get('concept');
  const concept = ['a', 'b', 'c'].includes(requested) ? requested : 'a';
  document.querySelector(`[href="?concept=${concept}"]`).setAttribute('aria-current', 'page');

  const fragment = name => `<div class="fragment viz2 ${name}" data-fragment="${name}"></div>`;
  const button = (text, action, value = '') => `<button type="button" data-action="${action}" data-value="${value}">${text}</button>`;
  const cuts = {
    a: {
      title: '结构化展陈',
      prompt: '从左侧矩阵选中一个 Task。右侧显示同组 Task，也可展开全部 Block 反查。',
      markup: `
        <div class="exhibit-title"><h1>多级并行调度的<br>空间化映射</h1><p data-overview></p></div>
        <figure class="matrix-sheet"><figcaption>Batch × KV Head <span>输入矩阵</span></figcaption>${fragment('matrix')}</figure>
        <figure class="task-specimen"><figcaption data-task-title></figcaption>${fragment('specimen')}<p data-source></p></figure>
        <figure class="block-section"><figcaption><span data-block-title></span>${button('全部 Block','blocks')}</figcaption>${fragment('destination')}<p data-rule></p></figure>
        <output class="sr-only" data-status aria-live="polite"></output>`
    },
    b: {
      title: '关联路径叙事',
      prompt: '按「来源 → Task → Block」推进，也可点选源片段或 Block 内的其他 Task。',
      markup: `
        <svg class="route-lines" viewBox="0 0 1600 900" preserveAspectRatio="none" aria-hidden="true">
          <path class="route-guide" d="M 550 265 C 670 265 676 426 800 426 S 973 603 1095 603"/>
          <path class="route-progress" pathLength="100" d="M 550 265 C 670 265 676 426 800 426 S 973 603 1095 603"/>
        </svg>
        <figure class="source-window"><figcaption data-source></figcaption>${fragment('origin')}</figure>
        <div class="travel-object"><div data-task-title></div>${fragment('specimen')}<p data-rule></p></div>
        <figure class="route-destination"><figcaption data-block-title></figcaption>${fragment('destination')}</figure>
        <div class="route-caption"><h1>跟着一个 Task<br>找到执行归属</h1><output data-status aria-live="polite"></output></div>
        <div class="route-controls" role="group" aria-label="路径关键帧">
          ${button('来源','route',0)}${button('Task','route',1)}${button('Block','route',2)}${button('返回整体','overview')}
        </div>
        <figure class="route-overview" hidden><figcaption data-overview></figcaption>${fragment('matrix')}</figure>`
    },
    c: {
      title: '焦点式讲述',
      prompt: '从完整 Assignment 矩阵中点选 Task，或用 Task 48 开始。切到 Block 时观察原组件重排。',
      markup: `
        ${fragment('matrix')}
        <div class="focus-title"><h1>从整个矩阵<br>定位一个 Task</h1><p data-overview></p></div>
        <figure class="focus-lens" hidden><figcaption data-task-title></figcaption>${fragment('lens')}<p data-source></p></figure>
        <div class="focus-controls" role="group" aria-label="焦点关键帧">
          ${button('Task 48','reference')}${button('查看归属','focus-block')}${button('返回整体','overview')}
        </div>
        <output class="focus-status" data-status aria-live="polite"></output>`
    }
  };

  slide.className = `concept-${concept}`;
  slide.innerHTML = cuts[concept].markup;
  document.title = `P05 · ${cuts[concept].title} · PyPTO Assignment`;
  prompt.textContent = cuts[concept].prompt;

  if (!screen.requestFullscreen) fullscreen.hidden = true;
  fullscreen.addEventListener('click', async () => {
    try {
      await screen.requestFullscreen();
    } catch {
      prompt.textContent = '当前窗口无法进入全屏。可在独立浏览器标签页打开本页。';
    }
  });

  try {
    mountStory();
  } catch (error) {
    document.querySelector('#error').hidden = false;
    console.error(error);
  }

  function mountStory() {
    const model = window.PtoPagedAttentionIntro.model;
    const morph = window.PtoVisualization2AssignmentMorph;
    const seqLens = model.defaultSeqLens;
    const {kvHeads, blocks, qPerKv} = model.constants;
    const config = {batch: seqLens.length, kvHeads, blocks, qPerKv};
    const layout = morph.createLayout(config);
    const reference = 48;
    let selection = concept === 'c' ? null : selectionForTask(reference);
    let routeStep = 0;
    let overview = concept === 'c';
    let destinationWide = false;
    let focusBlock = false;

    const views = Object.fromEntries([...slide.querySelectorAll('[data-fragment]')].map(host => {
      const view = morph.mount(host, {config, seqLens});
      view.pause();
      view.update(selection, true);
      // The stage owns the camera. Preserve native SVG hits and selection.
      view.viewport.addEventListener('wheel', event => event.stopImmediatePropagation(), {capture:true});
      view.viewport.addEventListener('pointerdown', event => {
        if (!event.target.closest('[data-pa-task],[data-am-block-hit]')) event.stopImmediatePropagation();
      }, {capture:true});
      view.viewport.setAttribute('tabindex','-1');
      if (host.classList.contains('specimen')) host.inert = true;
      return [host.dataset.fragment, {api:view, host, svg:host.querySelector('.viz2-am-svg')}];
    }));

    function selectionForTask(id) {
      return {kind:'task', task:id, block:model.taskInfo(id).block};
    }

    function taskNode(view, id) {
      return view.api.scene.querySelector(`[data-pa-task="${id}"]`);
    }

    function box(view, x, y, w, h) {
      view.svg.setAttribute('viewBox', [x,y,w,h].join(' '));
      view.svg.style.setProperty('--view-ratio', String(w/h));
    }

    function cropTask(view, task, context = false) {
      const r = layout.sources[task.id];
      if (context) {
        const w = r.w * 2.45, h = r.h * 4.25;
        const x = Math.max(0, Math.min(r.x - r.w * .56, layout.width - w));
        const y = Math.max(0, Math.min(r.y - r.h * 1.55, layout.height - h));
        box(view, x, y, w, h);
      } else {
        box(view, r.x - 2, r.y - 2, r.w + 4, r.h + 4);
      }
    }

    function cropBlock(view, block) {
      const rect = layout.blockRects[block];
      const blockNode = view.api.scene.querySelector(`[data-pa-block="${block}"]`);
      // Read the native renderer's settled layer offset for Block Assignment.
      const offset = blockNode.parentNode.transform.baseVal.getItem(0).matrix.e;
      box(view, rect.x + offset - 6, rect.y - 6, rect.w + 12, rect.h + 12);
    }

    function fit(view) {
      box(view, 0, 0, layout.width, layout.height);
    }

    function write(selector, text) {
      slide.querySelectorAll(selector).forEach(node => {node.textContent = text;});
    }

    function render({animate = false} = {}) {
      const block = selection?.block ?? model.taskInfo(reference).block;
      const id = selection?.task ?? model.tasksForBlock(block)[0];
      const task = model.taskInfo(id);
      const count = model.tasksForBlock(block).length;
      const hasTask = selection?.kind === 'task';
      slide.dataset.step = String(routeStep);
      slide.dataset.overview = String(overview);
      slide.dataset.focus = focusBlock || selection?.kind === 'block' ? 'block' : hasTask ? 'task' : 'overall';
      slide.dataset.wide = String(destinationWide);
      slide.dataset.lensSide = task.kv >= Math.ceil(kvHeads / 2) ? 'left' : 'right';

      write('[data-overview]', `${config.batch} Batch × ${kvHeads} KV Head · ${config.batch * kvHeads} Task`);
      write('[data-task-title]', hasTask ? `Task ${task.id}` : `Block ${block}`);
      write('[data-source]', hasTask ? `Batch ${task.batch} / KV Head ${task.kv} · ${seqLens[task.batch]} tokens` : '点选同组 Task 查看输入来源');
      write('[data-block-title]', `Block ${String(block).padStart(2,'0')} · ${count} Tasks`);
      write('[data-rule]', hasTask ? `${task.id} % ${blocks} = ${block}` : `Task ID % ${blocks} → Block`);
      const status = hasTask ? `Task ${task.id} · Batch ${task.batch} / KV Head ${task.kv} → Block ${task.block}` :
        selection ? `Block ${block} · ${count} Tasks` : '点选任意 Task，定位来源与归属。';
      write('[data-status]', status);

      Object.values(views).forEach(view => {
        view.api.update(selection, true);
        view.host.querySelectorAll('[data-pa-task][tabindex="0"]').forEach(node => node.setAttribute('tabindex', '-1'));
        if (!view.host.inert) taskNode(view,id)?.setAttribute('tabindex', '0');
      });

      function limitBlockStops(view) {
        view.host.querySelectorAll('[data-am-block-hit]').forEach(hit => {
          if (Number(hit.dataset.amBlockHit) !== block) hit.setAttribute('tabindex','-1');
        });
      }

      if (concept === 'a') {
        views.destination.api.settle(1);
        if (destinationWide) fit(views.destination);
        else {
          cropBlock(views.destination,block);
          limitBlockStops(views.destination);
        }
        cropTask(views.specimen,task);
        views.specimen.host.hidden = !hasTask;
        const control = slide.querySelector('[data-action="blocks"]');
        control.textContent = destinationWide ? '当前 Block' : '全部 Block';
        control.setAttribute('aria-expanded', String(destinationWide));
      }

      if (concept === 'b') {
        views.destination.api.settle(1);
        cropBlock(views.destination,block);
        limitBlockStops(views.destination);
        cropTask(views.origin,task,true);
        if (hasTask && routeStep < 2) write('[data-rule]', `${task.batch} × ${kvHeads} + ${task.kv} = ${task.id}`);
        cropTask(views.specimen,task);
        slide.querySelector('.route-overview').hidden = !overview;
        slide.querySelector('.travel-object').hidden = !hasTask;
        slide.querySelectorAll('[data-action="route"]').forEach(control => {
          control.setAttribute('aria-pressed', String(!overview && Number(control.dataset.value) === routeStep));
        });
        if (overview) write('[data-status]', `${blocks} 个逻辑 Block · 点选 Task 开始追踪`);
      }

      if (concept === 'c') {
        const phase = focusBlock ? 1 : 0;
        if (animate) views.matrix.api.manual(phase);
        views.lens.api.settle(phase);
        if (focusBlock) {
          cropBlock(views.lens,block);
          limitBlockStops(views.lens);
        } else if (hasTask) cropTask(views.lens,task);
        slide.querySelector('.focus-lens').hidden = overview;
        slide.querySelector('[data-action="focus-block"]').disabled = !selection;
        if (focusBlock) write('[data-task-title]', `Block ${String(block).padStart(2,'0')} · ${count} Tasks`);
      }
    }

    function pick(id, view) {
      selection = selectionForTask(id);
      overview = false;
      if (concept === 'b') {
        routeStep = view === views.destination ? 2 : 0;
      }
      render();
    }

    slide.addEventListener('click', event => {
      const host = event.target.closest('[data-fragment]');
      if (host) {
        const task = event.target.closest('[data-pa-task]');
        const block = event.target.closest('[data-am-block-hit],[data-pa-block]');
        if (task) pick(Number(task.dataset.paTask),views[host.dataset.fragment]);
        else if (block) {
          selection = {kind:'block', block:Number(block.dataset.amBlockHit ?? block.dataset.paBlock)};
          if (concept === 'b') routeStep = 2;
          destinationWide = false;
          render();
        }
        return;
      }
      const control = event.target.closest('[data-action]');
      if (!control) return;
      switch (control.dataset.action) {
        case 'blocks': destinationWide = !destinationWide; break;
        case 'route':
          overview = false;
          routeStep = Number(control.dataset.value);
          if (selection?.kind !== 'task') selection = selectionForTask(selection ? model.tasksForBlock(selection.block)[0] : reference);
          break;
        case 'reference':
          selection = selectionForTask(reference);
          overview = false;
          focusBlock = false;
          break;
        case 'focus-block':
          focusBlock = !focusBlock;
          control.textContent = focusBlock ? '返回输入' : '查看归属';
          break;
        case 'overview':
          overview = true;
          focusBlock = false;
          selection = null;
          if (concept === 'c') slide.querySelector('[data-action="focus-block"]').textContent = '查看归属';
          break;
      }
      if (control.dataset.action === 'reference') slide.querySelector('[data-action="focus-block"]').textContent = '查看归属';
      render({animate:concept === 'c'});
    });

    slide.addEventListener('keydown', event => {
      const node = event.target.closest('[data-pa-task],[data-am-block-hit]');
      if (!node) return;
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        node.dispatchEvent(new MouseEvent('click',{bubbles:true}));
        return;
      }
      const host = node.closest('[data-fragment]');
      if (!node.matches('[data-pa-task]') || host.dataset.fragment !== 'matrix') return;
      const delta = {ArrowLeft:-1,ArrowRight:1,ArrowUp:-kvHeads,ArrowDown:kvHeads}[event.key];
      if (delta === undefined) return;
      event.preventDefault();
      const id = Math.max(0,Math.min(config.batch*kvHeads-1,Number(node.dataset.paTask)+delta));
      pick(id,views.matrix);
      taskNode(concept === 'b' ? views.origin : views.matrix,id).focus({preventScroll:true});
    });

    render();
  }
})();
