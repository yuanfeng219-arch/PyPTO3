/* The live Concept C scene, shared by the review page and the Deck. */
(function (global) {
  'use strict';

  function mount(root, {model = global.PtoPagedAttentionIntro.model, title = true,
    keyboardTasks = false, onSelect = () => {}} = {}) {
    const morph = global.PtoVisualization2AssignmentMorph;
    const seqLens = model.defaultSeqLens;
    const {kvHeads, blocks, qPerKv} = model.constants;
    const config = {batch:seqLens.length, kvHeads, blocks, qPerKv};
    const layout = morph.createLayout(config);
    const total = config.batch * kvHeads;
    const fragment = kind => `<div class="assignment-focus-fragment viz2" data-focus-fragment="${kind}"></div>`;
    root.classList.add('assignment-focus');
    root.innerHTML = `
      ${fragment('matrix')}
      <div class="assignment-focus-title">
        ${title ? '<h1>从整个矩阵定位一个 Task</h1>' : ''}
        <p data-focus-overview></p>
      </div>
      <figure class="assignment-focus-lens" hidden>
        <figcaption></figcaption>${fragment('lens')}<p data-focus-source></p>
      </figure>`;
    const lens = root.querySelector('figure');
    const narrative = global.PtoPhotoNarrative;
    narrative.text(root.querySelector('[data-focus-overview]'),
      `${config.batch} Batch × ${kvHeads} KV Head · ${total} Task`);

    const views = Object.fromEntries([...root.querySelectorAll('[data-focus-fragment]')].map(host => {
      const view = morph.mount(host,{config,seqLens});
      view.pause();
      // The Deck bundles the SVG renderer without the workbench skin adapter.
      // Keep the same compact identity glyph using the existing VisualSystem.
      if (!global.PtoAIInfraPresentation?.taskGlyph) {
        view.scene.querySelectorAll('[data-pa-task]').forEach(node => {
          const id = Number(node.dataset.paTask);
          const {h} = layout.sources[id];
          const side = Math.min(14,h-6);
          const glyph = global.VisualSystem.node('task','task-'+id,'',0,0);
          node.insertAdjacentHTML('beforeend',`<svg class="ai-task-identity" pointer-events="none"
            x="4" y="${(h-side)/2}" width="${side}" height="${side}" viewBox="10 12 36 36"
            preserveAspectRatio="xMidYMid meet" overflow="hidden">${glyph}</svg>`);
        });
      }
      view.viewport.setAttribute('tabindex','-1');
      view.viewport.addEventListener('wheel',event => event.stopImmediatePropagation(),{capture:true});
      view.viewport.addEventListener('pointerdown',event => {
        if (!event.target.closest('[data-pa-task],[data-am-block-hit]')) event.stopImmediatePropagation();
      },{capture:true});
      return [host.dataset.focusFragment,{view,host,svg:host.querySelector('.viz2-am-svg')}];
    }));
    const {matrix,lens:detail} = views;
    let assignment = false;
    let active = false;

    function setCrop(view, {x,y,w,h}, padding) {
      const width = w + 2 * padding;
      const height = h + 2 * padding;
      view.svg.setAttribute('viewBox', [x-padding,y-padding,width,height].join(' '));
      view.svg.style.setProperty('--focus-view-ratio',String(width/height));
    }

    function update({selection = null, focus = 'overall', animate = false, visible = true} = {}) {
      const selectedBlock = selection?.block;
      const task = selection?.task == null ? null : model.taskInfo(selection.task);
      const nextAssignment = focus === 'block';
      root.dataset.focus = focus;
      root.dataset.lensSide = (task?.kv ?? layout.columns.findIndex(column =>
        column.x === layout.blockRects[selectedBlock]?.x)) >= Math.ceil(kvHeads / 2) ? 'left' : 'right';
      lens.hidden = focus !== 'detail' && !nextAssignment;

      for (const {view,host} of Object.values(views)) {
        view.update(selection,visible);
        host.querySelectorAll('[data-pa-task][tabindex="0"]').forEach(node => node.setAttribute('tabindex','-1'));
        const id = task?.id ?? (selectedBlock == null ? 0 : model.tasksForBlock(selectedBlock)[0]);
        host.querySelector(`[data-pa-task="${id}"]`)?.setAttribute('tabindex','0');
      }

      if (!active || assignment !== nextAssignment) {
        const target = Number(nextAssignment);
        if (animate && active) matrix.view.manual(target);
        else matrix.view.settle(target);
      }
      assignment = nextAssignment;
      active = visible;

      detail.view.settle(Number(assignment));
      if (assignment && selectedBlock != null) {
        const rect = layout.blockRects[selectedBlock];
        const block = detail.view.scene.querySelector(`[data-pa-block="${selectedBlock}"]`);
        const offset = block.parentNode.transform.baseVal.getItem(0).matrix.e;
        setCrop(detail,{...rect,x:rect.x + offset},6);
        detail.host.querySelectorAll('[data-am-block-hit]').forEach(hit => {
          if (Number(hit.dataset.amBlockHit) !== selectedBlock) hit.setAttribute('tabindex','-1');
        });
        narrative.text(lens.querySelector('figcaption'),
          `Block ${String(selectedBlock).padStart(2,'0')} · ${model.tasksForBlock(selectedBlock).length} Tasks`);
      } else if (task) {
        setCrop(detail,layout.sources[task.id],2);
        narrative.text(lens.querySelector('figcaption'),`Task ${task.id}`);
      }
      narrative.text(lens.querySelector('[data-focus-source]'),task
        ? `Batch ${task.batch} / KV Head ${task.kv} · ${seqLens[task.batch]} tokens`
        : '点选同组 Task 查看输入来源');
    }

    function selectTask(id) {
      const task = model.taskInfo(id);
      onSelect({kind:'task',task:id,block:task.block});
    }

    root.addEventListener('click',event => {
      const task = event.target.closest('[data-pa-task]');
      const block = event.target.closest('[data-am-block-hit],[data-pa-block]');
      if (task) selectTask(Number(task.dataset.paTask));
      else if (block) onSelect({kind:'block',block:Number(block.dataset.amBlockHit ?? block.dataset.paBlock)});
    });
    root.addEventListener('keydown',event => {
      const node = event.target.closest('[data-pa-task],[data-am-block-hit]');
      if (!node) return;
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        node.dispatchEvent(new MouseEvent('click',{bubbles:true}));
        return;
      }
      if (!keyboardTasks || !node.matches('[data-pa-task]') || !matrix.host.contains(node)) return;
      const delta = {ArrowLeft:-1,ArrowRight:1,ArrowUp:-kvHeads,ArrowDown:kvHeads}[event.key];
      if (delta === undefined) return;
      event.preventDefault();
      const id = Math.max(0,Math.min(total-1,Number(node.dataset.paTask)+delta));
      selectTask(id);
      matrix.host.querySelector(`[data-pa-task="${id}"]`).focus({preventScroll:true});
    });
    return {update,pause() {matrix.view.pause(); matrix.view.settle(Number(assignment));}};
  }

  global.PtoAssignmentFocus = {mount};
})(window);
