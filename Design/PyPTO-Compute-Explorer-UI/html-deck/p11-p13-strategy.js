(function (global) {
  'use strict';

  function create({graph, seqLens, initialTask, onChange}) {
    const value = name => graph.constants[name].value;
    const config = {
      batch:seqLens.length, kvHeads:value('NUM_KV_HEADS'), blocks:value('ATTN_SPMD_BLOCKS'),
      qPerKv:value('NUM_HEADS') / value('NUM_KV_HEADS'),
      numHeads:value('NUM_HEADS'), headDim:value('HEAD_DIM'), rowTile:value('ROW_TILE'),
      pageTokens:value('BLOCK_SIZE'), pagesPerStack:value('STACK_PAGES'),
      transferSlots:value('TRANSFER_SLOTS'), seqLens,
    };
    const preLaunch = value('PRE_LAUNCH');
    const anatomyAPI = global.PtoVisualization2Anatomy;
    const pipelineAPI = global.PtoVisualization2PipelineIntra;
    const scenarioAPI = global.PtoComputeScenario;
    const anatomy = anatomyAPI.createModel(config);
    const pipeline = pipelineAPI.createModel({anatomy, preLaunch});
    const source = global.PTO_PAGED_ATTENTION_PYPTO_SOURCE.split('\n');
    const loopLine = source.findIndex(line => line.includes('for tick, (m_iter'));
    const consumeLine = source.findIndex((line,index) => index > loopLine && line.includes('if tick >= PRE_LAUNCH:'));
    const loopExcerpt = [...source.slice(loopLine,loopLine + 6),source[loopLine].match(/^\s*/)[0] + '    # …',...source.slice(consumeLine,consumeLine + 2)]
      .map(line => line.slice(source[loopLine].search(/\S/))).join('\n');

    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let taskId = anatomy.tasks[initialTask] ? initialTask : anatomy.tasks[0].id;
    let scenarioState = scenarioAPI.create(anatomy.tasks[taskId]);
    const scenario = {get:()=>scenarioState, limits:()=>scenarioAPI.limits(scenarioState), task:()=>null};

    function chooseTask(id) {
      if (!anatomy.tasks[id]) return false;
      if (id !== taskId) {
        taskId = id;
        scenarioState = scenarioAPI.create(anatomy.tasks[id]);
      }
      return true;
    }

    const scripts = {
      assignment:[
        '工作项先由 Batch 和 KV Head 的组合决定。',
        '每个交点生成一个可定位的 Task。',
        '同一批 Task 按源码规则重排到逻辑执行块。',
        '选中 Task，直接看到它的输入归属与执行分组。',
        '从索引推算，转变为可观察的任务映射。',
      ],
      pipeline:[
        '从任务分配进入同一个 Task 的逻辑流水。',
        '沿一个 Stack 看结果如何从 QK 传给后续阶段。',
        '不同 Stack 的阶段按逻辑 Tick 交错推进。',
        '沿 READY 事件找到生产者与等待其结果的消费者。',
        '沿复用线查看哪些 Stack 共享同一 Transfer Slot。',
        '把隐藏在代码中的依赖和同步规则，转化为可追踪的执行过程。',
      ],
      scale:[
        '从源码的循环范围开始，区分工作批次和循环轮次。',
        '展开 Task 的真实 Stack，每层显示它负责的 KV 数据。',
        '同时看到 Stack、Logical Tick 和 Transfer Slot 的规模。',
        '从选中的 Stack 查看对应生产、消费与阶段依赖。',
        '返回整体，同一个 Stack 仍保留在它原来的位置。',
        '让代码中的循环规则和重复工作规模同时可见。',
      ],
    };

    function mount(slide) {
      const kind = slide.dataset.strategy;
      if (kind === 'scale') return global.PtoDeckStackScale.mount(slide,{
        anatomy,preLaunch,getTask:()=>taskId,chooseTask,onChange,
      });
      const captions = scripts[kind];
      const $ = selector => slide.querySelector(selector);
      const host = $('[data-strategy-primary]');
      const detail = $('[data-strategy-detail]');
      const controls = $('.strategy-controls');
      const buttons = captions.map((_, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = 'S' + index;
        button.onclick = () => set(index);
        controls.append(button);
        return button;
      });
      const reset = document.createElement('button');
      reset.type = 'button';
      reset.textContent = '重置';
      reset.onclick = () => set(0);
      controls.append(reset);
      let beat = 0, active = false, assignment, anatomyView, pipelineView;
      let focus = null, anatomyFocus = null, inspector = false, selection, pipelineTask;
      let cameras = {}, animations = [], rendering = false, listeners;

      function updateSummary() {
        const task = pipeline.tasks[taskId];
        const block = 'Block ' + String(task.block).padStart(2,'0');
        $('.strategy-context').textContent = kind === 'assignment' && beat < 3
          ? config.batch + ' Batch × ' + config.kvHeads + ' KV Head = ' + anatomy.tasks.length + ' Task'
          : 'Task ' + task.id + ' · Batch ' + task.batch + ' · KV Head ' + task.kvHead + ' · ' + block;
        if (kind === 'assignment' && selection?.kind === 'task') {
          $('.strategy-context').textContent = 'Task ' + task.id + ' · Batch ' + task.batch
            + ' · KV Head ' + task.kvHead + ' · ' + block;
        }
        if (kind === 'scale') {
          $('.strategy-counts').textContent = beat === 0
            ? '循环范围：stack_count + PRE_LAUNCH'
            : task.stacks.length + ' Stack + ' + preLaunch + ' PRE_LAUNCH = '
              + task.ticks.length + ' Logical Tick · ' + task.transferSlots + ' Transfer Slot';
          const stack = task.stacks[anatomyFocus?.index];
          if (beat === 3 && stack) $('.strategy-context').textContent +=
            ' / Stack ' + stack.index + ' · T' + stack.produceTick + ' → T' + stack.consumeTick;
          $('.strategy-stack').textContent = stack
            ? 'Stack ' + stack.index + ' · 生产 T' + stack.produceTick + ' → 消费 T' + stack.consumeTick + ' · Slot ' + stack.slot
            : 'Stack 表示数据工作批次；同一 Tick 可同时推进不同 Stack。';
        }
      }

      function cancel() {
        animations.forEach(animation => animation.cancel());
        animations = [];
      }

      function appear(element, delay, duration = 260) {
        if (!element || reduced.matches) return;
        animations.push(element.animate([{opacity:0},{opacity:1}],{delay,duration,fill:'backwards'}));
      }

      function makePipeline(target) {
        return pipelineAPI.mount(target,{
          model:pipeline, scenario,
          onTask: id => pick(id),
          onFocus(value) {
            if (!active) return;
            focus = value;
            inspector = !!value;
            if (kind === 'scale' && ['stack','stage','event'].includes(value?.kind)) {
              anatomyFocus = {kind:'stack',index:value.index};
            }
            render();
            publish();
          },
          onCamera(id,camera) { cameras.pipeline = camera; if (active && !rendering) render(); },
          onInspector(open) { inspector = open; if (active) render(); },
          onExit() { set(kind === 'scale' ? 4 : 0); },
        });
      }

      function ensureViews() {
        if (assignment || anatomyView || pipelineView) return;
        if (kind === 'assignment') {
          assignment = global.PtoVisualization2AssignmentMorph.mount(host,{config,seqLens,onEnter:id=>pick(id)});
          assignment.pause();
        } else if (kind === 'pipeline') {
          pipelineView = makePipeline(host);
        } else {
          anatomyView = anatomyAPI.mount(host,{
            model:anatomy,scenario,onTask:id=>pick(id),
            onFocus(value) {
              if (!active) return;
              if (value?.kind === 'stack' || value?.kind === 'page') {
                const index = value.kind === 'stack' ? value.index : anatomy.tasks[taskId].pages[value.index].stack;
                set(3,{stack:index});
              } else {
                anatomyFocus = value;
                render();
              }
            },
            onCamera(id,camera) { cameras.anatomy = camera; if (active) render(); },
            onInspector() {},
            onExit:()=>set(4),
          });
          pipelineView = makePipeline(detail);
          $('.strategy-source pre').textContent = loopExcerpt;
        }
      }

      function bind() {
        listeners = new AbortController();
        const options = {signal:listeners.signal};
        if (kind !== 'assignment') return;
        host.addEventListener('click',event => {
          const task = event.target.closest('[data-pa-task]');
          const block = event.target.closest('[data-am-block-hit],[data-pa-block]');
          if (task) pick(Number(task.dataset.paTask));
          else if (block) {
            selection = {kind:'block',block:Number(block.dataset.amBlockHit ?? block.dataset.paBlock)};
            render();
          }
        },options);
        host.addEventListener('keydown',event => {
          const node = event.target.closest('[data-pa-task],[data-am-block-hit]');
          if (!node) return;
          if (['Enter',' '].includes(event.key)) {
            event.preventDefault();
            node.dispatchEvent(new MouseEvent('click',{bubbles:true}));
          }
          const delta = {ArrowLeft:-1,ArrowRight:1,ArrowUp:-config.kvHeads,ArrowDown:config.kvHeads}[event.key];
          if (delta === undefined || node.dataset.paTask === undefined) return;
          event.preventDefault();
          const id = Math.max(0,Math.min(anatomy.tasks.length - 1,Number(node.dataset.paTask) + delta));
          pick(id);
          host.querySelector('[data-pa-task="' + id + '"]').focus();
        },options);
      }

      function render() {
        slide.dataset.beat = beat;
        buttons.forEach((button,i) => {
          if (i === beat) button.setAttribute('aria-current','step');
          else button.removeAttribute('aria-current');
        });
        $('.strategy-caption').textContent = captions[beat];
        updateSummary();
        if (assignment) {
          assignment.update(selection,true);
          host.querySelectorAll('[data-pa-task]').forEach(node => {
            const tabbable = beat > 0 && Number(node.dataset.paTask) === (selection?.task ?? taskId);
            node.setAttribute('tabindex',tabbable ? '0' : '-1');
          });
          const task = anatomy.tasks[taskId];
          host.querySelector('[data-viz2-selection]').textContent = selection?.kind === 'task'
            ? 'Task ' + taskId + ' · Batch ' + task.batch + ' / KV Head ' + task.kvHead
              + ' · Block ' + task.block + ' · ' + task.seqLen + ' tokens'
            : selection?.kind === 'block' ? 'Block ' + selection.block + ' · '
              + anatomy.tasks.filter(item=>item.block === selection.block).length + ' Tasks'
              : beat >= 2 ? 'Task ID % ' + config.blocks + ' → 逻辑 Block' : '按 Batch × KV Head 生成任务矩阵';
          return;
        }
        rendering = true;
        try {
          const detailOpen = kind === 'pipeline' || beat === 3;
          if (detail) {
            detail.parentElement.inert = !detailOpen;
            $('.strategy-source').inert = detailOpen;
          }
          const stackScale = Math.min(.74, 360 / (anatomy.tasks[taskId].stacks.length * 104));
          anatomyView?.update(taskId,anatomyFocus,cameras.anatomy ?? {scale:stackScale},{active,inspectorOpen:false});
          if (!detailOpen) {
            pipelineView.update(taskId,null,null,{active:false});
            return;
          }
          if (pipelineTask !== taskId) {
            const camera = kind === 'pipeline' ? {scale:.59,x:0,y:0} : null;
            pipelineView.update(taskId,null,null,{active});
            if (camera) cameras.pipeline = camera;
            pipelineTask = taskId;
          }
          pipelineView.update(taskId,focus,cameras.pipeline,{active,inspectorOpen:inspector});
        } finally {
          rendering = false;
        }
      }

      function pick(id) {
        if (!active || !chooseTask(id)) return;
        cameras = {};
        if (kind === 'assignment') {
          selection = {kind:'task',task:id,block:anatomy.tasks[id].block};
          render();
          publish();
        } else set(beat);
      }

      function publish() { onChange(slide); }

      function animateBeat() {
        if (kind === 'assignment' && beat === 1) {
          host.querySelectorAll('[data-pa-task]').forEach((node,i)=>appear(node,Math.floor(i/config.kvHeads)*38));
        }
        if (kind === 'scale' && beat === 1) {
          host.querySelectorAll('[data-anatomy-stack-container]').forEach((node,i)=>appear(node,i*115));
        }
        if (kind === 'pipeline' && beat === 2) {
          const task = pipeline.tasks[taskId];
          for (const op of task.operations) {
            appear(host.querySelector('[data-intra-operation="' + op.stack + ':' + op.stage + '"]'),op.tick*95);
          }
          for (const link of task.dependencies) {
            const producer = task.operations.find(op => op.stack === link.stack && op.stage === link.producer);
            const delay = producer.tick * 95;
            appear(host.querySelector('[data-intra-dependency="' + link.stack + ':' + link.buffer + '"]'),delay);
            appear(host.querySelector('[data-intra-index="' + link.stack + '"][data-intra-event="' + link.event + '"]'),delay);
          }
        }
        if (kind === 'pipeline' && beat === 1) {
          const stack = pipeline.tasks[taskId].stacks[0].index;
          host.querySelectorAll('[data-intra-stack-path="' + stack + '"] [data-intra-dependency]')
            .forEach((node,i)=>appear(node,i*220,400));
        }
      }

      function set(index,{stack, context, quiet = false} = {}) {
        if (!active || !captions[index] || !Number.isInteger(index)) return false;
        cancel();
        const previous = beat;
        beat = index;
        if (context) chooseTask(context.task);
        const task = pipeline.tasks[taskId];
        if (assignment) {
          selection = beat === 3 || context?.selected ? {kind:'task',task:taskId,block:task.block} : null;
          const destination = Number(beat >= 2);
          if (previous !== beat && !quiet) assignment.manual(destination);
          else assignment.settle(destination);
          if (beat === 0) assignment.fit();
        } else if (kind === 'pipeline') {
          const event = task.dependencies[0];
          const reuse = task.reuse[0];
          focus = [
            null,{kind:'stack',index:0},null,
            event && {kind:'event',index:event.stack,event:event.event,buffer:event.buffer},
            reuse && {kind:'slot',index:reuse.slot,buffer:reuse.buffer,from:reuse.from,to:reuse.to},
            null,
          ][beat] ?? null;
          inspector = beat === 3 || (beat === 4 && !!reuse);
        } else {
          const index = stack ?? context?.stack ?? anatomyFocus?.index ?? Math.min(2,task.stacks.length - 1);
          const chosen = task.stacks[index] ?? task.stacks[0];
          anatomyFocus = beat >= 3 ? {kind:'stack',index:chosen.index} : null;
          focus = beat === 3 ? anatomyFocus : null;
          inspector = false;
          if (beat === 3 && previous !== 3) cameras.pipeline = null;
        }
        if (context && kind !== 'assignment') {
          focus = context.focus;
          inspector = context.inspector === true;
        }
        render();
        if (kind === 'pipeline' && beat === 4 && !task.reuse.length) {
          $('.strategy-caption').textContent = '当前 Task 没有跨 Stack 的同槽复用。';
        }
        if (kind === 'scale' && beat === 3 && reduced.matches) fitDetail();
        if (!quiet) { animateBeat(); publish(); }
        return true;
      }

      function fitDetail() {
        if (!active || beat !== 3 || kind !== 'scale') return;
        detail.querySelector('[data-intra-fit]').click();
      }
      if (detail) {
        $('.strategy-workspace').addEventListener('transitionend',event => {
          if (event.propertyName === 'grid-template-columns' && event.target === $('.strategy-workspace')) fitDetail();
        });
      }
      return {
        last:captions.length - 1,
        get beat() {return beat;},
        get context() {return {task:taskId,stack:anatomyFocus?.index,focus,inspector,selected:selection?.kind === 'task'};},
        enter(index=0,context) {
          active = true;
          ensureViews();
          bind();
          cameras = {};
          anatomyFocus = null;
          assignment?.fit();
          set(index,{context,quiet:true});
        },
        leave() {
          active = false;
          cancel();
          listeners?.abort();
          assignment?.pause();
          assignment?.update(null,false);
          anatomyView?.update(taskId,null,null,{active:false});
          pipelineView?.update(taskId,null,null,{active:false});
        },
        set,
      };
    }
    return {mount};
  }

  global.PtoDeckStrategy = {create};
})(window);
