(function (global) {
  'use strict';

  function mount(slide, {graph, model, taskId, onChange}) {
    const live = slide.dataset.p14 === 'explore';
    const $ = selector => slide.querySelector(selector);
    const source = global.PTO_PAGED_ATTENTION_PYPTO_SOURCE;
    const lines = source.split('\n');
    const constants = graph.constants;
    const value = name => constants[name].value;
    const pageSize = value('BLOCK_SIZE'), stackPages = value('STACK_PAGES');
    const preLaunch = value('PRE_LAUNCH'), stackSize = pageSize * stackPages;
    const api = global.PtoComputeScenario;
    const anatomyAPI = global.PtoVisualization2Anatomy;
    const pipelineAPI = global.PtoVisualization2PipelineIntra;
    const anatomy = live && anatomyAPI.createModel({
      seqLens:model.defaultSeqLens, blocks:value('ATTN_SPMD_BLOCKS'),
      numHeads:value('NUM_HEADS'),kvHeads:value('NUM_KV_HEADS'),headDim:value('HEAD_DIM'),
      rowTile:value('ROW_TILE'),pageTokens:pageSize,pagesPerStack:stackPages,
      transferSlots:value('TRANSFER_SLOTS'),
    });
    const pipeline = live && pipelineAPI.createModel({anatomy,preLaunch});
    const entry = model.taskInfo(taskId);
    const defaultLength = model.defaultSeqLens[entry.batch];
    const example = model.partition(defaultLength);
    const captions = live ? [
      '把上一页的输入带进工具，计算组织直接呈现。',
      '在左下角调整 Sequence Length，输入框与滑杆都可操作。',
      '改动输入，计算模型立即重新划分工作。',
      '核对新输入对应的 Page、Stack 和尾部有效数据。',
      '同一输入同步改变 Assignment 的工作集合和 Pipeline 的逻辑轮次。',
      '来回比较两种合法输入，追踪组织变化。',
      '带入真实参数，让规则落到实例；主动调整条件，探索不同的计算组织。',
    ] : [
      '从真实源码中找到决定工作规模的规则。',
      '每 Stack 容纳 4 个 Page，容量由两项源码常量推导。',
      '找到当前 Batch 的输入长度，才知道规则作用于多少数据。',
      '不足一页的数据也需要一个 Page。',
      '同样，最后一个未满的 Stack 也需要保留。',
      '如果直接把这些参数带入计算模型，会怎样？',
    ];
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let beat = 0, active = false, state, baseline, comparison, projected;
    let assignment, timeline, timelineTask, assignmentFocus, pipelineFocus;
    let inspector = false, rendering = false, cameraA, cameraP, motions = [];
    const notify = () => onChange(slide);

    if (!live) {
      const seq = lines.findIndex(line=>line.includes('seq_len = pl.read(seq_lens, [batch])'));
      const indices = ['BLOCK_SIZE','STACK_PAGES','STACK_TOKENS'].map(name=>constants[name].line-1);
      indices.push(seq,seq+1,seq+2);
      indices.forEach((line,i)=>{
        const row = document.createElement('div');
        row.dataset.p14Highlight = [1,1,1,2,3,4][i];
        const number = document.createElement('small');
        number.textContent = String(line+1).padStart(3,'0');
        const code = document.createElement('code');
        code.textContent = lines[line].trim();
        row.append(number,code);
        $('[data-p14-source]').append(row);
      });
      const facts = [
        [1,'Stack 容量',pageSize+' × '+stackPages+' = '+stackSize+' tokens','每 Page 的 Token 容量 × 每 Stack 的 Page 容量'],
        [2,'当前输入 · seq_len',defaultLength+' tokens','Task '+taskId+' · Batch '+entry.batch+' · KV Head '+entry.kv],
        [3,'Page Count','⌈ '+defaultLength+' / '+pageSize+' ⌉ = '+example.pageCount,'按页容量划分，余数进入尾 Page'],
        [4,'Stack Count','⌈ '+defaultLength+' / '+stackSize+' ⌉ = '+example.stackCount,'最多 '+stackPages+' Pages 一组，保留不足一组的尾 Stack'],
        [5,'没有输入，缺少“计算现场”，难以带入结果','规则 → 输入 → 工作规模','开发者需要自行找参数、代入计算，再还原组织。'],
      ];
      facts.forEach(([step,label,result,detail])=>{
        const row = document.createElement('article');
        row.dataset.p14Min = step;
        for (const [tag,copy] of [['small',label],['strong',result],['p',detail]]) {
          const el = document.createElement(tag);
          el.textContent = copy;
          row.append(el);
        }
        $('[data-p14-derivation]').append(row);
      });
    }

    const scenario = {
      get:()=>state,
      limits:()=>api.limits(state),
      task:(kind,id)=>id===state.taskId ? projected[kind] : null,
      edit(kind,value) {
        const result = api.edit(state,kind,value);
        if (!result.error) apply(result.scenario,{remember:true});
        return result;
      },
      reset() {apply(baseline);},
    };

    function init(id) {
      baseline = api.create(anatomy.tasks[id]);
      const length = baseline.effectiveSeqLen > 1153 ? 1153 : 3072;
      comparison = api.edit(baseline,'sequence',String(length)).scenario;
      assignmentFocus = pipelineFocus = null;
      timelineTask = null;
      cameraA = cameraP = null;
      inspector = false;
    }

    function pick(id) {
      if (!active || !anatomy.tasks[id] || id===state.taskId) return;
      init(id);
      apply(baseline);
    }

    function mountViews() {
      assignment = anatomyAPI.mount($('[data-p14-assignment]'),{
        model:anatomy,scenario,onTask:pick,
        onFocus(focus) { assignmentFocus = focus; render(); },
        onCamera(id,camera) { cameraA = camera; if (!rendering) render(); },
        onInspector(open) { inspector = open; if (!rendering) render(); },
        onExit:()=>set(0),
      });
      timeline = pipelineAPI.mount($('[data-p14-pipeline]'),{
        model:pipeline,scenario,onTask:pick,
        onFocus(focus) { pipelineFocus = focus; render(); },
        onCamera(id,camera) { cameraP = camera; if (!rendering) render(); },
        onInspector() {},
        onExit:()=>set(0),
      });
    }

    function summarize(s) {
      const task = s === state ? projected.assignment : null;
      const tail = (s.effectiveSeqLen-1)%pageSize+1;
      const tailPages = task?.stacks.at(-1).pages.length ?? (s.pageCount-1)%stackPages+1;
      return s.effectiveSeqLen+' tokens · '+s.pageCount+' Pages · '+s.stackCount+' Stacks · '
        +(s.stackCount+preLaunch)+' Logical Ticks · 末组 '+tailPages+' Pages / 末页 '+tail+' tokens';
    }

    function render() {
      if (!active || !live || !state) return;
      rendering = true;
      try {
        assignment.update(state.taskId,assignmentFocus,cameraA ?? {scale:Math.min(.8,340/(state.stackCount*104))},{active,inspectorOpen:inspector});
        if (timelineTask !== projected.pipeline) {
          timeline.update(state.taskId,null,cameraP,{active});
          timelineTask = projected.pipeline;
        }
        timeline.update(state.taskId,pipelineFocus,cameraP,{active,inspectorOpen:!!pipelineFocus});
        $('[data-p14-current]').textContent = 'Task '+state.taskId+' · '+summarize(state);
        $('[data-p14-before]').textContent = '原配置 · '+summarize(baseline);
        $('[data-p14-after]').textContent = '对比 · '+summarize(comparison);
        for (const [selector,config] of [['[data-p14-before]',baseline],['[data-p14-after]',comparison]]) {
          $(selector).setAttribute('aria-pressed',String(state.effectiveSeqLen===config.effectiveSeqLen));
        }
      } finally { rendering = false; }
    }

    function cancel() {
      motions.forEach(motion=>motion.cancel());
      motions = [];
      $('[data-p14-pause]').textContent = '暂停过渡';
    }

    function apply(next,{quiet=false,remember=false}={}) {
      const changed = state?.taskId !== next.taskId || state.effectiveSeqLen !== next.effectiveSeqLen;
      if (changed) {
        state = next;
        projected = api.project(state,anatomy,source,preLaunch);
        assignmentFocus = pipelineFocus = null;
        cameraA = cameraP = null;
        inspector = false;
      }
      if (remember && next.isCustom) comparison = next;
      render();
      if (changed && !quiet) notify();
    }

    function set(index,{context,quiet=false}={}) {
      if (!active || !Number.isInteger(index) || !captions[index]) return false;
      cancel();
      const previous = beat;
      beat = index;
      slide.dataset.beat = index;
      $('.strategy-caption').textContent = captions[index];
      buttons.forEach((button,i)=>button.toggleAttribute('aria-current',i===index));
      if (live) {
        if (context) {
          if (context.task !== baseline.taskId && anatomy.tasks[context.task]) init(context.task);
          comparison = api.edit(baseline,'sequence',String(context.comparison)).scenario ?? comparison;
          apply(api.edit(baseline,'sequence',String(context.sequence)).scenario ?? baseline,{quiet:true});
        } else if (index === 0 || index === 1 && previous >= 2) apply(baseline,{quiet:true});
        else if (index === 2 && state.effectiveSeqLen === baseline.effectiveSeqLen && !quiet) {
          // Exercise the original Sequence Length input handler for the scripted step.
          const input = $('[data-p14-assignment] [data-compute-pages]');
          input.value = comparison.effectiveSeqLen;
          input.dispatchEvent(new Event('input',{bubbles:true}));
        } else if (index >= 2 && index <= 4 && state.effectiveSeqLen === baseline.effectiveSeqLen) {
          apply(comparison,{quiet:true});
        }
      } else {
        slide.querySelectorAll('[data-p14-min]').forEach(el=>el.hidden=Number(el.dataset.p14Min)>index);
        slide.querySelectorAll('[data-p14-highlight]').forEach(el=>el.classList.toggle('current',Number(el.dataset.p14Highlight)===index));
      }
      if (!quiet && !reduced.matches) {
        const target = live ? index===1 ? $('[data-compute-toolbar]') : $('[data-p14-current]')
          : slide.querySelector('[data-p14-min="'+index+'"]');
        if (target) motions.push(target.animate([{opacity:.45},{opacity:1}],{duration:430}));
      }
      if (!quiet) notify();
      return true;
    }

    const buttons = captions.map((_,i)=>{
      const button = document.createElement('button');
      button.type='button';
      button.textContent='S'+i;
      button.onclick=()=>set(i);
      $('.strategy-controls').append(button);
      return button;
    });
    $('[data-p14-pause]').onclick = event => {
      const ongoing = motions.filter(m=>['running','paused'].includes(m.playState));
      if (!ongoing.length) return;
      const resume = ongoing.some(m=>m.playState==='paused');
      ongoing.forEach(m=>resume?m.play():m.pause());
      event.currentTarget.textContent=resume?'暂停过渡':'继续过渡';
    };
    $('[data-p14-reset]').onclick = () => {
      if (live) init(taskId);
      set(0);
    };
    if (live) {
      $('[data-p14-before]').onclick=()=>apply(baseline);
      $('[data-p14-after]').onclick=()=>apply(comparison);
    }

    return {
      last:captions.length-1,set,
      get beat(){return beat;},
      get context(){return live ? {task:state.taskId,sequence:state.effectiveSeqLen,comparison:comparison.effectiveSeqLen} : null;},
      enter(index=0,context) {
        active = true;
        if (live) {
          init(taskId);
          state = undefined;
          if (!assignment) mountViews();
          apply(baseline,{quiet:true});
        }
        set(index,{context,quiet:true});
      },
      leave() {
        active = false;
        cancel();
        if (!live) return;
        assignment.update(state.taskId,null,null,{active:false});
        timeline.update(state.taskId,null,null,{active:false});
      },
    };
  }

  global.PtoDeckWhatIf = {mount};
})(window);
