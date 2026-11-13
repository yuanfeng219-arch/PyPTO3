(function (global) {
  'use strict';

  function mountAssignment(slide, {graph, model, taskId, onChange}) {
    const $ = selector => slide.querySelector(selector);
    const api = global.PtoComputeScenario;
    const anatomyAPI = global.PtoVisualization2Anatomy;
    const value = name => graph.constants[name].value;
    const anatomy = anatomyAPI.createModel({
      seqLens:model.defaultSeqLens, blocks:value('ATTN_SPMD_BLOCKS'),
      numHeads:value('NUM_HEADS'), kvHeads:value('NUM_KV_HEADS'), headDim:value('HEAD_DIM'),
      rowTile:value('ROW_TILE'), pageTokens:value('BLOCK_SIZE'), pagesPerStack:value('STACK_PAGES'),
      transferSlots:value('TRANSFER_SLOTS'),
    });
    const captions = [
      '带入 Task 的真实输入，查看 Block 内的 Page 与 Stack。',
      '用原生数值框或滑杆调整 Sequence Length。',
      '查看 Page / Stack 和尾页有效数据，再切回原配置比较。',
    ];
    const host = $('[data-p14-assignment]');
    let beat = 0, active = false, state, baseline, comparison, effective, assignment;
    let focus = null, camera = null, inspector = false;
    const notify = () => onChange(slide);

    const scenario = {
      get: () => state,
      limits: () => api.limits(state),
      task: (kind,id) => kind==='assignment' && id===state.taskId ? effective : null,
      edit(kind,value) {
        const result = api.edit(state,kind,value);
        if (!result.error) apply(result.scenario,{remember:true});
        return result;
      },
      reset: () => apply(baseline),
    };

    function init(id) {
      baseline = api.create(anatomy.tasks[id]);
      const length = baseline.effectiveSeqLen > 1153 ? 1153 : 3072;
      comparison = api.edit(baseline,'sequence',String(length)).scenario;
      state = undefined;
      focus = camera = null;
      inspector = false;
    }

    function render() {
      if (!active) return;
      const viewport = host.querySelector('.viz2-anatomy-viewport');
      const height = Math.max(84,effective.stacks.length*104);
      const fitted = Math.max(.25,Math.min(1,(viewport.clientHeight-8)/height,(viewport.clientWidth-24)/368));
      assignment.update(state.taskId,focus,camera ?? {scale:fitted},{active,inspectorOpen:inspector});
      const tail = effective.pages.at(-1);
      $('[data-p14-current]').textContent = `尾页 P${tail.index} · ${tail.tokens} / ${effective.pageTokens} tokens`;
      for (const [selector,label,config] of [['[data-p14-before]','原配置',baseline],['[data-p14-after]','对比配置',comparison]]) {
        const button = $(selector);
        button.textContent = `${label} · ${config.effectiveSeqLen} tokens · ${config.pageCount} Pages / ${config.stackCount} Stacks`;
        button.setAttribute('aria-pressed',String(state.effectiveSeqLen===config.effectiveSeqLen));
      }
    }

    function apply(next,{quiet=false,remember=false}={}) {
      const changed = state?.effectiveSeqLen !== next.effectiveSeqLen;
      if (changed) {
        const original = anatomy.tasks[next.taskId];
        // Match the Assignment branch of the existing Scenario projection.
        effective = next.isCustom
          ? anatomyAPI.createModel({...anatomy,seqLens:Array.from({length:original.batch+1},()=>next.effectiveSeqLen),blockTable:null}).tasks[next.taskId]
          : original;
        state = next;
        if ((focus?.kind==='page' && !effective.pages[focus.index]) ||
            (focus?.kind==='stack' && !effective.stacks[focus.index])) {
          focus = null;
          inspector = false;
        }
      }
      if (remember && next.isCustom) comparison = next;
      render();
      if (!quiet) notify();
    }

    function set(index,{context,quiet=false}={}) {
      if (!active || !Number.isInteger(index) || !captions[index]) return false;
      const previous = beat;
      beat = index;
      slide.dataset.beat = index;
      $('.strategy-caption').textContent = captions[index];
      buttons.forEach((button,i)=>button.toggleAttribute('aria-current',i===index));
      if (context) {
        if (context.task !== baseline.taskId && anatomy.tasks[context.task]) init(context.task);
        comparison = api.edit(baseline,'sequence',String(context.comparison)).scenario ?? comparison;
        apply(api.edit(baseline,'sequence',String(context.sequence)).scenario ?? baseline,{quiet:true});
      } else if (index===0 || index===1 && previous===2 || !state) {
        apply(baseline,{quiet:true});
      } else if (index===2 && !state.isCustom) {
        apply(comparison,{quiet:true});
      }
      if (!quiet) notify();
      return true;
    }

    const buttons = captions.map((_,i)=>{
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = 'S'+i;
      button.onclick = () => set(i);
      $('.strategy-controls').append(button);
      return button;
    });
    $('[data-p14-reset]').onclick = () => { init(taskId); set(0); };
    $('[data-p14-before]').onclick = () => apply(baseline);
    $('[data-p14-after]').onclick = () => apply(comparison);

    return {
      last:captions.length-1,set,
      get beat() { return beat; },
      get context() { return {task:state.taskId,sequence:state.effectiveSeqLen,comparison:comparison.effectiveSeqLen}; },
      enter(index=0,context) {
        active = true;
        init(taskId);
        if (!assignment) {
          assignment = anatomyAPI.mount(host,{
            model:anatomy,scenario,
            onTask(id) {
              if (!active || !anatomy.tasks[id] || id===state.taskId) return;
              init(id);
              apply(baseline);
            },
            onFocus(next) { focus = next; inspector = !!next; render(); },
            onCamera(_id,next) { camera = next; render(); },
            onInspector(open) { inspector = open; render(); },
            onExit:()=>{},
          });
          new ResizeObserver(()=>{ if (active && state && !camera) render(); }).observe(host.querySelector('.viz2-anatomy-viewport'));
        }
        apply(baseline,{quiet:true});
        set(index,{context,quiet:true});
      },
      leave() {
        active = false;
        assignment.update(state.taskId,null,null,{active:false});
      },
    };
  }

  function mountDerivation(slide,{graph,model,taskId,onChange}) {
    const $ = selector => slide.querySelector(selector);
    const lines = global.PTO_PAGED_ATTENTION_PYPTO_SOURCE.split('\n');
    const constants = graph.constants;
    const pageSize = constants.BLOCK_SIZE.value, stackPages = constants.STACK_PAGES.value;
    const stackSize = pageSize*stackPages;
    const entry = model.taskInfo(taskId);
    const defaultLength = model.defaultSeqLens[entry.batch];
    const example = model.partition(defaultLength);
    const captions = [
      '从真实源码中找到决定工作规模的规则。',
      '每 Stack 容纳 4 个 Page，容量由两项源码常量推导。',
      '找到当前 Batch 的输入长度，才知道规则作用于多少数据。',
      '不足一页的数据也需要一个 Page。',
      '同样，最后一个未满的 Stack 也需要保留。',
      '如果直接把这些参数带入计算模型，会怎样？',
    ];
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
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let beat = 0, active = false, motion;
    function cancel() {
      motion?.cancel();
      motion = null;
      $('[data-p14-pause]').textContent = '暂停过渡';
    }
    function set(index,{quiet=false}={}) {
      if (!active || !Number.isInteger(index) || !captions[index]) return false;
      cancel();
      beat = index;
      slide.dataset.beat = index;
      $('.strategy-caption').textContent = captions[index];
      buttons.forEach((button,i)=>button.toggleAttribute('aria-current',i===index));
      slide.querySelectorAll('[data-p14-min]').forEach(el=>el.hidden=Number(el.dataset.p14Min)>index);
      slide.querySelectorAll('[data-p14-highlight]').forEach(el=>el.classList.toggle('current',Number(el.dataset.p14Highlight)===index));
      if (!quiet && !reduced.matches) {
        motion = slide.querySelector('[data-p14-min="'+index+'"]')?.animate([{opacity:.45},{opacity:1}],{duration:430});
      }
      if (!quiet) onChange(slide);
      return true;
    }
    const buttons = captions.map((_,i)=>{
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = 'S'+i;
      button.onclick = ()=>set(i);
      $('.strategy-controls').append(button);
      return button;
    });
    $('[data-p14-pause]').onclick = event => {
      if (motion?.playState==='paused') {
        motion.play();
        event.currentTarget.textContent = '暂停过渡';
      } else if (motion?.playState==='running') {
        motion.pause();
        event.currentTarget.textContent = '继续过渡';
      }
    };
    $('[data-p14-reset]').onclick = ()=>set(0);
    return {
      last:captions.length-1,set,
      get beat() { return beat; },
      get context() { return null; },
      enter(index=0) { active=true; set(index,{quiet:true}); },
      leave() { active=false; cancel(); },
    };
  }

  global.PtoDeckWhatIf = {mount(slide,options) {
    return slide.dataset.p14==='explore' ? mountAssignment(slide,options) : mountDerivation(slide,options);
  }};
})(window);
