const app = document.querySelector('#app');
const titles = ['多级并行调度的空间化映射', '执行流水的动态演绎', '主动计算推演'];
const numbers = ['P05', 'P07', 'P13'];
// Each entry pairs an interaction with its rationale and review criteria.
const concepts = [
  ['coordinates','任务坐标','在 Batch 与 KV Head 的交点定位任务。','以输入坐标为入口，选点后放大 Task，保留两个维度的方向感。',
   '从两个循环索引找到某个 Task，再算出它的 Block。','行列交叉高亮，把被选中的 Task 与它在 Block 内的同组任务同时呈现。','坐标和总规模始终可见，讲者可以直接指向任意 Task。',
   '16 × 8 组输入、128 个 Task、取模分配及 Block 内顺序。','矩阵行列只编码输入坐标。','适合讲清 Task 从哪里来。高保真时可接源码定位，密集矩阵需要放大模式。'],
  ['blocks','执行块索引','从一个 Block 反查它负责的输入。','以执行分组为入口，展开五到六个同组 Task，查看每项的 Batch 与 KV Head。',
   '从跨步循环枚举一个逻辑 Block 的工作项，再还原输入。','选择 Block 展开一条 Task 序列，再选择其中一项读出输入坐标。','24 个分组及 5/6 的数量差异可以扫视，局部序列揭示跨步规律。',
   '24 个逻辑 Block，Block b 处理 b、b+24、b+48…；每项保留原始 Task ID。','分组高度编码 Task 数，Task 的数据量随 Batch 变化。','适合从负载分组提问。高保真时可接分组搜索与各 Task 的 Stack 数。'],
  ['remap','原位重排','让同一批 Task 从输入坐标移动到执行分组。','跟踪身份和位置的变化，核心动作是切换编排。',
   '把输入枚举顺序和 Block 分配顺序在脑中对应起来。','两种排列使用同一批 DOM 对象，通过位移过渡保持身份。','选中 Task 后追踪它的一次移动，亲眼看见来源、去向和邻居变化。',
   '全部 128 个 Task，源序 Batch × KV Head，目标序 Block 内按 Task ID 递增。','位移路径与时长是视觉过渡。','现场适合先选一项再切换。高保真时可加跨 Graph / Pipeline 的持续选中。'],
  ['relay','结果接力','沿一个 Stack 的数据交接逐次点亮下游阶段。','镜头固定在一个 Stack，阶段之间直接标出传递对象及 READY 条件。',
   '沿生产、同步与读取代码，找出同一 Stack 的四个阶段。','选择 Stack 后步进 QK、Softmax、PV、Accumulate，突出当前交接。','听众可以同时读出生产者、数据和消费者的开始条件。',
   '同 Stack 的 score / probability / pv transfer 与三组 SET、WAIT，生产与消费相隔 2 个逻辑 Tick。','逐阶段点亮用于讲解因果，Tick 无耗时单位。','适合第一次介绍 Pipeline。高保真时可扩展到源码片段和数据对象定位。'],
  ['gate','等待条件台','先尝试消费，再检查哪个 READY 条件尚未满足。','把同步变成可触发的因果实验，观众选择检查点与操作顺序。',
   '从生产端 SET 和消费端 WAIT 推断消费者何时可以读数据。','尝试消费会给出 WAIT 条件，发布 READY 后重新尝试并看到条件满足。','用户亲手构造一次尚未就绪和一次就绪，能复述两种结果的原因。',
   '三组 READY 事件及对应 transfer，同一 Stack 的逻辑位置与 Slot 归属。','独立检查一条边，省略其他先决条件及硬件执行细节。','适合现场让听众预测。高保真时可用真实事件日志检验等待区间。'],
  ['slots','Slot 周转簿','推进逻辑 Tick，观察三个 Slot 何时交给后续 Stack。','资源地址固定，Stack 标签随逻辑迭代更新，复用发生在同一行。',
   '同时核对生产索引、消费索引和取模地址。','播放或拨动 Tick，查看生产／消费焦点及每个 Slot 对应的 Stack。','固定地址让 Stack 0 → 3 的资源复用可以原位比较。',
   'Task 48 的 6 Stacks、8 Logical Ticks、3 Slots；slot = stack % 3。','窗口按逻辑生产到消费显示，阶段实际驻留时长需运行证据。','适合讲解资源复用。高保真时可加入 transfer buffer 地址及事件对照。'],
  ['ruler','Token 标尺','在同一标尺上比较两个长度的 Page 与 Stack 划分。','连续输入范围对齐离散格子，尾页有效量用填充比例表示。',
   '判断长度变化落在同页、跨页还是跨 Stack。','并排输入基线和假设，用同一横轴读出两组分块与尾页填充。','精确位置和新增边界可直接对照，保留不足一页的差异。',
   '128 tokens / Page，4 Pages / Stack，按上取整推导并保留尾部。','横轴固定为当前场景上限 3584。','适合讲局部差异，高保真时可加入大范围缩放与区间标注。'],
  ['derive','规则逐层求值','逐步查看长度如何经两次划分传导到逻辑轮次。','公式可单独选中，输入、计算与对象在每一步绑定。',
   '找源码常量和输入、分别推导页、组、循环范围。','点选 Page、Stack 或 Tick 规则，展示基线及假设的代入式和对象。','每次只检查一次推导，便于指出分块增长在哪里转化成重复计算。',
   '页和组上取整、组容量 512、循环总轮次 Stack count + 2。','点选顺序是讲述次序。','适合解释模型来源。高保真时可把每条公式定位到源码行。'],
  ['boundary','边界实验','用边界前后一 Token 的输入检验工作规模何时改变。','优先操作规则的临界值，逐条显示最小变化的后果。',
   '预测接下来加多少 Token 会新增一页或一组。','输入任意基线，应用下一 Page 或 Stack 边界作为假设，再验证差量。','一 Token 的成对实验把输入变化与结构跳变连起来。',
   '实际边界是 128n + 1 和 512n + 1，推导保留末页和逻辑轮次。','比较只针对单个 Task，工作规模以 Page、Stack、Tick 计。','适合邀请观众先预测再验证。高保真时可保存多组实验配置。']
];
const groupOf = index => Math.floor(index / 3);
const link = id => 'index.html?concept=' + id;
const btn = (label, action, value, pressed) => `<button type="button" data-action="${action}" data-value="${value}"${pressed === undefined ? '' : ` aria-pressed="${pressed}"`}>${label}</button>`;
const mark = (label, value) => `<span class="readout"><small>${label}</small><b>${value}</b></span>`;
const taskChip = (task, selected) => btn(String(task.id).padStart(2, '0'), 'task', task.id, task.id === selected);
const stageNames = ['QK', 'Softmax', 'PV', 'Accumulate'];
const transfers = ['score_transfer', 'probability_transfer', 'pv_transfer'];
const events = ['QK_READY', 'SOFTMAX_READY', 'PV_READY'];
const pipe = model.partition(model.tasks[48].length);
const seed = index => ({task: 48, mapped: false, stack: 0, step: 0, edge: 0,
  ready: false, attempted: false, tick: 0, playing: false, before: index === 8 ? 512 : 3072,
  after: index === 8 ? 513 : 3073, rule: 0});
let state, renderActive, timer;

function assignment(index, s) {
  const task = model.tasks[s.task];
  const group = model.groups[task.block];
  const detail = `<div class="task-detail">
    <div class="object-label">Task <b>${task.id}</b></div>
    <div class="readouts">${mark('Batch',task.batch)}${mark('KV Head',task.head)}${mark('逻辑 Block',task.block)}</div>
    <p class="formula">${task.batch} × 8 + ${task.head} = ${task.id}<br>${task.id} mod 24 = ${task.block}</p></div>`;
  if (index === 0) {
    const rows = Array.from({length: model.batches}, (_, batch) =>
      `<span class="axis ${batch === task.batch ? 'hot' : ''}">${batch}</span>` +
      model.tasks.slice(batch * 8, batch * 8 + 8).map(item =>
        `<span class="cell ${item.batch === task.batch || item.head === task.head ? 'cross' : ''}">${taskChip(item,s.task)}</span>`).join('')).join('');
    return `<div class="coordinates"><div><div class="diagram-label">Batch ↓ / KV Head →</div>
      <div class="matrix"><span></span>${Array.from({length:8},(_,head)=>`<span class="axis ${head===task.head?'hot':''}">${head}</span>`).join('')}${rows}</div></div>
      <div class="coordinate-detail">${detail}<div class="diagram-label">Block ${task.block} · 同组 Task</div>
      <div class="chips">${group.map(item=>taskChip(item,s.task)).join('')}</div></div></div>`;
  }
  if (index === 1) {
    return `<div class="block-board"><div class="diagram-label">24 个逻辑 Block · 每格条纹对应一个 Task</div>
      <div class="block-index">${model.groups.map((items, block) => btn(
        `<span>Block ${block}</span><i style="--count:${items.length}"></i><b>${items.length}</b>`, 'block', block, block === task.block)).join('')}</div>
      <div class="block-detail"><div><div class="diagram-label">Block ${task.block} · 步长 24</div>
      <div class="block-sequence">${group.map(item=>taskChip(item,s.task)).join('<span>+24</span>')}</div>
      <p>Task ${task.id} · Batch ${task.batch} / KV Head ${task.head}</p></div>
      <div class="formula">for task in range(${task.block}, 128, 24)</div></div></div>`;
  }
  const pieces = model.tasks.map(item => {
    const x = s.mapped ? item.block % 6 * 15.9 + Math.floor(item.id / 24) * 2.2 : item.head * 11.7;
    const y = s.mapped ? Math.floor(item.block / 6) * 24 + 8 : item.batch * 5.7;
    return `<span class="piece ${item.id === s.task ? 'picked' : ''}" data-piece="${item.id}" style="--x:${x}%;--y:${y}%">${taskChip(item,s.task)}</span>`;
  });
  return `<div class="remap"><div class="toolbar">${btn('输入坐标','map',0,!s.mapped)}${btn('逻辑 Block','map',1,s.mapped)}
     <span>${s.mapped ? '24 个 Block · 每组从左至右按 ID 递增' : '16 个 Batch（纵向） × 8 个 KV Head（横向）'}</span></div>
     <div class="remap-board ${s.mapped ? 'mapped' : ''}">${s.mapped ? model.groups.map((_,i)=>`<div class="block-outline" style="--x:${i%6*15.9}%;--y:${Math.floor(i/6)*24}%">Block ${i}</div>`).join('') : ''}${pieces.join('')}</div>
     <div class="remap-note">Task <b>${task.id}</b><span>Batch ${task.batch} × KV Head ${task.head}</span><span>→ Block ${task.block}</span></div></div>`;
}

function pipeline(index, s) {
  const stack = pipe.stacks[s.stack];
  const label = `Stack ${s.stack} · Slot ${stack.slot} · 生产 T${stack.produce} / 消费 T${stack.consume}`;
  if (index === 3) {
    return `<div class="relay">
      <div class="toolbar"><span>追踪对象</span>${pipe.stacks.map(item=>btn('S'+item.id,'stack',item.id,item.id===s.stack)).join('')}</div>
      <div class="relay-track">${stageNames.map((name,i)=>`<div class="relay-node ${i <= s.step ? 'lit' : ''} ${i === s.step ? 'now' : ''}">
        <small>Logical Tick ${i < 2 ? stack.produce : stack.consume}</small><strong>${name}</strong></div>` +
        (i < 3 ? `<div class="relay-edge ${i < s.step ? 'lit' : ''}"><code>${transfers[i]}</code><i></i><small>SET → WAIT</small></div>` : '')).join('')}</div>
      <div class="relay-status"><b>${label}</b><p>${[
        'QK 写入 score_transfer，发布 QK_READY 后，Softmax 才能读取。',
        'Softmax 通过 WAIT 后读取 score，写入 probability 并发布 SOFTMAX_READY。PV 在消费轮检查它。',
        'PV 通过 WAIT 后读取 probability，写入 pv_transfer 并发布 PV_READY。',
        'Accumulate 通过 PV_READY WAIT 后读取结果，更新累积状态。所有 Stack 处理后写出 Task Output。'
      ][s.step]}</p></div>
      <div class="toolbar">${btn('上一阶段','phase',s.step-1)}${btn(s.step===3?'重播接力':'下一阶段','phase',s.step===3?0:s.step+1)}</div></div>`;
  }
  if (index === 4) {
    const i = s.edge;
    return `<div class="gate">
      <div class="toolbar">${transfers.map((_,edge)=>btn(stageNames[edge]+' → '+stageNames[edge+1],'edge',edge,i===edge)).join('')}</div>
      <div class="gate-flow"><div><small>生产端 · T${i<2?0:2}</small><strong>${stageNames[i]}</strong>
      ${btn('写入并 SET','publish',0)}</div>
      <div class="gate-channel ${s.ready?'ready':''}"><code>${transfers[i]}</code><i></i><b>${events[i]}</b><small>${s.ready?'已 SET':'待 SET'}</small></div>
      <div><small>消费端 · T${i===0?0:2}</small><strong>${stageNames[i+1]}</strong>${btn('尝试 WAIT / 读取','attempt',0)}</div></div>
      <output class="gate-result" aria-live="polite">${s.attempted === 'pass' ? 'WAIT 条件满足，消费者可以读取。' :
        s.attempted === 'blocked' ? (s.ready ? 'READY 已发布。再次尝试 WAIT，验证读取条件。' : 'WAIT 条件未满足。先让生产端写入并 SET。') :
        (s.ready ? 'READY 已发布。尝试 WAIT，验证读取条件。' : '先尝试读取，观察 READY 未发布时的结果。')}</output>
      <div class="toolbar"><span>Stack 0 · Slot 0 · 下一次同槽分配为 Stack 3</span>${btn('重置条件','edge',i)}</div></div>`;
  }
  const previous = pipe.stacks.find(item => item.produce === s.tick - model.slots);
  return `<div class="slot-demo">
    <div class="toolbar"><span>Logical Tick</span>${Array.from({length:pipe.ticks},(_,tick)=>btn('T'+tick,'tick',tick,tick===s.tick)).join('')}${btn(s.playing?'暂停':'播放','play',0)}</div>
    <div class="slot-list">${Array.from({length:model.slots},(_,slot)=>{
      const history = pipe.stacks.filter(item=>item.slot===slot);
      const current = history.find(item=>s.tick>=item.produce && s.tick<=item.consume);
      return `<div class="slot-row ${current?'active':''}"><strong>Slot ${slot}</strong>
        <div class="slot-history">${history.map(item=>`<div class="${item===current?'current':''} ${s.tick<item.produce?'future':''}">S${item.id}<small>T${item.produce} → T${item.consume}</small></div>`).join('<span>复用 →</span>')}</div>
        <div class="slot-state">${!current?'此轮无工作':s.tick===current.produce?'QK → Softmax · 生产':s.tick===current.consume?'PV → Accumulate · 消费':'待进入逻辑消费轮'}</div></div>`;
    }).join('')}</div>
    <p class="slot-note" aria-live="polite">${s.tick < 2 ? '预启动 · 先生产，消费从 T2 开始。' : s.tick >= pipe.stacks.length ? '排空 · 生产结束，继续消费已有 Stack。' :
      previous ? `S${s.tick} 复用 S${previous.id} 的 Slot ${previous.slot}。前者此轮生产，同时 S${s.tick-2} 进入消费轮。` : 'T2 同时包含 S2 的生产和 S0 的消费。'}
    <br><small>READY 交接 · QK → Softmax → PV → Accumulate，各边先 SET 再通过对应 WAIT</small></p></div>`;
}

function workScale(a, b) {
  return [[a.pages.length,b.pages.length,'Pages'],[a.stacks.length,b.stacks.length,'Stacks'],[a.ticks,b.ticks,'Logical Ticks']];
}
const delta = n => n > 0 ? '+' + n : n;
function tokenBand(partition, reference) {
  return `<div class="token-band">${partition.stacks.map(stack=>`<div class="stack-band" style="--width:${stack.pages.length}">
    <span>S${stack.id}</span><div>${stack.pages.map(page=>{
      const change = reference && reference.pages[page.id]?.tokens !== page.tokens;
      return `<i class="${change?'changed':''}" style="--fill:${page.tokens/model.pageSize*100}%"
        title="Page ${page.id} · ${page.tokens} / 128 tokens"></i>`;
    }).join('')}</div></div>`).join('')}</div>`;
}

function whatIf(index, s) {
  const a = model.partition(s.before), b = model.partition(s.after);
  const scale = workScale(a,b);
  const tail = p => p.pages.at(-1).tokens;
  const field = (label,key) => `<label>${label}<input type="number" name="${key}" min="1" max="${model.max}" step="1" required value="${s[key]}"></label>`;
  const controls = `<form class="toolbar compare-form">${field('基线 tokens','before')}${field('假设 tokens','after')}<button type="submit">比较</button>
    ${btn('用假设作基线','pin',0)}</form>`;
  if (index === 6) {
    return `<div class="ruler">${controls}
      <div class="ruler-chart">
      <div class="ruler-axis"><span>Token</span><div>${[0,512,1024,1536,2048,2560,3072,3584].map((n,i)=>`<span style="--position:${i/7*100}%">${n}</span>`).join('')}</div></div>
      ${[[a,'基线'],[b,'假设']].map(([p,name],i)=>`<div class="ruler-row"><span>${name}<b>${p.length}</b></span>${tokenBand(p,i?a:null)}</div>`).join('')}</div>
      <div class="ruler-bottom"><p>每格 128 tokens · 每组 4 页<br>末页有效量 <b>${tail(a)} → ${tail(b)} / 128</b></p>
      <div class="readouts">${scale.map(([x,y,name])=>mark(name,`${x} → ${y}`)).join('')}</div></div></div>`;
  }
  if (index === 7) {
    const formulas = ['ceil(seq_len / 128)', 'ceil(page_count / 4)', 'stack_count + 2'];
    const substitutions = (p) => [
      `ceil(${p.length} / 128)`, `ceil(${p.pages.length} / 4)`, `${p.stacks.length} + 2`
    ][s.rule];
    const objectStrip = p => {
      if (s.rule === 2) return `<div class="tick-strip">${Array.from({length:p.ticks},(_,i)=>`<i>T${i}</i>`).join('')}</div>`;
      if (s.rule === 1) return tokenBand(p);
      return `<div class="page-strip">${p.pages.map(page=>`<i style="--fill:${page.tokens/model.pageSize*100}%" title="Page ${page.id} · ${page.tokens} tokens"></i>`).join('')}</div>`;
    };
    return `<div class="derive">${controls}
      <div class="rule-chain">${formulas.map((formula,i)=>btn(`<b>${scale[i][2]}</b><code>${formula}</code>`,'rule',i,i===s.rule)).join('<span>→</span>')}</div>
      <div class="derivation-results">${[[a,'基线'],[b,'假设']].map(([p,name],i)=>`<div>
       <small>${name} · ${p.length} tokens</small><code>${substitutions(p)} = <b>${scale[s.rule][i]}</b></code>${objectStrip(p)}</div>`).join('')}</div>
      <p>${['128 tokens 一页。尾页有效量 '+tail(a)+' → '+tail(b)+' tokens。',
        '4 页一组。末组 '+a.stacks.at(-1).pages.length+' → '+b.stacks.at(-1).pages.length+' 页。',
        '每个 Stack 有一轮生产位置和一轮消费位置。消费滞后 2 个 Tick，循环范围是 Stack 数加 2。'][s.rule]}</p></div>`;
  }
  const pageNext = Math.ceil(a.length / model.pageSize) * model.pageSize + 1;
  const stackNext = Math.ceil(a.length / model.stackSize) * model.stackSize + 1;
  function threshold(label, next) {
    return `<div class="threshold"><small>${label}</small>${next<=model.max?
      `<strong>${next-1} <span>→</span> ${next}</strong>${btn('应用边界前后','boundary',next,s.before===next-1&&s.after===next)}`:
      '<p>当前范围内无更高边界</p>'}</div>`;
  }
  // At an exact boundary the next input is the first token in the new group.
  return `<div class="boundary">${controls}<div class="thresholds">${threshold('下一 Page 边界',pageNext)}${threshold('下一 Stack 边界',stackNext)}</div>
    <div class="boundary-results">${[['Tokens',a.length,b.length],...scale.map(([x,y,name])=>[name,x,y])].map(([name,x,y])=>
      `<div><small>${name}</small><strong>${x} <span>→</span> ${y}</strong><b>${delta(y-x)}</b></div>`).join('')}</div>
    <div class="boundary-tail"><span>末页有效量 ${tail(a)} → ${tail(b)} / 128 tokens</span>
    <span>末组 ${a.stacks.at(-1).pages.length} → ${b.stacks.at(-1).pages.length} / 4 Pages</span></div></div>`;
}
function scene(index, s) {
  const group = groupOf(index);
  return `<div class="frame"><section class="stage tone-${group} variant-${concepts[index][0]}" aria-label="${concepts[index][1]} 演示画布">
    <div class="stage-head"><h2>${concepts[index][1]}</h2><span>${group===0?'16 Batch × 8 KV Heads · 128 Tasks':group===1?'Paged Attention · Task 48':'Paged Attention · 单 Task'}</span></div>
    <div class="scene">${[assignment,pipeline,whatIf][group](index,s)}</div>
    <div class="stage-foot"><span>${['Assignment · 逻辑映射','Pipeline · 源码逻辑位置','What-if · 运行前规则推演'][group]}</span><span>${numbers[group]}</span></div>
    </section></div>`;
}
function gallery() {
  app.className = 'gallery';
  app.innerHTML = `<header class="masthead"><a href="index.html">PyPTO Compute Explorer</a><a href="review.html">业务依据与评审</a></header>
  <div class="intro"><h1>Feature<br>Storytelling<span>.</span></h1>
    <p>把代码里的计算关系，变成台上可以指给观众看的对象。<br>三个设计命题，各用三种方式演绎。</p></div>
  <nav class="jump" aria-label="设计命题">${numbers.map((n,i)=>`<a href="#${n}">${n} <span>${['任务映射','流水演绎','计算推演'][i]}</span></a>`).join('')}</nav>
  ${titles.map((title,group)=>`<section class="collection" id="${numbers[group]}"><h2><code>${numbers[group]}</code> ${title}</h2>
    <div class="tiles">${concepts.slice(group*3,group*3+3).map(([id,name,method,difference],j)=>{
      const index=group*3+j, preview=seed(index);
      if(index===2) preview.mapped=true;
      if(index===3) preview.step=1;
      if(index===5) preview.tick=3;
      return `<article><div class="thumbnail"><div inert>${scene(index,preview)}</div>
        <a href="${link(id)}" aria-label="打开 ${name}"></a></div>
        <h3><a href="${link(id)}">${name} <span aria-hidden="true">↗</span></a></h3><p>${method}</p><small>${difference}</small></article>`;
    }).join('')}</div></section>`).join('')}
  <footer class="gallery-footer">Paged Attention <a href="review.html">查看业务约束、来源和设计评审</a></footer>`;
}

function concept(index) {
  const [id,name,method,,...notes]=concepts[index];
  const group=groupOf(index);
  state=seed(index);
  app.className='concept';
  app.innerHTML=`<header class="masthead"><a href="index.html#${numbers[group]}">← 全部概念</a><span>${numbers[group]} / ${titles[group]}</span><a href="review.html">业务依据与评审</a></header>
    <div class="presentation"><div id="canvas"></div><div class="presentation-tools"><span>${method}</span>
    <button type="button" id="fullscreen" hidden>全屏演示</button></div></div>
    <section class="notes"><h1>${name}</h1>
    <dl>${['原理解任务','设计表达','理解收益','真实关系','演示范围','评审判断'].map((label,i)=>`<div><dt>${label}</dt><dd>${notes[i]}</dd></div>`).join('')}</dl>
    <nav aria-label="同题其他概念">${concepts.slice(group*3,group*3+3).filter(item=>item[0]!==id).map(item=>`<a href="${link(item[0])}">${item[1]} ↗</a>`).join('')}</nav></section>`;
  const canvas=document.querySelector('#canvas');
  const fullscreen=document.querySelector('#fullscreen');
  fullscreen.hidden=!canvas.requestFullscreen;
  fullscreen.addEventListener('click',()=>canvas.requestFullscreen());
  renderActive = ({animate = false, focus} = {}) => {
    const start = animate ? new Map([...canvas.querySelectorAll('[data-piece]')].map(el=>[
      el.dataset.piece, {el, rect:el.getBoundingClientRect()}
    ])) : null;
    canvas.innerHTML=scene(index,state);
    const previous=canvas.querySelector('[data-action="phase"][data-value="-1"]');
    if(previous) previous.disabled=true;
    if(start) {
      canvas.querySelectorAll('[data-piece]').forEach(el=>{
        const previous=start.get(el.dataset.piece);
        previous.el.style.cssText=el.style.cssText;
        el.replaceWith(previous.el);
        const from=previous.rect, to=previous.el.getBoundingClientRect();
        if(!matchMedia('(prefers-reduced-motion: reduce)').matches) previous.el.animate([{transform:`translate(${from.x-to.x}px,${from.y-to.y}px)`},{transform:'none'}],
          {duration:690,easing:'cubic-bezier(.2,.75,.24,1)'});
      });
    }
    if(focus) {
      const [action,value]=focus;
      const options=[...canvas.querySelectorAll(`[data-action="${action}"]`)];
      (options.find(el=>el.dataset.value===value&&!el.disabled)||options.find(el=>!el.disabled))?.focus({preventScroll:true});
    }
  };
  renderActive();
  function stop() {
    clearInterval(timer);
    state.playing=false;
  }
  canvas.addEventListener('click',event=>{
    const control=event.target.closest('[data-action]');
    if(!control) return;
    const {action,value}=control.dataset;
    const n=Number(value);
    const form=canvas.querySelector('form');
    if(form && action==='rule' && !form.reportValidity()) return;
    if(form && action==='pin') {
      if(!form.reportValidity()) return;
      state.after=Number(form.elements.after.value);
    }
    switch(action) {
      case 'task': state.task=n; break;
      case 'block': state.task=model.groups[n][0].id; break;
      case 'map': if(state.mapped===Boolean(n)) return; state.mapped=Boolean(n); break;
      case 'stack': state.stack=n; state.step=0; break;
      case 'phase': state.step=n; break;
      case 'edge': state.edge=n; state.ready=false; state.attempted=false; break;
      case 'publish': state.ready=true; break;
      case 'attempt': state.attempted=state.ready?'pass':'blocked'; break;
      case 'tick': stop(); state.tick=n; break;
      case 'play':
        if(state.playing) stop();
        else {
          if(state.tick===pipe.ticks-1) state.tick=0;
          state.playing=true;
          timer=setInterval(()=>{
            state.tick++;
            if(state.tick===pipe.ticks-1) stop();
            const focused = document.activeElement?.dataset.action==='play';
            renderActive({focus:focused?['play','0']:undefined});
          },1000);
        }
        break;
      case 'pin': state.before=state.after; break;
      case 'rule': state.rule=n; break;
      case 'boundary': state.before=n-1; state.after=n; break;
    }
    if(form && action==='rule') {
      state.before=Number(form.elements.before.value);
      state.after=Number(form.elements.after.value);
    }
    renderActive({animate:action==='map',focus: event.detail===0?[action,value]:undefined});
  });
  canvas.addEventListener('submit',event=>{
    event.preventDefault();
    state.before=Number(event.target.elements.before.value);
    state.after=Number(event.target.elements.after.value);
    renderActive();
    canvas.querySelector('[type="submit"]').focus({preventScroll:true});
  });
  window.addEventListener('pagehide',stop);
}

const index=concepts.findIndex(item=>item[0]===new URLSearchParams(location.search).get('concept'));
if(index<0) gallery();
else concept(index);
