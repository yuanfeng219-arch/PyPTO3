// Original evidence is supplied by build_memory_reuse_case.mjs; geometry is schematic.
window.MEMORY_REUSE_CASE = (() => {
  const esc = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const kb = bytes => (bytes / 1024).toLocaleString('en', {maximumFractionDigits: 3}) + ' KiB';
  const disclosure = (title, sources, source) => '<details class="pass-disclosure"><summary>' + title + '</summary>' + sources.map(source).join('') + '</details>';
  function render(data, index, bufferId, valueId, source) {
    const proof = disclosure('原始诊断与 kernel 位置映射', [data.warning, data.mapping], source);
    if (index === 0) {
      const v=window.VisualSystem;
      const body='<text x="24" y="30" fill="#aaa" font-size="14">P29 · 请求深度 2</text><text x="530" y="30" fill="#aaa" font-size="14">P35 · 满足深度 1</text>'
        +v.memory('stage0','stage 0','独立存储',24,65,190,100,true)+v.memory('stage1','stage 1','独立存储',24,200,190,100,true)
        +v.edge(214,115,530,182,'mapping')+v.edge(214,250,530,182,'mapping')
        +v.memory('shared','stage 0 / stage 1','共享分配 · 有序访问',530,125,300,114,true);
      return '<div class="mr-diagram" style="--diagram-width:880px">'+v.wrap(body,880,330)+'</div><p class="mr-caption">两个阶段从独立存储映射到共享物理分配。隔离深度为 1；kernel 内仍可包含多个缓冲。</p>'+proof;
    }
    if (index === 1) {
      const buffer = data.buffers.find(b => b.id === bufferId) || data.buffers[0];
      const selected = buffer.values.find(v => v.id === valueId) || buffer.values[0];
      const node = value => '<button type="button" class="mr-value mr-value-object' + (selected.id === value.id ? ' selected' : '') + '" data-memory-value="' + esc(value.id) + '" aria-label="' + esc(value.label+' · '+value.id) + '" aria-pressed="' + (selected.id === value.id) + '">' + window.VisualSystem.wrap(window.VisualSystem.data({id:value.id,label:({load:'载入结果',mul:'加权结果',add:'累加结果'}[value.label]||value.label),detail:'stage '+value.stage+' · '+kb(value.bytes),role:'activation'},0,0),160,64) + '</button>';
      const physical = (id, values) => '<div class="mr-physical" style="--value-count:' + values.length + '"><header>' + window.VisualSystem.wrap(window.VisualSystem.memory(id,id,'物理分配',0,0,180,64,true),180,64) + '</header>' + values.map(node).join('') + '</div>';
      const before = [...new Set(buffer.values.map(v => v.beforeBuffer))];
      return '<div class="mr-buffer-tabs" role="group" aria-label="选择跨阶段共享缓冲">' + data.buffers.map(b => '<button type="button" class="ctl' + (b.id === buffer.id ? ' on' : '') + '" data-memory-buffer="' + b.id + '" aria-pressed="' + (b.id === buffer.id) + '">' + b.id + '<small>' + kb(b.bytes) + ' · ' + b.values.length + ' 个计算值</small></button>').join('') + '</div><p class="mr-caption">绿色图元：计算值　蓝色网格：物理存储　点击计算值联动两侧与代码证据。</p><div class="mr-comparison"><article><h4>P34 · 复用前 / ' + before.length + ' 个物理缓冲</h4>' + before.map(id => physical(id, buffer.values.filter(v => v.beforeBuffer === id))).join('') + '</article><article><h4>P35 · 复用后 / 1 个物理缓冲</h4>' + physical(buffer.id, buffer.values) + '<p class="mr-alias-note">两个阶段落到同一物理缓冲；访问必须服从生命周期与数据依赖顺序。</p></article></div><section class="object-inspector"><small>当前观测对象 · stage ' + selected.stage + ' · ' + esc(selected.op) + '</small><h4><code>' + esc(selected.id) + '</code></h4><div class="object-property"><span>' + esc(selected.beforeBuffer) + '</span><b>→</b><span>' + esc(selected.afterBuffer) + '</span></div>' + disclosure('查看该值的 P34 / P35 原始映射与物理分配', [selected.before, selected.after, buffer.allocation], source) + '</section><p class="mr-caption">图中只列计算值；view / reshape 别名未重复列出。阶段编号来自 P34 的 pipeline_membership，P35 中该属性已被消费。复用关系不等于每个访问都是可去除的假依赖。</p>';
    }
    const v=window.VisualSystem;
    const text=(x,y,label,size=18,color='#ddd')=>`<text x="${x}" y="${y}" font-size="${size}" fill="${color}">${esc(label)}</text>`;
    if (index === 2) {
      const body=text(24,30,'单组需求',18)+text(24,70,'64 KiB',24,'#fff')+text(24,98,'2 个阶段 × 32 KiB',16,'#aaa')
        +v.memory('group','当前流水组','64 KiB',24,138,230,106,true)
        +v.memory('others','同驻缓冲','峰值集合待补齐',24,282,230,106,true)
        +v.edge(254,191,400,245,'binding')+v.edge(254,335,400,245,'binding')
        +v.memory('pool','184 KiB','可规划容量 · 非剩余空闲',400,190,350,112,true)
        +text(400,145,'共同竞争存储空间',18,'#fff')
        +text(400,345,'诊断：同驻资源压力',20,'#f2c45f')
        +text(400,386,'隔离深度  2 → 1',24,'#fff');
      return '<p class="mr-takeaway">单组能放下，整体规划仍因同驻压力回退。</p><div class="mr-diagram mr-large-diagram" style="--diagram-width:820px">'+v.wrap(body,820,430)+'</div><details class="pass-disclosure"><summary>容量口径与缺失证据</summary><p>PH-MR-001 报告 co-resident pressure。184 KiB 是扣除保留区后的可规划容量，不是当时剩余空闲量。图中对象尺寸不编码各组占用比例。</p><p>缺少失败候选的峰值存活集合与冲突明细，尚不能确定具体挤占者；终态分配总量不能代替该证据。</p></details>'+proof;
    }
    const body=text(24,32,'已确认 · 编译结构',18,'#9ef04b')
      +v.memory('requested','深度 2','P29 · 请求隔离',24,80,210,106,true)
      +v.edge(234,133,315,133,'mapping')
      +v.memory('actual','深度 1','P35 · 实际满足',315,80,210,106,true)
      +text(24,235,data.buffers.length+' 个物理缓冲跨阶段共享',18,'#fff')
      +text(24,270,'同址访问需要满足依赖与生命周期顺序',16,'#aaa')
      +'<path d="M575 38V350" stroke="#555" stroke-dasharray="3 6"/>'
      +text(620,32,'待验证 · 执行影响',18,'#f2c45f')
      +text(620,109,'?',28,'#f2c45f')+text(670,96,'谁造成峰值压力？',20)+text(670,125,'峰值存活集合 / 冲突记录',15,'#aaa')
      +text(620,207,'?',28,'#f2c45f')+text(670,194,'增加了多少等待？',20)+text(670,223,'指令调度 / 停顿轨迹',15,'#aaa')
      +text(620,305,'?',28,'#f2c45f')+text(670,292,'恢复深度 2 能否加速？',20)+text(670,321,'同输入与资源约束下的运行对照',15,'#aaa');
    return '<p class="mr-takeaway">结构退化已确认，性能收益尚未验证。</p><div class="mr-diagram mr-large-diagram" style="--diagram-width:1020px">'+v.wrap(body,1020,370)+'</div><details class="pass-disclosure"><summary>查看结论依据与验证边界</summary><p>P29 请求 stage=2，P35 在 group 0 / Vec 只满足深度 1；P34 → P35 的 SSA 映射确认 '+data.buffers.length+' 个物理缓冲跨阶段共享。诊断指向同驻资源压力。</p><p>同址访问约束中可能包含真实数据依赖。目前不能量化延迟损失或加速比例，也未认定为编译器缺陷。</p></details>'+proof;
  }
  return { render };
})();
