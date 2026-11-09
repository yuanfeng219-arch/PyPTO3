/* Semantic presentation projection. Coordinates are stable across selection/focus.
   Ribbons have fixed visual width: they encode relationships, never counts. */
window.LINEAGE_VIEW = {
 width:2110,height:500,
 groups:[
 {id:'requirements',stage:1,x:24,y:210,w:152,h:112,label:'Softmax SPEC',sub:'fp16 · [B,N] → [B,N]',items:[['e1','数值稳定行归约','需求合同']]},
 {id:'constraints',stage:2,x:218,y:175,w:178,h:205,label:'知识 / API / 硬件约束',sub:'设计依据',items:[['d-kb','知识选择','归约 · 尾块 · 生命周期'],['d-api','VF API 适配','ZERO / ONE · fp32 sum'],['d-constraint','A5 / 950 · UB 248KB','初始设备为默认假设']]},
 {id:'design',stage:3,x:438,y:140,w:205,h:295,label:'采用的设计',sub:'Design Decision Lineage',items:[['d-vf','Vector / VF','纯 Vector · 无 Cube'],['d-precision','Mixed Precision','fp16 → fp32 → fp16'],['d-tiling','Single Tile 3×8192','整行归约 · 无跨 tile 状态'],['d-buffer','UB 192KB / 248KB','Double Buffer · in/out 双槽'],['d-schedule','Scheduling · 行块分核','单 kernel launch']]},
 {id:'kernel',stage:4,x:687,y:210,w:153,h:112,label:'PyPTO Kernel',sub:'生成的实现',items:[['i-kernel','三趟 max / sum / div','softmax_rows_vf']]},
 {id:'correctness',stage:4,x:875,y:210,w:146,h:112,label:'Correctness PASS',sub:'报告记录 · 非本机复测',items:[['x-verify','✓ 8/8 PASS','与 Baseline 精度一致']]},
 {id:'baseline',stage:5,x:1057,y:210,w:120,h:112,label:'Baseline',sub:'[32,4096] · kernel',items:[['x-base','5.926 μs','round_001']]},
 {id:'optimization',stage:5,x:1188,y:190,w:190,h:170,label:'Mask Lifting',sub:'✓ 采用',items:[['x-mask','Full Mask Reuse','核心优化 · 循环外创建'],['x-tail','Empty-tail-loop','采用 · 尾寄存器处理']]},
 {id:'final',stage:5,x:1408,y:210,w:105,h:112,label:'Final',sub:'[32,4096]',items:[['x-final','4.946 μs','1.198×']]},
 ],
 branches:[
 {id:'d-online',parent:'d-tiling',x:447,y:502,w:192,h:72,title:'跨 tile 流式归约',subtitle:'− 设计未采用 · 整行已容纳',mode:'design',relation:'selected over'}
 ],
 // Fixed-width hierarchy, never population/throughput. Junctions are presentation only.
 river:{
   flow:{y:266,mainHalfWidth:16,otherHalfWidth:19,padding:24},
   finding:{id:'f-mask',x:1057,y:128,w:137,h:54,label:'Finding · mask 重复',sub:'update_mask 开销线索'},
   failed:{id:'x-if',x:1405,y:120,w:260,h:60,label:'Runtime if',sub:'× tried & failed · rejected end'},
   // Product-side post-run summary, never an orchestrator stage.
   postRunOpportunities:{status:'not_attempted',evidenceRefs:[{sourceId:'PERFORMANCE_REPORT.md',location:'§7 Candidate Coverage Ledger'}],categories:[
     {categoryId:'target-met',categoryName:'目标已达成',candidateIds:['x-rows'],x:1770,y:154,w:315,h:94,reason:'达到目标后停止进一步调参'},
     {categoryId:'design-change',categoryName:'需要改设计',candidateIds:['x-reuse','x-fusion'],x:1770,y:290,w:315,h:156,reason:'需先重新评估 Stage 3 冻结设计'}
   ]},
   opportunities:[
     {id:'x-rows',categoryId:'target-met',x:1780,y:192,w:295,h:48,label:'TILE_ROWS Adjustment',sub:'当前性能达标，未继续调参'},
     {id:'x-reuse',categoryId:'design-change',x:1780,y:330,w:295,h:48,label:'Exp Result Reuse',sub:'需修改三趟计算结构'},
     {id:'x-fusion',categoryId:'design-change',x:1780,y:388,w:295,h:48,label:'Pass Fusion',sub:'需改变现有计算算法'}
   ]
 },
 /* The story connections below are sourced presentation relations, not new runs. */
 links:[
 ['requirements','constraints','workflow handoff','EXPLORE_REPORT.md','§1.1 输入摘要'],
 ['constraints','design','constrained by','DESIGN.md','Knowledge Bindings / §1–3'],
 ['design','kernel','produces','test_softmax.py','softmax_rows_vf / softmax_tile_group_kernel'],
 ['kernel','correctness','validated by','PERFORMANCE_REPORT.md','§8'],
 ['correctness','baseline','workflow handoff','PERFORMANCE_REPORT.md','§3 Stage 4 implementation'],
 ['baseline','optimization','improved by','PERFORMANCE_REPORT.md','§3–4'],
 ['optimization','final','measured as','PERFORMANCE_REPORT.md','§5'],
 ]
};

// Single geometry authority. Both renderers consume these anchors and monotone curves.
window.FlowGeometry={
 itemBox(group,index){return {x:group.x+10,y:group.y+(group.id==='optimization'?(index===0?67:132):67+index*43),w:group.w-20,h:40,group:group.id}},
 internalPorts(group){return {tailHandling:{x:group.x+group.w,y:group.y+104}}},
 failedBranch(view){const group=view.groups.find(g=>g.id==='optimization'),a=this.internalPorts(group).tailHandling,p=view.river.failed,b={x:p.x,y:p.y+p.h/2},d=(b.x-a.x)*.45;return {a,b,d:`M${a.x} ${a.y} C${a.x+d} ${a.y},${a.x+d} ${b.y},${b.x} ${b.y}`}},
 anchor(view,group,side){return {x:group.x+(side==='out'?group.w:0),y:group.flowY??view.river.flow.y}},
 curve(a,b){const dx=Math.max(0,b.x-a.x),d=Math.min(64,dx*.4);return `M${a.x} ${a.y} C${a.x+d} ${a.y},${b.x-d} ${b.y},${b.x} ${b.y}`},
 band(a,b,h){const dx=Math.max(0,b.x-a.x),d=Math.min(64,dx*.4);return `M${a.x} ${a.y-h} C${a.x+d} ${a.y-h},${b.x-d} ${b.y-h},${b.x} ${b.y-h} L${b.x} ${b.y+h} C${b.x-d} ${b.y+h},${a.x+d} ${a.y+h},${a.x} ${a.y+h} Z`},
 junction(view){const a=view.groups.find(g=>g.id==='baseline'),b=view.groups.find(g=>g.id==='optimization');const p=this.anchor(view,a,'out'),q=this.anchor(view,b,'in');return {x:(p.x+q.x)/2,y:(p.y+q.y)/2}},
 fitText(node,width){const full=node.textContent;node.setAttribute('data-full-text',full);if(node.getComputedTextLength()<=width)return;let text=full;while(text.length&&node.getComputedTextLength()>width){text=text.slice(0,-1);node.textContent=text+'…'}const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=full;node.append(title)},
 bounds(elements,padding=24){const boxes=elements.filter(Boolean).map(e=>e.getBBox()).filter(b=>b.width||b.height);if(!boxes.length)return [0,0,1,1];const step=4,x=Math.floor((Math.min(...boxes.map(b=>b.x))-padding)/step)*step,y=Math.floor((Math.min(...boxes.map(b=>b.y))-padding)/step)*step;return [x,y,Math.ceil((Math.max(...boxes.map(b=>b.x+b.width))+padding)/step)*step-x,Math.ceil((Math.max(...boxes.map(b=>b.y+b.height))+padding)/step)*step-y]}
};
