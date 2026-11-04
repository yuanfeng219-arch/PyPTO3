/* Reusable SVG grammar. No DOM or report runtime dependency. */
(function(root){
'use strict';
const tokens=Object.freeze({font:14,gap:24,dataWidth:160,dataHeight:64,opWidth:260,opHeight:48,cell:24,role:{activation:'#B1EF6B',weight:'#BCA0F3',aggregate:'#F2C45F',gradient:'#79D4F2',optimizer:'#B0B0B0'},parallel:{DP:'#8CE580',TP:'#B58AF7',SP:'#80B7F7',CP:'#80E5D3',PP:'#E580CF',EP:'#F7B080'}});
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const text=(x,y,s,size=14,color='#E8E8E8',anchor='start')=>`<text x="${x}" y="${y}" font-size="${size}" fill="${color}" text-anchor="${anchor}" dominant-baseline="middle">${esc(s)}</text>`;
const rect=(x,y,w,h,rx=4,stroke='#888',fill='#202020',extra='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" stroke="${stroke}" fill="${fill}" ${extra}/>`;
function rankBoard(x,y,w,h,fill='#393939'){
const compact=h<64,inset=compact?9:16,r=compact?3:5;
let body=`<rect class="rank-outline outline" x="${x}" y="${y}" width="${w}" height="${h}" rx="14" fill="${fill}"/>`;
for(const px of [x+inset,x+w-inset])for(const py of [y+inset,y+h-inset])body+=`<circle class="rank-pin" cx="${px}" cy="${py}" r="${r}" fill="#2b2b2b"/>`;
return body;
}
function data(o,x,y,opts={}){
 const expanded=opts.expanded,shape=o.shape||[2,4],rows=shape.length===1?1:shape[0],cols=shape.length===1?shape[0]:shape[1],w=expanded?cols*tokens.cell:tokens.dataWidth,h=expanded?rows*tokens.cell:tokens.dataHeight,color=tokens.role[o.role]||'#B0B0B0';
 let body=expanded?'':rect(x,y,w,h,6,'none','#2d2d2d');
 if(expanded){for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){body+=rect(x+c*24+2,y+r*24+2,20,20,1,'none',color,'fill-opacity=".18" data-element="true"');if(o.values)body+=text(x+c*24+12,y+r*24+12,o.values[r*cols+c],11,color,'middle')}body+=text(x,y-14,`${o.label} · [${shape}]`,14)}
 else {
  for(let i=0;i<4;i++)body+=rect(x+8,y+8+i*12,10,10,0,'none',color,`fill-opacity="${[1,.8,.6,.4][i]}"`);
  body+=text(x+27,y+22,o.label,16,'#fff')+text(x+27,y+46,`${shape.join(' × ')}  ${o.dtype||'FP16'}`,12,'#ccc');
 }
 if(opts.parallel&&opts.parallel!=='none'){let pc=tokens.parallel[opts.parallel];body+=`<path data-decoration="distribution" d="M${x},${y+h+8}v6h${w}v-6" stroke="${pc}" fill="none"/>`+text(x+w/2,y+h+30,`${opts.parallel} · ${opts.placement||'分片'}`,12,pc,'middle')}
 return `<g data-object="${esc(o.id)}" data-kind="${o.kind||'value'}" data-role="${esc(o.role||'activation')}" data-representation="${expanded?'expanded':'collapsed'}">${body}</g>`;
}
// Memory grid is a surface texture, not an element or capacity scale.
function memory(id,label,detail,x,y,w=300,h=106,physical=false){
 let b=rect(x,y,w,h,12,'none',physical?'#087CFA':'#2d2d2d');
 const cols=17,rows=5,gap=3,pad=9,cw=(w-pad*2-gap*(cols-1))/cols,ch=(h-pad*2-gap*(rows-1))/rows;
 for(let r=0;r<rows;r++)for(let c=0;c<cols;c++)b+=rect(x+pad+c*(cw+gap),y+pad+r*(ch+gap),cw,ch,1.5,physical?'#B3D7FF':'#aaa','#000','fill-opacity=".10" style="stroke-width:.65;stroke-opacity:.30"');
 b+=text(x+24,y+h*.36,label,Math.min(32,w*.09),'#fff')+text(x+24,y+h*.74,detail,Math.min(18,w*.047),'#eee');
 return `<g data-object="${esc(id)}" data-kind="${physical?'storage':'memref'}">${b}</g>`;
}
function node(kind,id,label,x,y){let w=180,h=64,b='';
 if(kind==='op'||kind==='communication'){w=tokens.opWidth;h=tokens.opHeight;b=rect(x,y,w,h,24,'#858585','#505050',kind==='communication'?'stroke-dasharray="5 4"':'')+`<g font-weight="650">${text(x+w/2,y+24,label,18,'#fff','middle')}</g>`}
 else if(kind==='add'||kind==='multiply'){w=h=48;const cx=x+24,cy=y+24;b=`<circle cx="${cx}" cy="${cy}" r="24" stroke="#ddd" fill="#111" style="stroke-width:1.5"/><path d="${kind==='add'?`M${cx-8} ${cy}H${cx+8} M${cx} ${cy-8}V${cy+8}`:`M${cx-6} ${cy-6}L${cx+6} ${cy+6} M${cx-6} ${cy+6}L${cx+6} ${cy-6}`}" stroke="#eee" fill="none" style="stroke-width:2" stroke-linecap="round"/>`}
 else if(kind==='memref'||kind==='storage'){return memory(id,label,kind==='memref'?'MemRef':'内存分配',x,y,w,h,kind==='storage')}
 else if(kind==='instruction'||kind==='task'){
 w=260;h=60;const green='#9EF04B';
 b=rect(x,y,w,h,0,'none',green)+rect(x+10,y+12,36,36,3,'none','#080b05');
 const icon=kind==='instruction'?'M9 9 L16 17 L9 25 M20 25 H29':'M9 10 H27 M9 18 H20 M9 26 H16 M23 22 L28 26 L23 30';
 b+=`<path data-icon="${kind}" transform="translate(${x+10} ${y+12})" d="${icon}" fill="none" stroke="${green}" style="stroke-width:2.6"/>`;
 b+=`<g font-weight="650">${text(x+62,y+30,label,20,'#080b05')}</g>`;
 }
 else if(kind==='rank'){b=rankBoard(x,y,w,h)+text(x+24,y+27,label,14)}
 else if(kind==='pass'){h=32;b=rect(x,y,w,h,0,'#aaa')+text(x+w/2,y+16,label,14,'#eee','middle')}
 else {const purple=['kernel','task','function','loop'].includes(kind),color=purple?'#9B91FF':'#999';b=rect(x,y,w,h,kind==='task'?8:2,color,'#202020',kind==='loop'?'stroke-dasharray="5 4"':'')+text(x+12,y+17,kind.toUpperCase(),10,color)+text(x+12,y+43,label);if(['kernel','function','rank','storage'].includes(kind))b+=`<path d="M${x},${y+28}h${w}" stroke="${color}"/>`;if(kind==='loop')b+=`<path d="M${x},${y}v${h}" stroke="${color}" stroke-width="3"/>`}
 return `<g data-object="${esc(id)}" data-kind="${kind}">${b}</g>`;
}
function edge(x1,y1,x2,y2,kind='data',label=''){let binding=kind==='binding',mapping=kind==='mapping',dash=kind==='call'?'8 5':mapping?'2 5':'';return `<path data-relation="${kind}" d="M${x1} ${y1} H${(x1+x2)/2} V${y2} H${x2}" fill="none" stroke="#aaa" ${dash?`stroke-dasharray="${dash}"`:''} marker-end="url(#${binding?'bind':mapping?'map':'arrow'})"/>`+(label?text((x1+x2)/2,(y1+y2)/2-12,label,12,'#aaa','middle'):'')}
function wrap(body,w=900,h=340){return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="对象与关系示例"><style>text{font-family:Inter,'PingFang SC',system-ui,sans-serif}path,rect,circle{stroke-width:1.5}</style><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M1 1L9 5L1 9" stroke="#aaa" fill="none"/></marker><marker id="bind" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7"><circle cx="5" cy="5" r="3" stroke="#aaa" fill="#202020"/></marker><marker id="map" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8"><path d="M1 5L5 1L9 5L5 9Z" stroke="#aaa" fill="#202020"/></marker></defs>${body}</svg>`}
function render(id,o={}){let b='',ex=!!o.expanded;const v=(id,label,role,shape,values)=>({id,label,role,shape,values});
 if(id==='pattern-compute'){
 // Expanded dimensions: X 2x4, W 4x3, Y 2x3. Fixed cell pitch throughout.
 b=edge(ex?136:200,190,350,190)+edge(480,ex?126:94,480,166,'data','参数输入')+edge(610,190,680,190)+data(v('X','X','activation',[2,4]),40,166,{expanded:ex})+data(v('W','W','weight',[4,3]),ex?444:400,30,{expanded:ex})+node('op','matmul','MatMul',350,166)+data(v('Y','Y','activation',[2,3]),680,166,{expanded:ex})+text(40,290,'X [2,4] × W [4,3] → Y [2,3]；示例尺寸，格子不编码数值。',14);
 }else if(id==='pattern-distribution'){
 const mode=o.mode||'shard',lab={shard:'分片',replica:'复制',partial:'部分结果'}[mode];
 b=text(40,26,mode==='shard'?'按列切分 W：局部 [4,2]，全局 [4,4]':mode==='replica'?'每个 Rank 保存完整 W [4,4] 的副本':'沿 K 维切分 MatMul：Y = Y₀ + Y₁，各部分结果均为 [2,4]',16);
 for(let i=0;i<2;i++){let x=40+i*440;b+=rankBoard(x,55,390,235)+text(x+24,82,'Rank '+i,14)+data(v(mode==='replica'?'W':mode==='partial'?'Y'+i:'W'+i,mode==='replica'?'W · 副本 '+i:mode==='partial'?'Y'+i+' · 待求和':'W'+i+' · 列 '+(i===0?'[0,2)':'[2,4)'),mode==='partial'?'activation':'weight',mode==='shard'?[4,2]:mode==='partial'?[2,4]:[4,4]),x+45,110,{expanded:ex,parallel:'TP',placement:lab})}b+=text(40,320,mode==='partial'?'部分结果不是输出的互斥分片；求和归约后得到完整输出。':'角色色位于数据内部；TP 标记位于对象外侧。Rank 表示进程身份。',13);
 }else if(id==='pattern-storage'){
 b=data(v('V','V','activation',[4,4]),40,70,{expanded:ex})+data({...v('view','V 的视图','activation',[2,4]),kind:'view'},40,225,{expanded:ex})+node('memref','M0','M0 · offset 0 B',345,70)+node('memref','M1','M1 · offset 16 B',345,225)+node('storage','B0','B0 · 32 B',660,135)+edge(ex?136:200,102,345,102,'binding')+edge(ex?136:200,257,345,257,'binding')+edge(525,102,660,167,'binding')+edge(525,257,660,183,'binding')+text(660,240,'FP16 · 连续行布局',13)+text(660,265,'视图引用 [16,32) B',13)+text(40,335,'同一底层分配 B0；视图改变索引范围，不创建新的数据副本。示例偏移以字节计。',13);
 return wrap(b,900,375);
 }else if(id==='system-value') b=data(v(o.vector?'vector-0':'value-0',o.vector?'v':'X',o.role||'activation',o.vector?[6]:[2,4],o.vector?[1,2,3,4,5,6]:undefined),80,65,{...o,expanded:ex});
 else if(id.startsWith('relation-')) b=edge(60,100,330,100,id.slice(9))+text(195,155,{data:'生产者 → 消费者',call:'调用方 → 被调用函数',binding:'值 / 视图 — 存储引用',mapping:'跨阶段对应（需要证据）'}[id.slice(9)],13,'#aaa','middle');
 else b=node(id.replace('system-',''),id,'示例 '+id.replace('system-',''),85,75);
 return wrap(b,id.startsWith('pattern-')?900:420,id.startsWith('pattern-')?350:280);
}
const api={tokens,rankBoard,data,memory,node,edge,wrap,render};if(typeof module!=='undefined')module.exports=api;root.VisualSystem=api;
})(typeof globalThis!=='undefined'?globalThis:this);
