/* Render-source narrative bindings. Only explicit UI sinks register expressions.
 * Source, model objects, selectors and data attributes never enter this layer. */
(function(global){
'use strict';
let enabled=false,revision=0;
const bindings=new Set(),byNode=new WeakMap();
const themes=Object.freeze(['宠物','家人','朋友','旅行','美食','风景','日常','纪念日']);
const themeName=id=>themes[Number(id)]??`主题 ${id}`;
const phrases={
 'Logical Tick':'流水轮次','logical ticks':'流水轮次','Tick':'流水轮次','tick':'流水轮次',
 'QK / Softmax':'主题匹配 / 筛选去重','PV / Context Accumulate':'打印 / 装帧','QK':'主题匹配','Softmax':'筛选去重','PV':'打印',
 'score_transfer':'分析结果周转资料','probability_transfer':'整理结果周转资料','pv_transfer':'加工结果周转资料',
 'Transfer buffers':'周转资料','Transfer buffer':'周转资料','Normalize / Store':'装帧与保存','Store':'保存',
 'Inputs':'输入资料','Computation':'处理内容','Outputs':'输出资料','Dependencies':'工序交接','Upstream':'上道工序','Downstream':'下道工序','Resource / Synchronization':'周转与交接',
 'Activation / KV Data':'原始照片与匹配资料','Intermediate / Reduction':'处理结果与汇总','Metadata':'订单资料','photo-narrative':'叙事工序',
 '数据依赖':'工序交接','produce_stack':'前段处理批次','consume_stack':'后段处理批次','produce_row / consume_row':'前后段交接位置',
 'Slot':'周转台位','Slots':'周转台位',
 '任务矩阵：一格 = Q Head（固定 5 个）· Block 分组：一格 = Stack（最多 512 tokens）':'订单矩阵：一格 = 匹配子项（固定 5 个）· 工作小组分组：一格 = 处理批次（最多 512 张照片）',
 'Static Execution Weight · AIC / 每 AIV Lane 分别统计 · 非耗时':'加工步骤的逻辑数量 · 叙事投影，不表示实际耗时',
 '当前 Task KV Cache，纵向浏览 Stacks':'当前工单的照片，纵向浏览处理批次',
 '结构模板 · 各 Batch 使用各自的 Q 数据':'匹配模板 · 每位顾客使用各自的匹配条件',
 '结构模板 · 各 Batch 的 Q 数值独立':'匹配模板 · 每位顾客的匹配条件独立',
 '共用结构示意；不同 Batch 的 Q Tensor 数值独立':'共用匹配模板；每位顾客的匹配条件独立',
 '不是执行顺序':'表示工单归属，不表示完成先后',
 '静态映射，不表示执行时序':'订单归属示意，不表示真实加工时间',
 '物理 Page 未知':'照片原始存放位置未提供',
 '未知 · 未提供实际 block_table 数据':'未提供原始存放位置',
 '未提供实际 block_table，未知':'未提供原始存放位置',
 '不推断 Bytes 或实测耗时':'不推断传输大小或实际耗时',
 '源码静态 READY 关系，不表示运行耗时。':'交接就绪的逻辑关系，不表示真实加工耗时。',
 '每个 Task 对应':'每张工单对应',
 'Task Generation':'生成工单','Work Generation':'准备工作','Block Assignment':'分配工作小组',
 'QKV Preparation':'订单准备','Paged Attention':'照片加工','Task Inputs':'工单输入','Task Output':'工单成品',
 'QK Matmul':'主题匹配','Online Softmax':'筛选去重','Scale & Mask':'照片范围筛选','Scale / Mask':'照片范围筛选',
 'PV Matmul':'打印','Context Accumulate':'装帧','Output Normalize':'装帧','BF16 Output':'冲印成品',
 'Partial Context':'本批已打印照片','Probability':'入选权重','Scores':'匹配评分','Q Tile':'匹配条件',
 'K Cache Pages':'待匹配标准照片包','V Cache Pages':'待打印标准照片包','K Cache':'待匹配照片','V Cache':'待打印照片',
 'KV Cache':'顾客照片库','Rescale FIFO':'批次校准交接队列','Delayed Rescale':'延后交接的校准信息',
 'Attention Output':'冲印成品','Attention':'照片加工','Projection':'订单整理','Cache Append':'照片入库',
 'Sequence Length':'照片总数','Seq Len':'照片总数','seq_lens':'照片数量',
 'Task input':'工单输入','Task output':'工单成品','Task ID':'工单编号','Page Count':'标准照片包数量','Stacks Count':'处理批次数量',
 'KV Sequence Length':'顾客照片数量','Per Task':'每张工单','per Task':'每张工单',
 'Static Execution Weight':'逻辑加工数量','Fit to View':'适应画布','Reset':'恢复原始订单',
 'Block 间':'工作小组间','Block 内':'工作小组内','Graph':'工作流程','Pipeline':'冲印流水线','Assignment':'订单分配',
 'Phase0':'订单准备','Phase 0':'订单准备','Q Heads':'匹配子项','Q Head':'匹配子项','Q Heads':'匹配子项',
 'KV Heads':'主题','KV Head':'主题','KV head':'主题','Q Head':'匹配子项',
 'KV':'主题','Batches':'位顾客','BATCH':'顾客','Batch':'顾客',
 'SPMD Blocks':'工作小组','Active Blocks':'活跃工作小组','Blocks':'工作小组','Block':'工作小组',
 'Tasks':'工单','Task':'工单','Pages':'标准照片包','Page':'标准照片包','Stacks':'处理批次','Stack':'处理批次',
 'tokens':'张照片','Tokens':'张照片','Token':'照片','token':'照片',
 'AIC':'匹配与打印台','AIV Lane':'整理通道','AIV':'整理台','Lane':'通道',
 'READY':'交接就绪','SET':'发出就绪','WAIT':'等待就绪','Buffer':'中转资料','Producer':'上道工序','Consumer':'下道工序',
 'Output':'成品','Input':'输入资料','Context':'成品汇总','Cache':'照片库','Custom':'自定义订单',
 'Shape':'结构','Valid':'有效','Tile':'处理单元','Memory':'存放区域','Dtype':'资料格式',
 'DATA':'资料','STATE':'批次记录','OPERATION':'加工操作','COMPOSITE':'加工阶段','State':'批次记录','Control':'交接控制',
 'Previous':'上一批','Current':'当前批','Updated / Next':'更新后 / 下一批','FIFO':'交接队列',
 'Matmul Accumulate':'汇总已打印照片','Matmul':'照片匹配','Accumulate':'装帧',
 'Scale':'缩放整理','Mask':'范围筛选','Rescale':'批次校准','Normalize':'装帧',
 'QKV':'订单资料','BF16':'成品格式','FP32':'加工格式','GM':'共享资料区','Mat':'加工台','Vec':'整理台',
 'QK_READY_EVENT':'照片分析就绪','SOFTMAX_READY_EVENT':'照片整理就绪','PV_READY_EVENT':'后段加工就绪',
 'QK_READY':'照片分析就绪','SOFTMAX_READY':'照片整理就绪','PV_READY':'后段加工就绪',
 'Q':'匹配条件','K':'待匹配照片','V':'原始照片',
 'Items':'准备项','Work':'准备工作','Output Projection':'装帧','External Inputs':'顾客提交资料',
 'BLK':'工作小组','rope_core':'准备处理位','g_idx':'准备项编号','lane_row':'通道位置','position':'照片位置','Source Evidence':'来源依据','Model-derived':'逻辑推导','Source-confirmed':'源码依据','Stall':'停顿','Index':'编号','ID':'编号','Compute Explorer':'冲印工作室'
};
const keys=Object.keys(phrases).sort((a,b)=>b.length-a.length);
const pattern=new RegExp(keys.map(k=>k.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')).join('|'),'g');
function translate(value,active=enabled){
 let t=String(value??'');if(!active)return t;
 if(/^g_idx = KV Head/.test(t))return '订单准备：为每位顾客按主题整理匹配条件与照片资料，再交给对应工单。';
 if(/^rope_core = g_idx/.test(t))return '准备项按处理位循环分配到工作小组；这里只表示逻辑分组，不表示真实物理排列。';
 // Customer names are one-based; all other identifiers retain their model IDs.
 t=t.replace(/\bBatch\s+(\d+)\b/g,(_,n)=>`顾客 ${Number(n)+1}`).replace(/^B(\d+)$/,(_,n)=>`顾客${Number(n)+1}`);
 t=t.replace(/\bB(\d+)\b/g,(_,n)=>`顾客${Number(n)+1}`);
 t=t.replace(/\bKV\s*(\d+)/g,'主题 $1').replace(/\bQ(\d+(?:[–−-]\d+)?)/g,'子项$1');
 t=t.replace(/\bStack\s+(\d+)\s+到\s+(\d+)/g,(_,a,b)=>`第 ${Number(a)+1} 批 到 第 ${Number(b)+1} 批`);
 t=t.replace(/\b(?:Page|P)\s*(\d+)\b/g,(_,n)=>`标准照片包 ${Number(n)+1}`).replace(/\b(?:Stack|S)\s*(\d+)\b/g,(_,n)=>`第 ${Number(n)+1} 批`);
 t=t.replace(/B%(\d+)/g,'轮转 $1 组 ').replace(/\bSPMD\b/g,'并行').replace(/\bLogical\b/g,'逻辑');
 return t.replace(pattern,(m,offset,s)=>{
  if(/^[A-Za-z]+$/.test(m)&&/[A-Za-z_]/.test(s[offset-1]||''))return m;
  if(/^[A-Za-z]+$/.test(m)&&/[A-Za-z_]/.test(s[offset+m.length]||''))return m;
  return phrases[m];
 }).replace(/主题\s*([0-7])(?!\d)/g,(_,id)=>themeName(id));
}
function translateFor(node,source){
 const element=node.nodeType===1?node:node.parentElement;
 return translate(source,enabled||!!element?.closest('[data-photo-narrative-scope]'));
}
function register(node,key,source,write){
 let entries=byNode.get(node);if(!entries){entries=new Map();byNode.set(node,entries);}
 const previous=entries.get(key);if(previous)bindings.delete(previous);
 const binding={node,source,write};entries.set(key,binding);bindings.add(binding);write(translateFor(node,source));
}
function raw(node){return !!node.parentElement?.closest('pre,code,script,style,[data-narrative-raw]');}
function text(node,value){
 if(!node)return value;
 const content=String(value??'');
 if(node.nodeType===3)node.nodeValue=content;
 else if(node.childNodes.length===1&&node.firstChild.nodeType===3)node.firstChild.nodeValue=content;
 else node.textContent=content;
 if(node.nodeType===3){if(!raw(node))register(node,'text',String(value??''),v=>{node.nodeValue=v;});}
 else {const leaf=node.firstChild;if(leaf&&!raw(leaf))register(leaf,'text',String(value??''),v=>{leaf.nodeValue=v;});}
 return value;
}
function attr(node,name,value){
 if(!node)return;node.setAttribute(name,value);
 if(['title','aria-label','data-ai-tip','label','data-display-label'].includes(name))register(node,name,String(value),v=>node.setAttribute(name,v));
}
function html(node,source){
 // Parse at the explicit render boundary, BEFORE attaching to the live document.
 // Bind only text and human-facing attributes. Never touch IDs, code, or API data.
 const range=document.createRange();range.selectNodeContents(node);
 const fragment=range.createContextualFragment(String(source??''));
 function bind(child){
  if(child.nodeType===3&&!raw(child))register(child,'text',child.nodeValue,v=>{child.nodeValue=v;});
  if(child.nodeType===1){if(child.matches('pre,code,script,style,[data-narrative-raw]'))return;
   for(const name of ['title','aria-label','data-ai-tip','label','data-display-label'])if(child.hasAttribute(name))attr(child,name,child.getAttribute(name));
   [...child.childNodes].forEach(bind);
  }
 }
 [...fragment.childNodes].forEach(bind);node.replaceChildren(fragment);return source;
}
function setMode(value){
 if(enabled===!!value)return;enabled=!!value;revision++;
 for(const b of bindings){if(!b.node.isConnected){bindings.delete(b);continue;}b.write(translateFor(b.node,b.source));}
}
function setScope(root,value){
 if(root.hasAttribute('data-photo-narrative-scope')===!!value)return;
 root.toggleAttribute('data-photo-narrative-scope',!!value);
 for(const b of bindings)if(root.contains(b.node))b.write(translateFor(b.node,b.source));
}
const boundary='工作流程是叙事投影：主题匹配 → 筛选去重 → 打印 → 装帧。用于理解组织、分批和交接，不代表源码实现这些照片算法；流水线为逻辑编排，非真实运行记录。';
function taskSummary(task){return `顾客 ${task.batch+1} · ${themeName(task.kvHead??task.kv)} · 工单 ${task.id} · 工作小组 ${task.block} · ${task.seqLen} 张照片 · ${task.pages.length} 个标准照片包 · ${task.stacks.length} 个处理批次 · 周转台位 ${task.stacks.map(s=>s.slot).join(' → ')}`;}
global.PtoPhotoNarrative=Object.freeze({text,html,attr,translate,setMode,setScope,taskSummary,boundary,themes,themeName,get enabled(){return enabled;},get revision(){return revision;}});
})(window);
