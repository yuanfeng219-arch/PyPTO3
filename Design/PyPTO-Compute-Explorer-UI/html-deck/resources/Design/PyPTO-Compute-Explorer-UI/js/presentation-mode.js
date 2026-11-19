/* Optional read-only narration. Never dispatches into the visualization store. */
(() => {
'use strict';
let root, state, enabled=false, lastKey='', pending=0;
const gear=document.createElement('button');
gear.type='button';gear.className='ai-help ai-settings';gear.dataset.liquidGlass='icon';
gear.setAttribute('aria-label','显示设置');gear.setAttribute('aria-expanded','false');gear.setAttribute('aria-controls','ai-display-settings');
gear.dataset.aiTip='开启当前视图的讲解说明';
gear.innerHTML=`<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.09a2 2 0 0 1-1-1.74v-.51a2 2 0 0 1 1-1.72l.15-.1a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path>
                  <circle cx="12" cy="12" r="3"></circle>
                </svg>`;
const settings=document.createElement('aside');settings.id='ai-display-settings';settings.className='ai-display-settings';settings.hidden=true;settings.dataset.liquidGlass='panel';settings.setAttribute('aria-label','显示设置');
settings.innerHTML='<header><b>显示设置</b><button type="button" aria-label="关闭显示设置">×</button></header><label><span>讲解模式<small>叠加当前视图的标题、对象和阅读提示</small></span><input type="checkbox" data-guide-toggle></label>';
document.body.append(settings);
const overlay=document.createElement('section');overlay.className='ai-presentation-overlay';overlay.hidden=true;overlay.setAttribute('aria-label','当前视图讲解');
function showSettings(open){settings.hidden=!open;gear.setAttribute('aria-expanded',String(open));}
gear.onclick=()=>showSettings(settings.hidden);settings.querySelector('button').onclick=()=>{showSettings(false);gear.focus();};
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!settings.hidden){showSettings(false);gear.focus();}});
document.addEventListener('pointerdown',e=>{if(!settings.hidden&&!settings.contains(e.target)&&!gear.contains(e.target))showSettings(false);});
function attach(){root=document.querySelector('.viz2');const header=root?.querySelector('.viz2-header');if(!header)return;header.insertBefore(gear,header.querySelector('.ai-help'));root.append(overlay);watch.disconnect();
 new MutationObserver(records=>{if(enabled&&records.some(r=>r.target.closest?.('[data-am-title],[data-am-state]')))schedule();}).observe(root,{subtree:true,childList:true,attributes:true,attributeFilter:['aria-pressed']});
}
const watch=new MutationObserver(attach);watch.observe(document.querySelector('#ideWorkarea'),{childList:true,subtree:true});attach();
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function copy(){
 const tab=state?.activeTab||root?.querySelector('[data-viz2-tab][aria-selected="true"]')?.dataset.viz2Tab||'assignment';
 const view=state?.views?.[tab]||{},selection=state?.selection,inner=view.subview==='block';
 if(inner){const t=selection?.task;return {tab,title:tab==='assignment'?'Task 的数据分块':tab==='pipeline'?'Task 内的流水依赖':'Task 内的计算依赖',object:`Task ${t??''}${selection?.block!=null?' · Block '+String(selection.block).padStart(2,'0'):''}`,body:tab==='assignment'?'从 Q Tile 到 Stack、Page，查看当前 Task 负责的 Head 与 KV 范围。':tab==='pipeline'?'沿逻辑 Tick 查看同一 Stack 的 QK、Softmax、PV 与累积关系。': '展开 OP，追踪当前 Task 的 Tensor、状态传递和源码关联。'};}
 if(tab==='assignment'){
  if(view.phase==='phase0')return {tab,diagramKind:'phase0',title:'准备阶段的工作归属',object:'Batch × KV Head → AIV Lane',body:'每个方格是一项预处理工作；按 Block 和 AIV Lane 查看归属，再通过同步进入 Attention。'};
  const currentTitle=[...root.querySelectorAll('[data-am-title]')].find(e=>e.getClientRects().length)?.textContent||'';
  const currentButton=[...root.querySelectorAll('[data-am-state="assignment"]')].find(e=>e.getClientRects().length);
  const grouped=currentTitle.includes('执行归属')||(!currentTitle&&currentButton?.getAttribute('aria-pressed')==='true');
  return grouped?{tab,title:'执行归属',object:'Attention Task 的 SPMD 工作划分',body:'每个 Task 处理一个 Batch 与一个 KV Head 对应的 5 个 Q Heads。源码以 24 个逻辑 Block 执行，Block b 依次领取 b、b+24、b+48…号 Task；卡片展示这组静态归属。24 来自 ATTN_SPMD_BLOCKS 配置，不代表设备物理核数。',source:'paged_attention_pypto.py · L30、L110–124、L261–263'}:{tab,title:'任务矩阵',object:'16 Batches × 8 KV Heads → 128 Tasks',body:'每个 Task 对应一个 Batch 与一个 KV Head，负责该 KV Head 对应的 5 个 Q Heads。'};
 }
 return tab==='pipeline'?{tab,title:'Block 间的流水结构',object:'准备阶段 → Attention Tasks → Output',body:'逐行查看 Block 的工作；Attention 的宽度随 Stack 数变化，末尾 N↓ 表示归一化与写回，不是实测耗时。'}:{tab,title:'计算与数据依赖',object:'Tensor · OP · State · Block',body:'从输入沿连线追踪计算与状态；展开 OP 看内部结构，通过图层切换数据、形状、精度和同步信息。'};
}
function render(){if(!root)return;root.classList.toggle('has-presentation',enabled);overlay.hidden=!enabled||root.hidden;if(!enabled)return;const c=copy(),key=JSON.stringify(c);if(key===lastKey)return;lastKey=key;overlay.innerHTML=`<h2>${escape(c.title)}</h2><p class="ai-guide-object">${escape(c.object)}</p><p>${escape(c.body)}</p>${c.source?`<p class="ai-guide-source">来源：${escape(c.source)}</p>`:''}`;}
function schedule(){if(!pending)pending=requestAnimationFrame(()=>{pending=0;render();});}
settings.querySelector('[data-guide-toggle]').onchange=e=>{enabled=e.target.checked;lastKey='';render();};
document.addEventListener('compute-view-change',e=>{state=e.detail;if(enabled)schedule();});
})();
