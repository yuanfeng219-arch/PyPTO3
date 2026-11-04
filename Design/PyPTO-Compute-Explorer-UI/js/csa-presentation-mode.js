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
 new MutationObserver(records=>{if(enabled&&records.some(r=>true))schedule();}).observe(root,{subtree:true,childList:true,attributes:true,attributeFilter:['aria-pressed','aria-selected']});
}
const watch=new MutationObserver(attach);watch.observe(document.querySelector('#ideWorkarea'),{childList:true,subtree:true});attach();
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function copy(){
 const s=window.PtoCsaExplorer?.getState(),tab=s?.tab||'assignment';
 const phase=root?.querySelector('[data-qk-phase][aria-pressed="true"]')?.dataset.qkPhase;
 const content=tab==='assignment'?(phase==='generation'?['Work Item 生成','Query × Sparse Block','每 Query 对应 5 个 Sparse Blocks；每格保留同一 Work Item 身份。']:phase==='ordering'?['qk_order 编排','Valid-first 稳定排序','valid 工作项在前，invalid 在后；两组均保留，不删除工作项。']:['SPMD 工作分配','Queue position → Lane / iteration','队列位置按 24 个 Lane 分配；点击 Work Item 查看归属。']):tab==='pipeline'?['QK/PV 逻辑依赖','Valid / invalid 分支','追踪同一 Work Item 的计算路径；invalid 项仅写入中性 m/l/o。']:['CSA 计算依赖','Tensor · OP · State · Communication','展开模块，追踪跨 Rank 数据归属、Task ID 和控制依赖。'];
 const f=s?.focus;const identity=f?.kind==='work-item'?' · Work Item '+f.workItemId+(f.valid===false?' · invalid':''):f?.kind==='lane'?' · Lane '+f.laneId:'';
 return {tab,title:content[0],object:content[1]+' · Rank '+(s?.rank??0)+identity,body:content[2]};
}
function render(){if(!root)return;root.classList.toggle('has-presentation',enabled);overlay.hidden=!enabled||root.hidden;if(!enabled)return;const c=copy(),key=JSON.stringify(c);if(key===lastKey)return;lastKey=key;overlay.innerHTML=`<h2>${escape(c.title)}</h2><p class="ai-guide-object">${escape(c.object)}</p><p>${escape(c.body)}</p>${c.source?`<p class="ai-guide-source">来源：${escape(c.source)}</p>`:''}`;}
function schedule(){if(!pending)pending=requestAnimationFrame(()=>{pending=0;render();});}
settings.querySelector('[data-guide-toggle]').onchange=e=>{enabled=e.target.checked;lastKey='';render();};
document.addEventListener('csa-view-change',()=>{if(enabled)schedule();});
document.addEventListener('compute-view-change',e=>{state=e.detail;if(enabled)schedule();});
})();
