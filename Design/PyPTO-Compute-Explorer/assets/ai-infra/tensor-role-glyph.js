/* Tensor role glyph: dataTiles copied from AI Infra training report P02.
 * Source: reports/llm-compute/training-parallel-communication-v1.html
 * Formal library variant: system-value / data-1 (数据角色).
 * Uses the host role palette; texture does not encode element count. */
(function(root){
'use strict';
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function dataTiles(x,y,w,h,color,opacity=.22){const pad=5,cell=Math.min(11,w-2*pad,h-2*pad),baseGap=3,iw=w-2*pad,ih=h-2*pad;if(cell<=0)return '';const cols=Math.max(1,Math.floor((iw+baseGap)/(cell+baseGap))),rows=Math.max(1,Math.floor((ih+baseGap)/(cell+baseGap))),gx=cols>1?(iw-cols*cell)/(cols-1):0,gy=rows>1?(ih-rows*cell)/(rows-1):0,x0=x+pad+(cols===1?(iw-cell)/2:0),y0=y+pad+(rows===1?(ih-cell)/2:0);let b='';for(let r=0;r<rows;r++)for(let c=0;c<cols;c++)b+=`<rect class="data-tile" x="${x0+c*(cell+gx)}" y="${y0+r*(cell+gy)}" width="${cell}" height="${cell}" fill="${color}" opacity="${opacity}"/>`;return `<g pointer-events="none">${b}</g>`}
function render({id,label,metadata='',color,width=280}){
 const w=width,h=metadata?64:35;
 // Four rows of square cells; the texture remains independent of tensor shape.
 const fit=(value,size)=>Math.min(size,(w-24)/Math.max(1,[...String(value)].reduce((n,c)=>n+(c.charCodeAt(0)>255?1:.56),0)));
 const text=(value,y,size)=>`<text x="${w/2}" y="${y}" text-anchor="middle" fill="${color}" style="font-size:${fit(value,size)}px;paint-order:stroke;stroke:#131313;stroke-width:2px;stroke-linejoin:round">${esc(value)}</text>`;
 return `<svg class="ai-role-tensor" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${esc([label,metadata].filter(Boolean).join(' · '))}"><g data-object="${esc(id)}"><rect class="ai-role-outline" x=".6" y=".6" width="${w-1.2}" height="${h-1.2}" rx="6" fill="#131313" stroke="${color}" stroke-width="1.2"/>${dataTiles(0,0,w,h,color)}${text(label,metadata?27:23,14)}${metadata?text(metadata,47,12):''}</g></svg>`;
}
root.PtoTensorRoleGlyph=Object.freeze({render});
})(window);
