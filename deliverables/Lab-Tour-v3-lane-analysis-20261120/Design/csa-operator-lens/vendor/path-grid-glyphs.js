/* Reference variants: memory = outlined grid; tensor = solid role grid.
 * Grids are textures; only explicit labels and the UB address bar encode quantities.
 */
(function(root){
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 function memory(label,detail,w=200,h=72,physical=false){
  const pad=5,gap=2.2,rows=5,cell=(h-pad*2-gap*(rows-1))/rows;
  const cols=Math.max(1,Math.floor((w-pad*2+gap)/(cell+gap))),gx=(w-pad*2-cols*cell)/Math.max(1,cols-1);
  let s=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(label+' · '+detail)}"><rect width="${w}" height="${h}" rx="8" fill="${physical?'#087CFA':'#2d2d2d'}"/>`;
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++)s+=`<rect x="${pad+c*(cell+gx)}" y="${pad+r*(cell+gap)}" width="${cell}" height="${cell}" rx="1" fill="none" stroke="${physical?'#3894ed':'#454545'}" stroke-width=".8"/>`;
  return s+`<g font-family="Inter,PingFang SC,sans-serif" fill="#fff"><text x="14" y="${h*.44}" font-size="${Math.min(18,h*.26)}" font-weight="600">${esc(label)}</text><text x="14" y="${h*.79}" font-size="${Math.min(14,h*.21)}" fill="${physical?'#fff':'#d1d1d1'}">${esc(detail)}</text></g></svg>`;
 }
 function tensor(id,label,shape,x,y,w,h,color){
  const pad=2.5,gap=.9,rows=4,cell=(h-pad*2-gap*(rows-1))/rows,cols=Math.max(1,Math.floor((w-pad*2+gap)/(cell+gap))),gx=(w-pad*2-cols*cell)/Math.max(1,cols-1);
  let s=`<g data-tensor="${esc(id)}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${color}"/><rect x="${x+1.3}" y="${y+1.3}" width="${w-2.6}" height="${h-2.6}" rx="3" fill="#000" fill-opacity=".22"/>`;
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++)s+=`<rect x="${x+pad+c*(cell+gx)}" y="${y+pad+r*(cell+gap)}" width="${cell}" height="${cell}" fill="${color}"/>`;
  return s+`<g fill="#080808" font-family="Inter,PingFang SC,sans-serif" text-anchor="middle"><text x="${x+w/2}" y="${y+11}" font-size="9" font-weight="700">${esc(label)}</text><text x="${x+w/2}" y="${y+21}" font-size="7.5" font-weight="600">${esc(shape)}</text></g></g>`;
 }
 root.PathGridGlyphs={memory,tensor};
})(window);
