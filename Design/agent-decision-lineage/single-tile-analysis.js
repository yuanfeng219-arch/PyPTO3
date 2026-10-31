(() => {
  const shared=window.SoftmaxTileModel, presets=shared.presets;
  let preset = 0, selected = 0, hovered = null, code = null;
  const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

  function preview(key) {
    const source = window.DECISION_EVIDENCE?.['test_softmax.py'];
    if (!source) return '<p>源码不可用</p>';
    const lines = source.split('\n');
    const patterns = {shape:/^MAX_N\s*=/,valid:/pl\.set_validshape\(in_slot/,btail:/valid_rows\s*=\s*pl\.min/,ntail:/tail_mreg_fp16\s*=\s*vf\.update_mask/};
    let index = lines.findIndex(l => patterns[key].test(l));
    if (index < 0 && key === 'ntail') index = lines.findIndex(l => /vf\.update_mask\(valid/.test(l));
    if (index < 0) return '<p>未找到对应源码</p>';
    const start = Math.max(0,index-(key==='btail'?1:0)), end = Math.min(lines.length,index+(key==='shape'?3:key==='ntail'?5:3));
    return `<div class="st-code"><div>test_softmax.py · L${start+1}–${end} <button class="btn btn-ghost" data-close-code>关闭</button></div><pre>${lines.slice(start,end).map((l,i)=>`<span class="${start+i===index?'code-highlight':''}">${start+i+1}  ${esc(l)}</span>`).join('\n')}</pre></div>`;
  }
  function render(host) {
    const [b,n]=presets[preset], tiles=shared.tiles(preset), t=tiles[hovered ?? selected], tail=n%128, full=Math.floor(n/128);
    host.innerHTML = `<div class="st-root"><div class="st-head"><div><span class="st-eyebrow">TENSOR → TILE MAPPING</span><h3>一整行，放入一个 Tile</h3><p>沿 B 轴每 3 行分组 · Softmax 沿 N 轴归约 →</p></div><div class="st-presets" role="group" aria-label="选择输入形状">${presets.map((p,i)=>`<button data-preset="${i}" aria-pressed="${preset===i}">${i?'P0':'边界'} [${p}]</button>`).join('')}</div></div><div class="st-mapping"><section class="st-panel"><div class="st-panel-head"><div><small>LOGICAL TENSOR</small><h4>输入 [${b}, ${n}]</h4></div><span>${tiles.length} tiles</span></div><div class="st-axis">B ↓ <span>N = ${n} · 归约方向 →</span></div><div class="st-tiles">${tiles.map((x,i)=>`<button class="st-tile ${i===(hovered??selected)?'is-active':''}" data-tile="${i}" aria-pressed="${i===selected}"><span>Tile ${i}</span><span class="st-grid">${Array.from({length:x.validShape[0]},()=>'<i></i>').join('')}</span><span>rows ${x.rowRange[0]}–${x.rowRange[1]}</span></button>`).join('')}</div></section><section class="st-panel"><div class="st-panel-head"><div><small>TILE ANATOMY</small><h4>Tile ${t.id.split('-')[1]}</h4></div><span>Rows ${t.rowRange.join('–')} · Cols 0–${n-1}</span></div><div class="st-shapes"><button class="btn btn-ghost" data-code="shape">Allocated <b>[3, 8192]</b> ↗</button><button class="btn btn-ghost" data-code="valid">Valid <b>[${t.validShape}]</b> ↗</button><button class="btn btn-ghost" data-code="btail">B Tail <b>${3-t.validShape[0]} 无效行</b> ↗</button></div><div class="st-anatomy">${Array.from({length:3},(_,i)=>`<div class="st-anatomy-row ${i>=t.validShape[0]?'is-invalid':''}">${i<t.validShape[0]?'<span class="st-valid">有效数据</span><span class="st-unused">未使用物理列</span>':'<span>无效行</span>'}</div>`).join('')}</div><p class="st-note">有效 ${t.validShape[0]*n} 个元素 · 物理列 8192；图中列宽压缩表达，非真实比例（有效列 ${n}/8192）</p><div class="st-register"><div><small>FP16 REGISTER · 128 LANES</small><button class="btn btn-ghost" data-code="ntail">N Tail 实现 ↗</button></div><p>${full} 个完整寄存器 × 128${tail?` · 尾寄存器 ${tail} 有效 / ${128-tail} 无效`:' · 无寄存器尾块'}</p>${tail?`<div class="st-lane-label">完整寄存器 · 128 valid</div><div class="st-lane-grid">${Array.from({length:128},()=>"<i></i>").join("")}</div><div class="st-lane-label">尾寄存器 · ${tail} valid / ${128-tail} invalid</div><div class="st-lane-grid">${Array.from({length:128},(_,i)=>`<i class="${i>=tail?"is-invalid":""}"></i>`).join("")}</div>`:'<div class="st-lanes"><span style="width:100%">128 valid × '+full+'</span></div>'}</div></section></div>${code?preview(code):''}<div class="st-rationale"><small>DESIGN RATIONALE</small><p>完整归约行放入单个 Tile，无需跨 N 轴分块，也无需保存跨 Tile 的 Max / Sum 状态。Online Softmax 跨 Tile 路线在此容量条件下未触发。</p><div><b>TILE_ROWS = 3</b><b>MAX_N = 8192</b><b>48 KiB / Slot</b></div><p>3 × 8192 × 2 bytes = 49,152 bytes；四个 Slot 共 192 KiB ≤ 248 KiB UB。</p><details><summary>设计依据 · DESIGN §2、§3、§7</summary><p><a href="evidence/DESIGN.md" target="_blank" rel="noopener">查看设计约束</a> · <a href="evidence/DESIGN_BINDINGS.json" target="_blank" rel="noopener">查看设计绑定</a></p></details></div></div>`;
    host.querySelectorAll('[data-tile]').forEach(el=>{const i=+el.dataset.tile; el.addEventListener('mouseenter',()=>{hovered=i; updateVisual(host)}); el.addEventListener('mouseleave',()=>{hovered=null; updateVisual(host)}); el.addEventListener('click',()=>{selected=i;shared.setState({selectedPreset:preset,selectedTileId:`tile-${i}`}); hovered=null; render(host)});});
    host.querySelectorAll('[data-preset]').forEach(el=>el.onclick=()=>{preset=+el.dataset.preset;selected=0;shared.setState({selectedPreset:preset,selectedTileId:'tile-0'});hovered=null;code=null;render(host)});
    host.querySelectorAll('[data-code]').forEach(el=>el.onclick=()=>{code=el.dataset.code;render(host)});
    host.querySelector('[data-close-code]')?.addEventListener('click',()=>{code=null;render(host)});
  }
  function updateVisual(host){
    const temporary = document.createElement('div');
    render(temporary);
    const panels = host.querySelectorAll('.st-mapping > .st-panel');
    const replacement = temporary.querySelectorAll('.st-mapping > .st-panel')[1];
    panels[1].replaceWith(replacement);
    replacement.querySelectorAll("[data-code]").forEach(el=>el.onclick=()=>{code=el.dataset.code;render(host)});
    host.querySelectorAll('[data-tile]').forEach(el => el.classList.toggle('is-active', +el.dataset.tile === (hovered ?? selected)));
  }
  window.SingleTileAnalysis={open(id){if(id!=='d-tiling')return false; const host=document.getElementById('detail'); if(!host)return false; host.classList.add('design-detail'); const state=shared.getState();preset=state.selectedPreset;selected=Math.max(0,shared.tiles(preset).findIndex(t=>t.id===state.selectedTileId));hovered=null;code=null;render(host);return true;}};
})();
