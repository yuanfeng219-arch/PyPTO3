// SVG layered renderer for the decode_layer compute graph.
// Reads NODES / CARRY_EDGES / TENSOR from graph-data.js.
(function () {
  "use strict";
  const SVGNS = "http://www.w3.org/2000/svg";
  const KIND = {
    cube:  { fill: "var(--dl-cube)",  label: "Cube 矩阵 (AIC)" },
    vec:   { fill: "var(--dl-vec)",   label: "Vector 向量 (AIV)" },
    mixed: { fill: "var(--dl-mixed)", label: "Cube+Vec 融合" },
    seed:  { fill: "var(--dl-seed)",  label: "累加器清零 (seed)" },
    io:    { fill: "var(--dl-io)",    label: "跨层 I/O carry" },
  };
  const NW = 172, NH = 56, COL_GAP = 96, ROW_GAP = 20, PAD = 60;

  const state = { scale: 1, tx: 0, ty: 0, selected: null, showCarry: true, showLabels: true };
  const byId = {};
  NODES.forEach((n) => (byId[n.id] = n));

  // ── layout: assign each node an (x,y). Group by column, stack within column. ──
  function layout() {
    const cols = {};
    NODES.forEach((n) => (cols[n.col] = cols[n.col] || []).push(n));
    const colKeys = Object.keys(cols).map(Number).sort((a, b) => a - b);
    let maxRows = 0;
    colKeys.forEach((c) => (maxRows = Math.max(maxRows, cols[c].length)));
    const colStep = NW + COL_GAP;
    const laneH = NH + ROW_GAP;
    const fullH = maxRows * laneH;
    colKeys.forEach((c, ci) => {
      const group = cols[c];
      const gh = group.length * laneH;
      const y0 = PAD + (fullH - gh) / 2;
      group.forEach((n, ri) => {
        n._x = PAD + ci * colStep;
        n._y = y0 + ri * laneH;
        n._ci = ci;
      });
    });
    return {
      w: PAD * 2 + (colKeys.length - 1) * colStep + NW,
      h: PAD * 2 + fullH,
      colKeys, cols,
    };
  }
  const L = layout();

  const svg = document.getElementById("graph");
  const wrap = document.getElementById("canvasWrap");
  let gRoot; // <g> holding the pan/zoom transform

  function el(tag, attrs, parent) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  // curved connector from right edge of `a` to left edge of `b`
  function edgePath(a, b) {
    const x1 = a._x + NW, y1 = a._y + NH / 2;
    const x2 = b._x, y2 = b._y + NH / 2;
    const mx = (x1 + x2) / 2;
    return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
  }

  function renderGraph() {
    svg.innerHTML = "";
    svg.setAttribute("viewBox", `0 0 ${L.w} ${L.h}`);

    const defs = el("defs", {}, svg);
    for (const arrow of [["arrow", "var(--dl-edge)"], ["arrowHi", "var(--primary)"], ["arrowCarry", "var(--dl-carry)"]]) {
      const m = el("marker", { id: arrow[0], viewBox: "0 0 10 10", refX: "9", refY: "5",
        markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 Z", fill: arrow[1] }, m);
    }

    gRoot = el("g", { id: "gRoot" }, svg);

    // column band backgrounds + stage labels
    L.colKeys.forEach((c, ci) => {
      const x = PAD + ci * (NW + COL_GAP) - COL_GAP / 2;
      if (ci > 0) el("line", { x1: x, y1: 24, x2: x, y2: L.h - 24,
        stroke: "var(--border-subtle)", "stroke-width": 1 }, gRoot);
    });

    const edgeLayer = el("g", { class: "dl-edges" }, gRoot);
    const carryLayer = el("g", { class: "dl-edges-carry" }, gRoot);
    const nodeLayer = el("g", { class: "dl-nodes" }, gRoot);

    // dependency edges
    NODES.forEach((n) => {
      (n.deps || []).forEach((d) => {
        const a = byId[d];
        if (!a) return;
        el("path", { d: edgePath(a, n), class: "dl-edge", "data-from": d, "data-to": n.id,
          fill: "none", stroke: "var(--dl-edge)", "stroke-width": 1.5, "marker-end": "url(#arrow)" }, edgeLayer);
      });
    });

    // cross-iteration carry edges (dashed, curve back to the left)
    CARRY_EDGES.forEach((ce) => {
      const a = byId[ce.from], b = byId[ce.to];
      if (!a || !b) return;
      const x1 = a._x + NW / 2, y1 = a._y + NH;
      const x2 = b._x + NW / 2, y2 = b._y + NH;
      const dip = Math.max(y1, y2) + 46;
      el("path", { d: `M${x1},${y1} C${x1},${dip} ${x2},${dip} ${x2},${y2}`,
        class: "dl-carry", fill: "none", stroke: "var(--dl-carry)", "stroke-width": 1.5,
        "stroke-dasharray": "5 4", "marker-end": "url(#arrowCarry)" }, carryLayer);
    });

    // nodes
    NODES.forEach((n) => {
      const g = el("g", { class: "dl-node", "data-id": n.id, transform: `translate(${n._x},${n._y})` }, nodeLayer);
      el("rect", { class: "dl-node-box", width: NW, height: NH, rx: 10,
        fill: (KIND[n.kind] || KIND.io).fill, stroke: "var(--border-strong)", "stroke-width": 1 }, g);
      const t = el("text", { class: "dl-node-title", x: 12, y: 22 }, g);
      t.textContent = n.label;
      const sub = el("text", { class: "dl-node-sub", x: 12, y: 40 }, g);
      sub.textContent = (n.unit || "") + (n.count ? " · " + n.count : "");
      if (n.scope === "manual") el("rect", { class: "dl-scope-dot", x: NW - 16, y: 8, width: 8, height: 8, rx: 2 }, g);
      g.addEventListener("click", (e) => { e.stopPropagation(); select(n.id); });
    });

    applyCarryVisibility();
    applyLabelVisibility();
    renderTensorLabels();
  }

  // ── tensor labels on dependency edges (midpoint chips) ──
  function renderTensorLabels() {
    const old = gRoot.querySelector(".dl-tensor-labels");
    if (old) old.remove();
    const layer = el("g", { class: "dl-tensor-labels" }, gRoot);
    NODES.forEach((n) => {
      (n.deps || []).forEach((d) => {
        const a = byId[d];
        if (!a) return;
        // pick the tensor that a writes and n reads
        const shared = (a.writes || []).filter((w) => (n.reads || []).includes(w));
        if (!shared.length) return;
        const name = TENSOR[shared[0]] ? TENSOR[shared[0]].t.split(" ")[0] : shared[0];
        const mx = (a._x + NW + n._x) / 2;
        const my = (a._y + n._y) / 2 + NH / 2;
        const g = el("g", { class: "dl-tlabel", transform: `translate(${mx},${my})` }, layer);
        const w = Math.max(28, name.length * 6.2 + 10);
        el("rect", { x: -w / 2, y: -9, width: w, height: 16, rx: 5 }, g);
        const t = el("text", { x: 0, y: 3 }, g);
        t.textContent = name;
      });
    });
  }

  function applyCarryVisibility() {
    const l = gRoot && gRoot.querySelector(".dl-edges-carry");
    if (l) l.style.display = state.showCarry ? "" : "none";
  }
  function applyLabelVisibility() {
    const l = gRoot && gRoot.querySelector(".dl-tensor-labels");
    if (l) l.style.display = state.showLabels ? "" : "none";
  }

  // ── selection: highlight node + its upstream/downstream, fill inspector ──
  function ancestry(id, dir, acc) {
    // dir: "up" follows deps, "down" follows dependents
    NODES.forEach((n) => {
      if (dir === "up" && n.id === id) {
        (n.deps || []).forEach((d) => { if (!acc.has(d)) { acc.add(d); ancestry(d, "up", acc); } });
      }
      if (dir === "down" && (n.deps || []).includes(id) && !acc.has(n.id)) {
        acc.add(n.id); ancestry(n.id, "down", acc);
      }
    });
    return acc;
  }

  function select(id) {
    state.selected = id;
    const up = ancestry(id, "up", new Set());
    const down = ancestry(id, "down", new Set());
    const related = new Set([id, ...up, ...down]);

    gRoot.querySelectorAll(".dl-node").forEach((g) => {
      const nid = g.getAttribute("data-id");
      g.classList.toggle("is-selected", nid === id);
      g.classList.toggle("is-related", related.has(nid) && nid !== id);
      g.classList.toggle("is-dim", !related.has(nid));
    });
    gRoot.querySelectorAll(".dl-edge").forEach((p) => {
      const f = p.getAttribute("data-from"), t = p.getAttribute("data-to");
      const on = related.has(f) && related.has(t);
      p.classList.toggle("is-hot", on);
      p.classList.toggle("is-dim", !on);
      p.setAttribute("marker-end", on ? "url(#arrowHi)" : "url(#arrow)");
    });
    renderInspector(byId[id], up, down);
  }

  function clearSelection() {
    state.selected = null;
    gRoot.querySelectorAll(".dl-node").forEach((g) => g.classList.remove("is-selected", "is-related", "is-dim"));
    gRoot.querySelectorAll(".dl-edge").forEach((p) => {
      p.classList.remove("is-hot", "is-dim");
      p.setAttribute("marker-end", "url(#arrow)");
    });
    renderInspectorEmpty();
  }

  // ── inspector (right panel) ──
  const inspector = document.getElementById("inspector");
  function tensorRows(ids) {
    if (!ids || !ids.length) return '<span class="dl-none">—</span>';
    return ids.map((id) => {
      const t = TENSOR[id];
      if (!t) return `<div class="dl-trow"><code>${id}</code></div>`;
      return `<div class="dl-trow">
        <div class="dl-trow-h"><code>${t.t}</code><span class="dl-dt dl-dt-${t.dt.toLowerCase()}">${t.dt}</span></div>
        <div class="dl-trow-shape">${t.shape}</div>
        <div class="dl-trow-role">${t.role}</div>
      </div>`;
    }).join("");
  }
  function nodeChips(ids) {
    if (!ids || !ids.size) return '<span class="dl-none">—</span>';
    return [...ids].map((id) => {
      const n = byId[id]; if (!n) return "";
      return `<button class="dl-chip dl-chip-${n.kind}" data-goto="${id}">${n.label}</button>`;
    }).join("");
  }
  function renderInspector(n, up, down) {
    if (!n) return renderInspectorEmpty();
    const k = KIND[n.kind] || KIND.io;
    inspector.innerHTML = `
      <div class="dl-insp-head">
        <div class="dl-insp-badge" style="background:${k.fill}"></div>
        <div>
          <div class="dl-insp-title">${n.label}</div>
          <div class="dl-insp-meta">${k.label}${n.count ? " · " + n.count + " tasks" : ""} · ${n.scope === "manual" ? "manual_scope" : "auto-dep"}</div>
        </div>
      </div>
      ${n.formula ? `<div class="dl-insp-formula">${n.formula}</div>` : ""}
      <p class="dl-insp-desc">${n.desc || ""}</p>
      <div class="dl-insp-sec"><h4>读取张量</h4>${tensorRows(n.reads)}</div>
      <div class="dl-insp-sec"><h4>写入张量</h4>${tensorRows(n.writes)}</div>
      <div class="dl-insp-sec"><h4>直接依赖 (deps)</h4><div class="dl-chips">${nodeChips(new Set(n.deps || []))}</div></div>
      <div class="dl-insp-sec"><h4>全部上游 (${up.size})</h4><div class="dl-chips">${nodeChips(up)}</div></div>
      <div class="dl-insp-sec"><h4>全部下游 (${down.size})</h4><div class="dl-chips">${nodeChips(down)}</div></div>
    `;
    inspector.querySelectorAll("[data-goto]").forEach((b) =>
      b.addEventListener("click", () => select(b.getAttribute("data-goto"))));
  }
  function renderInspectorEmpty() {
    const groups = [
      ["Scope 1 · RMSNorm + QKV", ["rms_recip", "q_seed", "q_proj", "k_proj", "v_proj", "qk_norm", "rope_qkv"]],
      ["Scope 2 · 分页注意力", ["fa_work_build", "fa_fused", "online_softmax"]],
      ["Scope 3 · out_proj + MLP", ["out_proj", "residual_rms_cast", "post_rms_reduce", "gate_proj", "up_proj", "silu", "down_proj"]],
      ["融合输出 (auto-dep)", ["dcr_xgamma"]],
    ];
    inspector.innerHTML = `
      <div class="dl-insp-intro">
        <h3>decode_layer 单层计算图</h3>
        <p>Qwen3-14B decode kernel 的一层前向：<b>FP32 层间 carry</b> + 融合的层输出/下一层 x·γ。
        节点按依赖阶段从左到右排列；点击任意节点可查看它读写的张量与完整上下游链路。</p>
        <p class="dl-insp-note">■ 右上角小方块 = 该任务在 <code>manual_scope</code> 内（tensormap 自动依赖被抑制，只有显式 <code>deps=</code> 生效）。虚线 = 跨层 carry 边（dcr_xgamma → 下一层）。</p>
      </div>
      ${groups.map(([title, ids]) => `
        <div class="dl-insp-sec"><h4>${title}</h4>
          <div class="dl-chips">${ids.map((id) => {
            const n = byId[id]; return n ? `<button class="dl-chip dl-chip-${n.kind}" data-goto="${id}">${n.label}</button>` : "";
          }).join("")}</div>
        </div>`).join("")}
    `;
    inspector.querySelectorAll("[data-goto]").forEach((b) =>
      b.addEventListener("click", () => select(b.getAttribute("data-goto"))));
  }

  // ── legend ──
  function renderLegend() {
    const legend = document.getElementById("legend");
    legend.innerHTML = Object.keys(KIND).map((k) =>
      `<span class="dl-leg-item"><i style="background:${KIND[k].fill}"></i>${KIND[k].label}</span>`).join("");
  }

  // ── pan / zoom ──
  function applyTransform() {
    gRoot.setAttribute("transform", `translate(${state.tx},${state.ty}) scale(${state.scale})`);
    const z = document.getElementById("zoomVal");
    if (z) z.textContent = Math.round(state.scale * 100) + "%";
  }
  function setScale(s, cx, cy) {
    s = Math.min(2.4, Math.max(0.2, s));
    const rect = svg.getBoundingClientRect();
    // keep the point under (cx,cy) fixed while zooming
    const px = (cx - rect.left - state.tx) / state.scale;
    const py = (cy - rect.top - state.ty) / state.scale;
    state.scale = s;
    state.tx = cx - rect.left - px * s;
    state.ty = cy - rect.top - py * s;
    applyTransform();
  }
  function fit() {
    const rect = wrap.getBoundingClientRect();
    // The graph is a long horizontal pipeline (15 stages). Fitting the full width
    // shrinks nodes to an unreadable ~0.2x, so fit to HEIGHT for legible nodes and
    // let the user pan horizontally. Only fall back to width-fit if the whole graph
    // already fits (short graphs / very wide canvas).
    const sH = (rect.height - 48) / L.h;
    const sW = (rect.width - 48) / L.w;
    state.scale = Math.min(1.1, Math.max(0.3, sW >= sH ? sW : sH));
    state.tx = state.scale >= sW ? 24 : (rect.width - L.w * state.scale) / 2;
    state.ty = (rect.height - L.h * state.scale) / 2;
    applyTransform();
  }

  function wireViewport() {
    let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;
    svg.addEventListener("mousedown", (e) => {
      if (e.target.closest(".dl-node")) return;
      dragging = true; sx = e.clientX; sy = e.clientY; ox = state.tx; oy = state.ty;
      svg.classList.add("is-panning");
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      state.tx = ox + (e.clientX - sx); state.ty = oy + (e.clientY - sy);
      applyTransform();
    });
    window.addEventListener("mouseup", () => { dragging = false; svg.classList.remove("is-panning"); });
    svg.addEventListener("click", (e) => { if (!e.target.closest(".dl-node")) clearSelection(); });
    wrap.addEventListener("wheel", (e) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      setScale(state.scale * (e.deltaY < 0 ? 1.12 : 0.89), e.clientX, e.clientY);
    }, { passive: false });

    document.getElementById("zoomIn").addEventListener("click", () => {
      const r = wrap.getBoundingClientRect(); setScale(state.scale * 1.15, r.left + r.width / 2, r.top + r.height / 2);
    });
    document.getElementById("zoomOut").addEventListener("click", () => {
      const r = wrap.getBoundingClientRect(); setScale(state.scale * 0.87, r.left + r.width / 2, r.top + r.height / 2);
    });
    document.getElementById("zoomFit").addEventListener("click", fit);
  }

  // ── toggles ──
  function wireToggles() {
    const c = document.getElementById("tglCarry");
    const l = document.getElementById("tglLabels");
    c.addEventListener("change", () => { state.showCarry = c.checked; applyCarryVisibility(); });
    l.addEventListener("change", () => { state.showLabels = l.checked; applyLabelVisibility(); });
  }

  // ── side panel: stage index (jump list) ──
  const STAGES = [
    ["1", "RMSNorm + QKV 投影", ["rms_recip", "q_proj", "k_proj", "v_proj", "qk_norm", "rope_qkv"]],
    ["2", "分页 GQA 注意力", ["fa_work_build", "fa_fused", "online_softmax"]],
    ["3a", "out_proj + 残差", ["out_proj", "residual_rms_cast", "post_rms_reduce"]],
    ["3b", "MLP (gate/up/silu/down)", ["gate_proj", "up_proj", "silu", "down_proj"]],
    ["→", "融合输出 · 跨层 carry", ["dcr_xgamma", "out_next"]],
  ];
  function renderSide() {
    const side = document.getElementById("sidePanel");
    side.innerHTML = `
      <div class="dl-side-head">计算阶段</div>
      <div class="dl-side-sub">${NODES.length} 个任务组 · 点击定位</div>
      <div class="dl-stage-list">
        ${STAGES.map(([tag, name, ids]) => `
          <div class="dl-stage">
            <div class="dl-stage-h"><span class="dl-stage-tag">${tag}</span>${name}</div>
            ${ids.map((id) => { const n = byId[id]; return n
              ? `<button class="dl-stage-node dl-chip-${n.kind}" data-goto="${id}"><i></i>${n.label}${n.count ? `<em>${n.count}</em>` : ""}</button>` : ""; }).join("")}
          </div>`).join("")}
      </div>`;
    side.querySelectorAll("[data-goto]").forEach((b) =>
      b.addEventListener("click", () => { select(b.getAttribute("data-goto")); focusNode(b.getAttribute("data-goto")); }));
  }
  function focusNode(id) {
    const n = byId[id]; if (!n) return;
    const rect = wrap.getBoundingClientRect();
    state.tx = rect.width / 2 - (n._x + NW / 2) * state.scale;
    state.ty = rect.height / 2 - (n._y + NH / 2) * state.scale;
    applyTransform();
  }

  // ── view tabs: layer graph is the SVG; host + math swap the inspector intro ──
  function wireTabs() {
    const tabs = document.querySelectorAll(".dl-tabs [role=tab]");
    tabs.forEach((t) => t.addEventListener("click", () => {
      tabs.forEach((x) => x.classList.remove("is-active"));
      t.classList.add("is-active");
      const v = t.getAttribute("data-view");
      document.body.setAttribute("data-view", v);
      if (v === "host") renderHostView();
      else if (v === "math") renderMathView();
      else { clearSelection(); }
    }));
  }
  function renderHostView() {
    clearSelection();
    inspector.innerHTML = `
      <div class="dl-insp-intro">
        <h3>多层编排 (decode_fwd)</h3>
        <p><code>_decode_layer</code> 是 <code>@pl.jit.inline</code> 体，被两个 host 循环复用：</p>
      </div>
      <div class="dl-insp-sec"><h4>decode_fwd · 单派发 40 层 + LM head</h4>
        <p class="dl-insp-desc">① <b>copy_hidden</b>：BF16 embed → FP32 cur（首层边界，一次转换）。
        ② <b>x_gamma0</b>：layer0 的 x·γ（其余层由上一层 dcr_xgamma 产出）。
        ③ 循环 40 次 <b>_decode_layer</b>，carry_tids / carry_normed_tids 逐层由 dcr 刷新。
        ④ <b>cast_lmhead_in</b>：末层 FP32 → BF16（一次）。⑤ <b>rms_lm_head</b> 出 logits [16, VOCAB]。</p>
      </div>
      <div class="dl-insp-sec"><h4>decode_fwd_layers · 分块 8 层，无 LM head</h4>
        <p class="dl-insp-desc">单次 40 层派发会超过 AICPU stream-sync 超时 (2000ms)，所以按 <code>_CHUNK_NLAYERS=8</code> 分块跑。
        块首 copy_hidden（BF16→FP32）、块尾 copy_out（FP32→BF16），块间 host 传 BF16。</p>
      </div>
      <div class="dl-insp-sec"><h4>跨层依赖 carry</h4>
        <p class="dl-insp-desc"><code>prev_out_tids</code>（DOWN_ON=5 个 slab writer of hidden）门控 rms_recip / 残差 / seeds；
        <code>prev_normed_tids</code>（5 个 slab writer of normed_in）门控 QKV。两者每层都由本层的 dcr_xgamma（同一任务写 out + normed_out）刷新。</p>
      </div>`;
  }
  function renderMathView() {
    clearSelection();
    inspector.innerHTML = `
      <div class="dl-insp-intro"><h3>数学数据流</h3>
      <p>一层 Qwen3 decode 的等价数学链（省略 batch/head 下标）：</p></div>
      ${[
        ["输入 RMSNorm", "x̂ = x / √(mean(x²)+ε)  ·  γ_in", "1/rms 延迟折进 rope，x·γ 先行喂 QKV"],
        ["Q/K/V 投影", "q = x̂ @ Wq,  k = x̂ @ Wk,  v = x̂ @ Wv", "split-K 原子加，FP32 累加"],
        ["QK-norm + RoPE", "q,k ← RoPE(qknorm(q,k))", "per-head 1/rms + 旋转 lo/hi 半维；K/V 写分页 cache"],
        ["注意力", "o = softmax(qkᵀ/√d) · v", "分块 flash-decoding + 在线 softmax 跨块归并"],
        ["out_proj + 残差", "h1 = x + o @ Wo", "FP32 残差；post-RMS 的 1/rms 延迟到 silu"],
        ["MLP", "h2 = (SiLU(ĥ1@Wg) · (ĥ1@Wu)) @ Wd", "ĥ1 = h1·post_γ；gate/up split-K，silu，down split-K"],
        ["层输出 (融合)", "out = h1 + h2;  x·γ_next = out · γ_in[next]", "dcr_xgamma 一个任务同时产出残差与下一层归一化输入"],
      ].map(([t, f, d], i) => `
        <div class="dl-math-step">
          <div class="dl-math-n">${i + 1}</div>
          <div><div class="dl-math-t">${t}</div>
          <div class="dl-math-f">${f}</div>
          <div class="dl-math-d">${d}</div></div>
        </div>`).join("")}`;
  }

  // ── init ──
  renderGraph();
  renderTensorLabels();
  applyLabelVisibility();
  renderLegend();
  renderSide();
  renderInspectorEmpty();
  wireViewport();
  wireToggles();
  wireTabs();
  fit();
  window.addEventListener("resize", fit);
})();
