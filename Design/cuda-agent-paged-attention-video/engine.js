// 确定性时间轴引擎：render(t) 只依赖 t，便于逐帧截图合成视频。
(function () {
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (p) => (p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2);
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // ---------------- timeline builder ----------------
  const B = { now: 0, tl: [], cur: { x: 960, y: 700 }, clock: 0 };
  const at = (t0, d, fn) => B.tl.push({ t0, d, fn, i: B.tl.length });
  const seq = (d, fn) => { at(B.now, d, fn); B.now += d; };
  const par = (d, fn, off = 0) => at(B.now + off, d, fn);
  const wait = (d) => { B.now += d; };
  const set = (fn) => seq(0, fn);
  const parts = (text, p) => text.slice(0, Math.round(text.length * p));

  function move(x, y, d = 0.7) {
    const fx = B.cur.x, fy = B.cur.y;
    par(d, (s, p) => { const e = ease(p); s.cursor.x = fx + (x - fx) * e; s.cursor.y = fy + (y - fy) * e; });
    B.cur = { x, y }; wait(d);
  }
  function click() {
    par(0.4, (s, p) => { s.cursor.ripple = p < 1 ? p : null; });
    wait(0.18);
  }
  const caption = (step, text) => set((s) => { s.caption = text ? { step, text } : null; });
  const ff = (txt) => set((s) => { s.ff = txt; });
  function clockTo(sec, d) {
    const from = B.clock; B.clock = sec;
    par(d, (s, p) => { s.agent.clock = from + (sec - from) * p; });
  }

  // terminals: target = 'wt' | 'term'
  function prompt(target, html) { set((s) => { s[target].pending = html; }); }
  function typeCmd(target, cmd, d) {
    seq(d, (s, p) => {
      const pr = s[target].pending || ''; s[target].pending = null;
      s[target].lines.push(pr + esc(parts(cmd, p)));
      s[target].typing = p < 1;
    });
  }
  function out(target, lines, d = 0) {
    seq(d, (s, p) => {
      const n = d ? Math.ceil(lines.length * p) : lines.length;
      for (let i = 0; i < n; i++) s[target].lines.push(lines[i]);
    });
  }

  // agent panel
  function msg(s, id, init) {
    let m = s.agent.byId[id];
    if (!m) { m = Object.assign({ id }, init); s.agent.byId[id] = m; s.agent.msgs.push(m); }
    return m;
  }
  function userMsg(id, text) { set((s) => { msg(s, id, { kind: 'user', text }); }); }
  function say(id, md, d, first = false) {
    seq(d, (s, p) => { const m = msg(s, id, { kind: 'text', first }); m.text = parts(md, p); s.agent.thinking = false; });
  }
  function thinking(on = true) { set((s) => { s.agent.thinking = on; }); }
  function toolStart(id, tool, arg) {
    set((s) => { s.agent.thinking = false; s.agent.turn += 1; msg(s, id, { kind: 'tool', tool, arg, status: 'run', out: [] }); });
  }
  function toolEnd(id, status, meta, outLines, diff) {
    set((s) => { const m = s.agent.byId[id]; m.status = status; m.meta = meta; if (outLines) m.out = outLines; if (diff) m.diff = diff; });
  }

  // file system / editor
  const lang = (p) => (p.endsWith('.py') ? 'py' : p.endsWith('.md') ? 'md' : p.endsWith('.sh') ? 'sh' : /\.(cu|cpp|h)$/.test(p) ? 'cpp' : 'txt');
  function openTab(s, path) {
    if (!s.editor.tabs.includes(path)) s.editor.tabs.push(path);
    s.editor.active = path; s.treeSel = path;
  }
  function openFile(path) { set((s) => openTab(s, path)); }
  function fileStatus(path, st) { set((s) => { s.paths[path] = st; }); }
  function writeFile(path, text, d, status = 'U') {
    const lines = text.split('\n');
    seq(d, (s, p) => {
      openTab(s, path);
      const n = Math.ceil(lines.length * p);
      s.fs[path] = lines.slice(0, n).map((t) => ({ t, k: 'add' }));
      s.editor.follow[path] = n; s.paths[path] = status;
      s.editor.caret = { path, line: n - 1, col: (lines[n - 1] || '').length };
    });
  }
  function setFile(path, text) { set((s) => { s.fs[path] = text.split('\n').map((t) => ({ t, k: '' })); }); }
  function settle(path) {
    set((s) => {
      const f = s.fs[path] || []; f.forEach((l) => (l.k = ''));
      if (s.editor.follow[path] != null) s.editor.scroll[path] = Math.max(0, f.length - 28);
      delete s.editor.follow[path]; s.editor.caret = null;
    });
  }
  // replace lines containing `oldLines` (exact match sequence) with `newLines`
  function editFile(path, oldLines, newLines, d, from = 0) {
    seq(d, (s, p) => {
      openTab(s, path);
      const f = s.fs[path];
      let idx = -1;
      for (let i = from; i < f.length; i++) if (f[i].t === oldLines[0]) { idx = i; break; }
      if (idx < 0) return;
      s.editor.center[path] = idx;
      if (p < 1) {
        const del = oldLines.map((t) => ({ t, k: 'del' }));
        const shown = Math.ceil(newLines.length * clamp(p * 2 - 0.3));
        const add = newLines.slice(0, shown).map((t) => ({ t, k: 'add' }));
        f.splice(idx, oldLines.length, ...del, ...add);
      } else {
        f.splice(idx, oldLines.length, ...newLines.map((t) => ({ t, k: 'add' })));
      }
    });
  }

  // ---------------- state ----------------
  function initState() {
    return {
      intro: 1, outro: 0, scene: 'desktop',
      wt: { vis: 0, lines: [], pending: null, typing: false, title: 'Windows PowerShell' },
      term: { lines: [], pending: null, typing: false, name: 'bash' },
      vs: { vis: 0, connecting: false, view: 'explorer' },
      paths: {}, openDirs: { kernels: true }, treeSel: null,
      fs: {},
      editor: { tabs: [], active: null, follow: {}, center: {}, scroll: {}, caret: null, selAll: false },
      agent: { msgs: [], byId: {}, thinking: false, turn: 0, clock: 0, running: false, input: '', focus: false },
      cursor: { x: 960, y: 700, ripple: null, vis: 1 },
      caption: null, ff: null, taskbarActive: null, outroData: null,
    };
  }

  // ---------------- tree ----------------
  function treeRows(paths, openDirs) {
    const root = { dirs: {}, files: [] };
    Object.keys(paths).forEach((p) => {
      const parts = p.split('/');
      let node = root;
      for (let i = 0; i < parts.length - 1; i++) {
        node.dirs[parts[i]] = node.dirs[parts[i]] || { dirs: {}, files: [], path: parts.slice(0, i + 1).join('/') };
        node = node.dirs[parts[i]];
      }
      if (parts[parts.length - 1] !== '') node.files.push(p);
    });
    const cmp = (a, b) => a.localeCompare(b, 'en', { numeric: true, sensitivity: 'base' });
    const rows = [];
    (function walk(node, depth) {
      Object.keys(node.dirs).sort(cmp).forEach((d) => {
        const dn = node.dirs[d];
        const open = !!openDirs[dn.path];
        const sts = Object.keys(paths).filter((p) => p.startsWith(dn.path + '/') && paths[p] && paths[p] !== 'ign').map((p) => paths[p]);
        rows.push({ name: d, path: dn.path, dir: true, open, depth, status: sts.length ? '•' : '', ign: paths[dn.path + '/'] === 'ign' });
        if (open) walk(dn, depth + 1);
      });
      node.files.slice().sort((a, b) => cmp(a.split('/').pop(), b.split('/').pop())).forEach((f) => {
        rows.push({ name: f.split('/').pop(), path: f, dir: false, depth, status: paths[f] || '' });
      });
    })(root, 0);
    return rows;
  }
  const ICON = {
    cu: ['CU', '#76b900', '#0f1a00'], cpp: ['C++', '#519aba', '#0b1720'], h: ['h', '#a074c4', '#170f20'], py: ['py', '#ffd43b', '#1d2a3a'],
    md: ['M↓', '#519aba', '#0b1720'], sh: ['$_', '#89e051', '#14200a'], so: ['so', '#9a9a9a', '#222'], bak: ['~', '#9a9a9a', '#222'],
  };
  function fileIcon(name) {
    const ext = name.split('.').pop();
    const ic = ICON[ext] || ['·', '#9a9a9a', '#222'];
    return `<span class="fi" style="color:${ic[1]};background:${ic[2]}">${ic[0]}</span>`;
  }

  // ---------------- syntax highlight ----------------
  const KW = {
    cpp: new Set('if else for while do return break continue switch case default const constexpr static inline template typename struct class namespace using auto extern sizeof true false nullptr unsigned signed'.split(' ')),
    py: new Set('import from as class def return if else elif for in not and or None True False with lambda pass raise try except while'.split(' ')),
  };
  const CTRL = new Set('if else for while do return break continue switch case import from return'.split(' '));
  const TY = new Set('void int float double half half2 float2 float4 uint4 size_t bool char dim3 cudaStream_t Tensor Half __global__ __device__ __host__ __shared__ __forceinline__ __restrict__ __launch_bounds__ __align__'.split(' '));
  const hlCache = new Map();
  function hl(line, lg) {
    const key = lg + '\u0000' + line;
    if (hlCache.has(key)) return hlCache.get(key);
    let html = '';
    if (lg === 'md') {
      html = /^#/.test(line) ? `<span class="h1">${esc(line)}</span>` : esc(line).replace(/`([^`]+)`/g, '<span class="s">`$1`</span>');
    } else if (lg === 'cpp' || lg === 'py' || lg === 'sh') {
      const pp = lg === 'cpp' && line.match(/^(\s*)(#\s*\w+)(.*)$/);
      if (pp) {
        const rest = pp[3].replace(/^(\s*)(<[^>]*>|"[^"]*")/, (m, a, b) => a + '\u0001' + b + '\u0002');
        html = esc(pp[1]) + `<span class="pp">${esc(pp[2])}</span>` + esc(rest).replace(/\u0001(.*?)\u0002/, '<span class="s">$1</span>');
      } else {
        const re = lg === 'cpp'
          ? /(\/\/.*$)|("(?:[^"\\]|\\.)*")|(\b0x[0-9a-fA-F]+\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?f?\b)|([A-Za-z_]\w*)|(\s+)|([\s\S])/g
          : /(#.*$)|((?:f|r)?"(?:[^"\\]|\\.)*"|(?:f|r)?'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)|([A-Za-z_]\w*)|(\s+)|([\s\S])/g;
        let m; const kw = KW[lg === 'sh' ? 'py' : lg];
        while ((m = re.exec(line))) {
          if (m[1]) html += `<span class="c">${esc(m[1])}</span>`;
          else if (m[2]) html += `<span class="s">${esc(m[2])}</span>`;
          else if (m[3]) html += `<span class="n">${esc(m[3])}</span>`;
          else if (m[4]) {
            const w = m[4]; const next = line.slice(re.lastIndex).match(/^\s*([(<])/);
            let cls = 'v';
            if (kw.has(w)) cls = CTRL.has(w) ? 'kc' : 'k';
            else if (TY.has(w)) cls = /^__/.test(w) ? 'k' : 'ty';
            else if (lg === 'py' && w === 'self') cls = 'k';
            else if (/^[A-Z][A-Z0-9_]+$/.test(w)) cls = 'cn';
            else if (next && next[1] === '(') cls = 'fn';
            else if (/^[A-Z]/.test(w) || w === 'torch' || w === 'nn' || w === 'at' || w === 'py' || w === 'c10' || w === 'cuda') cls = 'ty';
            html += `<span class="${cls}">${esc(w)}</span>`;
          } else html += esc(m[5] || m[6]);
        }
      }
    } else html = esc(line);
    hlCache.set(key, html);
    return html;
  }

  // terminal markup:  \u0001cls\u0002text\u0003
  const C = (cls, text) => `\u0001${cls}\u0002${text}\u0003`;
  function tmark(line) {
    let html = '', last = 0;
    line.replace(/\u0001(\w+)\u0002([\s\S]*?)\u0003/g, (m, cls, txt, off) => {
      html += line.slice(last, off) + `<span class="${cls}">${txt}</span>`;
      last = off + m.length; return m;
    });
    return html + line.slice(last);
  }

  // mini markdown for agent text
  function md(text) {
    const inline = (s) => esc(s)
      .replace(/\*\*(.+?)(\*\*|$)/g, '<b>$1</b>')
      .replace(/`([^`]+)(`|$)/g, '<code>$1</code>');
    const lines = text.split('\n'); let html = ''; let i = 0;
    while (i < lines.length) {
      const l = lines[i];
      if (/^\|/.test(l)) {
        html += '<table>';
        while (i < lines.length && /^\|/.test(lines[i])) {
          const cells = lines[i].split('|').slice(1, -1);
          if (!/^\s*-+/.test(cells[0] || '')) html += '<tr>' + cells.map((c) => `<td>${inline(c.trim())}</td>`).join('') + '</tr>';
          i++;
        }
        html += '</table>';
      } else if (/^\d+\. /.test(l)) {
        html += '<ol>';
        while (i < lines.length && /^\d+\. /.test(lines[i])) { html += `<li>${inline(lines[i].replace(/^\d+\. /, ''))}</li>`; i++; }
        html += '</ol>';
      } else { if (l.trim()) html += `<p>${inline(l)}</p>`; i++; }
    }
    return html;
  }

  // ---------------- icons ----------------
  const svg = (d, w = 24) => `<svg width="${w}" height="${w}" viewBox="0 0 24 24" fill="none" stroke="#d7d7d7" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
  const ACT = {
    explorer: svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>'),
    search: svg('<circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/>'),
    scm: svg('<circle cx="6" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="8" r="2"/><path d="M6 7v10M18 10c0 4-6 3-11.5 7.5"/>'),
    run: svg('<path d="M7 4l12 8-12 8z"/>'),
    ext: svg('<rect x="4" y="4" width="7" height="7"/><rect x="13" y="13" width="7" height="7"/><rect x="4" y="13" width="7" height="7"/><path d="M14 4h6v6"/>'),
    remote: svg('<rect x="3" y="5" width="18" height="12" rx="1"/><path d="M8 21h8M12 17v4"/>'),
    agent: svg('<path d="M12 3l2.2 5.3L20 9l-4.4 3.8L17 19l-5-3-5 3 1.4-6.2L4 9l5.8-.7z"/>'),
  };
  const TB = [
    ['start', '<svg width="20" height="20" viewBox="0 0 20 20"><rect x="2" y="2" width="7" height="7" rx="1" fill="#4cc2ff"/><rect x="11" y="2" width="7" height="7" rx="1" fill="#4cc2ff"/><rect x="2" y="11" width="7" height="7" rx="1" fill="#4cc2ff"/><rect x="11" y="11" width="7" height="7" rx="1" fill="#4cc2ff"/></svg>'],
    ['search', '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#eee" stroke-width="2"><circle cx="11" cy="11" r="6"/><path d="M20 20l-4.5-4.5"/></svg>'],
    ['files', '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M3 6a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" fill="#f8c537"/></svg>'],
    ['terminal', '<svg width="22" height="22" viewBox="0 0 24 24"><rect x="2" y="4" width="20" height="16" rx="3" fill="#2b2b2b" stroke="#888"/><path d="M6 9l3 3-3 3M11 15h6" stroke="#fff" stroke-width="1.8" fill="none"/></svg>'],
    ['vscode', '<svg width="22" height="22" viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="20" rx="5" fill="#2489ca"/><path d="M9 8l-4 4 4 4M15 8l4 4-4 4" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round"/></svg>'],
    ['browser', '<svg width="22" height="22" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" fill="#3b82f6"/><path d="M2 12h20M12 2c3 3 3 17 0 20M12 2c-3 3-3 17 0 20" stroke="#cfe3ff" stroke-width="1.3" fill="none"/></svg>'],
  ];

  // ---------------- render ----------------
  const $ = (id) => document.getElementById(id);
  const cache = {};
  function put(id, html) { if (cache[id] !== html) { cache[id] = html; $(id).innerHTML = html; return true; } return false; }
  const fmtClock = (sec) => { sec = Math.floor(sec); return `${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`; };

  function compute(t) {
    const s = initState();
    for (const a of TL) {
      if (t < a.t0) break;
      const p = a.d <= 0 ? 1 : clamp((t - a.t0) / a.d);
      a.fn(s, p, t);
    }
    return s;
  }

  function drawTerminal(id, st, t, caretOn) {
    const lines = st.lines.slice();
    if (st.pending != null) lines.push(st.pending);
    const blink = st.typing || Math.floor(t * 1.8) % 2 === 0;
    let html = lines.map(tmark).join('\n');
    if (caretOn && blink) html += '<span class="tcaret"></span>';
    if (put(id, html)) { const el = $(id); el.scrollTop = el.scrollHeight; }
  }

  function drawVS(s, t) {
    // activity bar
    const scmCount = Object.values(s.paths).filter((v) => v === 'U' || v === 'M' || v === 'D').length;
    put('vs-act', ['explorer', 'search', 'scm', 'run', 'ext', 'remote', 'agent'].map((k) =>
      `<div class="act-i ${s.vs.view === k || (k === 'agent') ? 'on' : ''}">${ACT[k]}${k === 'scm' && scmCount ? `<span class="act-badge">${scmCount}</span>` : ''}</div>`).join(''));

    // sidebar
    let side = '';
    if (s.vs.view === 'scm') {
      const items = Object.keys(s.paths).filter((p) => ['U', 'M', 'D'].includes(s.paths[p])).sort();
      side = `<div class="side-title">SOURCE CONTROL</div><div class="scm-box">Message (Ctrl+Enter to commit)</div><div class="scm-btn">✓ Commit</div>
        <div class="side-sec">⌄ Changes <span class="side-badge">${items.length}</span></div>` +
        items.map((p) => `<div class="tree-row ${s.paths[p]} ${s.treeSel === p ? 'sel' : ''}" style="padding-left:20px">${fileIcon(p)}<span class="nm">${esc(p.split('/').pop())} <span style="color:#777;font-size:11px">${esc(p.includes('/') ? p.split('/').slice(0, -1).join('/') : '')}</span></span><span class="st">${s.paths[p]}</span></div>`).join('');
    } else {
      side = `<div class="side-title">EXPLORER</div><div class="side-sec">⌄ AGENT_WORKDIR [SSH: ${SRC.HOST.toUpperCase()}]</div>`;
      if (s.vs.connecting) side += `<div class="connecting"><span class="spin" style="transform:rotate(${t * 360}deg)"></span>&nbsp; Setting up SSH Host ${SRC.HOST}:<br>Connecting…</div>`;
      else side += treeRows(s.paths, s.openDirs).map((r) => {
        const st = r.dir ? '' : r.status;
        const cls = [st === 'U' || st === 'M' || st === 'D' ? st : '', st === 'ign' || r.ign ? 'ign' : '', s.treeSel === r.path ? 'sel' : ''].join(' ');
        const icon = r.dir ? '' : fileIcon(r.name);
        return `<div class="tree-row ${cls}" style="padding-left:${8 + r.depth * 12}px"><span class="tw">${r.dir ? (r.open ? '⌄' : '›') : ''}</span>${icon}<span class="nm">${esc(r.name)}</span><span class="st">${r.dir ? (r.status ? '<span style="color:#e2c08d">●</span>' : '') : (['U', 'M', 'D'].includes(st) ? st : '')}</span></div>`;
      }).join('');
    }
    put('vs-side', side);

    // tabs
    put('vs-tabs', s.editor.tabs.map((p) => {
      const st = s.paths[p];
      return `<div class="tab ${p === s.editor.active ? 'on' : ''} ${st === 'U' || st === 'M' ? st : ''}">${fileIcon(p)}<span class="tn">${esc(p.split('/').pop())}</span>${st === 'U' || st === 'M' ? `<span style="font-size:11px;opacity:.8">${st}</span>` : ''}<span class="x">✕</span></div>`;
    }).join(''));
    const act = s.editor.active;
    put('vs-crumbs', act ? act.split('/').map(esc).join(' <span style="color:#666">›</span> ') : '');

    // code
    let code = '';
    if (!act) {
      code = `<div class="empty-ed"><svg width="120" height="120" viewBox="0 0 24 24" opacity=".25"><path d="M9 8l-4 4 4 4M15 8l4 4-4 4" stroke="#888" stroke-width="2" fill="none" stroke-linecap="round"/></svg><div>Show All Commands&nbsp;&nbsp; Ctrl+Shift+P</div><div>Go to File&nbsp;&nbsp; Ctrl+P</div></div>`;
      put('vs-mini', '');
    } else {
      const f = s.fs[act] || [];
      const VIS = 32;
      let top = 0;
      if (s.editor.scroll[act] != null) top = s.editor.scroll[act];
      if (s.editor.center[act] != null) top = Math.max(0, s.editor.center[act] - 12);
      if (s.editor.follow[act] != null) top = Math.max(0, s.editor.follow[act] - VIS + 4);
      top = Math.min(top, Math.max(0, f.length - 8));
      const lg = lang(act);
      let no = 0; const nums = f.map((l) => (l.k === 'del' ? '' : ++no));
      for (let i = top; i < Math.min(f.length, top + VIS); i++) {
        const l = f[i];
        const caret = s.editor.caret && s.editor.caret.path === act && s.editor.caret.line === i;
        code += `<div class="ln ${l.k} ${s.editor.selAll ? 'sel' : ''} ${caret ? 'cur hl' : ''}"><span class="no">${nums[i]}</span><span class="gut"></span><span class="tx">${hl(l.t, lg) || ' '}${caret ? '<span class="caret"></span>' : ''}</span></div>`;
      }
      put('vs-mini', f.slice(0, 220).map((l) => `<div style="width:${Math.min(66, l.t.replace(/\s+$/, '').length * 0.6)}px;margin-left:${(l.t.match(/^\s*/)[0].length) * 0.6}px;${l.k === 'add' ? 'background:#6a9b2a' : l.k === 'del' ? 'background:#b33' : ''}"></div>`).join(''));
    }
    put('vs-code', code);

    // terminal
    put('term-name', `⌨ ${esc(s.term.name)}`);
    drawTerminal('term-body', s.term, t, s.term.caret);

    // agent
    const a = s.agent;
    put('ag-meta', a.turn || a.running ? `<span>Turn ${a.turn}</span><span>⏱ ${fmtClock(a.clock)}</span><span style="color:${a.running ? '#73c991' : '#888'}">● ${a.running ? 'running' : 'idle'}</span>` : '<span style="color:#888">● idle</span>');
    let mh = '';
    if (!a.msgs.length) mh = `<div style="color:#777;font-size:12px;text-align:center;margin-top:220px;line-height:22px">CUDA Agent 已就绪<br>工作目录：agent_workdir · 规则：SKILL.md</div>`;
    for (const m of a.msgs) {
      if (m.kind === 'user') mh += `<div class="m-user"><div class="who"><span class="av" style="background:#6b5bd6">D</span>dev</div>${md(m.text)}</div>`;
      else if (m.kind === 'text') mh += `<div class="m-ag">${m.first ? '<div class="who"><span class="av" style="background:#76b900;color:#0b1400">⚡</span>CUDA Agent</div>' : ''}${md(m.text)}</div>`;
      else if (m.kind === 'tool') {
        const st = m.status === 'run' ? `<span class="spin" style="transform:rotate(${(t * 400) % 360}deg)"></span>running`
          : m.status === 'ok' ? `<span class="ok">✓</span>${esc(m.meta || '')}` : `<span class="err">✗</span>${esc(m.meta || '')}`;
        let body = '';
        if (m.diff) body = `<div class="tool-o tool-d">${m.diff}</div>`;
        else if (m.out && m.out.length) body = `<div class="tool-o">${m.out.map(tmark).join('\n')}</div>`;
        mh += `<div class="tool"><div class="tool-h"><span class="tn">${m.tool}</span><span class="arg">${esc(m.arg)}</span><span class="stt">${st}</span></div>${body}</div>`;
      }
    }
    if (a.thinking) mh += `<div class="thinking"><span class="spin" style="transform:rotate(${(t * 400) % 360}deg)"></span><span class="shimmer" style="background-position:${(-t * 120) % 200}% 0">Thinking…</span></div>`;
    if (put('ag-msgs', mh)) { const el = $('ag-msgs'); el.scrollTop = el.scrollHeight; }
    const inputBlink = Math.floor(t * 1.8) % 2 === 0;
    put('ag-input', (a.input ? esc(a.input) + (a.focus && inputBlink ? '<span class="caret" style="height:15px;vertical-align:-2px"></span>' : '')
      : (a.focus && inputBlink ? '<span class="caret" style="height:15px;vertical-align:-2px"></span>' : '') + '<span class="ph">向 CUDA Agent 描述要实现的算子…（@ 引用文件）</span>') +
      `<div class="bar"><span>＠ model.py</span><span>＠ SKILL.md</span><span style="margin-left:6px">Agent 模式 ⌄</span><span class="send ${a.running ? 'stop' : ''}">${a.running ? '■' : '➤'}</span></div>`);
    $('ag-input').classList.toggle('focus', a.focus);

    // status bar
    const lgName = act ? ({ py: 'Python', cpp: act.endsWith('.cu') ? 'CUDA C++' : 'C++', md: 'Markdown', sh: 'Shell Script', txt: 'Plain Text' })[lang(act)] : '';
    put('vs-status', `<div class="remote">⇄ ${s.vs.connecting ? 'Opening Remote…' : 'SSH: ' + SRC.HOST}</div><span>⎇ main${Object.values(s.paths).some((v) => v === 'U' || v === 'M' || v === 'D') ? '*' : ''}</span><span>⊗ 0 ⚠ 0</span>
      <div class="r">${act ? `<span>Ln ${s.editor.caret ? s.editor.caret.line + 1 : 1}, Col ${s.editor.caret ? s.editor.caret.col + 1 : 1}</span><span>Spaces: 4</span><span>UTF-8</span><span>LF</span><span>${lgName}</span>` : ''}<span>Python 3.10.12</span><span>CUDA 12.4</span><span>🔔</span></div>`);
  }

  function drawOutro(d) {
    if (!d) return '';
    const max = Math.max(...d.bars.map((b) => b.v));
    return `<div class="card">
      <h2>${d.title}</h2><div class="lead">${d.lead}</div>
      <div class="kpis">${d.kpis.map((k) => `<div class="kpi"><div class="l">${k.l}</div><div class="v">${k.v}<small>${k.u || ''}</small></div></div>`).join('')}</div>
      <div class="bars">${d.bars.map((b) => `<div class="lab">${b.l}</div><div class="track"><div class="fill" style="width:${(b.v / max) * 100}%;background:${b.c}"></div></div><div class="val">${b.v.toFixed(1)} µs</div>`).join('')}</div>
      <div class="path">${d.path.map((p, i) => `${i ? '<span class="ar">→</span>' : ''}<span class="p ${p[1] || ''}">${p[0]}</span>`).join('')}</div>
      <div class="src">${d.src}</div></div>`;
  }

  function draw(s, t) {
    $('intro').style.opacity = s.intro; $('intro').style.display = s.intro > 0 ? 'flex' : 'none';
    $('wt').style.opacity = s.wt.vis; $('wt').style.visibility = s.wt.vis > 0 ? 'visible' : 'hidden';
    $('wt').style.transform = `scale(${0.97 + 0.03 * s.wt.vis})`;
    put('wt-title', esc(s.wt.title));
    drawTerminal('wt-body', s.wt, t, true);
    $('vs').style.opacity = s.vs.vis; $('vs').style.visibility = s.vs.vis > 0 ? 'visible' : 'hidden';
    $('vs').style.transform = `scale(${0.9 + 0.1 * ease(s.vs.vis)})`;
    if (s.vs.vis > 0) drawVS(s, t);
    put('tb-icons', TB.map(([k, ic]) => `<div class="tb-icon ${s.taskbarActive === k ? 'active' : (k === 'terminal' && s.wt.vis > 0) || (k === 'vscode' && s.vs.vis > 0) ? 'open' : ''}">${ic}</div>`).join(''));
    // overlays
    const cap = s.caption;
    put('caption', cap ? `<span class="stp">${esc(cap.step)}</span><span>${esc(cap.text)}</span>` : '');
    $('caption').style.display = cap ? 'flex' : 'none';
    $('caption').style.left = s.vs.vis > 0.5 ? '854px' : '960px';
    put('ff', s.ff ? esc(s.ff) : ''); $('ff').style.display = s.ff ? 'block' : 'none';
    $('cursor').style.transform = `translate(${s.cursor.x}px, ${s.cursor.y}px)`;
    $('cursor').style.opacity = s.cursor.vis;
    const r = s.cursor.ripple;
    $('ripple').style.display = r != null ? 'block' : 'none';
    if (r != null) { $('ripple').style.left = s.cursor.x + 'px'; $('ripple').style.top = s.cursor.y + 'px'; $('ripple').style.transform = `scale(${0.4 + r})`; $('ripple').style.opacity = 1 - r; }
    $('outro').style.display = s.outro > 0 ? 'flex' : 'none'; $('outro').style.opacity = s.outro;
    put('outro', drawOutro(s.outroData));
    $('watermark').style.display = s.intro < 1 && s.outro < 1 ? 'block' : 'none';
  }

  let TL = [];
  function render(t) { draw(compute(t), t); }

  function boot() {
    TL = B.tl.slice().sort((a, b) => a.t0 - b.t0 || a.i - b.i);
    const dur = B.now;
    window.__duration = dur;
    window.__render = render;
    const params = new URLSearchParams(location.search);
    const capture = params.has('capture');
    if (capture) document.body.classList.add('capture');
    const fit = () => {
      const k = capture ? 1 : Math.min(innerWidth / 1920, (innerHeight - 26) / 1080);
      $('stage').style.transform = `scale(${k})`;
      $('viewport').style.bottom = capture ? '0' : '26px';
    };
    fit(); addEventListener('resize', fit);
    let t = parseFloat(params.get('t') || '0');
    render(t);
    window.__ready = true;
    if (capture) return;
    let playing = !params.has('t'); let last = performance.now();
    const tick = (now) => {
      if (playing) { t += (now - last) / 1000; if (t > dur) t = dur, playing = false; render(t); }
      last = now;
      $('hud-t').textContent = `${t.toFixed(1)}s / ${dur.toFixed(1)}s`;
      $('hud-fill').style.width = (t / dur) * 100 + '%';
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    addEventListener('keydown', (e) => {
      if (e.code === 'Space') { playing = !playing; if (t >= dur) t = 0; e.preventDefault(); }
      if (e.code === 'ArrowRight') { t = Math.min(dur, t + 5); render(t); }
      if (e.code === 'ArrowLeft') { t = Math.max(0, t - 5); render(t); }
    });
    $('hud-bar').addEventListener('click', (e) => { const r = e.currentTarget.getBoundingClientRect(); t = ((e.clientX - r.left) / r.width) * dur; render(t); });
  }

  window.ENGINE = {
    B, seq, par, wait, set, move, click, caption, ff, clockTo, prompt, typeCmd, out, userMsg, say, thinking, toolStart, toolEnd,
    openFile, fileStatus, writeFile, setFile, settle, editFile, treeRows, C, esc, boot, ease, clamp,
  };
})();
