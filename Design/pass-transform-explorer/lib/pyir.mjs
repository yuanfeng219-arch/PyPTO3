// PyPTO IR dump parser.
//
// The `passes_dump/*.py` files are a very regular subset of Python: one
// statement per line, indentation-structured, every value carrying an explicit
// type annotation. We parse them properly (tokenizer + recursive descent over
// expressions) instead of regexing, so every downstream analysis - op
// histograms, def-use graphs, memory tables, task DAGs - works on real
// structure rather than on text that happens to look right.

const KEYWORDS = new Set(['and', 'or', 'not', 'in', 'is', 'None', 'True', 'False', 'for', 'if', 'else', 'lambda']);

const PUNCT = [
  '**', '//', '==', '!=', '<=', '>=', '->', '<<', '>>',
  '(', ')', '[', ']', '{', '}', ',', ':', '.', '=', '+', '-', '*', '/', '%',
  '<', '>', '|', '&', '^', '~', '@',
];

const QUOTES = new Set(['"', "'"]);

export function tokenize(src) {
  const toks = [];
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') { i++; continue; }
    if (c === '#') break;
    if (QUOTES.has(c)) {
      const quote = c;
      let j = i + 1;
      let val = '';
      while (j < n) {
        if (src[j] === '\\') { val += src[j] + (src[j + 1] ?? ''); j += 2; continue; }
        if (src[j] === quote) break;
        val += src[j]; j++;
      }
      toks.push({ t: 'str', v: val });
      i = j + 1;
      continue;
    }
    if (/[0-9]/.test(c) || (c === '.' && /[0-9]/.test(src[i + 1] || ''))) {
      let j = i;
      while (j < n && /[0-9a-fA-FxXoObB_.]/.test(src[j])) j++;
      if (src[j] === 'e' || src[j] === 'E') {
        let k = j + 1;
        if (src[k] === '+' || src[k] === '-') k++;
        if (/[0-9]/.test(src[k] || '')) { j = k; while (j < n && /[0-9]/.test(src[j])) j++; }
      }
      toks.push({ t: 'num', v: src.slice(i, j) });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(c)) {
      let j = i;
      while (j < n && /[A-Za-z0-9_]/.test(src[j])) j++;
      const word = src.slice(i, j);
      toks.push({ t: KEYWORDS.has(word) ? 'kw' : 'name', v: word });
      i = j;
      continue;
    }
    let matched = null;
    for (const p of PUNCT) {
      if (src.startsWith(p, i)) { matched = p; break; }
    }
    if (matched) { toks.push({ t: 'op', v: matched }); i += matched.length; continue; }
    toks.push({ t: 'op', v: c });
    i++;
  }
  toks.push({ t: 'eof', v: '' });
  return toks;
}

class ExprParser {
  constructor(toks) { this.toks = toks; this.p = 0; }
  peek(k = 0) { return this.toks[this.p + k] ?? { t: 'eof', v: '' }; }
  next() { return this.toks[this.p++] ?? { t: 'eof', v: '' }; }
  at(t, v) { const k = this.peek(); return k.t === t && (v === undefined || k.v === v); }
  eat(t, v) { if (this.at(t, v)) return this.next(); return null; }
  expect(t, v) {
    const got = this.eat(t, v);
    if (!got) throw new Error('expected ' + (v ?? t) + ', got ' + this.peek().t + ':' + this.peek().v);
    return got;
  }

  parseExpr() { return this.parseTernary(); }

  parseTernary() {
    const body = this.parseOr();
    if (this.at('kw', 'if')) {
      this.next();
      const cond = this.parseOr();
      this.expect('kw', 'else');
      return { k: 'ifexp', cond, body, orelse: this.parseTernary() };
    }
    return body;
  }

  parseOr() {
    let l = this.parseAnd();
    while (this.at('kw', 'or')) { this.next(); l = { k: 'bool', op: 'or', l, r: this.parseAnd() }; }
    return l;
  }

  parseAnd() {
    let l = this.parseNot();
    while (this.at('kw', 'and')) { this.next(); l = { k: 'bool', op: 'and', l, r: this.parseNot() }; }
    return l;
  }

  parseNot() {
    if (this.at('kw', 'not')) { this.next(); return { k: 'unary', op: 'not', v: this.parseNot() }; }
    return this.parseCompare();
  }

  parseCompare() {
    let l = this.parseAdd();
    for (;;) {
      let op = null;
      for (const o of ['==', '!=', '<=', '>=', '<', '>']) if (this.at('op', o)) { op = o; break; }
      if (op) { this.next(); l = { k: 'cmp', op, l, r: this.parseAdd() }; continue; }
      if (this.at('kw', 'in')) { this.next(); l = { k: 'cmp', op: 'in', l, r: this.parseAdd() }; continue; }
      if (this.at('kw', 'is')) {
        this.next();
        const neg = this.eat('kw', 'not');
        l = { k: 'cmp', op: neg ? 'is not' : 'is', l, r: this.parseAdd() };
        continue;
      }
      return l;
    }
  }

  parseAdd() {
    let l = this.parseMul();
    for (;;) {
      let op = null;
      for (const o of ['+', '-', '|', '^', '&', '<<', '>>']) if (this.at('op', o)) { op = o; break; }
      if (!op) return l;
      this.next();
      l = { k: 'bin', op, l, r: this.parseMul() };
    }
  }

  parseMul() {
    let l = this.parseUnary();
    for (;;) {
      let op = null;
      for (const o of ['*', '//', '/', '%', '@']) if (this.at('op', o)) { op = o; break; }
      if (!op) return l;
      this.next();
      l = { k: 'bin', op, l, r: this.parseUnary() };
    }
  }

  parseUnary() {
    for (const o of ['-', '+', '~']) {
      if (this.at('op', o)) { this.next(); return { k: 'unary', op: o, v: this.parseUnary() }; }
    }
    return this.parsePower();
  }

  parsePower() {
    const base = this.parsePostfix();
    if (this.at('op', '**')) { this.next(); return { k: 'bin', op: '**', l: base, r: this.parseUnary() }; }
    return base;
  }

  parsePostfix() {
    let node = this.parseAtom();
    for (;;) {
      if (this.at('op', '.')) {
        this.next();
        node = { k: 'attr', on: node, name: this.next().v };
        continue;
      }
      if (this.at('op', '(')) {
        this.next();
        const { args, kwargs } = this.parseArgs(')');
        node = { k: 'call', fn: node, args, kwargs };
        continue;
      }
      if (this.at('op', '[')) {
        this.next();
        node = { k: 'sub', on: node, index: this.parseSeq(']') };
        continue;
      }
      return node;
    }
  }

  parseArgs(close) {
    const args = [];
    const kwargs = {};
    while (!this.at('op', close) && !this.at('eof')) {
      if (this.at('name') && this.peek(1).t === 'op' && this.peek(1).v === '=') {
        const key = this.next().v;
        this.next();
        kwargs[key] = this.parseExpr();
      } else if (this.at('op', '*') || this.at('op', '**')) {
        this.next();
        args.push(this.parseExpr());
      } else {
        args.push(this.parseExpr());
      }
      if (!this.eat('op', ',')) break;
    }
    this.expect('op', close);
    return { args, kwargs };
  }

  parseSeq(close) {
    const items = [];
    while (!this.at('op', close) && !this.at('eof')) {
      if (this.at('op', ':')) { this.next(); items.push({ k: 'slicecolon' }); continue; }
      items.push(this.parseExpr());
      if (!this.eat('op', ',')) break;
    }
    this.expect('op', close);
    return items;
  }

  parseAtom() {
    const tk = this.peek();
    if (tk.t === 'num') { this.next(); return { k: 'num', v: tk.v }; }
    if (tk.t === 'str') { this.next(); return { k: 'str', v: tk.v }; }
    if (tk.t === 'name') { this.next(); return { k: 'name', id: tk.v }; }
    if (tk.t === 'kw' && (tk.v === 'None' || tk.v === 'True' || tk.v === 'False')) {
      this.next();
      return { k: 'const', v: tk.v };
    }
    if (tk.t === 'op' && tk.v === '(') {
      this.next();
      const items = this.parseSeq(')');
      return items.length === 1 ? items[0] : { k: 'tuple', items };
    }
    if (tk.t === 'op' && tk.v === '[') {
      this.next();
      return { k: 'list', items: this.parseSeq(']') };
    }
    if (tk.t === 'op' && tk.v === '{') {
      this.next();
      const entries = [];
      while (!this.at('op', '}') && !this.at('eof')) {
        const key = this.parseExpr();
        this.expect('op', ':');
        entries.push([key, this.parseExpr()]);
        if (!this.eat('op', ',')) break;
      }
      this.expect('op', '}');
      return { k: 'dict', entries };
    }
    this.next();
    return { k: 'raw', v: tk.v };
  }
}

export function dotted(node) {
  if (!node) return '';
  if (node.k === 'name') return node.id;
  if (node.k === 'attr') {
    const base = dotted(node.on);
    return base ? base + '.' + node.name : node.name;
  }
  if (node.k === 'sub') return dotted(node.on);
  if (node.k === 'call') return dotted(node.fn);
  return '';
}

export function parseExpression(src) {
  return new ExprParser(tokenize(src)).parseExpr();
}

// ---------------------------------------------------------------------------
// Line / block structure
// ---------------------------------------------------------------------------

function indentOf(line) {
  let n = 0;
  while (n < line.length && line[n] === ' ') n++;
  return n;
}

/** Split on `sep` at bracket depth 0, respecting string literals. */
export function splitTop(src, sep) {
  const out = [];
  let depth = 0;
  let quote = null;
  let cur = '';
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (quote) {
      cur += c;
      if (c === '\\') { cur += src[i + 1] ?? ''; i++; continue; }
      if (c === quote) quote = null;
      continue;
    }
    if (QUOTES.has(c)) { quote = c; cur += c; continue; }
    if (c === '(' || c === '[' || c === '{') depth++;
    if (c === ')' || c === ']' || c === '}') depth--;
    if (depth === 0 && src.startsWith(sep, i)) {
      out.push(cur);
      cur = '';
      i += sep.length - 1;
      continue;
    }
    cur += c;
  }
  out.push(cur);
  return out;
}

/** Index of `needle` at bracket depth 0, or -1. */
export function findTop(src, needle, from = 0) {
  let depth = 0;
  let quote = null;
  for (let i = from; i < src.length; i++) {
    const c = src[i];
    if (quote) {
      if (c === '\\') { i++; continue; }
      if (c === quote) quote = null;
      continue;
    }
    if (QUOTES.has(c)) { quote = c; continue; }
    if (c === '(' || c === '[' || c === '{') { depth++; continue; }
    if (c === ')' || c === ']' || c === '}') { depth--; continue; }
    if (depth === 0 && src.startsWith(needle, i)) return i;
  }
  return -1;
}

let STAT_ERRORS = [];
export function parserErrors() { return STAT_ERRORS; }
export function resetParserErrors() { STAT_ERRORS = []; }

function safeExpr(src, where) {
  try {
    return parseExpression(src);
  } catch (err) {
    STAT_ERRORS.push({ where, src, msg: String((err && err.message) || err) });
    return { k: 'raw', v: src };
  }
}

/**
 * Parse a dump file into { name, imports, className, functions }.
 * Every statement keeps its 1-based source line and verbatim text so the UI can
 * always jump back to the exact line in the dump.
 */
export function parseDump(text, fileLabel = '') {
  const rawLines = text.split(/\r?\n/);
  const lines = rawLines.map((l, i) => ({ no: i + 1, text: l.replace(/\s+$/, ''), indent: indentOf(l) }));

  const program = {
    name: null,
    imports: [],
    className: null,
    functions: [],
    globals: [], // module-level `X = pl.dynamic("X")` shape symbols
    lineCount: rawLines.length,
  };
  const header = rawLines.find((l) => l.startsWith('# pypto.program:'));
  if (header) program.name = header.slice('# pypto.program:'.length).trim();

  let i = 0;
  const N = lines.length;

  while (i < N) {
    const l = lines[i];
    const t = l.text.trim();
    if (t === '' || t.startsWith('#')) { i++; continue; }
    if (t.startsWith('import ') || t.startsWith('from ')) { program.imports.push(t); i++; continue; }
    if (t.startsWith('class ')) {
      program.className = t.slice(6).split(/[(:]/)[0].trim();
      i++;
      continue;
    }
    if (t.startsWith('@pl.function')) {
      const deco = safeExpr(t.slice(1), fileLabel + ':' + l.no);
      let j = i + 1;
      while (j < N && !lines[j].text.trim().startsWith('def ')) j++;
      if (j >= N) { i++; continue; }
      const defLine = lines[j];
      const fn = parseDef(defLine, deco, fileLabel);
      const { body, next } = parseBlock(lines, j + 1, defLine.indent, fileLabel);
      fn.body = body;
      fn.line = defLine.no;
      fn.decoLine = l.no;
      fn.endLine = next < N ? lines[next].no - 1 : rawLines.length;
      program.functions.push(fn);
      i = next;
      continue;
    }
    if (l.indent === 0) {
      const eq = findTop(t, '=');
      if (eq > 0 && t[eq + 1] !== '=') {
        const rhs = t.slice(eq + 1).trim();
        program.globals.push({
          name: t.slice(0, eq).trim(),
          rhsSrc: rhs,
          expr: safeExpr(rhs, fileLabel + ':' + l.no),
          line: l.no,
          raw: l.text,
        });
      }
    }
    i++;
  }
  return program;
}

function matchParen(t, open) {
  let depth = 0;
  let quote = null;
  for (let k = open; k < t.length; k++) {
    const c = t[k];
    if (quote) { if (c === '\\') { k++; continue; } if (c === quote) quote = null; continue; }
    if (QUOTES.has(c)) { quote = c; continue; }
    if (c === '(' || c === '[' || c === '{') depth++;
    else if (c === ')' || c === ']' || c === '}') { depth--; if (depth === 0) return k; }
  }
  return -1;
}

function parseDef(defLine, deco, fileLabel) {
  const t = defLine.text.trim();
  const open = t.indexOf('(');
  const name = t.slice(4, open).trim();
  const close = matchParen(t, open);
  const paramSrc = t.slice(open + 1, close);
  const after = t.slice(close + 1);
  const arrow = after.indexOf('->');
  const retSrc = arrow >= 0 ? after.slice(arrow + 2).replace(/:\s*$/, '').trim() : '';

  const params = splitTop(paramSrc, ',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      const c = findTop(s, ':');
      if (c < 0) return { name: s, typeSrc: '', type: null };
      const typeSrc = s.slice(c + 1).trim();
      return { name: s.slice(0, c).trim(), typeSrc, type: safeExpr(typeSrc, fileLabel + ':' + defLine.no) };
    });

  const kw = deco && deco.k === 'call' ? deco.kwargs : {};
  return {
    name,
    params,
    retSrc,
    ret: retSrc ? safeExpr(retSrc, fileLabel + ':' + defLine.no) : null,
    deco: {
      type: kw.type ? dotted(kw.type) : null,
      level: kw.level ? dotted(kw.level) : null,
      role: kw.role ? dotted(kw.role) : null,
      attrs: kw.attrs ? literal(kw.attrs) : null,
    },
    defSrc: t,
    body: [],
  };
}

function parseBlock(lines, start, parentIndent, fileLabel) {
  const body = [];
  let i = start;
  const N = lines.length;
  let blockIndent = null;
  while (i < N) {
    const l = lines[i];
    if (l.text.trim() === '') { i++; continue; }
    if (l.indent <= parentIndent) break;
    if (blockIndent === null) blockIndent = l.indent;
    if (l.indent < blockIndent) break;
    if (l.text.trim().startsWith('#')) { i++; continue; }
    const { stmt, next } = parseStmt(lines, i, fileLabel);
    if (stmt) body.push(stmt);
    i = next;
  }
  return { body, next: i };
}

function parseStmt(lines, i, fileLabel) {
  const l = lines[i];
  const t = l.text.trim();
  const at = fileLabel + ':' + l.no;
  const base = { line: l.no, indent: l.indent, raw: l.text };

  if (t.startsWith('with ')) {
    const colon = findTop(t, ':');
    let head = t.slice(5, colon < 0 ? t.length : colon).trim();
    let asName = null;
    const asIdx = findTop(head, ' as ');
    if (asIdx >= 0) { asName = head.slice(asIdx + 4).trim(); head = head.slice(0, asIdx).trim(); }
    const ctx = safeExpr(head, at);
    const { body, next } = parseBlock(lines, i + 1, l.indent, fileLabel);
    return {
      stmt: { ...base, kind: 'with', ctxSrc: head, ctx, ctxName: dotted(ctx), as: asName, body },
      next,
    };
  }

  if (t.startsWith('for ')) {
    const colon = findTop(t, ':');
    const head = t.slice(4, colon < 0 ? t.length : colon).trim();
    const inIdx = findTop(head, ' in ');
    const targetSrc = head.slice(0, inIdx).trim();
    const iterSrc = head.slice(inIdx + 4).trim();
    const iter = safeExpr(iterSrc, at);
    const { body, next } = parseBlock(lines, i + 1, l.indent, fileLabel);
    const init = iter.k === 'call' && iter.kwargs.init_values ? iter.kwargs.init_values : null;
    return {
      stmt: {
        ...base,
        kind: 'for',
        targetSrc,
        targets: splitTop(targetSrc, ',').map((s) => s.trim()).filter(Boolean),
        iterSrc,
        iter,
        loopKind: dotted(iter).replace(/^pl\./, ''),
        tripCount: iter.k === 'call' && iter.args.length ? literal(iter.args[iter.args.length - 1]) : null,
        stage: iter.k === 'call' && iter.kwargs.stage ? literal(iter.kwargs.stage) : null,
        initValues: init ? (init.items || [init]).map((x) => dotted(x) || String(literal(x))) : null,
        body,
      },
      next,
    };
  }

  if (t.startsWith('if ')) {
    const colon = findTop(t, ':');
    const condSrc = t.slice(3, colon < 0 ? t.length : colon).trim();
    const { body, next } = parseBlock(lines, i + 1, l.indent, fileLabel);
    let orelse = [];
    let after = next;
    let k = next;
    while (k < lines.length && lines[k].text.trim() === '') k++;
    if (k < lines.length && lines[k].indent === l.indent && lines[k].text.trim().startsWith('else')) {
      const r = parseBlock(lines, k + 1, lines[k].indent, fileLabel);
      orelse = r.body;
      after = r.next;
    }
    return { stmt: { ...base, kind: 'if', condSrc, cond: safeExpr(condSrc, at), body, orelse }, next: after };
  }

  if (t === 'return' || t.startsWith('return ')) {
    const src = t.slice(6).trim();
    return {
      stmt: {
        ...base,
        kind: 'return',
        valueSrc: src,
        values: src ? splitTop(src, ',').map((s) => s.trim()).filter(Boolean) : [],
      },
      next: i + 1,
    };
  }

  const eq = findTop(t, '=');
  if (eq > 0 && t[eq + 1] !== '=' && !['!', '<', '>', '+', '-', '*', '/'].includes(t[eq - 1])) {
    const lhs = t.slice(0, eq).trim();
    const rhsSrc = t.slice(eq + 1).trim();
    const colon = findTop(lhs, ':');
    const targetSrc = colon >= 0 ? lhs.slice(0, colon).trim() : lhs;
    const typeSrc = colon >= 0 ? lhs.slice(colon + 1).trim() : '';
    const expr = safeExpr(rhsSrc, at);
    return {
      stmt: {
        ...base,
        kind: 'assign',
        targets: splitTop(targetSrc, ',').map((s) => s.trim()).filter(Boolean),
        typeSrc,
        type: typeSrc ? safeExpr(typeSrc, at) : null,
        rhsSrc,
        expr,
        op: expr.k === 'call' ? dotted(expr.fn) : null,
      },
      next: i + 1,
    };
  }

  const expr = safeExpr(t, at);
  return {
    stmt: { ...base, kind: 'expr', rhsSrc: t, expr, op: expr.k === 'call' ? dotted(expr.fn) : null },
    next: i + 1,
  };
}

export function literal(node) {
  if (!node) return null;
  switch (node.k) {
    case 'num': return Number(node.v.replace(/_/g, ''));
    case 'str': return node.v;
    case 'const': return node.v === 'None' ? null : node.v === 'True';
    case 'name': return node.id;
    case 'attr': return dotted(node);
    case 'list':
    case 'tuple': return node.items.map(literal);
    case 'dict': {
      const o = {};
      for (const [k, v] of node.entries) o[String(literal(k))] = literal(v);
      return o;
    }
    case 'unary': return node.op === '-' ? -literal(node.v) : literal(node.v);
    default: return null;
  }
}

/** Walk every expression node under `e`. */
export function walkExpr(e, fn) {
  if (!e || typeof e !== 'object') return;
  fn(e);
  switch (e.k) {
    case 'call':
      walkExpr(e.fn, fn);
      e.args.forEach((a) => walkExpr(a, fn));
      Object.values(e.kwargs).forEach((a) => walkExpr(a, fn));
      break;
    case 'attr': walkExpr(e.on, fn); break;
    case 'sub': walkExpr(e.on, fn); e.index.forEach((a) => walkExpr(a, fn)); break;
    case 'list':
    case 'tuple': e.items.forEach((a) => walkExpr(a, fn)); break;
    case 'dict': e.entries.forEach(([k, v]) => { walkExpr(k, fn); walkExpr(v, fn); }); break;
    case 'bin':
    case 'bool':
    case 'cmp': walkExpr(e.l, fn); walkExpr(e.r, fn); break;
    case 'unary': walkExpr(e.v, fn); break;
    case 'ifexp': walkExpr(e.cond, fn); walkExpr(e.body, fn); walkExpr(e.orelse, fn); break;
    default: break;
  }
}

/** Depth-first walk over statements, passing the chain of enclosing statements. */
export function walkStmts(body, fn, chain = []) {
  for (const s of body) {
    fn(s, chain);
    if (s.body) walkStmts(s.body, fn, [...chain, s]);
    if (s.orelse) walkStmts(s.orelse, fn, [...chain, s]);
  }
}

/** Names referenced (read) by an expression, excluding the callee paths. */
export function readsOf(expr) {
  const out = new Set();
  walkExpr(expr, (e) => {
    if (e.k === 'name') out.add(e.id);
  });
  // Callee dotted paths (pl.tile.load, self.foo) are not value reads.
  walkExpr(expr, (e) => {
    if (e.k === 'call') {
      const root = rootName(e.fn);
      if (root && (root === 'pl' || root === 'pld' || root === 'self')) out.delete(root);
    }
    if (e.k === 'attr') {
      const root = rootName(e);
      if (root === 'pl' || root === 'pld' || root === 'self') out.delete(root);
    }
  });
  out.delete('pl');
  out.delete('pld');
  out.delete('self');
  return out;
}

function rootName(node) {
  let cur = node;
  while (cur && (cur.k === 'attr' || cur.k === 'call' || cur.k === 'sub')) {
    cur = cur.on || cur.fn;
  }
  return cur && cur.k === 'name' ? cur.id : null;
}
