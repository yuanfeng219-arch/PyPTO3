// Patience diff over IR lines.
//
// Because the dumps put exactly one statement per line, a line diff *is* a
// statement diff. Patience diff is used rather than plain LCS because IR
// snapshots repeat many near-identical lines (`t__tile_3: ... = pl.tile.load(...)`),
// and LCS happily matches the wrong copies; anchoring on lines that are unique
// on both sides keeps hunks attributed to the statement that really changed.

/**
 * @returns {{tag:'=' | '-' | '+', a:number, b:number, line:string}[]}
 *          one entry per output row; `a`/`b` are 0-based source indices.
 */
export function diffLines(a, b) {
  const out = [];
  patience(a, b, 0, a.length, 0, b.length, out);
  return out;
}

function patience(a, b, a0, a1, b0, b1, out) {
  while (a0 < a1 && b0 < b1 && a[a0] === b[b0]) { out.push({ tag: '=', a: a0, b: b0, line: a[a0] }); a0++; b0++; }

  const tail = [];
  while (a1 > a0 && b1 > b0 && a[a1 - 1] === b[b1 - 1]) { a1--; b1--; tail.push({ tag: '=', a: a1, b: b1, line: a[a1] }); }

  if (a0 === a1 || b0 === b1) {
    for (let i = a0; i < a1; i++) out.push({ tag: '-', a: i, b: -1, line: a[i] });
    for (let j = b0; j < b1; j++) out.push({ tag: '+', a: -1, b: j, line: b[j] });
    for (let k = tail.length - 1; k >= 0; k--) out.push(tail[k]);
    return;
  }

  const anchors = uniqueAnchors(a, b, a0, a1, b0, b1);
  if (!anchors.length) {
    if ((a1 - a0) * (b1 - b0) <= 4_000_000) lcs(a, b, a0, a1, b0, b1, out);
    else {
      for (let i = a0; i < a1; i++) out.push({ tag: '-', a: i, b: -1, line: a[i] });
      for (let j = b0; j < b1; j++) out.push({ tag: '+', a: -1, b: j, line: b[j] });
    }
    for (let k = tail.length - 1; k >= 0; k--) out.push(tail[k]);
    return;
  }

  let pa = a0;
  let pb = b0;
  for (const [ai, bi] of anchors) {
    patience(a, b, pa, ai, pb, bi, out);
    out.push({ tag: '=', a: ai, b: bi, line: a[ai] });
    pa = ai + 1;
    pb = bi + 1;
  }
  patience(a, b, pa, a1, pb, b1, out);
  for (let k = tail.length - 1; k >= 0; k--) out.push(tail[k]);
}

/** Lines occurring exactly once on each side, kept in a longest increasing run. */
function uniqueAnchors(a, b, a0, a1, b0, b1) {
  const ca = new Map();
  for (let i = a0; i < a1; i++) {
    const e = ca.get(a[i]);
    if (e === undefined) ca.set(a[i], i);
    else ca.set(a[i], -1);
  }
  const pairs = [];
  const cb = new Map();
  for (let j = b0; j < b1; j++) {
    const e = cb.get(b[j]);
    if (e === undefined) cb.set(b[j], j);
    else cb.set(b[j], -1);
  }
  for (const [line, i] of ca) {
    if (i < 0) continue;
    const j = cb.get(line);
    if (j === undefined || j < 0) continue;
    pairs.push([i, j]);
  }
  pairs.sort((x, y) => x[0] - y[0]);

  // Longest increasing subsequence on the b index.
  const tails = [];
  const prev = new Array(pairs.length).fill(-1);
  const idx = [];
  for (let k = 0; k < pairs.length; k++) {
    const v = pairs[k][1];
    let lo = 0;
    let hi = tails.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (tails[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    tails[lo] = v;
    idx[lo] = k;
    prev[k] = lo > 0 ? idx[lo - 1] : -1;
  }
  const res = [];
  let k = tails.length ? idx[tails.length - 1] : -1;
  while (k >= 0) { res.push(pairs[k]); k = prev[k]; }
  res.reverse();
  return res;
}

function lcs(a, b, a0, a1, b0, b1, out) {
  const n = a1 - a0;
  const m = b1 - b0;
  const dp = new Uint32Array((n + 1) * (m + 1));
  const w = m + 1;
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i * w + j] = a[a0 + i] === b[b0 + j]
        ? dp[(i + 1) * w + j + 1] + 1
        : Math.max(dp[(i + 1) * w + j], dp[i * w + j + 1]);
    }
  }
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[a0 + i] === b[b0 + j]) { out.push({ tag: '=', a: a0 + i, b: b0 + j, line: a[a0 + i] }); i++; j++; }
    else if (dp[(i + 1) * w + j] >= dp[i * w + j + 1]) { out.push({ tag: '-', a: a0 + i, b: -1, line: a[a0 + i] }); i++; }
    else { out.push({ tag: '+', a: -1, b: b0 + j, line: b[b0 + j] }); j++; }
  }
  while (i < n) { out.push({ tag: '-', a: a0 + i, b: -1, line: a[a0 + i] }); i++; }
  while (j < m) { out.push({ tag: '+', a: -1, b: b0 + j, line: b[b0 + j] }); j++; }
}

/**
 * Group a row list into hunks with `context` unchanged lines around each change.
 * Rows carry absolute file line numbers via the `aBase` / `bBase` offsets.
 */
export function toHunks(rows, context = 3, aBase = 1, bBase = 1) {
  const changed = rows.map((r) => r.tag !== '=');
  const keep = new Array(rows.length).fill(false);
  for (let i = 0; i < rows.length; i++) {
    if (!changed[i]) continue;
    for (let k = Math.max(0, i - context); k <= Math.min(rows.length - 1, i + context); k++) keep[k] = true;
  }
  const hunks = [];
  let cur = null;
  for (let i = 0; i < rows.length; i++) {
    if (!keep[i]) { cur = null; continue; }
    const r = rows[i];
    if (!cur) {
      cur = { rows: [], aStart: r.a >= 0 ? r.a + aBase : null, bStart: r.b >= 0 ? r.b + bBase : null, add: 0, del: 0 };
      hunks.push(cur);
    }
    if (cur.aStart === null && r.a >= 0) cur.aStart = r.a + aBase;
    if (cur.bStart === null && r.b >= 0) cur.bStart = r.b + bBase;
    if (r.tag === '+') cur.add++;
    if (r.tag === '-') cur.del++;
    cur.rows.push([r.tag, r.a >= 0 ? r.a + aBase : 0, r.b >= 0 ? r.b + bBase : 0, r.line]);
  }
  return hunks;
}

export function countChanges(rows) {
  let add = 0;
  let del = 0;
  for (const r of rows) {
    if (r.tag === '+') add++;
    else if (r.tag === '-') del++;
  }
  return { add, del };
}

/**
 * Token-level diff of two single lines, for inline highlighting of the parts of
 * a statement a pass actually rewrote (`Vec` -> `Mat`, `FP32` -> `BF16`, ...).
 */
export function wordDiff(a, b) {
  const ta = splitTokens(a);
  const tb = splitTokens(b);
  const rows = [];
  patience(ta, tb, 0, ta.length, 0, tb.length, rows);
  const left = [];
  const right = [];
  for (const r of rows) {
    if (r.tag === '=') { left.push([0, r.line]); right.push([0, r.line]); }
    else if (r.tag === '-') left.push([1, r.line]);
    else right.push([1, r.line]);
  }
  return { left: mergeRuns(left), right: mergeRuns(right) };
}

function splitTokens(s) {
  return s.match(/[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|\s+|./g) || [];
}

/**
 * Aligned token-level edits between two lines: each entry is one contiguous
 * region that differs, as the old text and the new text. Used to aggregate the
 * substitutions a pass performs (`ND` -> `NZ`, `0` -> `4096`, `x` -> `x__ssa_v0`).
 */
export function tokenEdits(a, b) {
  const ta = splitTokens(a);
  const tb = splitTokens(b);
  const rows = [];
  patience(ta, tb, 0, ta.length, 0, tb.length, rows);

  const edits = [];
  let oldRun = '';
  let newRun = '';
  let ctx = '';
  const flush = () => {
    if (oldRun.trim() || newRun.trim()) edits.push({ old: oldRun.trim(), new: newRun.trim(), ctx: ctx.trim() });
    oldRun = '';
    newRun = '';
  };
  for (const r of rows) {
    if (r.tag === '=') {
      if (/\S/.test(r.line)) {
        flush();
        ctx = r.line;
      } else if (oldRun || newRun) {
        // whitespace inside a changed region stays part of it
        oldRun += r.line;
        newRun += r.line;
      }
      continue;
    }
    if (r.tag === '-') oldRun += r.line;
    else newRun += r.line;
  }
  flush();
  return edits;
}

function mergeRuns(parts) {
  const out = [];
  for (const [flag, text] of parts) {
    const last = out[out.length - 1];
    if (last && last[0] === flag) last[1] += text;
    else out.push([flag, text]);
  }
  return out;
}
