/* Python 高亮与 Compute Explorer 源码视图保持一致。 */
(function (global) {
'use strict';
  const PY_KEYWORDS = new Set([
    'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def',
    'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if',
    'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise',
    'return', 'try', 'while', 'with', 'yield', 'match', 'case'
  ]);

  // 内置函数 / 类型 / 异常。单独给一个颜色，和普通调用区分开。
  const PY_BUILTINS = new Set([
    'abs', 'aiter', 'all', 'anext', 'any', 'ascii', 'bin', 'bool', 'breakpoint',
    'bytearray', 'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
    'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter',
    'float', 'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help',
    'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
    'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object', 'oct', 'open',
    'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
    'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super', 'tuple',
    'type', 'vars', 'zip',
    'ArithmeticError', 'AssertionError', 'AttributeError', 'BaseException',
    'Exception', 'ImportError', 'IndexError', 'KeyError', 'KeyboardInterrupt',
    'MemoryError', 'NameError', 'NotImplementedError', 'OSError', 'OverflowError',
    'RuntimeError', 'StopIteration', 'TypeError', 'ValueError', 'ZeroDivisionError',
  ]);

  function highlightPythonLines(source) {
    const esc = (s) => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    const span = (cls, s) => `<span class="${cls}">${esc(s)}</span>`;
    const lines = source.split('\n');
    const out = [];
    let triple = null; // { quote, cls } — 跨行中的三引号字符串

    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];
      let html = '';
      let i = 0;

      // 同类相邻 token 合并成一个 span。decode_layer.py 有 1768 行，逐字符各出一个
      // span 会把 DOM 体积翻好几倍；plain 与 token 两个缓冲互斥，谁在用另一个必为空。
      let plain = '';
      let tokCls = null;
      let tokBuf = '';
      const flushPlain = () => { if (plain) { html += esc(plain); plain = ''; } };
      const flushTok = () => { if (tokBuf) { html += span(tokCls, tokBuf); tokCls = null; tokBuf = ''; } };
      const pushTok = (cls, s) => {
        if (tokCls !== cls) { flushPlain(); flushTok(); tokCls = cls; }
        tokBuf += s;
      };
      const pushPlain = (s) => { flushTok(); plain += s; };

      if (triple) {
        const close = line.indexOf(triple.quote);
        if (close === -1) { out.push(span(triple.cls, line)); continue; }
        html += span(triple.cls, line.slice(0, close + 3));
        i = close + 3;
        triple = null;
      }

      let expectName = null; // 'func' | 'class' — next identifier is a definition name
      let prev = '';         // 上一个有意义的字符，用于判断 x.y 是不是属性访问

      while (i < line.length) {
        const rest = line.slice(i);
        const ch = line[i];

        // comment
        if (ch === '#') { pushTok('tok-com', line.slice(i)); i = line.length; break; }

        // 字符串（可带 r/b/u/f 前缀）。三引号且顶在视觉行首 → 当 docstring 单独上色。
        const sm = /^([rRbBuUfF]{0,2})("""|'''|"|')/.exec(rest);
        if (sm) {
          const prefixLen = sm[1].length;
          const q = sm[2];
          const startQuote = i + prefixLen;
          const tokenStart = i;
          const cls = q.length === 3 && line.slice(0, tokenStart).trim() === '' ? 'tok-doc' : 'tok-str';
          let text;
          if (q.length === 3) {
            const after = line.indexOf(q, startQuote + 3);
            if (after === -1) {
              pushTok(cls, line.slice(tokenStart));
              triple = { quote: q, cls };
              i = line.length;
              break;
            }
            text = line.slice(tokenStart, after + 3);
            i = after + 3;
          } else {
            let j = startQuote + 1;
            while (j < line.length) {
              if (line[j] === '\\') { j += 2; continue; }
              if (line[j] === q) { j++; break; }
              j++;
            }
            text = line.slice(tokenStart, j);
            i = j;
          }
          // f-string：把 {...} 里的内容单独上色（只做一层，不解析格式串与嵌套花括号）
          const bodyFrom = prefixLen + q.length;
          if (/[fF]/.test(sm[1]) && text.length >= bodyFrom + q.length && text.endsWith(q)) {
            pushTok(cls, text.slice(0, bodyFrom));
            const body = text.slice(bodyFrom, text.length - q.length);
            const re = /\{[^{}]*\}/g;
            let at = 0;
            let brace;
            while ((brace = re.exec(body))) {
              if (brace.index > at) pushTok(cls, body.slice(at, brace.index));
              pushTok('tok-interp', brace[0]);
              at = brace.index + brace[0].length;
            }
            if (at < body.length) pushTok(cls, body.slice(at));
            pushTok(cls, text.slice(text.length - q.length));
          } else {
            pushTok(cls, text);
          }
          prev = text.slice(-1);
          continue;
        }

        // number
        const nm = /^(0[xXoObB][0-9a-fA-F_]+|(?:\d[\d_]*\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[jJ]?)/.exec(rest);
        if (nm && /\d/.test(nm[0])) { pushTok('tok-num', nm[0]); prev = nm[0].slice(-1); i += nm[0].length; continue; }

        // decorator (only at the visual start of a line)
        if (ch === '@') {
          const dm = /^@[A-Za-z_][\w.]*/.exec(rest);
          if (dm && line.slice(0, i).trim() === '') { pushTok('tok-dec', dm[0]); prev = dm[0].slice(-1); i += dm[0].length; continue; }
        }

        // identifier / keyword / 属性
        const im = /^[A-Za-z_]\w*/.exec(rest);
        if (im) {
          const word = im[0];
          const isCall = /^\s*\(/.test(line.slice(i + word.length));
          const isAttr = prev === '.';
          let cls = null;
          if (expectName) { cls = expectName === 'class' ? 'tok-cls' : 'tok-fn'; expectName = null; }
          else if (word === 'def') { cls = 'tok-kw'; expectName = 'func'; }
          else if (word === 'class') { cls = 'tok-kw'; expectName = 'class'; }
          else if (isAttr) cls = isCall ? 'tok-fn' : 'tok-prop';
          else if (PY_KEYWORDS.has(word)) cls = 'tok-kw';
          else if (word === 'True' || word === 'False' || word === 'None') cls = 'tok-const';
          else if (word === 'self' || word === 'cls') cls = 'tok-self';
          else if (PY_BUILTINS.has(word)) cls = 'tok-builtin';
          else if (isCall) cls = 'tok-fn';
          if (cls) pushTok(cls, word); else pushPlain(word);
          prev = word.slice(-1);
          i += word.length;
          continue;
        }

        // 运算符（相邻同类会合并成一个 span，所以不用穷举 **= / >>= 这类组合）
        if (/^(?::=|[-+*/%@&|^~<>=!])/.test(rest)) { pushTok('tok-op', ch); prev = ch; i++; continue; }
        // 括号与分隔符（. 不在此列，属性链保持干净）
        if (/[()[\]{},:;]/.test(ch)) { pushTok('tok-punc', ch); prev = ch; i++; continue; }

        pushPlain(ch);
        prev = ch;
        i++;
      }
      flushTok();
      flushPlain();
      out.push(html);
    }
    return out;
  }




  global.agentHighlightPythonLines = highlightPythonLines;
})(window);
