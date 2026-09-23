// A deliberately small markdown renderer for the PyPTO pass docs.
//
// It handles exactly what those docs use - paragraphs, headings, rules, lists,
// tables, fenced code, inline code/bold/links - and nothing else. Every block
// rule must consume at least one line so the loop can never stall.

export function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function mdInline(s) {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (_, text, href) {
      var safe = /^(https?:|#|\.\.?\/)/.test(href) ? href : '#';
      return '<a href="' + escapeHtml(safe) + '" target="_blank" rel="noreferrer">' + text + '</a>';
    });
}

function mdTable(rows) {
  const cells = rows
    .map((r) => r.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim()))
    .filter((r) => !r.every((c) => /^:?-+:?$/.test(c) || c === ''));
  if (!cells.length) return '';
  const head = cells.shift();
  return '<table class="ptx-table"><thead><tr>'
    + head.map((c) => '<th>' + mdInline(c) + '</th>').join('')
    + '</tr></thead><tbody>'
    + cells.map((r) => '<tr>' + r.map((c) => '<td>' + mdInline(c) + '</td>').join('') + '</tr>').join('')
    + '</tbody></table>';
}

const BLOCK_START = /^\s*(?:[-*+]\s|\d+\.\s|\||```|#{1,6}\s)/;

export function md(src) {
  const out = [];
  const lines = String(src).split(/\r?\n/);
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim().startsWith('```')) {
      const lang = line.trim().slice(3);
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) { buf.push(lines[i]); i++; }
      i++;
      out.push('<pre class="ptx-doc__code" data-lang="' + escapeHtml(lang) + '"><code>'
        + escapeHtml(buf.join('\n')) + '</code></pre>');
      continue;
    }

    if (/^\s*\|/.test(line)) {
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) { rows.push(lines[i]); i++; }
      out.push(mdTable(rows));
      continue;
    }

    if (/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { out.push('<hr>'); i++; continue; }

    if (/^\s*(?:[-*+]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      while (i < lines.length && /^\s*(?:[-*+]|\d+\.)\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*(?:[-*+]|\d+\.)\s+/, ''));
        i++;
        // Fold continuation lines into the item they belong to.
        while (i < lines.length && lines[i].trim() !== '' && /^\s{2,}\S/.test(lines[i]) && !BLOCK_START.test(lines[i])) {
          items[items.length - 1] += ' ' + lines[i].trim();
          i++;
        }
      }
      const tag = ordered ? 'ol' : 'ul';
      out.push('<' + tag + '>' + items.map((t) => '<li>' + mdInline(t) + '</li>').join('') + '</' + tag + '>');
      continue;
    }

    if (/^#{1,6}\s/.test(line)) { out.push('<h4>' + mdInline(line.replace(/^#+\s/, '')) + '</h4>'); i++; continue; }

    if (line.trim() === '') { i++; continue; }

    if (/^\s*>/.test(line)) {
      const quote = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) { quote.push(lines[i].replace(/^\s*>\s?/, '')); i++; }
      out.push('<blockquote>' + mdInline(quote.join(' ')) + '</blockquote>');
      continue;
    }

    // Paragraph. The current line is always consumed, so `i` always advances.
    const para = [lines[i]];
    i++;
    while (i < lines.length && lines[i].trim() !== '' && !BLOCK_START.test(lines[i])
      && !/^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(lines[i]) && !/^\s*>/.test(lines[i])) {
      para.push(lines[i]);
      i++;
    }
    out.push('<p>' + mdInline(para.join(' ')) + '</p>');
  }

  return out.join('');
}
