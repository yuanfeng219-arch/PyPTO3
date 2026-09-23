import fs from 'node:fs';

const root = '/Users/yin/PyPTO3-main/Insight';
const reportPath = `${root}/dpsk模型迁移-whitepaper/index.html`;
const sourcePath = `${root}/dpsk模型迁移.md`;

const escapeHTML = (value) => value
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;');

function inline(value) {
  let text = escapeHTML(value);
  text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2">$1 ↗</a>');
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  return text;
}

function renderMarkdown(markdown) {
  const lines = markdown.trim().split(/\r?\n/);
  let html = '';
  let paragraph = [];
  let listType = null;
  let code = false;
  let codeLines = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    html += `<p>${inline(paragraph.join(' '))}</p>`;
    paragraph = [];
  };
  const closeList = () => {
    if (!listType) return;
    html += `</${listType}>`;
    listType = null;
  };
  const flushAll = () => { flushParagraph(); closeList(); };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (/^```/.test(line)) {
      flushAll();
      if (!code) {
        code = true;
        codeLines = [];
      } else {
        html += `<div class="source-code"><div class="source-code-bar">SOURCE EXAMPLE</div><pre><code>${escapeHTML(codeLines.join('\n'))}</code></pre></div>`;
        code = false;
      }
      continue;
    }
    if (code) { codeLines.push(line); continue; }
    if (!line.trim()) { flushAll(); continue; }
    if (/^---+$/.test(line.trim())) { flushAll(); html += '<hr>'; continue; }

    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushAll();
      const level = Math.min(4, Math.max(3, heading[1].length + 1));
      html += `<h${level}>${inline(heading[2])}</h${level}>`;
      continue;
    }

    if (line.startsWith('>')) {
      flushAll();
      const quoteLines = [];
      let cursor = i;
      while (cursor < lines.length && lines[cursor].startsWith('>')) {
        quoteLines.push(lines[cursor].replace(/^>\s?/, ''));
        cursor += 1;
      }
      html += `<blockquote>${inline(quoteLines.filter(Boolean).join(' '))}</blockquote>`;
      i = cursor - 1;
      continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (bullet || ordered) {
      flushParagraph();
      const wanted = ordered ? 'ol' : 'ul';
      if (listType !== wanted) { closeList(); listType = wanted; html += `<${wanted}>`; }
      html += `<li>${inline((bullet || ordered)[1])}</li>`;
      continue;
    }

    if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1])) {
      flushAll();
      const rows = [line];
      let cursor = i + 2;
      while (cursor < lines.length && lines[cursor].includes('|') && lines[cursor].trim()) {
        rows.push(lines[cursor]);
        cursor += 1;
      }
      const cells = (row) => row.replace(/^\s*\||\|\s*$/g, '').split('|').map((cell) => cell.trim());
      html += '<div class="source-table-wrap"><table><thead><tr>';
      html += cells(rows[0]).map((cell) => `<th>${inline(cell)}</th>`).join('');
      html += '</tr></thead><tbody>';
      for (const row of rows.slice(1)) html += `<tr>${cells(row).map((cell) => `<td>${inline(cell)}</td>`).join('')}</tr>`;
      html += '</tbody></table></div>';
      i = cursor - 1;
      continue;
    }

    paragraph.push(line.trim());
  }
  flushAll();
  return `<article class="source-text" data-fidelity="verbatim"><div class="source-fidelity">SOURCE TEXT · 逐段保留</div>${html}</article>`;
}

function splitSource(markdown) {
  const buckets = { cover: [], '01-assets': [], '02-spine': [], '03-journey': [], '04-hotpaths': [], '05-ascend-stack': [], '06-observability': [], '07-acceptance': [], '08-synthesis': [] };
  let bucket = 'cover';
  for (const line of markdown.split(/\r?\n/)) {
    if (/^# 1\./.test(line)) bucket = '01-assets';
    else if (/^# 2\./.test(line)) bucket = '02-spine';
    else if (/^# 3\./.test(line)) bucket = '03-journey';
    else if (/^### 3\.4/.test(line)) bucket = '04-hotpaths';
    else if (/^# 4\./.test(line)) bucket = '05-ascend-stack';
    else if (/^# 5\./.test(line)) bucket = '06-observability';
    else if (/^# 6\./.test(line)) bucket = '07-acceptance';
    else if (/^# 结语/.test(line) || /^# 主要资料/.test(line)) bucket = '08-synthesis';
    buckets[bucket].push(line);
  }
  return buckets;
}

const captionNotes = {
  'fig-01': '读图：中心是保持不变的模型语义，外围是必须在 Ascend 上重新实现并验证的执行系统。边界：该图表达迁移对象关系，不代表工作量或性能占比。',
  'fig-02': '读图：从上到下观察优化知识由 Algorithm、Tile DSL 到 Backend 的落点。术语：Migration Seam 表示可复用知识与硬件相关实现之间的重新映射边界。',
  'fig-03': '读图：沿高亮 Request 自 Model 向下经过十个技术层；右侧标签表示该层可留下的验证证据。十层来自原文迁移对象表的结构化重绘。',
  'fig-04': '读图：横轴是迁移时间顺序，纵轴是问题所在技术层；同一阶段可同时影响多层。符号只表示主要工作、次要影响和证据输出，不表示定量权重。',
  'fig-05': '读图：从模型定义的 DSA/MLA 意图向右追踪 Indexer、Top-K KV、Cache 访问和 Kernel。GPU 与 Ascend 分支仅表示实现路径不同，图形不按计算量比例绘制。',
  'fig-06': '读图：16×16 网格对应 256 个 Routed Experts，高亮 8 格表示一个 Token 的 Top-8 选择；右侧 Rank 分布为定性示意。Shared Expert 独立显示，不能与 Routed Expert 数量混算。',
  'fig-07': '读图：逐列核对 Weight、Activation、Communication Tensor 与 KV Cache 的语义、dtype、scale、layout、kernel 和验证证据。“需验证”表示原文未提供可直接判定的等价关系。',
  'fig-08': '读图：上半部观察 Token 到 Expert/Rank 的空间分布，下半部沿相同 Rank 顺序观察 Dispatch、Grouped GEMM、Combine 与等待。所有条带长度均为示意，不是 profiling 实测。',
  'fig-09': '读图：Prefill 的“宽而短”和 Decode 的“窄而长”表示负载形状差异，再映射到独立资源池。533 tps 与 32 ms TPOT 仅属于图中注明的 A3、4-node、64K/3K 配置。',
  'fig-10': '读图：从左侧客户问题进入，依次定位技术层、Ascend 承接能力和可获得证据。箭头表示诊断路由，不表示这些产品之间存在固定调用链。',
  'fig-11': '读图：上方是一条连续客户问题链，下方是分散在不同组件中的数据孤岛；中间断点说明对象 ID、时间范围和证据关系在跨组件时丢失。该图是基于原文客户旅程的产品问题重绘。',
  'fig-12': '读图：四个视图共享同一 Request、Layer、Token、Expert、Rank 和证据 ID；切换视图不会改变被诊断对象。界面数据为产品概念与 MOCK SNAPSHOT，不代表真实测量结果。',
  'fig-13': '读图：五个工程维度围绕同一目标 workload 并行收集证据，外圈 Economics 是业务判断。环形布局表示可反复回访，不表示固定先后顺序或完成比例。',
  'fig-14': '读图：从中心 Workload 出发，依次回答位置、故障、变化、证据和完成度五个问题；每个方向都回指前文已经建立的图和证据边界，不新增事实。'
};

let html = fs.readFileSync(reportPath, 'utf8');
const markdown = fs.readFileSync(sourcePath, 'utf8');
const buckets = splitSource(markdown);

html = html.replace(/<article class="source-text" data-fidelity="verbatim">[\s\S]*?<\/article>/g, '');
html = html.replace(/<div class="figcap-note added-caption">[\s\S]*?<\/div>/g, '');

for (const [id, source] of Object.entries(buckets)) {
  const rendered = renderMarkdown(source.join('\n'));
  if (id === 'cover') {
    html = html.replace(/(<p class="lead">[\s\S]*?<\/p>)/, `$1${rendered}`);
    continue;
  }
  const sectionPattern = new RegExp(`(<section class="floor" id="${id}">[\\s\\S]*?<div class="section-head"><div>[\\s\\S]*?<\\/div><p>[\\s\\S]*?<\\/p><\\/div>)`);
  html = html.replace(sectionPattern, `$1${rendered}`);
}

for (const [id, note] of Object.entries(captionNotes)) {
  const figurePattern = new RegExp(`(<figure[^>]*id="${id}"[\\s\\S]*?<figcaption class="figcap">[\\s\\S]*?)(<\\/figcaption>)`);
  html = html.replace(figurePattern, `$1<div class="figcap-note added-caption"><b>图解说明：</b>${note}</div>$2`);
}

fs.writeFileSync(reportPath, html);
