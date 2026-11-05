/* Open path-interactions.html in a browser; no build or dependencies required. */
(function () {
  'use strict';
  const frame = document.getElementById('demo');
  const results = document.getElementById('results');
  const report = [];
  function check(condition, message) {
    if (!condition) throw new Error(message);
  }
  frame.addEventListener('load', async function () {
    const doc = frame.contentDocument, win = frame.contentWindow;
    if (!doc) {
      results.textContent = 'FAIL 请通过同源的本地 HTTP 服务打开本检查页。';
      results.className = 'fail';
      results.dataset.complete = 'true';
      return;
    }
    const settle = async () => {
      for (let i = 0; i < 4; i++) await new Promise(resolve => win.requestAnimationFrame(resolve));
    };
    const hover = element => element.dispatchEvent(new win.MouseEvent('mouseover', { bubbles: true }));
    const hot = selector => [...doc.querySelectorAll(selector + '.is-hot')];
    const tests = [
      ['40 行源码均映射到可见高亮，悬停不重建图元、不挤动硅片图', async () => {
        // Expected semantic mappings from hardware-native-ide/path.html.
        const groups = [
          [[728,729,730,732,733,734], ['f-spmd']], [[731], ['f-spmd']],
          [[741,744,745,746,747,748,749,751,752], ['f-stride']], [[764,765,766], ['f-pipe']],
          [[767,768], ['q']], [[769,796,797,798], ['out']], [[770], ['k','v']],
          [[753,781], ['valid']], [[758], ['tbl']], [[773], ['k']], [[774], ['qk']],
          [[775], ['scale']], [[782], ['fill']], [[783], ['max']], [[784], ['exp']],
          [[785], ['sum']], [[786], ['cast']], [[789], ['v']], [[790,794], ['sv']]
        ];
        const svg = doc.querySelector('#hpSvg > svg');
        const cube = doc.querySelector('[data-aic-node="cube:Cube"]');
        const height = doc.querySelector('#hpSvg').clientHeight;
        let count = 0;
        for (const [lines, nodes] of groups) for (const line of lines) {
          hover(doc.querySelector('.cl[data-line="' + line + '"]'));
          await settle();
          const actual = hot('.lg-hit').map(el => el.dataset.node).sort();
          check(JSON.stringify(actual) === JSON.stringify(nodes.slice().sort()), 'L' + line + ': ' + actual);
          for (const node of hot('.lg-hit')) {
            const scope = node.querySelector(':scope > .lg-frame');
            const tensor = node.querySelector('[data-tensor] > rect');
            const surface = scope || tensor || node.querySelector(':scope > .lg-node, [data-kind="op"] > rect');
            check(surface, '缺少可见图元 L' + line);
            const style = win.getComputedStyle(surface);
            if (scope || tensor) {
              check(style.stroke === 'rgb(232, 232, 232)' && parseFloat(style.strokeWidth) >= 1.5,
                'L' + line + ' 的 ' + node.dataset.node + ' 只有状态，没有可见高亮边框');
            } else {
              check(style.fill === 'rgb(230, 230, 230)', 'L' + line + ' 计算节点未显示高亮');
            }
          }
          check(hot('.cl').length === 1 && hot('.cl')[0].dataset.line === String(line), '源码焦点 L' + line);
          if (line !== 765 && line !== 766) check(hot('.sv-region').length || hot('.sv-buf').length, '硬件焦点 L' + line);
          check(doc.querySelector('#hpSvg > svg') === svg && cube.isConnected, '悬停重建了硬件图');
          check(doc.querySelector('#hpSvg').clientHeight === height, '说明挤动了画布 L' + line);
          count++;
        }
        check(count === 40, '未覆盖全部源码行');
      }],
      ['三个循环范围和三个计算阶段均可见高亮，含容量告警状态', async () => {
        for (const stage of [4, 2]) {
          doc.querySelector('.stage-ctl button[data-stage="' + stage + '"]').click();
          await settle();
          for (const id of ['f-spmd', 'f-stride', 'f-pipe', 'p-score', 'p-softmax', 'p-weight']) {
            const node = doc.querySelector('.lg-hit[data-node="' + id + '"]');
            hover(node); await settle();
            const shape = node.querySelector(':scope > rect');
            const style = win.getComputedStyle(shape);
            check(style.stroke === 'rgb(232, 232, 232)' && parseFloat(style.strokeWidth) >= 1.5,
              id + ' 在 stage=' + stage + ' 时不可见');
            if (id === 'f-pipe') {
              const bounds = shape.getBBox();
              doc.querySelectorAll('.lg-ghost').forEach(ghost => {
                const box = ghost.getBBox();
                check(box.y + box.height < bounds.y + bounds.height, '流水叠层遮挡了范围框底边');
              });
            }
            hover(doc.querySelector('.hp-top')); await settle();
            check(win.getComputedStyle(shape).stroke !== 'rgb(232, 232, 232)', id + ' 移开后未取消高亮');
          }
        }
        doc.querySelector('.stage-ctl button[data-stage="4"]').click(); await settle();
      }],
      ['逻辑、硬件、UB 缓冲可反向定位源码', async () => {
        const cases = [
          ['.lg-hit[data-node="qk"]', [774]],
          ['[data-aic-node="buffer:L0B"]', [770,773,789]],
          ['[data-aic-node="cube:Cube"]', [774,790,794]],
          ['.sv-buf[data-buf="exp_bf16"]', [786]],
          ['[data-aiv-node="buffer:UB"]', [764]]
        ];
        for (const [selector, lines] of cases) {
          hover(doc.querySelector(selector)); await settle();
          check(JSON.stringify(hot('.cl').map(el => Number(el.dataset.line)).sort()) === JSON.stringify(lines), selector);
        }
      }],
      ['新张量、存储与计算图元具有可见高亮', async () => {
        hover(doc.querySelector('.cl[data-line="768"]')); await settle();
        check(win.getComputedStyle(doc.querySelector('[data-tensor="q"] > rect')).stroke !== 'none', 'Q 未描边');
        check(win.getComputedStyle(doc.querySelector('[data-aic-node="buffer:L0A"] svg > rect')).stroke !== 'none', 'L0A 未描边');
        hover(doc.querySelector('.cl[data-line="774"]')); await settle();
        check(win.getComputedStyle(doc.querySelector('[data-aic-node="cube:Cube"]')).backgroundColor === 'rgb(232, 232, 232)', 'Cube 未高亮');
        check(doc.querySelector('[data-path-relations] [data-region="ch-c2v"]').classList.contains('is-hot'), 'C2V 未高亮');
      }],
      ['布局重算后连线保持当前焦点，装饰层不阻挡 UB 悬停', async () => {
        hover(doc.querySelector('.cl[data-line="786"]')); await settle();
        frame.style.height = '950px'; await settle();
        check(doc.querySelector('[data-path-relations] [data-region="ch-v2c"]').classList.contains('is-hot'), '重算后丢失 V2C 焦点');
        check(win.getComputedStyle(doc.querySelector('.ub-reserve')).pointerEvents === 'none', '保留区遮挡缓冲');
        frame.style.height = '1000px'; await settle();
      }],
      ['stage 参数切换与复用方案预览保持联动', async () => {
        const stage2 = [...doc.querySelectorAll('.stage-ctl button')].find(el => el.textContent === '2');
        check(stage2, '缺少 stage 控件'); stage2.click(); await settle();
        check(doc.querySelector('#hpDiagnostic').hidden, 'stage=2 仍显示容量错误');
        doc.querySelector('#hpStoryToggle').click();
        doc.querySelector('[data-act="toggle"][data-step="4"]').click();
        const option = doc.querySelector('.opt[data-opt="C"]');
        check(option, '缺少复用方案'); hover(option); await settle();
        check(!doc.querySelector('.sv-buf[data-buf="scores_valid"]'), '复用预览仍分配别名缓冲');
        hover(doc.querySelector('.cl[data-line="781"]')); await settle();
        check(hot('.sv-buf').some(el => el.dataset.buf === 'scores_valid'), '离开预览未恢复当前配置');
      }]
    ];
    try {
      await settle();
      for (const [name, run] of tests) {
        try { await run(); report.push('PASS ' + name); }
        catch (error) { report.push('FAIL ' + name + ': ' + error.message); }
        results.textContent = report.join('\n');
      }
    } finally {
      results.className = report.some(line => line.startsWith('FAIL')) ? 'fail' : 'pass';
      results.dataset.complete = 'true';
    }
  });
})();
