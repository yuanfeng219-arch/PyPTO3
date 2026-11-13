/* 开场三页（map → intro → index）之间的翻页：← / → 或 PageUp / PageDown。
 * 上一页 / 下一页取自 <body data-prev data-next>；输入框里按键不翻页。 */
(function () {
  'use strict';
  var b = document.body, prev = b.dataset.prev, next = b.dataset.next;
  document.addEventListener('keydown', function (e) {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    var t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    if ((e.key === 'ArrowRight' || e.key === 'PageDown') && next) location.href = next;
    else if ((e.key === 'ArrowLeft' || e.key === 'PageUp') && prev) location.href = prev;
  });
})();
