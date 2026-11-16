/* 开场页沿用 V4 的翻页样式，目标保持在 V3 演示中。 */
(() => {
  const pages = ['map.html', 'intro.html', 'agent.html'];
  const labels = ['计算领域体验设计创新', '项目背景', '进入算子开发'];
  const current = pages.indexOf(location.pathname.split('/').pop());
  if (current < 0) return;
  function go(delta) {
    const next = current + delta;
    if (next >= 0 && next < pages.length) location.href = pages[next];
  }
  document.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    if (event.target.closest('input,textarea,select,[contenteditable="true"],dialog,[role="dialog"]')) return;
    if (['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'PageDown', 'PageUp'].includes(event.key)) {
      event.preventDefault();
      go(['ArrowRight', 'ArrowDown', 'PageDown'].includes(event.key) ? 1 : -1);
    }
  }, true);
  const nav = document.createElement('nav');
  nav.className = 'tour-pages';
  nav.setAttribute('aria-label', '首页翻页');
  nav.innerHTML = pages.map((page, i) => `<a href="${page}" ${i === current ? 'aria-current="page"' : ''} aria-label="${labels[i]}">${i + 1}</a>`).join('');
  document.body.append(nav);
})();
