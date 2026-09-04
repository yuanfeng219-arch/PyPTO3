(() => {
  const MIN_FONT_PX = 12;
  const hasReadableContent = (element) => {
    if (element.matches('input, textarea, select, option, text, tspan')) return true;
    return Array.from(element.childNodes).some((node) =>
      node.nodeType === Node.TEXT_NODE && node.textContent.trim()
    );
  };

  const clampElement = (element) => {
    if (!(element instanceof Element) || element.matches('script, style, template')) return;
    if (!hasReadableContent(element)) return;
    const size = Number.parseFloat(getComputedStyle(element).fontSize);
    if (Number.isFinite(size) && size < MIN_FONT_PX) {
      element.style.setProperty('font-size', `${MIN_FONT_PX}px`, 'important');
      element.dataset.minReadableType = 'true';
    }
  };

  const clampTree = (root) => {
    if (root instanceof Element) clampElement(root);
    root.querySelectorAll?.('*').forEach(clampElement);
  };

  let queued = false;
  const clampDocument = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      clampTree(document);
    });
  };

  const narrowLayout = window.matchMedia('(max-width: 760px)');
  const explorerToggle = document.querySelector('[data-ide-toggle="explorer"]');
  let autoCollapsedExplorer = false;

  const syncNarrowExplorer = () => {
    if (!explorerToggle) return;
    const expanded = explorerToggle.getAttribute('aria-expanded') === 'true';
    if (narrowLayout.matches && expanded && !autoCollapsedExplorer) {
      explorerToggle.click();
      autoCollapsedExplorer = true;
    } else if (!narrowLayout.matches && autoCollapsedExplorer) {
      if (!expanded) explorerToggle.click();
      autoCollapsedExplorer = false;
    }
  };

  clampDocument();
  syncNarrowExplorer();

  new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      mutation.addedNodes.forEach((node) => {
        if (node instanceof Element) clampTree(node);
      });
    });
  }).observe(document.body, { childList: true, subtree: true });

  window.addEventListener('resize', () => {
    clampDocument();
    syncNarrowExplorer();
  }, { passive: true });
})();
