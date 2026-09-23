/*
 * PTO Liquid Glass adapter.
 * CSS-first fallback aligned with nikdelvin/liquid-glass, MIT:
 * https://github.com/nikdelvin/liquid-glass
 */
(() => {
  const TARGET_SELECTOR = '[data-liquid-glass]';

  function supportsNativeBackdropBlur() {
    if (typeof CSS === 'undefined' || !CSS.supports) return false;
    return CSS.supports('backdrop-filter', 'blur(24px) saturate(180%)')
      || CSS.supports('-webkit-backdrop-filter', 'blur(24px) saturate(180%)');
  }

  function initLiquidGlass() {
    const root = document.documentElement;
    const surfaces = Array.from(document.querySelectorAll(TARGET_SELECTOR));
    const hasNativeBackdrop = supportsNativeBackdropBlur();
    root.classList.toggle('has-native-liquid-glass', hasNativeBackdrop);
    root.classList.toggle('no-native-liquid-glass', !hasNativeBackdrop);

    surfaces.forEach((element) => {
      element.classList.add('liquid-glass-material');
    });

    window.PtoLiquidGlass = {
      refresh() {},
      destroy() {
        surfaces.forEach((element) => {
          element.classList.remove('liquid-glass-material');
        });
      },
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initLiquidGlass, { once: true });
  } else {
    initLiquidGlass();
  }
})();
