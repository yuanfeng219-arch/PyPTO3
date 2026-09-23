## [LRN-20260521-001] correction

**Logged**: 2026-05-21T11:58:00+08:00
**Priority**: medium
**Status**: pending
**Area**: frontend

### Summary
Design-system theme requests must be implemented in shared token/component sources, not only in the preview page.

### Details
The user asked to add a glass mode for the design system. I first added a `glass` mode directly to `design-system-preview.html`, which made the preview page look correct but did not make the component design system itself support `data-theme="glass"`.

### Suggested Action
When adding a theme, update `tokens/semantic.css`, `tokens/components.css`, token generation, and only then wire the preview page as a consumer.

### Metadata
- Source: user_feedback
- Related Files: design-system-preview.html, tokens/semantic.css, tokens/components.css
- Tags: design-system, theme, glass

---

## [LRN-20260521-002] correction

**Logged**: 2026-05-21T12:10:00+08:00
**Priority**: high
**Status**: pending
**Area**: frontend

### Summary
Liquid Glass themes should not turn large content surfaces into glass.

### Details
The user clarified that the PTO IDE Frame Pattern looked wrong because the glass skin was applied too broadly. Apple HIG treats Liquid Glass as a functional layer for controls and navigation, not as a content-layer material. PTO should reserve glass for tabs, buttons, selected states, floating controls, search fields, and small interactive chrome while keeping panes, code surfaces, canvases, and large content regions stable.

### Suggested Action
Keep `data-theme="glass"` token overrides conservative by default. Use corner-only ambient page backgrounds and apply glass variables only to functional controls or floating UI.

### Metadata
- Source: user_feedback
- Related Files: tokens/semantic.css, tokens/components.css, patterns/ide-frame/pattern.css
- Tags: design-system, liquid-glass, ide-frame

---
