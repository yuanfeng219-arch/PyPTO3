# Retrofit Container Audit

Use this audit during Workflow B before and after migrating an existing demo to PTO style.

The goal is not only to replace hard-coded values with tokens. The goal is to remove old visual language when PTO already has a different container model.

## What to inspect

Check every generic container:

- cards
- panels
- list rows
- inspector sections
- sidebars
- popups
- metric blocks
- empty states

Include styles from CSS, inline `style=""`, and JS-created DOM.

## Legacy decoration signals

Flag these as legacy decoration by default:

- full container borders on every card or row
- `border-left` or `border-inline-start`
- `box-shadow: inset ... 0 ...` used as a left rail
- `::before` or `::after` used as a vertical side bar
- `outline` used as a container border
- `linear-gradient(... to right ...)` or `linear-gradient(90deg, ...)` used as a side highlight
- colored accent strips that do not encode data
- stacked card borders inside an already framed pane

Do not token-swap these into `var(--border-*)` or `var(--primary)`. Delete or remap them to the PTO component listed below.

## PTO remap targets

Use these defaults:

| Legacy container | PTO target | Decoration rule |
|---|---|---|
| Page / workbench pane | `.workbench-pane` or `.panel-shell.panel-shell-quiet` | no full border, no left accent |
| Dense inspector rail | `.inspector-section` | section dividers only |
| Emphasized inspector block | `.inspector-soft-card` plus `.is-info`, `.is-warning`, `.is-danger`, or `.is-success` | tinted fill, no left rail |
| Compact repeated info card | `.card-demo` primitives or existing shared card class | shared border only if the class has one |
| Selected item | existing `.is-selected` state | selected fill or focus ring, not a permanent left rail |
| Real data encoding | data-viz exemption | document the encoded meaning |

## Anti-preservation rule

If the old UI has a card like this:

```css
.card {
  border: 1px solid #334155;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.08);
}

.card::before {
  width: 3px;
  background: #5b8cff;
}
```

Do not migrate it to:

```css
.card {
  border: 1px solid var(--border-default);
}

.card::before {
  background: var(--primary);
}
```

Instead, remove the old container frame and consume the closest PTO panel/card/inspector class. A leftover old rail is a failed migration unless it is a documented data-viz encoding.

## Required residue check

After edits, search the migrated module for:

```text
border-left
border-inline-start
box-shadow: inset
::before
::after
outline:
linear-gradient(90deg
linear-gradient(to right
```

For each match:

1. identify the selector and element it affects
2. mark it `removed`, `PTO-owned`, `data-viz-exempt`, or `needs-user-decision`
3. if it is `data-viz-exempt`, state the exact data meaning

Generic cards, panels, and inspector blocks should have zero unapproved `border-left`, pseudo-element rail, or inset-left-shadow residue.

## Final response requirement

Workflow B final responses must include:

- the migration table
- a `Container decoration residue` line
- any remaining selectors that need user decision
