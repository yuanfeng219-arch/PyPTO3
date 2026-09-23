#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const foundationFile = path.join(root, 'tokens', 'foundation.css');
const componentsFile = path.join(root, 'tokens', 'components.css');
const sharedCssFiles = [
  path.join(root, 'css', 'style.css'),
  ...fs.readdirSync(path.join(root, 'patterns'), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(root, 'patterns', entry.name, 'pattern.css'))
    .filter((file) => fs.existsSync(file)),
];

const requiredMinimums = {
  'font-size-body-md': 14,
  'font-size-body-sm': 12,
  'font-size-label-sm': 12,
  'font-size-label-xs': 11,
};
const typeRoleSizes = {
  'type-body': 14,
  'type-body-sm': 12,
  'type-label': 12,
  'type-micro': 11,
  'type-mono': 12,
};
const bodySelectorRe = /description|(?:^|[-_])prose|(?:^|[-_])help|empty-(?:sub|hint)|inspector[^,{]*>\s*p/i;
const rootScaleSelectorRe = /(^|,\s*)(?:html|body|:root|#(?:app|root)|\.(?:app|page|root|pto-ide-frame__pane-body|pto-ide-frame__workarea))(?:\b|[.#:[\]])/i;

const foundationCss = fs.readFileSync(foundationFile, 'utf8');
const fontSizeTokens = Object.fromEntries(
  [...foundationCss.matchAll(/--(font-size-[\w-]+)\s*:\s*([\d.]+)px\s*;/g)]
    .map((match) => [match[1], Number(match[2])]),
);

let failed = false;
let approvedDataVizCount = 0;
let microLabelCount = 0;

function relative(file) {
  return path.relative(root, file) || '.';
}

function lineNumber(source, offset) {
  return source.slice(0, offset).split('\n').length;
}

function fail(message) {
  failed = true;
  console.error(`ERROR ${message}`);
}

function note(message) {
  console.log(`- ${message}`);
}

function collectCssFiles(target) {
  const resolved = path.resolve(target);
  if (!fs.existsSync(resolved)) {
    fail(`target does not exist: ${target}`);
    return [];
  }
  if (fs.statSync(resolved).isFile()) {
    return resolved.endsWith('.css') ? [resolved] : [];
  }
  const files = [];
  for (const entry of fs.readdirSync(resolved, { withFileTypes: true })) {
    const child = path.join(resolved, entry.name);
    if (entry.isDirectory()) files.push(...collectCssFiles(child));
    else if (entry.isFile() && entry.name.endsWith('.css')) files.push(child);
  }
  return files;
}

function inferPx(value) {
  const typeRole = value.match(/var\(--(type-[\w-]+)\)/);
  if (typeRole && typeRoleSizes[typeRole[1]] !== undefined) {
    return typeRoleSizes[typeRole[1]];
  }

  const tokenMath = value.match(/var\(--(font-size-[\w-]+)\)\s*([+-])\s*([\d.]+)px/);
  if (tokenMath && fontSizeTokens[tokenMath[1]] !== undefined) {
    const delta = Number(tokenMath[3]);
    return fontSizeTokens[tokenMath[1]] + (tokenMath[2] === '+' ? delta : -delta);
  }

  const tokenOnly = value.match(/var\(--(font-size-[\w-]+)\)/);
  if (tokenOnly && fontSizeTokens[tokenOnly[1]] !== undefined) {
    return fontSizeTokens[tokenOnly[1]];
  }

  const explicit = value.match(/([\d.]+)px/);
  return explicit ? Number(explicit[1]) : null;
}

function fontDeclarations(source) {
  const declarations = [];
  const blockRe = /([^{}]+)\{([^{}]*)\}/gs;
  for (const block of source.matchAll(blockRe)) {
    const selector = block[1].trim().replace(/\s+/g, ' ');
    const body = block[2];
    const bodyOffset = (block.index || 0) + block[0].indexOf(body);
    const declarationRe = /(?:^|;)\s*(font-size|font)\s*:\s*([^;]+)/g;
    for (const declaration of body.matchAll(declarationRe)) {
      const size = inferPx(declaration[2]);
      if (size === null) continue;
      declarations.push({
        selector,
        property: declaration[1],
        value: declaration[2].trim(),
        size,
        offset: bodyOffset + (declaration.index || 0),
      });
    }
  }
  return declarations;
}

function patternTypographyContract(file) {
  const patternDir = path.dirname(file);
  const contractFile = path.join(patternDir, 'pattern.json');
  if (!fs.existsSync(contractFile)) return null;
  const contract = JSON.parse(fs.readFileSync(contractFile, 'utf8'));
  return contract.typography || null;
}

function auditTokenMinimums() {
  console.log('Typography token minimums');
  for (const [token, minimum] of Object.entries(requiredMinimums)) {
    const actual = fontSizeTokens[token];
    if (actual === undefined) {
      fail(`tokens/foundation.css is missing --${token}`);
      continue;
    }
    if (actual < minimum) fail(`--${token} is ${actual}px; minimum is ${minimum}px`);
    else note(`--${token} ${actual}px ok`);
  }

  const componentsCss = fs.readFileSync(componentsFile, 'utf8');
  for (const token of ['type-body', 'type-body-sm', 'type-label', 'type-micro', 'type-mono']) {
    if (!componentsCss.includes(`--${token}:`)) fail(`tokens/components.css is missing --${token}`);
  }
}

function auditTokenIntegrity(files) {
  console.log('\nTypography token integrity');
  for (const file of files) {
    const source = fs.readFileSync(file, 'utf8');
    if (file !== componentsFile) {
      for (const match of source.matchAll(/--(text-label|text-mono)\s*:/g)) {
        fail(`${relative(file)}:${lineNumber(source, match.index || 0)} redefines deprecated --${match[1]}`);
      }
    }
    for (const match of source.matchAll(/font-family\s*:\s*var\(--type-|color\s*:\s*var\(--(?:type|font)-|font\s*:\s*var\(--font-/g)) {
      fail(`${relative(file)}:${lineNumber(source, match.index || 0)} mixes typography token categories`);
    }
    for (const match of source.matchAll(/font\s*:\s*var\(--text-/g)) {
      fail(`${relative(file)}:${lineNumber(source, match.index || 0)} uses deprecated --text-* font shorthand`);
    }
  }

  const styleFile = path.join(root, 'css', 'style.css');
  const styleCss = fs.readFileSync(styleFile, 'utf8');
  const bodyRule = styleCss.match(/(?:^|\n)body\s*\{([^}]*)\}/);
  if (!bodyRule || !/font\s*:\s*var\(--type-body\)/.test(bodyRule[1])) {
    fail('css/style.css body must use font: var(--type-body)');
  } else {
    note('css/style.css body uses --type-body');
  }
}

function auditFontFloors(files, strictProductTarget = false) {
  for (const file of files) {
    const source = fs.readFileSync(file, 'utf8');
    const contract = file.includes(`${path.sep}patterns${path.sep}`)
      ? patternTypographyContract(file)
      : null;

    for (const declaration of fontDeclarations(source)) {
      if (bodySelectorRe.test(declaration.selector) && declaration.size < 14) {
        const location = `${relative(file)}:${lineNumber(source, declaration.offset)}`;
        fail(`${location} ${declaration.selector} is prose at ${declaration.size}px; body minimum is 14px`);
        continue;
      }

      if (declaration.size >= 12) continue;
      const location = `${relative(file)}:${lineNumber(source, declaration.offset)}`;

      if (declaration.size >= 11) {
        microLabelCount += 1;
        continue;
      }

      if (!strictProductTarget && contract?.dataVizMinimumPx !== undefined) {
        const minimum = Number(contract.dataVizMinimumPx);
        if (
          declaration.size >= minimum
          && contract.dataVizExceptionReason
          && contract.dataVizFallback
        ) {
          approvedDataVizCount += 1;
          continue;
        }
      }

      fail(`${location} ${declaration.selector} resolves to ${declaration.size}px without an approved data-viz contract`);
    }

    if (strictProductTarget) {
      for (const block of source.matchAll(/([^{}]+)\{([^{}]*)\}/gs)) {
        const selector = block[1].trim().replace(/\s+/g, ' ');
        if (!rootScaleSelectorRe.test(selector)) continue;
        const shrink = block[2].match(/(?:^|;)\s*zoom\s*:\s*0?\.\d+|scale\(\s*0?\.\d+/);
        if (!shrink) continue;
        fail(`${relative(file)}:${lineNumber(source, block.index || 0)} ${selector} shrinks product UI with zoom/scale`);
      }
    }
  }
}

auditTokenMinimums();
auditTokenIntegrity([componentsFile, ...sharedCssFiles]);

console.log('\nShared typography floors');
auditFontFloors(sharedCssFiles);
note(`${microLabelCount} shared 11px micro-label declarations`);
note(`${approvedDataVizCount} approved sub-11px data-viz declarations`);

const targets = process.argv.slice(2);
if (targets.length) {
  const targetFiles = [...new Set(targets.flatMap(collectCssFiles))];
  console.log('\nProduct target typography floors');
  auditFontFloors(targetFiles, true);
  note(`${targetFiles.length} target CSS files checked`);
}

if (failed) process.exitCode = 1;
