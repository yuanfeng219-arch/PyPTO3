#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const tokenFiles = ['foundation.css', 'semantic.css', 'components.css'].map((file) => path.join(root, 'tokens', file));
const patternDir = path.join(root, 'patterns');
let failed = false;

const blockRe = /([^{}]+)\{([^{}]*)\}/gs;
const varRe = /--([\w-]+)\s*:\s*(.*?);/gs;
const colorRe = /#[0-9A-Fa-f]{3,8}|rgba?\([^)]*\)|linear-gradient\([^)]*\)|radial-gradient\([^)]*\)/g;

function groupForSelector(selector) {
  if (!selector.includes(':root')) return null;
  if (selector.includes("data-theme='light'") || selector.includes('data-theme="light"')) return 'light';
  if (selector.includes("data-theme='glass'") || selector.includes('data-theme="glass"')) return 'glass';
  if (selector.includes("data-theme='dark'") || selector.includes('data-theme="dark"')) return 'dark';
  return 'base';
}

function parseTokenGroups(file) {
  const css = fs.readFileSync(file, 'utf8');
  const groups = {};
  for (const block of css.matchAll(blockRe)) {
    const group = groupForSelector(block[1]);
    if (!group) continue;
    groups[group] ||= {};
    for (const variable of block[2].matchAll(varRe)) {
      groups[group][variable[1]] = variable[2].trim().replace(/\s+/g, ' ');
    }
  }
  return groups;
}

function listPatternFiles(extension) {
  return fs.readdirSync(patternDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(patternDir, entry.name, `pattern.${extension}`))
    .filter((file) => fs.existsSync(file));
}

function rgbFromHex(hex) {
  let value = hex.replace('#', '');
  if (value.length === 8) value = value.slice(0, 6);
  if (value.length === 3) value = value.split('').map((char) => char + char).join('');
  return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16));
}

function blend(foreground, alpha, background) {
  return foreground.map((channel, index) => Math.round(channel * alpha + background[index] * (1 - alpha)));
}

function luminance(rgb) {
  return rgb
    .map((channel) => {
      const value = channel / 255;
      return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    })
    .reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
}

function contrast(a, b) {
  const high = Math.max(luminance(a), luminance(b));
  const low = Math.min(luminance(a), luminance(b));
  return (high + 0.05) / (low + 0.05);
}

function formatRatio(value) {
  return `${value.toFixed(2)}:1`;
}

function printTokenCoverage() {
  console.log('Token coverage');
  for (const file of tokenFiles) {
    const groups = parseTokenGroups(file);
    const parts = Object.entries(groups).map(([group, values]) => `${group}:${Object.keys(values).length}`);
    console.log(`- tokens/${path.basename(file)} ${parts.join(' ')}`);
  }
}

function printCssColorCounts() {
  console.log('\nShared CSS hard-coded color scan');
  const files = [path.join(root, 'css', 'style.css'), ...listPatternFiles('css')];
  for (const file of files) {
    const css = fs.readFileSync(file, 'utf8');
    const hard = css.match(colorRe)?.length || 0;
    const vars = css.match(/var\(--/g)?.length || 0;
    const lightRules = css.match(/data-theme=['"]light['"]/g)?.length || 0;
    console.log(`- ${path.relative(root, file)} hard:${hard} varRefs:${vars} lightRules:${lightRules}`);
  }
}

function printPatternThemeBridge() {
  console.log('\nPattern preview theme bridge');
  for (const file of listPatternFiles('html')) {
    const html = fs.readFileSync(file, 'utf8');
    const status = html.includes('pto-preview-theme') ? 'ok' : 'missing';
    if (status === 'missing') failed = true;
    console.log(`- ${path.relative(root, file)} ${status}`);
  }
}

function printContrastChecks() {
  const backgroundLight = rgbFromHex('#F5F5F5');
  const backgroundDark = rgbFromHex('#101010');
  const pairs = [
    ['light foreground on background', blend([0, 0, 0], 0.9, backgroundLight), backgroundLight, 4.5],
    ['light secondary on background', blend([0, 0, 0], 0.55, backgroundLight), backgroundLight, 4.5],
    ['light muted on background', blend([0, 0, 0], 0.42, backgroundLight), backgroundLight, 3],
    ['dark foreground on background', blend([255, 255, 255], 0.9, backgroundDark), backgroundDark, 4.5],
    ['dark secondary on background', blend([255, 255, 255], 0.6, backgroundDark), backgroundDark, 4.5],
    ['primary foreground on primary', rgbFromHex('#FFFFFF'), rgbFromHex('#4369EF'), 4.5],
  ];

  console.log('\nBaseline contrast checks');
  for (const [label, foreground, background, minimum] of pairs) {
    const ratio = contrast(foreground, background);
    const status = ratio >= minimum ? 'ok' : 'low';
    if (status === 'low') failed = true;
    console.log(`- ${label} ${formatRatio(ratio)} ${status}`);
  }
}

printTokenCoverage();
printCssColorCounts();
printPatternThemeBridge();
printContrastChecks();

if (failed) {
  process.exitCode = 1;
}
