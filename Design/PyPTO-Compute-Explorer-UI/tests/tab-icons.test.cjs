const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');

const demo = path.join(__dirname, '..');
const styles = fs.readFileSync(path.join(demo, 'styles/visualization2.css'), 'utf8');

test('navigation icons use self-contained masks with the original SVG geometry', () => {
  for (const name of ['graph', 'assignment', 'pipeline']) {
    const rule = new RegExp(`\\.viz2-tab-icon--${name}\\s*\\{([^}]+)\\}`);
    const mask = styles.match(rule)?.[1].match(/url\("data:image\/svg\+xml,([^"]+)"\)/)?.[1];
    assert.ok(mask, `${name} must render from file:// without an external mask request`);

    const asset = fs.readFileSync(path.join(demo, 'assets/viz2-tabs', `${name}.svg`), 'utf8');
    const geometry = svg => svg.match(/(?:viewBox|<path d)="[^"]+"/g);
    assert.deepEqual(geometry(decodeURIComponent(mask)), geometry(asset), name);
  }
});
