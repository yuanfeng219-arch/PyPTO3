/* eslint-disable @typescript-eslint/no-var-requires */
const fs = require('fs')
const path = require('path')

const projectRoot = path.resolve(__dirname, '..')
const buildRoot = path.join(projectRoot, 'build')
const inputPath = path.join(buildRoot, 'index.html')
const canonicalOutputPath = path.resolve(projectRoot, '..', 'llvmcfg-standalone.html')
const generatedOutputPath = path.resolve(projectRoot, '..', 'llvmcfg-standalone.generated.html')
const forceCanonical = process.argv.includes('--force')
const outputPath = forceCanonical ? canonicalOutputPath : generatedOutputPath

const mimeTypes = {
  '.gif': 'image/gif',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webp': 'image/webp',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
}

function localAssetPath(assetUrl) {
  const cleanUrl = decodeURIComponent(assetUrl.split(/[?#]/, 1)[0])
  if (/^(?:[a-z]+:)?\/\//i.test(cleanUrl) || cleanUrl.startsWith('data:')) {
    return null
  }
  return path.join(buildRoot, cleanUrl.replace(/^\.\//, '').replace(/^\//, ''))
}

function inlineCssAsset(assetUrl, cssPath) {
  if (
    !assetUrl ||
    assetUrl.startsWith('data:') ||
    assetUrl.startsWith('#') ||
    /^(?:[a-z]+:)?\/\//i.test(assetUrl)
  ) {
    return `url(${assetUrl})`
  }

  const cleanUrl = assetUrl.replace(/^['"]|['"]$/g, '')
  const assetPath = path.resolve(path.dirname(cssPath), cleanUrl.split(/[?#]/, 1)[0])
  if (!fs.existsSync(assetPath)) return `url(${assetUrl})`

  const mimeType = mimeTypes[path.extname(assetPath).toLowerCase()]
  if (!mimeType) return `url(${assetUrl})`
  const payload = fs.readFileSync(assetPath).toString('base64')
  return `url("data:${mimeType};base64,${payload}")`
}

function readCss(cssPath, seen = new Set()) {
  const resolvedPath = path.resolve(cssPath)
  if (seen.has(resolvedPath)) return ''
  seen.add(resolvedPath)

  let css = fs.readFileSync(resolvedPath, 'utf8')
  css = css.replace(
    /@import\s+(?:url\()?['"]([^'"]+)['"]\)?\s*;/gi,
    (_match, importUrl) => {
      const importPath = path.resolve(path.dirname(resolvedPath), importUrl)
      return fs.existsSync(importPath) ? readCss(importPath, seen) : ''
    },
  )
  return css.replace(/url\(([^)]+)\)/gi, (_match, assetUrl) =>
    inlineCssAsset(assetUrl.trim(), resolvedPath),
  )
}

if (!fs.existsSync(inputPath)) {
  throw new Error(`Build output is missing: ${inputPath}`)
}

let html = fs.readFileSync(inputPath, 'utf8')
const seenCss = new Set()
const inlineScripts = []

html = html.replace(/<link\b[^>]*>/gi, (tag) => {
  const rel = tag.match(/\brel=["']([^"']+)["']/i)?.[1]
  const href = tag.match(/\bhref=["']([^"']+)["']/i)?.[1]

  if (rel === 'stylesheet' && href) {
    const cssPath = localAssetPath(href)
    if (!cssPath || !fs.existsSync(cssPath)) {
      throw new Error(`Cannot inline stylesheet: ${href}`)
    }
    return `<style data-source="${href}">\n${readCss(cssPath, seenCss)}\n</style>`
  }

  if (['icon', 'apple-touch-icon', 'manifest'].includes(rel || '')) return ''
  return tag
})

html = html.replace(
  /<script\b([^>]*)\bsrc=["']([^"']+)["']([^>]*)><\/script>/gi,
  (_tag, beforeSrc, src, afterSrc) => {
    const scriptPath = localAssetPath(src)
    if (!scriptPath || !fs.existsSync(scriptPath)) {
      throw new Error(`Cannot inline script: ${src}`)
    }
    const script = fs
      .readFileSync(scriptPath, 'utf8')
      .replace(/^\/\/# sourceMappingURL=.*$/gm, '')
      .replace(/<\/script/gi, '<\\/script')
    const attributes = `${beforeSrc} ${afterSrc}`
      .replace(/\s*defer(?:=["'][^"']*["'])?/gi, '')
      .replace(/\s+/g, ' ')
      .trim()
    inlineScripts.push({
      priority: src.includes('/vendor/') ? 0 : 1,
      source: src,
      tag: `<script${attributes ? ` ${attributes}` : ''} data-source="${src}">\n${script}\n</script>`,
    })
    return ''
  },
)

inlineScripts.sort((left, right) => left.priority - right.priority)
html = html.replace(
  '</body>',
  () => `${inlineScripts.map(({ tag }) => tag).join('\n')}\n</body>`,
)

html = html
  .replace('<html lang="en"', '<html lang="zh-CN"')
  .replace('<title>LLVM FLOW</title>', '<title>PyPTO Control Flow Explorer</title>')
  .replace(
    '<head>',
    '<head>\n    <!-- Standalone export: all runtime assets are embedded in this file. -->',
  )

const unresolvedAssets = [
  ...html.matchAll(/<(?:script|link)\b[^>]*(?:src|href)=["'](?!data:)([^"']+)["']/gi),
]
if (unresolvedAssets.length > 0) {
  throw new Error(
    `Standalone HTML still contains external assets: ${unresolvedAssets
      .map((match) => match[1])
      .join(', ')}`,
  )
}
if (/@import\s+/i.test(html)) {
  throw new Error('Standalone HTML still contains CSS @import rules')
}

fs.writeFileSync(outputPath, html)
const sizeMiB = (fs.statSync(outputPath).size / 1024 / 1024).toFixed(2)
console.log(`Created ${outputPath} (${sizeMiB} MiB)`)
if (!forceCanonical) {
  console.log('Canonical llvmcfg-standalone.html was preserved; pass --force to replace it explicitly.')
}
