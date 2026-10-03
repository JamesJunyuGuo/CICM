// Render an HTML figure with Chromium, preserving vector text in the PDF.
// Requires Node >=18 and Playwright; it does not install either dependency.
// Usage: node render_html.cjs figure.html output_basename
const { chromium } = require('playwright');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

async function main() {
  const [html, output] = process.argv.slice(2);
  if (!html || !output) throw new Error('Expected HTML path and output basename');
  const browser = await chromium.launch({
    channel: 'chromium',
    headless: true,
    // Chromium 134's Fontations backend can emit Type 3 PDF fonts even for
    // static TrueType input. Use FreeType for this local figure export.
    args: ['--no-sandbox', '--disable-dev-shm-usage',
      '--disable-features=FontationsFontBackend,FontationsForSelectedFormats'],
  });
  try {
    const page = await browser.newPage({ deviceScaleFactor: 2 });
    await page.goto(pathToFileURL(path.resolve(html)).href, { waitUntil: 'networkidle' });
    await page.evaluate(() => document.fonts.ready);
    const fonts = await page.evaluate(() => [...document.fonts].map(font => ({
      family: font.family, weight: font.weight, status: font.status,
    })));
    if (fonts.some(font => font.status !== 'loaded')) {
      throw new Error('Figure fonts did not load: ' + JSON.stringify(fonts));
    }
    const size = await page.evaluate(() => ({
      width: Math.ceil(parseFloat(getComputedStyle(document.body).width)),
      height: Math.ceil(parseFloat(getComputedStyle(document.body).height)),
    }));
    await page.setViewportSize(size);
    const overflow = await page.evaluate(() =>
      [...document.querySelectorAll('.panel')].flatMap(panel => {
        const p = panel.getBoundingClientRect();
        return [...panel.querySelectorAll('*')].filter(el => {
          const b = el.getBoundingClientRect();
          return b.left < p.left - 1 || b.right > p.right + 1 ||
            b.top < p.top - 1 || b.bottom > p.bottom + 1;
        }).map(el => el.textContent.trim());
      }));
    if (overflow.length) throw new Error('Panel overflow: ' + overflow.join(' | '));
    await page.screenshot({ path: output + '.png', fullPage: true });
    await page.pdf({
      path: output + '.pdf', width: size.width + 'px', height: size.height + 'px',
      printBackground: true, margin: { top: 0, bottom: 0, left: 0, right: 0 },
    });
    console.log(JSON.stringify({ html, output, ...size, fonts, panelOverflow: overflow }));
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
