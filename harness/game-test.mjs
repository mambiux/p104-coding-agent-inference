// game-test.mjs <game.html> — load in headless Chromium, capture errors, exercise title → play → game over, screenshots.
import { chromium } from 'playwright';
import { pathToFileURL } from 'node:url';
import { resolve, dirname } from 'node:path';
const f = process.argv[2]; const out = dirname(resolve(f));
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 540, height: 960 }, deviceScaleFactor: 1 });
const page = await ctx.newPage(); const errors = [];
page.on('pageerror', e => errors.push('uncaught: ' + e.message.split('\n')[0]));
page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text().slice(0, 140)); });
const r = {};
await page.goto(pathToFileURL(resolve(f)).href, { waitUntil: 'load', timeout: 15000 }); await page.waitForTimeout(1200);
r.load_errors = errors.length;
const drawn = async () => page.evaluate(() => { const c = document.querySelector('canvas'); if (!c) return 'no canvas'; const x = c.getContext('2d'); const d = x.getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4 * 97) if (d[i]) n++; return n > 50 ? 'drawn' : 'blank'; });
r.canvas = await drawn(); r.title_text = (await page.evaluate(() => document.body.innerText)).replace(/\s+/g, ' ').slice(0, 120);
await page.screenshot({ path: `${out}/shot-title.png` });
// start + flap rhythm for ~6 s, screenshot mid-play, frame-rate probe
await page.keyboard.press('Space'); await page.waitForTimeout(300);
const fps = await page.evaluate(() => new Promise(res => { let n = 0; const t0 = performance.now(); const tick = () => { n++; if (performance.now() - t0 < 1000) requestAnimationFrame(tick); else res(n); }; requestAnimationFrame(tick); }));
r.fps = fps;
for (let i = 0; i < 14; i++) { await page.keyboard.press('Space'); await page.waitForTimeout(420); if (i === 5) await page.screenshot({ path: `${out}/shot-play.png` }); }
r.play_errors = errors.length - r.load_errors;
// stop flapping → fall → game over
await page.waitForTimeout(4000); await page.screenshot({ path: `${out}/shot-over.png` });
r.after_text = (await page.evaluate(() => document.body.innerText)).replace(/\s+/g, ' ').slice(0, 160);
r.best_saved = await page.evaluate(() => { try { return Object.keys(localStorage).join(',') || 'none'; } catch { return 'n/a'; } });
// restart with R, pause with P
await page.keyboard.press('r'); await page.waitForTimeout(400); await page.keyboard.press('Space'); await page.waitForTimeout(400); await page.keyboard.press('p'); await page.waitForTimeout(300);
r.total_errors = errors.length; r.errors = errors.slice(0, 5);
console.log(JSON.stringify(r, null, 1)); await browser.close();
