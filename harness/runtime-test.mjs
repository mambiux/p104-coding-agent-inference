// runtime-test.mjs <page.html>... — load each page in headless Chromium, capture console errors/uncaught exceptions, exercise the required features.
import { chromium } from 'playwright';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
const files = process.argv.slice(2);
const browser = await chromium.launch();
for (const f of files) {
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 900 } });
  const page = await ctx.newPage();
  const errors = []; page.on('pageerror', e => errors.push('uncaught: ' + e.message.split('\n')[0]));
  page.on('console', m => { if (m.type() === 'error') errors.push('console: ' + m.text().slice(0, 120)); });
  const r = { file: f.split('/').pop() };
  try {
    await page.goto(pathToFileURL(resolve(f)).href, { waitUntil: 'load', timeout: 15000 }); await page.waitForTimeout(1500);
    r.load_errors = errors.length;
    // clock: find text matching hh:mm that changes after 1.2 s
    const clockRe = /\d{1,2}:\d{2}/; const t1 = await page.evaluate(() => document.body.innerText); await page.waitForTimeout(1200); const t2 = await page.evaluate(() => document.body.innerText);
    r.clock = clockRe.test(t1) && t1.match(/\d{1,2}:\d{2}(:\d{2})?/g)?.join() !== t2.match(/\d{1,2}:\d{2}(:\d{2})?/g)?.join() ? 'ticks' : (clockRe.test(t1) ? 'static' : 'none');
    // canvas drawn? (any non-transparent pixel)
    r.canvas_drawn = await page.evaluate(() => { const c = document.querySelector('canvas'); if (!c) return 'no canvas'; try { const x = c.getContext('2d'); const d = x.getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) n++; return n > 50 ? 'drawn' : 'blank'; } catch (e) { return 'err'; } });
    // chart hover
    const cv = page.locator('canvas').first(); if (await cv.count()) { const b = await cv.boundingBox(); if (b) { const e0 = errors.length; await page.mouse.move(b.x + b.width * 0.3, b.y + b.height * 0.6); await page.mouse.move(b.x + b.width * 0.5, b.y + b.height * 0.5); await page.waitForTimeout(300); r.hover_errors = errors.length - e0; } }
    // task manager: type in the first text input near 'task', submit, see text appear
    const ti = page.locator('input[type="text"], input:not([type])').filter({ has: page.locator(':scope') }).first();
    if (await ti.count()) { const e0 = errors.length; await ti.fill('Buy GPU risers'); await ti.press('Enter'); await page.waitForTimeout(400);
      let shown = (await page.evaluate(() => document.body.innerText)).includes('Buy GPU risers');
      if (!shown) { const btn = page.locator('button', { hasText: /^add$/i }).first(); if (await btn.count()) { await ti.fill('Buy GPU risers'); await btn.click(); await page.waitForTimeout(400); shown = (await page.evaluate(() => document.body.innerText)).includes('Buy GPU risers'); } }
      r.task_add = shown ? 'shows' : 'NOT shown'; r.task_errors = errors.length - e0;
      // persistence
      await page.reload({ waitUntil: 'load' }); await page.waitForTimeout(800); r.task_persist = (await page.evaluate(() => document.body.innerText)).includes('Buy GPU risers') ? 'persists' : 'lost';
      const cb = page.locator('button', { hasText: /complete|done|toggle|✓|✔/i }).first(); if (await cb.count()) { const e1 = errors.length; await cb.click(); await page.waitForTimeout(200); r.task_complete_errors = errors.length - e1; }
      const fb = page.locator('button', { hasText: /^(active|completed|all)$/i }).first(); r.task_filter = (await fb.count()) ? 'present' : 'none';
    } else r.task_add = 'no text input';
    // algorithm demo: number input
    const ni = page.locator('input[type="number"], textarea').first();
    if (await ni.count()) { const e0 = errors.length; const norm = t => t.replace(/\d{1,2}:\d{2}(:\d{2})?( ?[AP]M)?/g,'').replace(/\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[^\n]{0,40}\d{4}/g,''); const before = norm(await page.evaluate(() => document.body.innerText)); await ni.fill('10'); await ni.press('Enter'); await page.waitForTimeout(300);
      let after = norm(await page.evaluate(() => document.body.innerText));
      if (after === before) { const b2 = page.locator('button', { hasText: /compute|find|run|calc|go/i }).first(); if (await b2.count()) { await b2.click({ force: true }).catch(() => {}); await page.waitForTimeout(300); after = norm(await page.evaluate(() => document.body.innerText)); } }
      const body = await page.evaluate(() => document.body.innerText);
      const pm = body.match(/prime[^\d\n]{0,60}?(\d+)/i);
      r.algo = /\b29\b/.test(body) ? 'correct (29 = 10th prime)' : (pm ? `answers ${pm[1]} (expected 29)` : (after !== before ? 'responds (no 29)' : 'no response')); r.algo_errors = errors.length - e0;
    } else r.algo = 'no number input';
    // theme toggle
    const tb = page.locator('button', { hasText: /theme|dark|light|🌙|☀/i }).first();
    if (await tb.count()) { const bg0 = await page.evaluate(() => getComputedStyle(document.body).backgroundColor); await tb.click(); await page.waitForTimeout(400); const bg1 = await page.evaluate(() => getComputedStyle(document.body).backgroundColor); r.theme_toggle = bg0 !== bg1 ? 'switches' : 'NO effect'; } else r.theme_toggle = 'no button';
    // mobile overflow
    await page.setViewportSize({ width: 360, height: 740 }); await page.waitForTimeout(300);
    r.mobile_overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1 ? 'horizontal scroll' : 'ok');
    r.rtl = await page.evaluate(() => getComputedStyle(document.documentElement).direction);
    r.total_errors = errors.length; r.error_samples = [...new Set(errors)].slice(0, 8);
  } catch (e) { r.harness_error = e.message.split('\n')[0]; r.total_errors = errors.length; r.error_samples = [...new Set(errors)].slice(0, 8); }
  console.log(JSON.stringify(r)); await ctx.close();
}
await browser.close();
