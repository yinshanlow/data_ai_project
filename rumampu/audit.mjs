#!/usr/bin/env node
/**
 * RuMampu self-audit.
 *   node audit.mjs [path-to-rumampu.html]
 * Static checks need only Node. DOM checks need the `playwright` package
 * (set NODE_PATH to a global install if required) and a Chromium binary.
 *
 * Checks: banned phrases per language, --short fills per screen, body word
 * count per screen, RM figures missing a provenance label, untranslated
 * strings, tag balance, console errors, tap-target floor.
 */
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';

const FILE = process.argv[2] || new URL('./rumampu.html', import.meta.url).pathname;
const src = readFileSync(FILE, 'utf8');
let failures = 0;
const fail = (msg) => { failures++; console.log('  FAIL  ' + msg); };
const ok = (msg) => console.log('  ok    ' + msg);

/* ---------- 1. banned phrases, per language, whole file (copy AND code) ---------- */
console.log('\n[1] banned phrases');
// Occurrences of the product name and CSS technicalities are not the banned senses.
const scrubbed = src.replace(/RuMampu/g, '').replace(/safe-area-inset/g, '');
const BANNED = {
  en: [/afford/i, /\bsafe\b/i, /\bunsafe\b/i, /\bapproved\b/i, /pre-approved/i, /\beligib\w*/i,
       /predict\w*/i, /forecast/i, /\bwill be\b/i, /recommend\w*/i, /you should/i, /\bconsider\b/i,
       /\b(low|moderate|high) risk\b/i, /stable income/i, /coefficient of variation/i, /standard deviation/i],
  ms: [/\bmampu\b/i, /\bselamat\b/i, /\bsyor\w*/i, /\bdisyor\w*/i, /\bcadang\w*/i, /\bramal\w*/i,
       /\bstabil\b/i, /\blayak\b/i, /\blulus\w*/i, /\bdijangka\w*/i],
  zh: [/负担/, /安全/, /建议/, /推荐/, /预测/, /预报/, /稳定/, /合格/, /批准/, /风险/, /应该/]
};
for (const [lang, pats] of Object.entries(BANNED)) {
  let count = 0;
  for (const re of pats) {
    const m = scrubbed.match(new RegExp(re.source, re.flags + 'g'));
    if (m) { count += m.length; fail(`${lang}: "${re.source}" x${m.length}`); }
  }
  if (!count) ok(`${lang}: 0 banned phrases`);
}

/* ---------- 2. tag balance (all markup appears literally in source) ---------- */
console.log('\n[2] tag balance');
for (const tag of ['div', 'span', 'p', 'button', 'label', 'main', 'nav', 'i', 'b', 'h3', 'select']) {
  const open = (src.match(new RegExp('<' + tag + '(?=[\\s>])', 'g')) || []).length;
  const close = (src.match(new RegExp('</' + tag + '>', 'g')) || []).length;
  if (open !== close) fail(`<${tag}> ${open} opened vs ${close} closed`);
}
if (!failures) ok('open/close counts match for all paired tags');

/* ---------- 3. untranslated strings ---------- */
console.log('\n[3] translations');
const stringsSrc = src.match(/const STRINGS = \{[\s\S]*?\n\};/)[0];
const STRINGS = new Function(stringsSrc + '; return STRINGS;')();
const SAME_OK = new Set(['langname', 'months', 'src_ehail', 'wc_petrol', 'cm_ptptn']);
const enKeys = Object.keys(STRINGS.en);
for (const lang of ['ms', 'zh']) {
  const missing = enKeys.filter(k => STRINGS[lang][k] === undefined);
  const extra = Object.keys(STRINGS[lang]).filter(k => STRINGS.en[k] === undefined);
  const same = enKeys.filter(k => !SAME_OK.has(k) && STRINGS[lang][k] === STRINGS.en[k]);
  if (missing.length) fail(`${lang}: missing keys: ${missing.join(', ')}`);
  if (extra.length) fail(`${lang}: extra keys: ${extra.join(', ')}`);
  if (same.length) fail(`${lang}: identical to en (untranslated?): ${same.join(', ')}`);
  if (!missing.length && !extra.length && !same.length) ok(`${lang}: ${enKeys.length} keys, all translated`);
}

/* ---------- 4. DOM pass ---------- */
console.log('\n[4] DOM pass (playwright)');
let chromium;
try {
  const require = createRequire(import.meta.url);
  ({ chromium } = require('playwright'));
} catch {
  fail('playwright not importable — DOM checks skipped (word counts, --short, provenance, console)');
}
if (chromium) {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  const errors = [];
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto(pathToFileURL(FILE).href);
  await page.waitForFunction(() => window.__audit);

  const routes = await page.evaluate(() => window.__audit.routes);
  const report = {};
  for (const lang of ['en', 'ms', 'zh']) {
    await page.evaluate(l => window.__audit.setLang(l), lang);
    for (const r of routes) {
      await page.evaluate(rr => window.__audit.goto(rr), r);
      const res = await page.evaluate(() => {
        const scr = document.getElementById('screen');
        const SHORT = 'rgb(241, 89, 42)';
        // words: visible text, letter-bearing tokens only (numbers/RM excluded)
        const words = (scr.innerText || '').split(/\s+/)
          .filter(w => /[\p{L}]/u.test(w) && !/^RM$/.test(w));
        // CJK: innerText has no spaces between characters; count chars/2 as word-equivalents
        const cjk = (scr.innerText || '').match(/[一-鿿]/g) || [];
        const latinWords = words.filter(w => !/[一-鿿]/.test(w)).length;
        const wordCount = latinWords + Math.round(cjk.length / 2);
        // --short usage grouped by component class
        const comps = new Set();
        for (const el of scr.querySelectorAll('*')) {
          const cs = getComputedStyle(el);
          if (cs.backgroundColor === SHORT || cs.borderColor === SHORT || cs.color === SHORT)
            comps.add(el.className.split(' ')[0] || el.tagName);
        }
        // provenance: every RM figure needs a .prov nearby
        const missing = [];
        const walker = document.createTreeWalker(scr, NodeFilter.SHOW_TEXT);
        while (walker.nextNode()) {
          const tn = walker.currentNode;
          if (!/RM\s?[\d]/.test(tn.textContent)) continue;
          let el = tn.parentElement, found = false;
          for (let hop = 0; el && hop < 4; hop++, el = el.parentElement) {
            if (el.id === 'screen' || el.classList.contains('stack') || el.classList.contains('stack-s')) break;
            if (el.querySelector('.prov')) { found = true; break; }
          }
          if (!found) {
            const blk = tn.parentElement.closest('.stack > *, .stack-s > *') || tn.parentElement;
            for (const sib of [blk.previousElementSibling, blk.nextElementSibling]) {
              if (sib && (sib.matches('.prov, .figrow') || sib.querySelector?.('.prov'))) { found = true; break; }
            }
          }
          if (!found) missing.push(tn.textContent.trim().slice(0, 40));
        }
        // tap targets
        const tiny = [...scr.querySelectorAll('button,input,select,a')]
          .filter(el => { const r = el.getBoundingClientRect(); return r.width > 0 && (r.height < 44 || r.width < 24); })
          .map(el => (el.className || el.tagName).toString().slice(0, 30));
        return { wordCount, shortComps: [...comps], missing, tiny };
      });
      (report[r] ||= {})[lang] = res;
    }
  }
  await browser.close();

  console.log('\n  screen           words(en/ms/zh)  --short comps  prov-missing  tiny-targets');
  for (const r of routes) {
    const e = report[r].en, m = report[r].ms, z = report[r].zh;
    const shortN = e.shortComps.length;
    const line = `  ${r.padEnd(16)} ${String(e.wordCount).padStart(3)}/${String(m.wordCount).padStart(3)}/${String(z.wordCount).padStart(3)}          ${shortN} ${JSON.stringify(e.shortComps)}  ${e.missing.length}  ${e.tiny.length}`;
    console.log(line);
    if (shortN > 1) fail(`${r}: --short used by ${shortN} components: ${e.shortComps.join(', ')}`);
    if (e.wordCount > 66) fail(`${r}: en default word count ${e.wordCount} > ~60`);
    for (const lang of ['en', 'ms', 'zh']) {
      const rr = report[r][lang];
      if (rr.missing.length) fail(`${r}/${lang}: RM figures without provenance: ${rr.missing.join(' | ')}`);
      if (rr.tiny.length) fail(`${r}/${lang}: tap targets under floor: ${rr.tiny.join(', ')}`);
    }
  }
  if (errors.length) fail('console errors: ' + errors.join(' | '));
  else ok('0 console errors across all screens and languages');
}

console.log('\n' + (failures ? `${failures} FAILURE(S)` : 'ALL CHECKS PASSED'));
process.exit(failures ? 1 : 0);
