#!/usr/bin/env node
// CPU-only tests of the EXACT shipped app.js Chat initialization and migration.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert/strict');
const source = fs.readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('const DEFAULTS = ');
const end = source.indexOf('let messages = store.get("chat", []);', start);
assert(start >= 0 && end > start);
const block = source.slice(start, end);
const MODEL = 'qwen3.8-flash-next-unsloth-v31-iq4_xs-ple-q8_0-projection-2slot-v0166';
const KEY = 'sampling.v0166.524k.ui64k.v1';
let passed = 0;
function fixture(saved = {}, blocked = false) {
  const data = JSON.parse(JSON.stringify(saved));
  const context = vm.createContext({store: {
    get(k, d) { return Object.hasOwn(data, k) ? JSON.parse(JSON.stringify(data[k])) : d; },
    set(k, v) { if (blocked) return false; data[k] = JSON.parse(JSON.stringify(v)); return true; },
  }, $: () => ({dataset: {open: 'false'}}), busy: false, loadDrawer() { throw Error('Unexpected drawer write'); }});
  vm.runInContext(block + '\nglobalThis.api = {apply: applyHostllmUiDefaults, save: saveHostllmUiSettings, settings: () => JSON.parse(JSON.stringify(settings)), defaults: () => JSON.parse(JSON.stringify(DEFAULTS))};', context);
  return {api: context.api, data};
}
{
  const {api, data} = fixture({chat: [{role: 'user', text: 'untouched synthetic history'}], apikey: 'synthetic-optional-pref'});
  api.apply(MODEL); assert.equal(api.settings().max, '65536'); assert.equal(api.defaults().max, '65536');
  assert.equal(data.sampling, undefined); assert.equal(data.chat[0].text, 'untouched synthetic history');
  assert.equal(data.apikey, 'synthetic-optional-pref'); passed++;
}
for (const legacy of ['16000', 16000, '16384', 16384, '']) {
  const sampling = {max: legacy, temperature: .75, top_p: .88, top_k: 30, thinking: 'medium', esp: false, mcp: false, seed: '9118'};
  const {api, data} = fixture({sampling}); api.apply(MODEL);
  const s = api.settings(); assert.equal(s.max, '65536');
  for (const key of Object.keys(sampling).filter(k => k !== 'max')) assert.equal(s[key], sampling[key]);
  assert.deepEqual(data.sampling, sampling); assert.equal(data[KEY].max, '65536'); passed++;
}
for (const custom of ['32768', '100000']) {
  const {api} = fixture({sampling: {max: custom}}); api.apply(MODEL); assert.equal(api.settings().max, custom); passed++;
}
for (const later of ['', '16000', '8192']) {
  const {api, data} = fixture({[KEY]: {max: later, thinking: 'low'}, sampling: {max: '16384'}});
  api.apply(MODEL); assert.equal(api.settings().max, later); assert.equal(api.settings().thinking, 'low');
  api.save({...api.settings(), max: later}); assert.equal(data[KEY].max, later); assert.equal(data.sampling.max, '16384'); passed++;
}
{
  const {api, data} = fixture({sampling: {max: '16000'}}); api.apply('another-model');
  assert.equal(api.settings().max, '16000'); assert.equal(api.defaults().max, ''); assert.equal(data[KEY], undefined); passed++;
}
{
  const {api, data} = fixture({sampling: {max: '16384'}}); api.apply(MODEL); api.apply('another-model');
  assert.equal(api.settings().max, '16384'); assert.equal(api.defaults().max, '');
  api.apply(MODEL); assert.equal(api.settings().max, '65536'); assert.equal(data.sampling.max, '16384'); passed++;
}
{
  const {api} = fixture({}, true); api.apply(MODEL); assert.equal(api.settings().max, '65536'); passed++;
}
assert(source.includes('    applyHostllmUiDefaults(health.model);'));
assert(source.includes('  saveHostllmUiSettings(settings);'));
assert(source.includes('  if (settings.max) body.max_tokens = +settings.max;'));
console.log(`${passed} actual-UI block scenarios passed: fresh/legacy/custom/blank/model scoping/storage/privacy; no backend/default clamp.`);
