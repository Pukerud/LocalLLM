// Pure DOM stubs: execute the actual patched renderConvCache function in Node.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const source = fs.readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('function renderConvCache(c) {');
const end = source.indexOf('\nfunction renderTotals(', start);
assert(start >= 0 && end > start);
const elements = {};
const context = {
  $: id => elements[id] || (elements[id] = {style: {}, textContent: '', hidden: false}),
  fmt: n => String(n), gb: n => String(n / 1e9), since: () => 'just now',
  facts: (element, rows) => { element.rows = rows; }
};
vm.createContext(context);vm.runInContext(source.slice(start, end), context);
context.renderConvCache({enabled: false});
assert(elements['cc-note'].textContent.includes('--conversation-cache-mib'));
context.renderConvCache({implementation:'deepseek-native',enabled:true,ram_slots_limit:4,disk_target_mib:8192,
  last_prompt:3013,last_reused:3000,parks:2,restores:1,parked_ram:2,parked_disk:3});
assert(elements['cc-sum'].textContent.includes('native conversation cache enabled'));
assert(!elements['cc-note'].textContent.includes('--conversation-cache-mib'));
assert(elements['cc-note'].textContent.includes('not a RAM-byte cap'));
assert(elements['cc-bars'].hidden);
assert(JSON.stringify(elements['cc-facts'].rows).includes('8192 MiB (not a RAM limit)'));
assert(JSON.stringify(elements['cc-facts'].rows).includes('3000 of 3013'));
context.renderConvCache({implementation:'deepseek-native',enabled:null,ram_slots_limit:null,disk_target_mib:null});
assert(elements['cc-sum'].textContent.includes('unavailable'));
context.renderConvCache({implementation:'deepseek-native',enabled:false,ram_slots_limit:0,disk_target_mib:8192});
assert(elements['cc-sum'].textContent.includes('disabled'));
context.renderConvCache({enabled:true,slots:4,parked:1,budget_mib:8192,bytes:1024,parks:1,restores:0});
assert(!elements['cc-bars'].hidden, 'Qwen renderer must retain its memory/slot bars');
console.log('DeepSeek native enabled/disabled/unknown rendering and unchanged Qwen fallback: PASS');
