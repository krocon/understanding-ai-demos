// Folge 6 – Wahrscheinlichkeit: echte Verteilung für das nächste Token + Temperatur-Vergleich.
// Aufruf: npm run wahrscheinlichkeit   (Ollama muss laufen, Modell: llama3.2)
// Braucht ein Ollama mit logprobs-Unterstützung (ab 0.12.11). Schreibt wahrscheinlichkeit.json.
import { writeFileSync } from 'node:fs';
const HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const MODEL = process.env.MODEL || 'llama3.2';
const PROMPT = 'Moin! Der Build ist';
async function gen(options, extra = {}) {
  const r = await fetch(`${HOST}/api/generate`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL, prompt: PROMPT, raw: true, stream: false, options, ...extra }) });
  if (!r.ok) throw new Error(`Ollama: ${r.status} ${await r.text()}`);
  return r.json();
}
// 1) Verteilung über das nächste Token (Platz 1–10)
const first = await gen({ temperature: 0, num_predict: 1 }, { logprobs: true, top_logprobs: 10 });
const top = (first.logprobs?.[0]?.top_logprobs || []).map(x => ({ token: x.token, p: Math.exp(x.logprob) }));
if (!top.length) console.log('Keine logprobs in der Antwort – Ollama aktualisieren (ab 0.12.11).');
console.log(`Text: „${PROMPT}“ – wahrscheinlichste nächste Token:`);
top.forEach((x, i) => console.log(`  Platz ${i + 1}: ${JSON.stringify(x.token)}  ${(x.p * 100).toFixed(1)} %`));
// 2) Je 5 Fortsetzungen bei 3 Temperaturen
const runs = {};
for (const temperature of [0, 0.7, 1.5]) {
  runs[temperature] = [];
  for (let i = 0; i < 5; i++) runs[temperature].push((await gen({ temperature, num_predict: 12, seed: 100 + i })).response.replace(/\s+/g, ' ').trim());
  console.log(`\nTemperatur ${temperature}:`); runs[temperature].forEach(s => console.log(`  ${PROMPT} ${s}`));
}
writeFileSync(new URL('./wahrscheinlichkeit.json', import.meta.url), JSON.stringify({ model: MODEL, prompt: PROMPT, top, runs }, null, 2));
console.log('\n→ wahrscheinlichkeit.json geschrieben');
