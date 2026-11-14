// Folge 4: Self-Attention im Kleinen – 5 Token, jeder Vektor hat 4 Zahlen. Nur Multiplizieren und Addieren.
//   npm run attention     (schreibt 04-attention/attention.json und zeigt die Tabelle)
// Die Vektoren und die Rechenmatrizen sind Zufallszahlen mit festem Startwert: Es geht um die Rechnung, nicht um Bedeutung.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
const rng = seed => () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const rnd = rng(42), r3 = x => Math.round(x * 1000) / 1000, D = 4;
const tokens = ['Server', 'antwortet', 'nicht', 'weil', 'er'];
const mat = () => Array.from({ length: D }, () => Array.from({ length: D }, () => rnd() * 2 - 1));
const mul = (M, v) => M.map(row => row.reduce((s, x, j) => s + x * v[j], 0));
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const vec = tokens.map(() => Array.from({ length: D }, () => rnd() * 2 - 1));        // 1 Vektor je Token
const WQ = mat(), WK = mat(), WV = mat();                                              // „gelernt“ wären sie im echten Modell
const Q = vec.map(v => mul(WQ, v)), K = vec.map(v => mul(WK, v)), V = vec.map(v => mul(WV, v));  // Schritt 1: Frage, Etikett, Inhalt
const scores = Q.map(q => K.map(k => dot(q, k) / Math.sqrt(D)));                       // Schritt 2: jede Frage mit jedem Etikett
const masked = scores.map((row, i) => row.map((s, j) => (j <= i ? s : -Infinity)));    //            nach vorn schauen ist gestrichen
const weights = masked.map(row => { const m = Math.max(...row), e = row.map(s => Math.exp(s - m)), z = e.reduce((a, b) => a + b, 0); return e.map(x => x / z); }); // Schritt 3: jede Zeile auf 100 %
const mixed = weights.map((w, i) => vec[i].map((x, d) => x + w.reduce((s, wj, j) => s + wj * V[j][d], 0)));                                              // Schritt 4: mischen, obendrauf
const out = { date: new Date().toISOString().slice(0, 10), tokens, dimensions: D, vectors: vec.map(v => v.map(r3)), scores: scores.map(r => r.map(r3)), weights: weights.map(r => r.map(r3)), result: mixed.map(v => v.map(r3)) };
writeFileSync(fileURLToPath(new URL('./attention.json', import.meta.url)), JSON.stringify(out, null, 1));
console.log('Gewichte in Prozent (Zeile = wer schaut, Spalte = wohin):');
console.log(' '.repeat(11) + tokens.map(t => t.slice(0, 9).padStart(10)).join(''));
weights.forEach((row, i) => console.log(tokens[i].padEnd(11) + row.map((w, j) => (j <= i ? `${Math.round(w * 100)} %` : '–').padStart(10)).join('')));
