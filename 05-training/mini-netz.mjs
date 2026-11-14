// Folge 5: Training im Kleinen – eine Gerade und ein winziges Netz lernen per Gradientenabstieg.
//   npm run training      (schreibt 05-training/training.json und zeigt die Kernzahlen)
// Ohne Abhängigkeiten. Fester Startwert → jedes Mal dieselben Zahlen wie im Video.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
const rng = seed => () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const r4 = x => Math.round(x * 10000) / 10000;

// ── Teil 1: eine Gerade durch Punkte (Wohnfläche → Miete). Die Punkte sind erfunden, das Verfahren ist echt.
const flats = [[32, 560], [41, 610], [48, 740], [55, 720], [63, 860], [70, 900], [78, 1010], [86, 1040], [95, 1210], [104, 1250]];
const xs = flats.map(p => p[0] / 100), ys = flats.map(p => p[1] / 1000);            // auf handliche Größen bringen
let m = -0.6, t = 1.4;                                                               // schlechte Startwerte
const lineLoss = () => xs.reduce((s, x, i) => s + (m * x + t - ys[i]) ** 2, 0) / xs.length;
const line = [];
for (let step = 0; step <= 4000; step++) {
  if ([0, 20, 100, 500, 4000].includes(step)) line.push({ step, m: r4(m * 10), t: r4(t * 1000), loss: r4(lineLoss()) }); // m in €/m², t in €
  let gm = 0, gt = 0; xs.forEach((x, i) => { const e = m * x + t - ys[i]; gm += 2 * e * x; gt += 2 * e; });
  m -= 0.5 * gm / xs.length; t -= 0.5 * gt / xs.length;
}

// ── Teil 2: ein winziges Netz (1 Eingang → H Neuronen mit Knick → 1 Ausgang) bildet eine Wellenlinie nach.
const X = Array.from({ length: 13 }, (_, i) => -3 + i * 0.5), Y = X.map(Math.sin);
const grid = Array.from({ length: 131 }, (_, i) => -6.5 + i * 0.1);                   // breiter als die Punkte: der Rand zeigt, wo nichts gelernt wurde
function train(H, steps, snaps, seed = 7) {
  const rnd = rng(seed), w1 = Array.from({ length: H }, () => (rnd() * 2 - 1) * 1.5), b1 = Array.from({ length: H }, () => (rnd() * 2 - 1) * 2), w2 = Array.from({ length: H }, () => (rnd() * 2 - 1) * 0.3); let b2 = 0;
  const f = x => w1.reduce((s, w, j) => s + w2[j] * Math.max(0, w * x + b1[j]), b2);
  const loss = () => X.reduce((s, x, i) => s + (f(x) - Y[i]) ** 2, 0) / X.length;
  const out = [], lossCurve = [];
  for (let step = 0; step <= steps; step++) {
    if (step % Math.max(1, Math.round(steps / 200)) === 0) lossCurve.push([step, r4(loss())]);
    if (snaps.includes(step)) out.push({ step, loss: r4(loss()), curve: grid.map(x => [r4(x), r4(f(x))]) });
    const g1 = new Array(H).fill(0), gb1 = new Array(H).fill(0), g2 = new Array(H).fill(0); let gb2 = 0;
    X.forEach((x, i) => { const hs = w1.map((w, j) => Math.max(0, w * x + b1[j])), e = 2 * (hs.reduce((s, h, j) => s + w2[j] * h, b2) - Y[i]) / X.length;
      gb2 += e; hs.forEach((h, j) => { g2[j] += e * h; if (h > 0) { g1[j] += e * w2[j] * x; gb1[j] += e * w2[j]; } }); });
    const lr = 0.02; for (let j = 0; j < H; j++) { w1[j] -= lr * g1[j]; b1[j] -= lr * gb1[j]; w2[j] -= lr * g2[j]; } b2 -= lr * gb2;
  }
  return { neurons: H, parameters: 3 * H + 1, snapshots: out, lossCurve };
}
const net = train(8, 20000, [0, 200, 2000, 20000]);
const sizes = [1, 2, 8].map(H => { const r = train(H, 20000, [20000]); return { neurons: H, parameters: r.parameters, loss: r.snapshots[0].loss, curve: r.snapshots[0].curve }; });

const out = { date: new Date().toISOString().slice(0, 10), line: { points: flats, snapshots: line }, points: X.map((x, i) => [r4(x), r4(Y[i])]), net, sizes };
writeFileSync(fileURLToPath(new URL('./training.json', import.meta.url)), JSON.stringify(out));
console.log('Gerade:', line.map(s => `Schritt ${s.step}: ${s.m} €/m² · ${s.t} € · Loss ${s.loss}`).join('\n        '));
console.log(`Netz mit ${net.neurons} Neuronen (${net.parameters} Regler):`, net.snapshots.map(s => `Schritt ${s.step}: Loss ${s.loss}`).join(' · '));
console.log('Am Rand (x = 6,5, wo kein Punkt liegt):', net.snapshots.at(-1).curve.at(-1)[1], '– echte Welle:', r4(Math.sin(6.5)));
console.log('Größe des Netzes:', sizes.map(s => `${s.neurons} Neuronen → Loss ${s.loss}`).join(' · '));
