// Folge 3, Kapitel 7: 20 Begriffe → echte Vektoren → Ähnlichkeit von jedem mit jedem → 2D-Bild.
//   Voraussetzung: Ollama läuft und das Modell ist da:   ollama pull bge-m3
//   Warum bge-m3 und diese 4 Gruppen: siehe begriffe-vergleich.mjs und begriffe-gruppen.mjs (nomic-embed-text trennt
//   einzelne deutsche Wörter kaum; mit bge-m3 liegt bei allen 20 Begriffen der ähnlichste in der eigenen Gruppe).
//   Start:         npm run begriffe        (schreibt 03-embeddings/begriffe.json und zeigt die Kernzahlen)
// Ohne Abhängigkeiten (Node 18+). Adresse/Modell änderbar: OLLAMA_URL, EMBEDDING_MODEL.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const URL_ = process.env.OLLAMA_URL || 'http://localhost:11434';
const MODEL = process.env.EMBEDDING_MODEL || 'bge-m3';
// bge-m3 braucht kein Aufgaben-Präfix (nomic-embed-text schon, z. B. EMBEDDING_PREFIX='clustering: ').
const PREFIX = process.env.EMBEDDING_PREFIX ?? '';
const groups = {
  'Tiere': ['Hund', 'Katze', 'Pferd', 'Kuh', 'Schaf'],
  'Programmiersprachen': ['Python', 'Java', 'Rust', 'Kotlin', 'TypeScript'],
  'Städte': ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt'],
  'Datenbanken': ['Datenbank', 'Tabelle', 'Index', 'Abfrage', 'Transaktion'],
};
const terms = Object.entries(groups).flatMap(([group, list]) => list.map(term => ({ term, group })));

const res = await fetch(`${URL_}/api/embed`, { method: 'POST', headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ model: MODEL, input: terms.map(t => PREFIX + t.term) }) }).catch(e => { console.error(`Ollama nicht erreichbar unter ${URL_}: ${e.message}`); process.exit(1); });
if (!res.ok) { console.error(`Ollama antwortet mit ${res.status}: ${await res.text()}`); process.exit(1); }
const { embeddings } = await res.json();
if (!Array.isArray(embeddings) || embeddings.length !== terms.length) { console.error('Unerwartete Antwort: Zahl der Vektoren passt nicht.'); process.exit(1); }

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const norm = a => Math.sqrt(dot(a, a));
const cos = (a, b) => dot(a, b) / (norm(a) * norm(b));
const r3 = x => Math.round(x * 1000) / 1000;
const matrix = embeddings.map(a => embeddings.map(b => r3(cos(a, b))));

// 2D-Bild: Hauptkomponentenanalyse (die 2 Richtungen mit der größten Streuung), per Potenzmethode.
const dim = embeddings[0].length, n = embeddings.length;
const mean = Array.from({ length: dim }, (_, j) => embeddings.reduce((s, v) => s + v[j], 0) / n);
let X = embeddings.map(v => v.map((x, j) => x - mean[j]));
const component = () => {
  let w = Array.from({ length: dim }, (_, j) => Math.sin(j + 1));           // fester Startwert → jedes Mal dasselbe Bild
  for (let it = 0; it < 300; it++) {
    const p = X.map(x => dot(x, w));                                         // X·w
    const nw = Array.from({ length: dim }, (_, j) => X.reduce((s, x, i) => s + x[j] * p[i], 0)); // Xᵀ·(X·w)
    const l = norm(nw); w = nw.map(x => x / l);
  }
  const scores = X.map(x => dot(x, w));
  X = X.map((x, i) => x.map((v, j) => v - scores[i] * w[j]));               // Richtung herausrechnen
  return scores;
};
const pc1 = component(), pc2 = component();
const points = terms.map((t, i) => ({ ...t, x: r3(pc1[i]), y: r3(pc2[i]) }));

const idx = t => terms.findIndex(x => x.term === t);
const pair = (a, b) => matrix[idx(a)][idx(b)];
const inGroup = [], between = [];
terms.forEach((a, i) => terms.forEach((b, j) => { if (i < j) (a.group === b.group ? inGroup : between).push(matrix[i][j]); }));
const avg = a => r3(a.reduce((s, x) => s + x, 0) / a.length);
const out = { model: MODEL, prefix: PREFIX, dimensions: dim, date: new Date().toISOString().slice(0, 10), groups, terms: terms.map(t => t.term), matrix, points,
  facts: { hundKatze: pair('Hund', 'Katze'), hundDatenbank: pair('Hund', 'Datenbank'), mittelInDerGruppe: avg(inGroup), mittelZwischenGruppen: avg(between), kleinsterWert: r3(Math.min(...between)), } };
const file = fileURLToPath(new URL('./begriffe.json', import.meta.url));
writeFileSync(file, JSON.stringify(out, null, 1));

console.log(`Modell ${MODEL}, ${dim} Dimensionen, ${n} Begriffe → ${file}`);
console.log(`Hund · Katze:   ${pair('Hund', 'Katze')}`);
console.log(`Hund · Datenbank: ${pair('Hund', 'Datenbank')}`);
const pairs = []; terms.forEach((a, i) => terms.forEach((b, j) => { if (i < j) pairs.push([matrix[i][j], a.term, b.term, a.group === b.group]); }));
pairs.sort((x, y) => y[0] - x[0]);
console.log('Ähnlichste Paare:   ' + pairs.slice(0, 5).map(p => `${p[1]}·${p[2]} ${p[0]}`).join('  '));
console.log('Unähnlichste Paare: ' + pairs.slice(-5).map(p => `${p[1]}·${p[2]} ${p[0]}`).join('  '));
console.log('Stärkste Paare über Gruppen hinweg: ' + pairs.filter(p => !p[3]).slice(0, 4).map(p => `${p[1]}·${p[2]} ${p[0]}`).join('  '));
console.log(`Mittel in der Gruppe: ${out.facts.mittelInDerGruppe} · zwischen den Gruppen: ${out.facts.mittelZwischenGruppen} · kleinster Wert: ${out.facts.kleinsterWert}`);
