// Welche 4 Gruppen von Begriffen trennt ein Embedding-Modell wirklich? Probiert 10 Gruppen (50 deutsche Wörter)
// und alle 210 Vierer-Kombinationen je Modell.   npm run begriffe:gruppen
// Schreibt nichts, rechnet nur. Modelle: alle installierten Embedding-Modelle (oder MODELS=a,b).
const URL_ = process.env.OLLAMA_URL || 'http://localhost:11434';
const pool = {
  Tiere: ['Hund', 'Katze', 'Pferd', 'Kuh', 'Schaf'],
  Obst: ['Apfel', 'Birne', 'Banane', 'Kirsche', 'Pflaume'],
  Städte: ['Berlin', 'Hamburg', 'München', 'Köln', 'Frankfurt'],
  Werkzeug: ['Hammer', 'Zange', 'Säge', 'Bohrer', 'Schraubenzieher'],
  Instrumente: ['Geige', 'Trompete', 'Klavier', 'Flöte', 'Gitarre'],
  Küche: ['Messer', 'Topf', 'Pfanne', 'Sieb', 'Löffel'],
  Wetter: ['Regen', 'Schnee', 'Nebel', 'Gewitter', 'Hagel'],
  Berufe: ['Ärztin', 'Bäcker', 'Lehrerin', 'Pilot', 'Tischler'],
  Programmiersprachen: ['Python', 'Java', 'Rust', 'Kotlin', 'TypeScript'],
  Datenbanken: ['Datenbank', 'Tabelle', 'Index', 'Abfrage', 'Transaktion'],
};
const names = Object.keys(pool), words = names.flatMap(n => pool[n]), G = i => Math.floor(i / 5);
const tags = await fetch(`${URL_}/api/tags`).then(r => r.json()).catch(e => { console.error(`Ollama nicht erreichbar unter ${URL_}: ${e.message}`); process.exit(1); });
const models = process.env.MODELS ? process.env.MODELS.split(',') : (tags.models || []).map(m => m.name).filter(n => /embed|bge|e5|minilm|arctic|gte/i.test(n));
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0), avg = a => a.reduce((s, x) => s + x, 0) / a.length;
const combos = []; for (let a = 0; a < 10; a++) for (let b = a + 1; b < 10; b++) for (let c = b + 1; c < 10; c++) for (let d = c + 1; d < 10; d++) combos.push([a, b, c, d]);

for (const model of models) for (const prefix of (/nomic/i.test(model) ? ['', 'clustering: '] : [''])) {
  const res = await fetch(`${URL_}/api/embed`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model, input: words.map(w => prefix + w) }) });
  if (!res.ok) { console.log(`${model}: Fehler ${res.status}`); continue; }
  const E = (await res.json()).embeddings.map(v => { const n = Math.sqrt(dot(v, v)); return v.map(x => x / n); });
  const S = E.map(a => E.map(b => dot(a, b)));
  const rows = combos.map(c => {
    const idx = c.flatMap(g => [0, 1, 2, 3, 4].map(k => g * 5 + k)); const inG = [], out = []; let nn = 0, worst = 1;
    idx.forEach(i => { let best = -2, bj = -1; idx.forEach(j => { if (i === j) return; if (i < j) (G(i) === G(j) ? inG : out).push(S[i][j]); if (S[i][j] > best) { best = S[i][j]; bj = j; } }); if (G(i) === G(bj)) nn++; });
    c.forEach(g => { const v = []; for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) v.push(S[g * 5 + i][g * 5 + j]); worst = Math.min(worst, avg(v)); });
    return { Gruppen: c.map(g => names[g]).join(' · '), 'Nachbar passt': nn, 'in Gruppe': avg(inG).toFixed(3), zwischen: avg(out).toFixed(3), Abstand: +(avg(inG) - avg(out)).toFixed(3), 'schwächste Gruppe': worst.toFixed(3), 'größter Wert zwischen': Math.max(...out).toFixed(3) };
  }).sort((a, b) => b['Nachbar passt'] - a['Nachbar passt'] || b.Abstand - a.Abstand);
  console.log(`\n=== ${model}${prefix ? ` · Präfix „${prefix}“` : ''} · ${E[0].length} Dimensionen · beste 6 von 210 Kombinationen ===`);
  console.table(rows.slice(0, 6).map(r => ({ ...r, 'Nachbar passt': `${r['Nachbar passt']}/20` })));
  console.log('Zusammenhalt je Gruppe (Mittel der Ähnlichkeit innerhalb) und Mittel zu allen anderen Wörtern:');
  console.log(names.map((n, g) => { const v = [], o = []; for (let i = 0; i < 5; i++) for (let j = 0; j < 50; j++) { if (g * 5 + i === j) continue; (G(j) === g ? v : o).push(S[g * 5 + i][j]); } return `  ${n.padEnd(20)} ${avg(v).toFixed(3)}  /  ${avg(o).toFixed(3)}`; }).join('\n'));
}
