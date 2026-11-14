// Welche Einstellung trennt die 4 Gruppen am besten? Probiert Modelle × Präfixe × Sprache und zeigt eine Tabelle.
//   npm run begriffe:vergleich
// Liest die bei Ollama installierten Embedding-Modelle selbst aus. Schreibt nichts, rechnet nur.
const URL_ = process.env.OLLAMA_URL || 'http://localhost:11434';
const sets = {
  de: [['Hund', 'Katze', 'Pferd', 'Kuh', 'Schaf'], ['Python', 'Java', 'Rust', 'Kotlin', 'TypeScript'], ['Messer', 'Topf', 'Pfanne', 'Sieb', 'Löffel'], ['Datenbank', 'Tabelle', 'Index', 'Abfrage', 'Transaktion']],
  en: [['dog', 'cat', 'horse', 'cow', 'sheep'], ['Python', 'Java', 'Rust', 'Kotlin', 'TypeScript'], ['knife', 'pot', 'pan', 'sieve', 'spoon'], ['database', 'table', 'index', 'query', 'transaction']],
  'de-Satz': [['Hund', 'Katze', 'Pferd', 'Kuh', 'Schaf'].map(x => `Ein ${x} ist ein Tier.`), ['Python', 'Java', 'Rust', 'Kotlin', 'TypeScript'].map(x => `${x} ist eine Programmiersprache.`),
    ['Messer', 'Topf', 'Pfanne', 'Sieb', 'Löffel'].map(x => `${x}: ein Werkzeug in der Küche.`), ['Datenbank', 'Tabelle', 'Index', 'Abfrage', 'Transaktion'].map(x => `${x}: ein Begriff aus der Welt der Datenbanken.`)],
  // kurze Sätze OHNE den Gruppennamen – fair, weil kein gemeinsames Wort die Gruppe verrät
  'de-neutral': [['Der Hund bellt am Zaun.', 'Die Katze schnurrt auf dem Sofa.', 'Das Pferd galoppiert über die Wiese.', 'Die Kuh gibt Milch.', 'Das Schaf wird geschoren.'],
    ['Python rückt Blöcke ein.', 'Java läuft auf der JVM.', 'Rust prüft den Speicher beim Kompilieren.', 'Kotlin kennt Null-Sicherheit.', 'TypeScript ergänzt JavaScript um Typen.'],
    ['Das Messer schneidet die Zwiebel.', 'Im Topf kocht die Suppe.', 'In der Pfanne brät das Ei.', 'Das Sieb lässt das Nudelwasser ablaufen.', 'Der Löffel rührt den Teig.'],
    ['Die Datenbank speichert alle Bestellungen.', 'Die Tabelle hat eine Spalte für das Datum.', 'Der Index beschleunigt die Suche nach der Kundennummer.', 'Die Abfrage liefert zehn Zeilen.', 'Die Transaktion wird zurückgerollt.']],
};
const prefixes = ['', 'clustering: ', 'classification: ', 'search_document: ', 'search_query: '];

const tags = await fetch(`${URL_}/api/tags`).then(r => r.json()).catch(e => { console.error(`Ollama nicht erreichbar unter ${URL_}: ${e.message}`); process.exit(1); });
const all = (tags.models || []).map(m => m.name);
const models = (process.env.MODELS ? process.env.MODELS.split(',') : all.filter(n => /embed|bge|e5|minilm|arctic|gte/i.test(n)));
console.log('Installierte Modelle:', all.join(', ') || '(keine)');
console.log('Getestet werden:', models.join(', ') || '(keins – z. B. ollama pull bge-m3)', '\n');

const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0), cos = (a, b) => dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));
const rows = [];
for (const model of models) for (const [lang, groups] of Object.entries(sets)) for (const prefix of (/nomic/i.test(model) ? prefixes : [''])) {
  const input = groups.flat().map(t => prefix + t);
  const res = await fetch(`${URL_}/api/embed`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model, input }) });
  if (!res.ok) { console.log(`${model}: Fehler ${res.status} ${(await res.text()).slice(0, 120)}`); break; }
  const { embeddings: E } = await res.json();
  const inG = [], out = []; let nn = 0;
  E.forEach((a, i) => { let best = -2, bj = -1; E.forEach((b, j) => { if (i === j) return; const c = cos(a, b); if (i < j) (Math.floor(i / 5) === Math.floor(j / 5) ? inG : out).push(c); if (c > best) { best = c; bj = j; } }); if (Math.floor(i / 5) === Math.floor(bj / 5)) nn++; });
  const avg = a => a.reduce((s, x) => s + x, 0) / a.length;
  rows.push({ Modell: model, Sprache: lang, Präfix: prefix || '(ohne)', Dim: E[0].length, 'in Gruppe': avg(inG).toFixed(3), zwischen: avg(out).toFixed(3), Abstand: (avg(inG) - avg(out)).toFixed(3), 'Nachbar passt': `${nn}/20`,
    'Hund·Katze': cos(E[0], E[1]).toFixed(3), 'Hund·Datenbank': cos(E[0], E[15]).toFixed(3) });
}
rows.sort((a, b) => b.Abstand - a.Abstand);
console.table(rows);
console.log('Abstand = Mittel in der Gruppe minus Mittel zwischen den Gruppen (größer = klarere Blöcke). „Nachbar passt“: bei wie vielen Begriffen liegt der ähnlichste in derselben Gruppe.');
