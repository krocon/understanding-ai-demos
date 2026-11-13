// Echte Zerlegung der Beispiele aus Folge 2 – genau diese Zahlen stehen im Video.
//   npm install && npm run tokens
import { encode, decode } from 'gpt-tokenizer/encoding/o200k_base';
const examples = {
  'Englisch': 'The server is not responding.',
  'Deutsch': 'Der Server antwortet nicht.',
  'Code': 'const user = await db.users.findById(id);',
  'Kompositum': 'Datenbankverbindungsfehler',
  'Erdbeere': 'Erdbeere',
};
for (const [name, text] of Object.entries(examples)) {
  const ids = encode(text);
  console.log(`\n${name}: ${JSON.stringify(text)}\n  ${ids.length} Token, ${[...text].length} Zeichen`);
  console.log('  ' + ids.map(id => `${JSON.stringify(decode([id]))} (${id})`).join('  '));
}
console.log('\nNachbarn im Vokabular:', [4821, 4822].map(id => `${id} = ${JSON.stringify(decode([id]))}`).join(', '));
