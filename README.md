# understanding-ai-demos

Demos zur Videoserie **„KI verstehen – vom Token zum Agenten“**. Alles läuft im Browser, ohne Build und ohne Server.

| Folge | Demo | Datei |
|---|---|---|
| 2 · Token | Tokenizer: eigenen Text zerlegen | `02-token/tokenizer.html` |
| 2 · Token | Die Token-Schleife, Schritt für Schritt | `02-token/schleife.html` |
| 2 · Token | Die Zahlen aus dem Video nachrechnen | `npm install && npm run tokens` |
| 3 · Embeddings | Bedeutung als Ort im 3D-Raum | `03-embeddings/raum-3d.html` |
| 3 · Embeddings | 20 Begriffe, echte Ähnlichkeiten (Ollama, `bge-m3`) | `npm run begriffe` |
| 3 · Embeddings | Modelle, Präfixe und Sprachen im Vergleich | `npm run begriffe:vergleich` · `npm run begriffe:gruppen` |
| 4 · Attention | Attention nachgerechnet: 5 Token, 4 Zahlen je Vektor | `npm run attention` |
| 4 · Attention | Attention in GPT-2 small (PyTorch, siehe Kopf der Datei) | `04-attention/gpt2_attention.py` |
| 5 · Training | Eine Gerade lernt, ein winziges Netz lernt eine Wellenlinie | `npm run training` |
| 6 · Wahrscheinlichkeit | Echte Verteilung für das nächste Token, 3 Temperaturen (Ollama, `llama3.2`) | `npm run wahrscheinlichkeit` |

## Starten

`index.html` im Browser öffnen, oder `npm start` und die angezeigte Adresse aufrufen.

## Was echt ist und was Schema

- **Echt:** die Token-Zerlegung. Sie kommt aus dem offenen Tokenizer `o200k_base` (OpenAI), eingebunden über die Bibliothek
  [gpt-tokenizer](https://github.com/niieani/gpt-tokenizer) (MIT, gebündelt in `02-token/vendor/o200k_base.js`).
  Andere Modelle, zum Beispiel Claude, verwenden einen eigenen Tokenizer und kommen auf andere Stücke und andere Zahlen.
- **Echt:** die Rechnungen der Skripte ab Folge 3 – Ähnlichkeiten aus einem lokalen Embedding-Modell, die Attention-Rechnung
  (mit Zufallsvektoren statt gelernter Werte), das Training der Geraden und des kleinen Netzes. Die Mietpunkte der Geraden
  sind erfunden, das Verfahren ist echt. Die Skripte schreiben ihre Ergebnisse als JSON neben sich.
- **Schema:** die Wahrscheinlichkeiten in der Token-Schleife und die Punkte im 3D-Raum. Sie sind von Hand gesetzt, um das Prinzip zu zeigen.
- **Größenordnungen:** Ein Vokabular hat grob 50000 bis 200000 Token. Ein Embedding hat Hunderte bis Tausende Dimensionen
  (zum Beispiel 768, 1536 oder 4096) – nicht drei.
