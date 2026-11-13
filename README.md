# understanding-ai-demos

Demos zur Videoserie **„KI verstehen – vom Token zum Agenten“**. Alles läuft im Browser, ohne Build und ohne Server.

| Folge | Demo | Datei |
|---|---|---|
| 2 · Token | Tokenizer: eigenen Text zerlegen | `02-token/tokenizer.html` |
| 2 · Token | Die Token-Schleife, Schritt für Schritt | `02-token/schleife.html` |
| 2 · Token | Die Zahlen aus dem Video nachrechnen | `npm install && npm run tokens` |
| 3 · Embeddings | Bedeutung als Ort im 3D-Raum | `03-embeddings/raum-3d.html` |

## Starten

`index.html` im Browser öffnen, oder `npm start` und die angezeigte Adresse aufrufen.

## Was echt ist und was Schema

- **Echt:** die Token-Zerlegung. Sie kommt aus dem offenen Tokenizer `o200k_base` (OpenAI), eingebunden über die Bibliothek
  [gpt-tokenizer](https://github.com/niieani/gpt-tokenizer) (MIT, gebündelt in `02-token/vendor/o200k_base.js`).
  Andere Modelle, zum Beispiel Claude, verwenden einen eigenen Tokenizer und kommen auf andere Stücke und andere Zahlen.
- **Schema:** die Wahrscheinlichkeiten in der Token-Schleife und die Punkte im 3D-Raum. Sie sind von Hand gesetzt, um das Prinzip zu zeigen.
- **Größenordnungen:** Ein Vokabular hat grob 50000 bis 200000 Token. Ein Embedding hat Hunderte bis Tausende Dimensionen
  (zum Beispiel 768, 1536 oder 4096) – nicht drei.
