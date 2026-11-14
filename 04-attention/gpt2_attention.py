"""Folge 4 – Attention in einem echten Modell (GPT-2 small: 12 Schichten, je 12 Köpfe).

Einmalig einrichten (lädt PyTorch und beim ersten Lauf das Modell, zusammen einige hundert MB):
    python3 -m venv .venv-attention
    .venv-attention/bin/pip install torch transformers
Aufruf:
    .venv-attention/bin/python 04-attention/gpt2_attention.py
Schreibt 04-attention/gpt2-attention.json.
"""
import json, sys
from pathlib import Path
import torch
from transformers import GPT2TokenizerFast, GPT2Model

SATZ = sys.argv[1] if len(sys.argv) > 1 else "The server does not respond because it is overloaded."
VON, NACH = " it", " server"   # Fürwort → Bezugswort

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
model.eval()
enc = tok(SATZ, return_tensors="pt")
tokens = [tok.decode([i]) for i in enc["input_ids"][0]]
with torch.no_grad():
    att = torch.stack(model(**enc).attentions).squeeze(1)   # [Schicht, Kopf, von, nach]
L, H, N, _ = att.shape
i_von, i_nach = tokens.index(VON), tokens.index(NACH)
print("Token:", tokens)
print(f"{L} Schichten, {H} Köpfe je Schicht\n")

# 1) Köpfe, in denen das Fürwort am stärksten auf das Bezugswort schaut
w = att[:, :, i_von, i_nach]
rang = sorted(((w[l, h].item(), l, h) for l in range(L) for h in range(H)), reverse=True)
print(f"Gewicht von {VON!r} auf {NACH!r} – die 8 stärksten Köpfe (Schicht und Kopf ab 1 gezählt):")
for g, l, h in rang[:8]:
    zeile = att[l, h, i_von, : i_von + 1]
    dick = tokens[int(zeile.argmax())]
    print(f"  Schicht {l + 1:2d}, Kopf {h + 1:2d}: {g * 100:5.1f} %   (dickster Bogen geht zu {dick!r})")

# Der gezeigte Kopf: der stärkste, bei dem der dickste Bogen wirklich zum Bezugswort führt
beste = next(((g, l, h) for g, l, h in rang if int(att[l, h, i_von, : i_von + 1].argmax()) == i_nach), rang[0])
g, l, h = beste
print(f"\n→ Für das Video: Schicht {l + 1}, Kopf {h + 1}, Gewicht {g * 100:.0f} %")

# 2) Was die anderen Köpfe tun: Vorgänger, Satzanfang, sonst
def art(l_, h_):
    a = att[l_, h_]
    vor = sum(a[i, i - 1].item() for i in range(1, N)) / (N - 1)
    anfang = sum(a[i, 0].item() for i in range(1, N)) / (N - 1)
    return ("Vorgänger" if vor > 0.5 else "Satzanfang" if anfang > 0.5 else "anderes", vor, anfang)
zaehler = {"Vorgänger": 0, "Satzanfang": 0, "anderes": 0}
koepfe = []
for l_ in range(L):
    for h_ in range(H):
        k, vor, anfang = art(l_, h_)
        zaehler[k] += 1
        koepfe.append({"schicht": l_ + 1, "kopf": h_ + 1, "art": k, "vorgaenger": round(vor, 3), "satzanfang": round(anfang, 3)})
print(f"\nAlle {L * H} Köpfe: {zaehler['Vorgänger']} schauen vor allem auf das Token davor, "
      f"{zaehler['Satzanfang']} vor allem auf den Satzanfang, {zaehler['anderes']} tun etwas anderes.")

out = {"modell": "gpt2 (small)", "satz": SATZ, "tokens": tokens, "schichten": L, "koepfe_je_schicht": H,
       "gezeigt": {"schicht": l + 1, "kopf": h + 1, "von": VON, "nach": NACH, "gewicht": round(g, 4),
                   "zeile": [round(x, 4) for x in att[l, h, i_von, : i_von + 1].tolist()]},
       "rangliste": [{"schicht": l_ + 1, "kopf": h_ + 1, "gewicht": round(g_, 4)} for g_, l_, h_ in rang[:12]],
       "arten": zaehler, "koepfe": koepfe}
ziel = Path(__file__).with_name("gpt2-attention.json")
ziel.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\n→ {ziel.name} geschrieben")
