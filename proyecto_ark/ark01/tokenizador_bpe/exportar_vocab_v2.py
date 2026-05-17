import json, sys
import sentencepiece as spm

MODEL = sys.argv[1] if len(sys.argv) > 1 else "tokenizador_bpe_32k_v2.model"
sp = spm.SentencePieceProcessor()
sp.Load(MODEL)
n = sp.GetPieceSize()
print(f"Vocab cargado: {n} tokens")

vocab  = {}
scores = {}
for i in range(n):
    piece = sp.IdToPiece(i)
    vocab[piece]  = i
    scores[piece] = sp.GetScore(i)

with open("vocab_sp_v2.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
with open("vocab_scores_v2.json", "w", encoding="utf-8") as f:
    json.dump(scores, f, ensure_ascii=False)

print(f"Generados: vocab_sp_v2.json y vocab_scores_v2.json ({n} tokens)")
