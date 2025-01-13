import os

import numpy as np
from sentence_transformers import SentenceTransformer

from src.metrics.content_preservation import compute_content_preservation_score as sim
from src.metrics.emd import calculate_emd
from src.metrics.towards_away import joint, away, towards
from src.text2vec.embedder import Embedder

camembedder = Embedder("__models/distilbert-class-ep3", finetuned=False)
clasembedder = Embedder("__models/distilbert-class-ep3", finetuned=True)
sentembedder = SentenceTransformer("Lajavaness/sentence-flaubert-base")


hugo = []
for file in os.listdir("data/paragraphs/masked/hugo_paragraphs"):
    with open(os.path.join("data/paragraphs/masked/hugo_paragraphs", file), "r", encoding="utf-8") as f:
        hugo.append(f.read().strip().replace("\n", " ").replace("  ", " "))
print(f"Read {len(hugo)} paragraphs of Victor Hugo.")


neutral = []
for file in os.listdir("data/paragraphs/masked/hugo2neutral"):
    with open(os.path.join("data/paragraphs/masked/hugo2neutral", file), "r", encoding="utf-8") as f:
        neutral.append(f.read().strip().replace("\n", " ").replace("  ", " "))
print(f"Read {len(neutral)} paragraphs of Neutralized Hugo.")


restored = []
for file in os.listdir("data/paragraphs/masked/restored_hugo"):
    with open(os.path.join("data/paragraphs/masked/restored_hugo", file), "r", encoding="utf-8") as f:
        restored.append(f.read().strip().replace("\n", " ").replace("  ", " "))
print(f"Read {len(restored)} paragraphs of Restored Hugo.")


other = []
for file in os.listdir("data/paragraphs/masked/other_paragraphs"):
    with open(os.path.join("data/paragraphs/masked/other_paragraphs", file), "r", encoding="utf-8") as f:
        other.append(f.read().strip().replace("\n", " ").replace("  ", " "))
print(f"Read {len(other)} paragraphs of Other Authors.")


other2hugo = []
for file in os.listdir("data/paragraphs/masked/other2hugo"):
    with open(os.path.join("data/paragraphs/masked/other2hugo", file), "r", encoding="utf-8") as f:
        other2hugo.append(f.read().strip().replace("\n", " ").replace("  ", " "))
print(f"Read {len(other2hugo)} paragraphs of Other2Hugo Authors.")


sem_s = sentembedder.encode(other)
sem_r = sentembedder.encode(other2hugo)
sem_t = sentembedder.encode(hugo)

sem = sim(sem_s, sem_t, sem_r)
print(f"SIM: {sem:.4f}")

prob_s = clasembedder.get_probs(other).numpy()
prob_r = clasembedder.get_probs(other2hugo).numpy()
prob_t = np.array([0, 0, 0, 0, 0, 0, 1])


emd_r = [calculate_emd(s, r, 6) for s, r in zip(prob_s, prob_r)]
emd_t = [calculate_emd(s, prob_t, 6) for s in prob_s]
emd_rt = [calculate_emd(r, prob_t, 6) for r in prob_r]

mean_emd_r = np.mean(emd_r)
mean_emd_t = np.mean(emd_t)
mean_emd_rt = np.mean(emd_rt)

print(f"EMD: {mean_emd_r:.4f}")
print(f"EMD-T: {mean_emd_t:.4f}")
print(f"EMD-RT: {mean_emd_rt:.4f}")

e_aw = max(0, min(mean_emd_r, mean_emd_t)) / mean_emd_t
print(f"E-AW: {e_aw:.4f}")

e_tow = max(0, mean_emd_t - mean_emd_rt) / mean_emd_t
print(f"E-TOW: {e_tow:.4f}")

e_j = joint(e_aw, e_tow, sem)
print(f"E-J: {e_j:.4f}")

emb_s = camembedder.encode(other).numpy()
emb_r = camembedder.encode(other2hugo).numpy()
emb_t = camembedder.encode(hugo).numpy()

uar = np.mean(emb_t, axis=0)

aw = np.mean([
    away(s, uar, r) for s, r in zip(emb_s, emb_r)
])
print(f"AW: {aw:.4f}")

tow = np.mean([
    towards(s, uar, r) for s, r in zip(emb_s, emb_r)
])
print(f"TOW: {tow:.4f}")

j = joint(aw, tow, sem)
print(f"J: {j:.4f}")
