import fasttext
import tempfile
import os

# ─────────────────────────────────────────────
# 1️⃣  Build a tiny text corpus (one sentence per line)
sentences = [
    "machine learning makes computers learn from data",
    "deep learning is a branch of machine learning",
    "large language models are impressive",
    "language models can generate text",
    "data science uses machine learning",
]

# Write the corpus to a temporary file (fastText needs a file)
with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as fp:
    corpus_path = fp.name
    for line in sentences:
        fp.write(line + "\n")

# ─────────────────────────────────────────────
# 2️⃣  Train a word-embedding model (skipgram ≈ Word2Vec)
model = fasttext.train_unsupervised(
    corpus_path,
    model="skipgram",   # or "cbow"
    dim=50,             # vector size
    epoch=5,            # small for demo; raise for real data
    minCount=1,         # keep every word in this tiny corpus
)

# ─────────────────────────────────────────────
# 3️⃣  Inspect the learned vectors
print("\nVector for 'learning' (first 5 dims):")
print(model.get_word_vector("learning")[:5])

print("\nTop 3 words similar to 'language':")
for word, score in model.get_nearest_neighbors("language")[:3]:
    print(f"{word:10s}   cos-sim = {score:.3f}")

# ─────────────────────────────────────────────
# 4️⃣  Clean up temporary file
os.remove(corpus_path)