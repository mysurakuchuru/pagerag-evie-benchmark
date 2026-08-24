from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "BAAI/bge-m3"

model = SentenceTransformer(MODEL_NAME)

query = "Did JPMorgan Chase repurchase more than half of its $30 billion stock repurchase program?"

sentences = [
    "JPMorgan Chase repurchased $18.841 billion of common stock in 2024.",
    "JPMorgan Chase paid $1.3 billion in preferred stock dividends.",
    "The company discussed long-term debt and funding."
]

query_embedding = model.encode(
    query,
    normalize_embeddings=True
)

sentence_embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

scores = sentence_embeddings @ query_embedding

print("\n--- SIMILARITY RESULTS ---")

for sentence, score in sorted(
    zip(sentences, scores),
    key=lambda x: x[1],
    reverse=True
):
    print(f"\nScore: {score:.4f}")
    print(sentence)
