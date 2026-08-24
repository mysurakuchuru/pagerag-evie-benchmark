from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"

print("Loading BGE-M3...")

model = SentenceTransformer(MODEL_NAME)

text = "JPMorgan Chase authorized a $30 billion common share repurchase program."

print("\nCreating embedding...")

embedding = model.encode(
    text,
    normalize_embeddings=True
)

print("\n--- RESULT ---")
print("Embedding shape:", embedding.shape)
print("First 10 values:")
print(embedding[:10])
