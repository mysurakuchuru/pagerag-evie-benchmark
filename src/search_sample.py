from sentence_transformers import SentenceTransformer
import json
import numpy as np


MODEL_NAME = "BAAI/bge-m3"
CORPUS_FILE = "data/sample_corpus.jsonl"
NUM_PAGES = 100
TOP_K = 10


QUERY = (
    "Did JPMorganChase execute more than half of its planned "
    "$30 billion stock repurchase program by year-end?"
)


# --------------------------------------------------
# 1. Load our locally saved document pages
# --------------------------------------------------

pages = []

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        pages.append(json.loads(line))

        if len(pages) >= NUM_PAGES:
            break


print(f"Loaded {len(pages)} document pages.")


# --------------------------------------------------
# 2. Extract the page text
# --------------------------------------------------

page_texts = [page["text"] for page in pages]


# --------------------------------------------------
# 3. Load BGE-M3
# --------------------------------------------------

print("\nLoading BGE-M3...")

model = SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# 4. Convert PDF-page text into embeddings
# --------------------------------------------------

print("\nCreating page embeddings...")

page_embeddings = model.encode(
    page_texts,
    normalize_embeddings=True,
    batch_size=1,
    show_progress_bar=True,
)


# --------------------------------------------------
# 5. Convert the user's question into an embedding
# --------------------------------------------------

print("\nCreating query embedding...")

query_embedding = model.encode(
    QUERY,
    normalize_embeddings=True,
)


# --------------------------------------------------
# 6. Calculate similarity
# --------------------------------------------------

scores = page_embeddings @ query_embedding


# --------------------------------------------------
# 7. Rank pages
# --------------------------------------------------

ranked_indices = np.argsort(scores)[::-1]


# --------------------------------------------------
# 8. Display top results
# --------------------------------------------------

print("\n" + "=" * 70)
print("QUERY")
print("=" * 70)

print(QUERY)


print("\n" + "=" * 70)
print(f"TOP {TOP_K} RETRIEVED PAGES")
print("=" * 70)


for rank, index in enumerate(ranked_indices[:TOP_K], start=1):

    page = pages[index]
    score = scores[index]

    print(f"\nRANK #{rank}")
    print(f"Similarity score: {score:.4f}")
    print(f"Corpus ID: {page['corpus_id']}")
    print(f"Document: {page['doc_id']}")
    print(f"PDF page: {page['page_number']}")

    preview = page["text"][:300].replace("\n", " ")

    print(f"Preview: {preview}...")
    print("-" * 70)


# --------------------------------------------------
# 9. Check where the known correct page ranked
# --------------------------------------------------

GROUND_TRUTH_CORPUS_ID = 10

ground_truth_rank = None

for rank, index in enumerate(ranked_indices, start=1):

    if pages[index]["corpus_id"] == GROUND_TRUTH_CORPUS_ID:
        ground_truth_rank = rank
        break


print("\n" + "=" * 70)
print("GROUND TRUTH CHECK")
print("=" * 70)

print("Expected relevant Corpus ID:", GROUND_TRUTH_CORPUS_ID)
print("BGE-M3 rank:", ground_truth_rank)
