from datasets import load_dataset

DATASET_NAME = "vidore/vidore_v3_finance_en"

print("Loading queries...")

queries = load_dataset(
    DATASET_NAME,
    "queries",
    split="test",
    streaming=True,
)

query = next(iter(queries))

print("\n--- SAMPLE QUERY ---")
print("Query ID:", query["query_id"])
print("Query:", query["query"])
print("Language:", query["language"])
print("Content type:", query["content_type"])

print("\nLoading relevance labels (qrels)...")

qrels = load_dataset(
    DATASET_NAME,
    "qrels",
    split="test",
    streaming=True,
)

relevant_pages = [
    row
    for row in qrels
    if row["query_id"] == query["query_id"]
]

print("\n--- CORRECT PAGES ---")

for row in relevant_pages:
    print(
        "Corpus ID:",
        row["corpus_id"],
        "| Relevance score:",
        row["score"],
        "| Content type:",
        row["content_type"],
    )
