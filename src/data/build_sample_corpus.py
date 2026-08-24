from datasets import load_dataset
import json
from pathlib import Path

DATASET_NAME = "vidore/vidore_v3_finance_en"
NUM_PAGES = 100

OUTPUT_FILE = Path("data/sample_corpus.jsonl")

print(f"Loading first {NUM_PAGES} document pages...")

corpus = load_dataset(
    DATASET_NAME,
    "corpus",
    split="test",
    streaming=True,
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

count = 0

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for page in corpus:
        record = {
            "corpus_id": page["corpus_id"],
            "doc_id": page["doc_id"],
            "page_number": page["page_number_in_doc"],
            "text": page["markdown"],
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        count += 1

        if count >= NUM_PAGES:
            break

print(f"\nSaved {count} pages to:")
print(OUTPUT_FILE)

print("\nSample corpus is ready.")
