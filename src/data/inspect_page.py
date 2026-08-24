from datasets import load_dataset

DATASET_NAME = "vidore/vidore_v3_finance_en"
TARGET_CORPUS_ID = 10

print("Loading document corpus...")

corpus = load_dataset(
    DATASET_NAME,
    "corpus",
    split="test",
    streaming=True,
)

target_page = None

for page in corpus:
    if page["corpus_id"] == TARGET_CORPUS_ID:
        target_page = page
        break

if target_page is None:
    raise ValueError(f"Corpus ID {TARGET_CORPUS_ID} not found")

print("\n--- PAGE INFORMATION ---")
print("Corpus ID:", target_page["corpus_id"])
print("Document:", target_page["doc_id"])
print("Page number:", target_page["page_number_in_doc"])

print("\n--- FULL EXTRACTED PAGE CONTENT ---")
print(target_page["markdown"])

# Save the actual page image
output_path = "results/sample_relevant_page.png"
target_page["image"].save(output_path)

print("\nSaved page image to:", output_path)
