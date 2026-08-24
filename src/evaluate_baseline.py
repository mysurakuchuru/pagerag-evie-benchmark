from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import csv
import json
import math
import time

import numpy as np


DATASET_NAME = "vidore/vidore_v3_finance_en"
MODEL_NAME = "BAAI/bge-m3"

CORPUS_FILE = "data/sample_corpus.jsonl"
OUTPUT_FILE = "results/bge_m3_10query_results.csv"

NUM_PAGES = 100
NUM_QUERIES = 10


# --------------------------------------------------
# Metric functions
# --------------------------------------------------

def recall_at_k(ranked_ids, relevance, k):

    relevant_ids = set(relevance.keys())

    retrieved = set(ranked_ids[:k])

    hits = len(relevant_ids.intersection(retrieved))

    return hits / len(relevant_ids)


def dcg_at_k(ranked_ids, relevance, k):

    dcg = 0.0

    for rank, corpus_id in enumerate(ranked_ids[:k], start=1):

        rel = relevance.get(corpus_id, 0)

        if rel > 0:

            dcg += (2 ** rel - 1) / math.log2(rank + 1)

    return dcg


def ndcg_at_k(ranked_ids, relevance, k):

    actual_dcg = dcg_at_k(
        ranked_ids,
        relevance,
        k
    )

    ideal_relevances = sorted(
        relevance.values(),
        reverse=True
    )

    ideal_dcg = 0.0

    for rank, rel in enumerate(
        ideal_relevances[:k],
        start=1
    ):

        ideal_dcg += (
            (2 ** rel - 1)
            / math.log2(rank + 1)
        )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# --------------------------------------------------
# 1. Load local 100-page corpus
# --------------------------------------------------

print("Loading local corpus...")

pages = []

with open(
    CORPUS_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        pages.append(json.loads(line))

        if len(pages) >= NUM_PAGES:
            break


corpus_ids = [
    page["corpus_id"]
    for page in pages
]

page_texts = [
    page["text"]
    for page in pages
]

print(f"Loaded {len(pages)} pages.")


# --------------------------------------------------
# 2. Load qrels including relevance scores
# --------------------------------------------------

print("\nLoading relevance judgments...")

qrels_dataset = load_dataset(
    DATASET_NAME,
    "qrels",
    split="test",
    streaming=True
)


qrels = {}


for row in qrels_dataset:

    query_id = row["query_id"]
    corpus_id = row["corpus_id"]
    relevance_score = row["score"]

    if corpus_id in corpus_ids:

        if query_id not in qrels:
            qrels[query_id] = {}

        qrels[query_id][corpus_id] = relevance_score


print(
    f"Found {len(qrels)} queries with relevant pages "
    "inside the sample corpus."
)


# --------------------------------------------------
# 3. Select benchmark queries
# --------------------------------------------------

print("\nLoading queries...")

queries_dataset = load_dataset(
    DATASET_NAME,
    "queries",
    split="test",
    streaming=True
)


queries = []


for row in queries_dataset:

    query_id = row["query_id"]

    if query_id in qrels:

        queries.append(
            {
                "query_id": query_id,
                "query": row["query"],
                "relevance": qrels[query_id],
            }
        )

    if len(queries) >= NUM_QUERIES:
        break


print(
    f"Selected {len(queries)} benchmark queries."
)


# --------------------------------------------------
# 4. Load model
# --------------------------------------------------

print("\nLoading BGE-M3...")

model = SentenceTransformer(
    MODEL_NAME
)


# --------------------------------------------------
# 5. Build document embedding index
# --------------------------------------------------

print("\nCreating document embeddings...")

index_start = time.perf_counter()


page_embeddings = model.encode(
    page_texts,
    normalize_embeddings=True,
    batch_size=1,
    show_progress_bar=True
)


index_time = time.perf_counter() - index_start


index_bytes = page_embeddings.nbytes
index_mb = index_bytes / (1024 ** 2)


print(
    f"\nDocument embedding shape: "
    f"{page_embeddings.shape}"
)

print(
    f"Embedding index size: "
    f"{index_mb:.2f} MB"
)

print(
    f"Indexing time: "
    f"{index_time:.2f} seconds"
)


# --------------------------------------------------
# 6. Run benchmark
# --------------------------------------------------

results = []

query_latencies = []


print("\nRunning retrieval benchmark...")


for i, item in enumerate(
    queries,
    start=1
):

    query_id = item["query_id"]
    query_text = item["query"]
    relevance = item["relevance"]

    # ----------------------------------------------
    # Start query timer
    # ----------------------------------------------

    start = time.perf_counter()


    query_embedding = model.encode(
        query_text,
        normalize_embeddings=True
    )


    scores = (
        page_embeddings
        @ query_embedding
    )


    ranked_indices = np.argsort(
        scores
    )[::-1]


    latency = (
        time.perf_counter()
        - start
    )

    query_latencies.append(latency)


    ranked_ids = [
        pages[index]["corpus_id"]
        for index in ranked_indices
    ]


    # ----------------------------------------------
    # First relevant page rank
    # ----------------------------------------------

    first_relevant_rank = None


    for rank, corpus_id in enumerate(
        ranked_ids,
        start=1
    ):

        if corpus_id in relevance:

            first_relevant_rank = rank
            break


    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    hit_at_1 = int(
        first_relevant_rank <= 1
    )

    hit_at_5 = int(
        first_relevant_rank <= 5
    )

    hit_at_10 = int(
        first_relevant_rank <= 10
    )


    recall_5 = recall_at_k(
        ranked_ids,
        relevance,
        5
    )

    recall_10 = recall_at_k(
        ranked_ids,
        relevance,
        10
    )


    ndcg_5 = ndcg_at_k(
        ranked_ids,
        relevance,
        5
    )

    ndcg_10 = ndcg_at_k(
        ranked_ids,
        relevance,
        10
    )


    reciprocal_rank = (
        1 / first_relevant_rank
    )


    print("\n" + "-" * 70)

    print(
        f"Query {i}/{len(queries)}"
    )

    print(
        "Query ID:",
        query_id
    )

    print(
        "Question:",
        query_text
    )

    print(
        "Relevant pages:",
        relevance
    )

    print(
        "First relevant rank:",
        first_relevant_rank
    )

    print(
        f"Recall@5={recall_5:.3f} | "
        f"Recall@10={recall_10:.3f}"
    )

    print(
        f"nDCG@5={ndcg_5:.3f} | "
        f"nDCG@10={ndcg_10:.3f}"
    )

    print(
        f"Latency={latency:.4f}s"
    )


    results.append(
        {
            "query_id": query_id,
            "query": query_text,
            "first_relevant_rank": first_relevant_rank,
            "hit_at_1": hit_at_1,
            "hit_at_5": hit_at_5,
            "hit_at_10": hit_at_10,
            "recall_at_5": recall_5,
            "recall_at_10": recall_10,
            "ndcg_at_5": ndcg_5,
            "ndcg_at_10": ndcg_10,
            "reciprocal_rank": reciprocal_rank,
            "query_latency_seconds": latency,
        }
    )


# --------------------------------------------------
# 7. Aggregate benchmark metrics
# --------------------------------------------------

hit_1 = np.mean(
    [r["hit_at_1"] for r in results]
)

hit_5 = np.mean(
    [r["hit_at_5"] for r in results]
)

hit_10 = np.mean(
    [r["hit_at_10"] for r in results]
)


recall_5 = np.mean(
    [r["recall_at_5"] for r in results]
)

recall_10 = np.mean(
    [r["recall_at_10"] for r in results]
)


ndcg_5 = np.mean(
    [r["ndcg_at_5"] for r in results]
)

ndcg_10 = np.mean(
    [r["ndcg_at_10"] for r in results]
)


mrr = np.mean(
    [r["reciprocal_rank"] for r in results]
)


avg_latency = np.mean(
    query_latencies
)


# --------------------------------------------------
# 8. Print final benchmark
# --------------------------------------------------

print("\n" + "=" * 70)

print("BGE-M3 DENSE BASELINE")

print("=" * 70)


print(
    f"Pages indexed:    {len(pages)}"
)

print(
    f"Queries:          {len(results)}"
)


print("\nRetrieval quality")

print(
    f"Hit@1:            {hit_1:.3f}"
)

print(
    f"Hit@5:            {hit_5:.3f}"
)

print(
    f"Hit@10:           {hit_10:.3f}"
)

print(
    f"Recall@5:         {recall_5:.3f}"
)

print(
    f"Recall@10:        {recall_10:.3f}"
)

print(
    f"nDCG@5:           {ndcg_5:.3f}"
)

print(
    f"nDCG@10:          {ndcg_10:.3f}"
)

print(
    f"MRR:              {mrr:.3f}"
)


print("\nPerformance")

print(
    f"Avg query latency: "
    f"{avg_latency:.4f} seconds"
)

print(
    f"Indexing time:     "
    f"{index_time:.2f} seconds"
)

print(
    f"Embedding size:    "
    f"{index_mb:.2f} MB"
)


# --------------------------------------------------
# 9. Save results
# --------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()

    writer.writerows(results)


print(
    f"\nDetailed results saved to: "
    f"{OUTPUT_FILE}"
)



























