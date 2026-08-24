# Single-query comparison: BGE-M3 vs EVIE

## Query
Did JPMorganChase execute more than half of its planned $30 billion stock repurchase program by year-end?

## Benchmark setup
- Dataset: `vidore/vidore_v3_finance_en`
- Candidate pages: 100
- Ground-truth corpus ID: 10
- Ground-truth PDF page: 107
- Document: `jpmorgan_chase_2024`

## Result
- **BGE-M3 dense baseline** ranked the correct page at **#17**
- **EVIE-Preview-4.5B** ranked the correct page at **#1**

## Takeaway
On this real financial-PDF retrieval example, the visual retrieval model (EVIE) significantly outperformed the text-only dense baseline (BGE-M3).
