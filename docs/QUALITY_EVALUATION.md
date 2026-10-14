# Exploratory quality evaluation: important caveats

The 40 ratings in `results/quality_scored_all_40.csv` were assigned with AI assistance, **not by independent human annotators**. Two identifier mappings were used: for prompts 1–5, IDs were assigned before shuffling and could reveal configuration; for prompts 6–20, IDs were assigned after shuffling. The final scores were matched back to model and prompt IDs. These are historical exploratory observations, not independently validated ground truth.

For the 100-word limit, words were counted using Python `len(answer.split())`. Bootstrap resampling (10,000 samples, seed 42) was performed over the 20 paired prompt-level score differences. This does not quantify evaluator bias or justify a claim of equivalent quality.

The original review mappings and pilot files are preserved in the **private research archive**, not included in this curated public-preview package. The scores cannot be independently re-created without a new documented evaluation process.
