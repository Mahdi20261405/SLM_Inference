# Engineering Practices for Efficient and Reliable Small Language Model Inference

**A reproducibility-oriented case study of FP16 vs. 4-bit NF4 inference**  
Independent research project — Seyedmahdi Masoumi, October 2026.

This repository compares `Qwen/Qwen2.5-0.5B-Instruct` in FP16 and bitsandbytes NF4 4-bit mode on an NVIDIA Tesla T4 in Google Colab. The contribution is a documented engineering case study, **not** a new quantization algorithm or a universal performance claim.

## Original experiment

| Metric | FP16 | 4-bit NF4 |
|---|---:|---:|
| PyTorch model-loading allocated-memory delta (GiB) | 0.9255 | 0.4421 |
| Mean throughput (generated tokens/s, 40 runs each) | 29.753 | 19.936 |
| Mean generation-call latency (s, 40 runs each) | 2.152 | 3.248 |
| Within 100 whitespace-separated words (of 20) | 10 | 17 |
| Ended with recognized end token (of 20) | 18 | 20 |
| Exploratory AI-assisted quality score (0–4) | 3.1 | 3.0 |

NF4 used approximately 52.2% less **PyTorch-reported allocated memory during model loading**, but was slower on the original benchmark. This memory measure is **not** total GPU VRAM, peak inference memory, power, or energy. The AI-assisted subjective scoring was neither independently human-rated nor fully blinded. Its mean paired difference was 0.10/4 (bootstrap 95% CI [-0.35, 0.55]); see `docs/QUALITY_EVALUATION.md`.

## Independent fresh-Colab reproduction (October 2026)

| Metric | FP16 | 4-bit NF4 |
|---|---:|---:|
| Model-loading allocated-memory delta (GiB) | 0.9272 | 0.4416 |
| Mean throughput (generated tokens/s, 40 runs each) | 30.716 | 23.420 |
| Within 100 words (of 20) | 10 | 17 |
| Ended with recognized end token (of 20) | 18 | 20 |

A separate Colab runtime reproduced the **direction** of the memory and speed trade-offs and the aggregate instruction-compliance counts. Exact timing was not identical. The second run did **not** independently repeat the historical AI-assisted quality ratings. Reproduction CSVs are stored under `reproduction_2026/` and **must not be merged into or substituted for** the original `results/` data.

## Repository layout

- `notebooks/reproduction_clean.ipynb` — cleaned notebook, executed end-to-end in a fresh Colab session; published copy has outputs cleared.
- `data/` — original 20 public base prompts and their length-instruction variants.
- `results/` — **original** experimental measurements and exploratory AI-assisted ratings.
- `reproduction_2026/data/` and `reproduction_2026/results/` — separate independent reproduction inputs and results.
- `figures/` — figures from the original report.
- `report/research_report.pdf` — October 2026 research report (14 pages).
- `scripts/verify_results.py` — recomputes descriptive statistics from original CSVs without a GPU.
- `docs/QUALITY_EVALUATION.md` — scoring methodology and limitations.

## Verify original descriptive results without a GPU

```bash
python -m pip install pandas numpy
python scripts/verify_results.py
```

## Re-run inference on a GPU

1. Open `notebooks/reproduction_clean.ipynb` in Google Colab with a GPU runtime (Tesla T4 was used).
2. Execute cells in order; the notebook installs pinned versions of Transformers, Accelerate, bitsandbytes, and pandas. The Colab-provided PyTorch/CUDA environment can change.
3. When prompted, upload `data/benchmark_prompts.csv` into the Colab working directory. The notebook derives the extended prompts from this same base set.
4. Export the new CSVs separately. The notebook does not re-create the historical AI-assisted subjective scores.

Model repository revision was not pinned in the original experiment, and exact numerical replication is not guaranteed.

## Scope, provenance and limitations

The benchmark uses a single model, one GPU type, 20 purpose-built English AI/software-engineering prompts and sequential rather than randomized precision runs. Generation latency excludes production serving overhead. No energy, carbon, or digital-sovereignty outcomes were directly measured. The public materials are a curated derivative of a separate private research archive; no private MSc thesis material is intended to be included.
