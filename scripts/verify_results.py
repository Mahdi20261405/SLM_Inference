"""Verify published descriptive summaries against archived CSVs. No model download required."""
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
R = BASE / "results"
mem = pd.read_csv(R / "memory_benchmark.csv").set_index("precision")
fp = pd.read_csv(R / "multiprompt_fp16_results.csv")
nf = pd.read_csv(R / "multiprompt_4bit_results.csv")
qfp = pd.read_csv(R / "extended_quality_fp16.csv")
qnf = pd.read_csv(R / "extended_quality_4bit.csv")
scores = pd.read_csv(R / "quality_scored_all_40.csv")
paired = pd.read_csv(R / "paired_quality_comparison.csv")
assert len(fp) == len(nf) == 40
assert len(qfp) == len(qnf) == 20
assert len(scores) == 40 and len(paired) == 20
print("GPU allocation delta (GiB):", mem["model_memory_gb"].to_dict())
print("NF4 model-load allocation reduction (%):", round(100*(1-mem.loc["4-bit NF4","model_memory_gb"]/mem.loc["FP16","model_memory_gb"]), 2))
for name, df in [("FP16",fp),("4-bit NF4",nf)]:
    print(name,"throughput",round(df.tokens_per_second.mean(),3),"latency",round(df.latency_seconds.mean(),3))
for name, df in [("FP16",qfp),("4-bit NF4",qnf)]:
    print(name,"within 100 words",int(df.within_100_words.sum()),"/",len(df),"normal ends",int(df.ended_normally.sum()))
print("Mean AI-assisted quality scores:",scores.groupby("precision").total_quality_score.mean().to_dict())
diff=paired["difference_FP16_minus_4bit"].to_numpy()
rng=np.random.default_rng(42)
means=np.array([rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(10000)])
print("Paired mean diff:",round(diff.mean(),3),"bootstrap CI:",np.percentile(means,[2.5,97.5]).round(3).tolist())
