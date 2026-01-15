from jobs.loader import load_hdfs_jobs
from prompts.templates import PROMPT_TEMPLATES
from predictors.mlbp import train_predictor, predict_probabilities
from optimizers.random_search import random_search
from evaluation.baseline import single_prompt_baseline
from evaluation.pareto import pareto_front
from visualization.plots import plot_pareto
from cost.cost_model import compute_cost
from evaluation.metrics import (
    pareto_front,
    compute_igd,
    compute_delta,
    compute_mn
)



# Load jobs
jobs = load_hdfs_jobs(
    "hdfs_data/HDFS.log",
    "hdfs_data/preprocessed/anomaly_label.csv",
    max_jobs=3000
)

prompts = PROMPT_TEMPLATES

# Train predictor
model = train_predictor(jobs, prompts)
probabilities = predict_probabilities(model, jobs, prompts)

# Baseline (RQ1)
# baseline = single_prompt_baseline(jobs, prompts[1], probabilities)
# print("[BASELINE]", baseline)
import pandas as pd

baseline_results = []

for prompt in prompts:
    result = single_prompt_baseline(jobs, prompt, probabilities)
    baseline_results.append(result)
    print("[BASELINE]", result)

baseline_df = pd.DataFrame(baseline_results)
baseline_df.to_csv(
    "results/tables/baseline_results.csv",
    index=False
)


# Optimization
# solutions = random_search(jobs, prompts, probabilities)
from optimizers.registry import OPTIMIZERS

# =============================
# SELECT OPTIMIZER BY INDEX
# =============================
OPTIMIZER_INDEX = 2   # <-- change ONLY this number

optimizer_entry = OPTIMIZERS[OPTIMIZER_INDEX]
optimizer_fn = optimizer_entry["fn"]

print(f"[OPTIMIZER] Using {optimizer_entry['name']}")

solutions = optimizer_fn(
    jobs,
    prompts,
    probabilities,
    compute_cost
)


# =============================
# Evaluation Metrics (Table 3)
# =============================

# Pareto front for this optimizer
pf = pareto_front(solutions)

# Reference Pareto (for now, use self-reference)
# Later this will be union of all optimizers
reference_pf = pf

igd = compute_igd(pf, reference_pf)
delta = compute_delta(pf)
mn = compute_mn(pf)

print(
    f"[METRICS] IGD={igd:.4f} | Δ={delta:.4f} | Mₙ={mn}"
)

import pandas as pd

# Save optimized Pareto solutions
pareto_df = pd.DataFrame(pf)
pareto_df.to_csv(
    "results/tables/optimized_pareto.csv",
    index=False
)


# Plot
plot_pareto(
    solutions,
    pf,
    baseline_results,
    "results/figures/pareto_with_baselines.png"
)


print("[DONE] Pareto plot saved")


