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
# from optimizers.registry import OPTIMIZERS

# =============================
# SELECT OPTIMIZER BY INDEX
# =============================
OPTIMIZER_INDEX = 2   # <-- change ONLY this number

# optimizer_entry = OPTIMIZERS[OPTIMIZER_INDEX]
# optimizer_fn = optimizer_entry["fn"]

# print(f"[OPTIMIZER] Using {optimizer_entry['name']}")

# solutions = optimizer_fn(
#     jobs,
#     prompts,
#     probabilities,
#     compute_cost
# )


# # =============================
# # Evaluation Metrics (Table 3)
# # =============================

# # Pareto front for this optimizer
# pf = pareto_front(solutions)

# # Reference Pareto (for now, use self-reference)
# # Later this will be union of all optimizers
# reference_pf = pf

# igd = compute_igd(pf, reference_pf)
# delta = compute_delta(pf)
# mn = compute_mn(pf)

# print(
#     f"[METRICS] IGD={igd:.4f} | Δ={delta:.4f} | Mₙ={mn}"
# )


#16012026:0103
# =============================
# SIDE-BY-SIDE OPTIMIZER EVAL
# =============================

from optimizers.registry import OPTIMIZERS
from evaluation.metrics import (
    pareto_front,
    compute_igd,
    compute_delta,
    compute_mn
)

all_pareto_points = []
optimizer_pareto = {}

print("\n[OPTIMIZATION] Running optimizers side-by-side\n")

# --- run each optimizer ---
for opt in OPTIMIZERS:
    name = opt["name"]
    fn = opt["fn"]

    print(f"[OPTIMIZER] Running {name}")

    solutions = fn(
        jobs,
        prompts,
        probabilities,
        compute_cost
    )

    pf = pareto_front(solutions)

    optimizer_pareto[name] = pf
    all_pareto_points.extend(pf)

    print(f"[OPTIMIZER] {name} produced {len(pf)} Pareto solutions\n")

# --- build reference Pareto ---
reference_pf = pareto_front(all_pareto_points)

print("[REFERENCE] Global reference Pareto size:", len(reference_pf))
print("\n[TABLE 3] Optimizer Comparison")
print("-" * 50)
print(f"{'Algorithm':12s} | {'IGD':>8s} | {'Δ':>8s} | {'Mₙ':>5s}")
print("-" * 50)

# --- compute metrics ---
table_rows = []

for name, pf in optimizer_pareto.items():
    igd = compute_igd(pf, reference_pf)
    delta = compute_delta(pf)
    mn = compute_mn(pf)

    table_rows.append({
        "algorithm": name,
        "IGD": igd,
        "Delta": delta,
        "Mn": mn
    })

    print(f"{name:12s} | {igd:8.4f} | {delta:8.4f} | {mn:5d}")

print("-" * 50)



import pandas as pd

# Save optimized Pareto solutions
# pareto_df = pd.DataFrame(pf)
# pareto_df.to_csv(
#     "results/tables/optimized_pareto.csv",
#     index=False
# )


import pandas as pd

df_metrics = pd.DataFrame(table_rows)
df_metrics.to_csv(
    "results/tables/table3_optimizer_comparison.csv",
    index=False
)

print("[DONE] Table 3 metrics saved to results/tables/table3_optimizer_comparison.csv")


# Plot
plot_pareto(
    solutions,
    pf,
    baseline_results,
    "results/figures/pareto_with_baselines.png"
)


print("[DONE] Pareto plot saved")


