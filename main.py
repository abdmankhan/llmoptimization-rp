from jobs.loader import load_hdfs_jobs
from prompts.templates import PROMPT_TEMPLATES
from predictors.mlbp import train_predictor, predict_probabilities, create_training_data
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

# =============================
# PREDICTOR CONFIG
# =============================
PREDICTOR_TYPE = "xgb"   # "rf" or "xgb"

# Train predictor
# Create training data
X, y = create_training_data(jobs, prompts)

# Train predictor (RF or XGB)
model = train_predictor(X, y, predictor_type=PREDICTOR_TYPE)


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

# from optimizers.registry import OPTIMIZERS
# from evaluation.metrics import (
#     pareto_front,
#     compute_igd,
#     compute_delta,
#     compute_mn
# )

# all_pareto_points = []
# optimizer_pareto = {}

# print("\n[OPTIMIZATION] Running optimizers side-by-side\n")

# # --- run each optimizer ---
# for opt in OPTIMIZERS:
#     name = opt["name"]
#     fn = opt["fn"]

#     print(f"[OPTIMIZER] Running {name}")

#     solutions = fn(
#         jobs,
#         prompts,
#         probabilities,
#         compute_cost
#     )

#     pf = pareto_front(solutions)

#     optimizer_pareto[name] = pf
#     all_pareto_points.extend(pf)

#     print(f"[OPTIMIZER] {name} produced {len(pf)} Pareto solutions\n")

# # --- build reference Pareto ---
# reference_pf = pareto_front(all_pareto_points)

# print("[REFERENCE] Global reference Pareto size:", len(reference_pf))
# print("\n[TABLE 3] Optimizer Comparison")
# print("-" * 50)
# print(f"{'Algorithm':12s} | {'IGD':>8s} | {'Δ':>8s} | {'Mₙ':>5s}")
# print("-" * 50)

# # --- compute metrics ---
# table_rows = []

# for name, pf in optimizer_pareto.items():
#     igd = compute_igd(pf, reference_pf)
#     delta = compute_delta(pf)
#     mn = compute_mn(pf)

#     table_rows.append({
#         "algorithm": name,
#         "IGD": igd,
#         "Delta": delta,
#         "Mn": mn
#     })

#     print(f"{name:12s} | {igd:8.4f} | {delta:8.4f} | {mn:5d}")

# print("-" * 50)



# import pandas as pd

# # Save optimized Pareto solutions
# # pareto_df = pd.DataFrame(pf)
# # pareto_df.to_csv(
# #     "results/tables/optimized_pareto.csv",
# #     index=False
# # )


# import pandas as pd

# df_metrics = pd.DataFrame(table_rows)
# df_metrics.to_csv(
#     "results/tables/table3_optimizer_comparison.csv",
#     index=False
# )

# print("[DONE] Table 3 metrics saved to results/tables/table3_optimizer_comparison.csv")
NUM_RUNS = 3


OPTIMIZER_NAMES = ["Random Search", "NSGA-II", "SPEA2"]

from collections import defaultdict
import numpy as np
import pandas as pd

from optimizers.registry import OPTIMIZERS
from evaluation.metrics import (
    pareto_front,
    compute_igd,
    compute_delta,
    compute_mn
)

# Filter optimizers by name (keeps code stable)
name_to_fn = {o["name"]: o["fn"] for o in OPTIMIZERS}
selected_opts = {name: name_to_fn[name] for name in OPTIMIZER_NAMES}

# Store metrics per run
metrics_runs = defaultdict(list)

print("\n[EXPERIMENT] Multi-run evaluation started\n")

for run_id in range(1, NUM_RUNS + 1):
    print(f"\n[RUN {run_id}/{NUM_RUNS}]")

    all_pareto_points = []
    run_pareto = {}

    # --- run each optimizer ---
    for name, fn in selected_opts.items():
        print(f"[OPTIMIZER] Running {name}")

        solutions = fn(
            jobs,
            prompts,
            probabilities,
            compute_cost
        )

        pf = pareto_front(solutions)
        run_pareto[name] = pf
        all_pareto_points.extend(pf)

        print(f"[OPTIMIZER] {name} Pareto size: {len(pf)}")

    # --- reference Pareto for this run ---
    reference_pf = pareto_front(all_pareto_points)
    print(f"[REFERENCE] Run {run_id} reference Pareto size: {len(reference_pf)}")

    # --- compute metrics for this run ---
    for name, pf in run_pareto.items():
        igd = compute_igd(pf, reference_pf)
        delta = compute_delta(pf)
        mn = compute_mn(pf)

        metrics_runs[name].append({
            "IGD": igd,
            "Delta": delta,
            "Mn": mn
        })

        print(
            f"[RUN {run_id}] {name:12s} | "
            f"IGD={igd:.4f} | Δ={delta:.4f} | Mₙ={mn}"
        )

# =============================
# AGGREGATE RESULTS (MEAN ± STD)
# =============================

rows = []

print("\n[TABLE 3] Mean ± Std over runs")
print("-" * 60)
print(f"{'Algorithm':12s} | {'IGD (μ±σ)':>16s} | {'Δ (μ±σ)':>16s} | {'Mₙ (μ±σ)':>16s}")
print("-" * 60)

for name, values in metrics_runs.items():
    igds = np.array([v["IGD"] for v in values])
    deltas = np.array([v["Delta"] for v in values])
    mns = np.array([v["Mn"] for v in values])

    row = {
        "algorithm": name,
        "IGD_mean": igds.mean(),
        "IGD_std": igds.std(ddof=1) if len(igds) > 1 else 0.0,
        "Delta_mean": deltas.mean(),
        "Delta_std": deltas.std(ddof=1) if len(deltas) > 1 else 0.0,
        "Mn_mean": mns.mean(),
        "Mn_std": mns.std(ddof=1) if len(mns) > 1 else 0.0,
    }
    rows.append(row)

    print(
        f"{name:12s} | "
        f"{row['IGD_mean']:.4f}±{row['IGD_std']:.4f} | "
        f"{row['Delta_mean']:.4f}±{row['Delta_std']:.4f} | "
        f"{row['Mn_mean']:.1f}±{row['Mn_std']:.1f}"
    )

print("-" * 60)

# Save Table 3
df = pd.DataFrame(rows)
df.to_csv("results/tables/table3_optimizer_comparison_mean_std.csv", index=False)
print("[DONE] Table 3 (mean ± std) saved to results/tables/table3_optimizer_comparison_mean_std.csv")


# Plot
plot_pareto(
    solutions,
    pf,
    baseline_results,
    "results/figures/pareto_with_baselines.png"
)


print("[DONE] Pareto plot saved")


