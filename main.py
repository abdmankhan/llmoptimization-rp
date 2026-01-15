from jobs.loader import load_hdfs_jobs
from prompts.templates import PROMPT_TEMPLATES
from predictors.mlbp import train_predictor, predict_probabilities
from optimizers.random_search import random_search
from evaluation.baseline import single_prompt_baseline
from evaluation.pareto import pareto_front
from visualization.plots import plot_pareto


# Load jobs
jobs = load_hdfs_jobs(
    "hdfs_data/HDFS.log",
    "hdfs_data/preprocessed/anomaly_label.csv",
    max_jobs=10000
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
solutions = random_search(jobs, prompts, probabilities)
pareto = pareto_front(solutions)
import pandas as pd

# Save optimized Pareto solutions
pareto_df = pd.DataFrame(pareto)
pareto_df.to_csv(
    "results/tables/optimized_pareto.csv",
    index=False
)


# Plot
plot_pareto(
    solutions,
    pareto,
    baseline_results,
    "results/figures/pareto_with_baselines.png"
)


print("[DONE] Pareto plot saved")
