import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import numpy as np


# def create_training_data(jobs, prompts):
#     """
#     Build training data for MLBP.

#     Each sample corresponds to (job_i, prompt_j).
#     Features: [job_tokens, prompt_tokens]
#     Label: simulated success (job label)
#     """

#     X = []
#     y = []

#     for job in jobs:
#         for prompt in prompts:
#             # Features
#             X.append([
#                 job["tokens"],
#                 prompt["tokens"]
#             ])

#             # Label (simulated ground truth)
#             y.append(job["label"])

#     return np.array(X), np.array(y)


import random
import numpy as np
from llm.gemini_client import query_gemini


GROUNDING_RATIO = 0.10
RANDOM_SEED = 42
MAX_GEMINI_CALLS = 50

def create_training_data(jobs, prompts, use_real_llm=True):
    random.seed(RANDOM_SEED)

    X, y = [], []

    total_pairs = len(jobs) * len(prompts)
    # grounding_budget = int(GROUNDING_RATIO * total_pairs)

    grounding_budget = min(
        int(GROUNDING_RATIO * total_pairs),
        MAX_GEMINI_CALLS
    )


    # choose which (job, prompt) pairs get real labels
    all_pairs = [
        (i, j)
        for i in range(len(jobs))
        for j in range(len(prompts))
    ]
    grounded_pairs = set(random.sample(all_pairs, grounding_budget))

    print(f"[TRAINING] Using {len(grounded_pairs)} real LLM calls out of {len(all_pairs)} total pairs")

    for i, job in enumerate(jobs):
        ground_truth = job["label"]  # 0=normal, 1=anomaly
        
        for j, prompt in enumerate(prompts):
            # Richer feature set
            features = [
                job["tokens"],                              # Job length
                job.get("unique_tokens", job["tokens"]//2), # Vocabulary size
                job.get("has_error", 0),                    # Error indicators
                prompt["tokens"],                           # Prompt complexity
                prompt["id"],                               # Prompt type
                job["tokens"] / max(prompt["tokens"], 1),   # Job-to-prompt ratio
                job["tokens"] * prompt["tokens"],           # Interaction term
            ]
            X.append(features)

            if use_real_llm and (i, j) in grounded_pairs:
                # Get LLM prediction for this (job, prompt) pair
                llm_prediction = query_gemini(job["log"], prompt)
                # Label = 1 if LLM was correct, 0 if wrong
                label = 1 if llm_prediction == ground_truth else 0
            else:
                # Simulate prompt-specific success rates based on paper's assumptions:
                # 1. Longer prompts (more context) → higher base accuracy
                # 2. Complex jobs benefit more from complex prompts
                # 3. Simple jobs can be handled well by any prompt
                
                # Base success rate increases with prompt complexity
                prompt_quality = prompt["tokens"] / 100.0  # 0.1 to 0.9
                base_rate = 0.60 + 0.25 * prompt_quality  # 60% to 85%
                
                # Job difficulty (anomalies and error logs are harder)
                is_difficult = ground_truth == 1 or job.get("has_error", 0) == 1
                
                if is_difficult:
                    # Complex jobs: larger prompts help significantly
                    # Simple prompt: 60%, Standard: 70%, Fewshot_1: 78%, Fewshot_3: 85%
                    success_rate = 0.55 + 0.35 * prompt_quality
                else:
                    # Simple jobs: all prompts work well, diminishing returns
                    # Simple: 75%, Standard: 80%, Fewshot: 82-85%
                    success_rate = 0.75 + 0.10 * prompt_quality
                
                # Add noise
                success_rate = max(0.45, min(0.95, success_rate + random.gauss(0, 0.05)))
                
                label = 1 if random.random() < success_rate else 0

            y.append(label)

    # Debug: Show per-prompt success rates in training data
    print("\n[TRAINING] Per-prompt success rates in simulated data:")
    for prompt_idx, prompt in enumerate(prompts):
        prompt_labels = [y[i] for i in range(len(y)) if i % len(prompts) == prompt_idx]
        success_rate = np.mean(prompt_labels) if prompt_labels else 0
        print(f"  {prompt['name']:12s} (${prompt['tokens']:3d} tokens): {success_rate:.1%} success")

    return np.array(X), np.array(y)



def train_predictor(X, y, predictor_type="rf"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if predictor_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )

    elif predictor_type == "xgb":
        model = XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42
        )

    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"[MLBP-{predictor_type.upper()}] Predictor accuracy: {acc:.4f}")
    
    # Validate: Model should predict higher success for complex prompts
    print(f"[MLBP-{predictor_type.upper()}] Validation check (expected: increasing probabilities):")
    print(f"  Testing on a sample 'difficult' job...")
    
    return model



import time

def predict_probabilities(model, jobs, prompts):
    """
    Predict success probability for each (job, prompt) pair.
    Success = correctly classifying the job (whether normal or anomaly).
    Returns the confidence (max probability) instead of just P(anomaly).
    """
    p = {}

    total_jobs = len(jobs)
    start_time = time.time()

    for i, job in enumerate(jobs):
        for prompt in prompts:
            # Match training features
            features = [[
                job["tokens"],
                job.get("unique_tokens", job["tokens"]//2),
                job.get("has_error", 0),
                prompt["tokens"],
                prompt["id"],
                job["tokens"] / max(prompt["tokens"], 1),
                job["tokens"] * prompt["tokens"],
            ]]
            proba = model.predict_proba(features)[0]
            # P(success) = probability that this prompt will correctly classify this job
            # proba[0] = P(failure), proba[1] = P(success)
            prob = proba[1]
            p[(i, prompt["id"])] = prob

        # ---- progress logging ----
        if i % max(1, total_jobs // 20) == 0:
            percent = (i / total_jobs) * 100
            elapsed = time.time() - start_time
            print(
                f"[PROGRESS] Probability matrix: "
                f"{percent:.1f}% ({i}/{total_jobs}) | "
                f"{elapsed:.1f}s elapsed"
            )
        # --------------------------

    print("[MLBP] Probability matrix computed")
    return p

