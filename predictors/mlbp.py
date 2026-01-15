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

import numpy as np


def create_training_data(jobs, prompts):
    """
    Build training data for MLBP.

    Each sample corresponds to (job_i, prompt_j).
    Features: [job_tokens, prompt_tokens]
    Label: simulated success (job label)
    """

    X = []
    y = []

    for job in jobs:
        for prompt in prompts:
            # Features
            X.append([
                job["tokens"],
                prompt["tokens"]
            ])

            # Label (simulated ground truth)
            y.append(job["label"])

    return np.array(X), np.array(y)


def train_predictor(X, y, predictor_type="rf"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if predictor_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42
        )

    elif predictor_type == "xgb":
        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
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

    return model



import time

def predict_probabilities(model, jobs, prompts):
    p = {}

    total_jobs = len(jobs)
    start_time = time.time()

    for i, job in enumerate(jobs):
        for prompt in prompts:
            features = [[job["tokens"], prompt["tokens"]]]
            prob = model.predict_proba(features)[0][1]
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

