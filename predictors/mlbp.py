import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train_predictor(jobs, prompts):
    X, y = [], []

    import random
    import math

    MAX_JOB_TOKENS = max(job["tokens"] for job in jobs)
    MAX_PROMPT_TOKENS = max(p["tokens"] for p in prompts)

    ALPHA = 4.0      # steepness (tunable)
    NOISE = 0.05     # realism noise

    HISTORICAL_FRACTION = 0.3 # 20%
    historical_jobs = random.sample(
        jobs,
        int(len(jobs) * HISTORICAL_FRACTION)
    )
    for job in historical_jobs:
        for prompt in prompts:
            job_norm = job["tokens"] / MAX_JOB_TOKENS
            prompt_norm = prompt["tokens"] / MAX_PROMPT_TOKENS

            logit = ALPHA * (prompt_norm - job_norm)
            success_prob = 1 / (1 + math.exp(-logit))

        # add noise so nothing is perfect
            success_prob = max(0.0, min(1.0, success_prob + random.uniform(-NOISE, NOISE)))

            success = 1 if random.random() < success_prob else 0

            X.append([job["tokens"], prompt["tokens"]])
            y.append(success)



    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=1,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"[MLBP] Predictor accuracy: {accuracy_score(y_test, preds):.4f}")
    return model


def predict_probabilities(model, jobs, prompts):
    p = {}

    for job in jobs:
        for prompt in prompts:
            features = np.array([[job["tokens"], prompt["tokens"]]])
            prob = model.predict_proba(features)[0][1]
            p[(job["id"], prompt["id"])] = prob

    print("[MLBP] Probability matrix computed")
    return p
