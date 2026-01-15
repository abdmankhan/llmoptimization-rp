import numpy as np
from scipy.spatial.distance import cdist


def pareto_front(points):
    """Return non-dominated points"""
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if (
                q["cost"] <= p["cost"]
                and q["accuracy"] >= p["accuracy"]
                and (q["cost"] < p["cost"] or q["accuracy"] > p["accuracy"])
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    return pareto


# -----------------------
# IGD
# -----------------------
def compute_igd(pareto, reference):
    """
    pareto: Pareto front of one algorithm
    reference: approximated true Pareto front
    """
    P = np.array([[p["cost"], p["accuracy"]] for p in pareto])
    R = np.array([[r["cost"], r["accuracy"]] for r in reference])

    distances = cdist(R, P)
    return np.mean(np.min(distances, axis=1))


# -----------------------
# Delta (Δ)
# -----------------------
def compute_delta(pareto):
    """
    Spread metric (Δ)
    """
    if len(pareto) < 2:
        return float("inf")

    pareto_sorted = sorted(pareto, key=lambda x: x["cost"])
    distances = []

    for i in range(len(pareto_sorted) - 1):
        p1 = pareto_sorted[i]
        p2 = pareto_sorted[i + 1]
        d = np.linalg.norm(
            [p1["cost"] - p2["cost"], p1["accuracy"] - p2["accuracy"]]
        )
        distances.append(d)

    distances = np.array(distances)
    d_mean = np.mean(distances)

    delta = np.sum(np.abs(distances - d_mean)) / (
        (len(distances)) * d_mean
    )
    return delta


# -----------------------
# M_n
# -----------------------
def compute_mn(pareto):
    return len(pareto)
