import random
from cost.cost_model import compute_cost


def random_search(
    jobs,
    prompts,
    probabilities,
    cost_fn=compute_cost,
    iterations=100
):
    """Simple random baseline that samples prompt assignments uniformly."""

    solutions = []

    for _ in range(iterations):
        total_cost = 0.0
        total_acc = 0.0
        assignment = []

        for job in jobs:
            prompt = random.choice(prompts)
            assignment.append(prompt["id"])
            total_cost += cost_fn(job["tokens"], prompt["tokens"])
            total_acc += probabilities[(job["id"], prompt["id"])]

        solutions.append({
            "cost": total_cost,
            "accuracy": total_acc / len(jobs),
            "assignment": assignment
        })

    print("[OPT] Random search completed")
    return solutions
