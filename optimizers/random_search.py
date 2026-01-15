import random
from cost.cost_model import compute_cost

def random_search(jobs, prompts, probabilities, iterations=100):
    solutions = []

    for _ in range(iterations):
        total_cost = 0
        total_acc = 0

        for job in jobs:
            prompt = random.choice(prompts)
            total_cost += compute_cost(job["tokens"], prompt["tokens"])
            total_acc += probabilities[(job["id"], prompt["id"])]

        solutions.append({
            "cost": total_cost,
            "accuracy": total_acc / len(jobs)
        })

    print("[OPT] Random search completed")
    return solutions
