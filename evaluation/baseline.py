from cost.cost_model import compute_cost

def single_prompt_baseline(jobs, prompt, probabilities):
    cost = 0
    acc = 0

    for job in jobs:
        cost += compute_cost(job["tokens"], prompt["tokens"])
        acc += probabilities[(job["id"], prompt["id"])]

    return {
        "prompt": prompt["name"],
        "cost": cost,
        "accuracy": acc / len(jobs)
    }
