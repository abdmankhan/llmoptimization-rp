import random
import copy


def mopso_optimize(jobs, prompts, p_matrix, cost_fn,
                   swarm_size=30, iterations=30):

    num_jobs = len(jobs)
    num_prompts = len(prompts)

    swarm = [
        [random.randrange(num_prompts) for _ in range(num_jobs)]
        for _ in range(swarm_size)
    ]

    archive = []

    def evaluate(sol):
        cost = 0
        acc = 0
        for i, pidx in enumerate(sol):
            cost += cost_fn(jobs[i]["tokens"], prompts[pidx]["tokens"])
            acc += p_matrix[(i, pidx)]
        return cost, acc / num_jobs

    for _ in range(iterations):
        for sol in swarm:
            cost, acc = evaluate(sol)
            archive.append({
                "cost": cost,
                "accuracy": acc,
                "assignment": sol.copy()
            })

        # keep non-dominated
        archive = pareto_filter(archive)
        swarm = random.sample(
            [a["assignment"] for a in archive],
            min(swarm_size, len(archive))
        )

    return archive
