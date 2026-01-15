import random
from deap import base, creator, tools, algorithms


def spea2_optimize(jobs, prompts, p_matrix, cost_fn,
                   population_size=40, generations=40):

    num_jobs = len(jobs)
    num_prompts = len(prompts)

    creator.create("FitnessSPEA2", base.Fitness, weights=(-1.0, 1.0))
    creator.create("IndividualSPEA2", list, fitness=creator.FitnessSPEA2)

    toolbox = base.Toolbox()
    toolbox.register("attr_prompt", random.randrange, num_prompts)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.IndividualSPEA2,
        toolbox.attr_prompt,
        n=num_jobs
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        cost = 0
        acc = 0
        for i, pidx in enumerate(ind):
            cost += cost_fn(jobs[i]["tokens"], prompts[pidx]["tokens"])
            acc += p_matrix[(i, pidx)]
        return cost, acc / num_jobs

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt,
                     low=0, up=num_prompts - 1,
                     indpb=1.0 / num_jobs)
    toolbox.register("select", tools.selSPEA2)

    pop = toolbox.population(population_size)
    archive = []

    for gen in range(generations):
        offspring = algorithms.varAnd(pop, toolbox, 0.9, 0.1)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + archive, population_size)
        archive = tools.selBest(pop, population_size // 2)

    solutions = []

    for ind in archive:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

        c, a = ind.fitness.values
        solutions.append(
            {
            "cost": c,
            "accuracy": a,
            "assignment": list(ind),
            }
        )

    return solutions