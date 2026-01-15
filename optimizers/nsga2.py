import random
import numpy as np
from deap import base, creator, tools, algorithms


def nsga2_optimize(
    jobs,
    prompts,
    p_matrix,
    cost_fn,
    population_size=40,
    generations=40,
    crossover_prob=0.9,
    mutation_prob=0.1,
):
    """
    NSGA-II optimizer for prompt scheduling.

    Each individual = list of prompt indices (one per job)

    Objectives:
      - Minimize total cost
      - Maximize expected success
    """

    num_jobs = len(jobs)
    num_prompts = len(prompts)

    # ---------- DEAP setup ----------
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)


    toolbox = base.Toolbox()

    # Gene = prompt index
    toolbox.register("attr_prompt", random.randrange, num_prompts)

    # Individual = prompt assignment for all jobs
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_prompt,
        n=num_jobs,
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ---------- Evaluation ----------
    def evaluate(individual):
        total_cost = 0.0
        total_success = 0.0

        for i, prompt_idx in enumerate(individual):
            job = jobs[i]
            prompt = prompts[prompt_idx]

            total_cost += cost_fn(job["tokens"], prompt["tokens"])
            total_success += p_matrix[(i, prompt_idx)]

        avg_success = total_success / num_jobs
        return total_cost, avg_success

    toolbox.register("evaluate", evaluate)

    # ---------- Operators ----------
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register(
        "mutate",
        tools.mutUniformInt,
        low=0,
        up=num_prompts - 1,
        indpb=1.0 / num_jobs,
    )
    toolbox.register("select", tools.selNSGA2)

    # ---------- Run NSGA-II ----------
    population = toolbox.population(n=population_size)

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(generations):
        offspring = algorithms.varAnd(
            population,
            toolbox,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
        )

        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        population = toolbox.select(
            population + offspring,
            population_size,
        )

    # ---------- Extract solutions ----------
    solutions = []
    for ind in population:
        cost, acc = ind.fitness.values
        solutions.append(
            {
                "cost": cost,
                "accuracy": acc,
                "assignment": list(ind),
            }
        )

    return solutions
