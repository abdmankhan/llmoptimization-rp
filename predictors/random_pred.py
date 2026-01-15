import random

def random_probabilities(jobs, prompts):
    return {
        (job["id"], prompt["id"]): random.random()
        for job in jobs
        for prompt in prompts
    }
