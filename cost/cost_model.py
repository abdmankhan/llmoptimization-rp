def compute_cost(job_tokens, prompt_tokens, price=0.0001):
    return (job_tokens + prompt_tokens) * price
