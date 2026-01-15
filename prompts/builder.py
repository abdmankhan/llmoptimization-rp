def build_prompt(job_text, template_name):
    return f"[{template_name.upper()} PROMPT]\n{job_text}"
