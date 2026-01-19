import os
import json
import hashlib
from ollama import chat

CACHE_DIR = "results/ollama_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _hash(job_text, prompt_name):
    key = f"{prompt_name}::{job_text}"
    return hashlib.md5(key.encode()).hexdigest()


def query_gemini(job_text, prompt, instruction="standard"):
    """
    Local LLM (Ollama) grounding.
    
    Args:
        job_text: The log text to classify
        prompt: Prompt template dict with 'name' and 'tokens'
        instruction: Type of instruction to use (simple, standard, fewshot_1, fewshot_3)
    
    Returns: 1 (anomaly) or 0 (normal)
    """

    key = _hash(f"{job_text}_{instruction}", prompt["name"])
    cache_path = f"{CACHE_DIR}/{key}.json"

    # ---------- cache ----------
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)["label"]

    # ---------- Build prompt based on complexity ----------
    if instruction == "simple" or prompt.get("name") == "simple":
        full_prompt = f"""Classify this log as Anomaly or Normal.
Output JSON: {{"label": "Anomaly"}} or {{"label": "Normal"}}

LOG: {job_text}"""
    
    elif instruction == "standard" or prompt.get("name") == "standard":
        full_prompt = f"""You are a log analysis expert. Classify this system log.

TASK: Determine if the log indicates an anomaly or normal operation.
OUTPUT FORMAT (JSON ONLY): {{"label": "Anomaly"}} or {{"label": "Normal"}}

LOG:
{job_text}"""
    
    elif instruction == "fewshot_1" or prompt.get("name") == "fewshot_1":
        full_prompt = f"""You are a log analysis expert. Classify system logs as Anomaly or Normal.

EXAMPLE:
LOG: "081109 203518 148 INFO dfs.DataNode: PacketResponder 2 for block blk_-6952295868487656571 terminating"
CLASSIFICATION: {{"label": "Normal"}}

NOW CLASSIFY:
LOG: {job_text}
OUTPUT (JSON ONLY): {{"label": "Anomaly"}} or {{"label": "Normal"}}"""
    
    else:  # fewshot_3
        full_prompt = f"""You are a log analysis expert. Classify system logs as Anomaly or Normal.

EXAMPLES:
1. LOG: "081109 203518 148 INFO dfs.DataNode: PacketResponder 2 terminating"
   {{"label": "Normal"}}

2. LOG: "081109 204655 352 ERROR dfs.DataNode: DataNode shutdown started"  
   {{"label": "Anomaly"}}

3. LOG: "081109 203621 63 INFO dfs.DataNode: Receiving block blk_123 from /10.0.0.1"
   {{"label": "Normal"}}

NOW CLASSIFY:
LOG: {job_text}
OUTPUT (JSON ONLY): {{"label": "Anomaly"}} or {{"label": "Normal"}}"""


    # ---------- Ollama call ----------
    response = chat(
        model="gemma3",
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    )

    # text = response["message"]["content"].lower()
    # label = 1 if "anomaly" in text else 0

    content = response["message"]["content"]

    try:
        parsed = json.loads(content)
        label_str = parsed.get("label", "").lower()
        label = 1 if label_str == "anomaly" else 0
    except Exception:
    # fallback if model misbehaves
        label = 0


    # ---------- save ----------
    with open(cache_path, "w") as f:
        json.dump({"label": label}, f)

    return label
