import os
import json
import hashlib
from ollama import chat

CACHE_DIR = "results/ollama_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _hash(job_text, prompt_name):
    key = f"{prompt_name}::{job_text}"
    return hashlib.md5(key.encode()).hexdigest()


def query_gemini(job_text, prompt):
    """
    Local LLM (Ollama) grounding.
    Returns: 1 (anomaly) or 0 (normal)
    """

    key = _hash(job_text, prompt["name"])
    cache_path = f"{CACHE_DIR}/{key}.json"

    # ---------- cache ----------
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)["label"]

    # ---------- prompt ----------
    full_prompt = f"""
You are given a system log.

TASK:
Classify the log strictly.

OUTPUT FORMAT (JSON ONLY):
{{"label": "Anomaly"}} or {{"label": "Normal"}}

LOG:
{job_text}
"""


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

    import json

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
