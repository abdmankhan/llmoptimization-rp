#!/usr/bin/env python3
"""
Generate fully cached real LLM results for all (job, prompt) pairs.
Run this ONCE to build a complete dataset, then experiments are instant.
"""

import os
import sys
from tqdm import tqdm
from jobs.loader import load_hdfs_jobs
from prompts.templates import PROMPT_TEMPLATES
from llm.gemini_client import query_gemini

print("=" * 60)
print("REAL LLM DATA GENERATOR")
print("=" * 60)

# Load jobs
max_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
print(f"\nLoading {max_jobs} jobs...")
jobs = load_hdfs_jobs(
    "hdfs_data/HDFS.log",
    "hdfs_data/preprocessed/anomaly_label.csv",
    max_jobs=max_jobs
)

prompts = PROMPT_TEMPLATES

# Count existing cache
cache_dir = "results/ollama_cache"
existing_cache = len([f for f in os.listdir(cache_dir) if f.endswith('.json')]) if os.path.exists(cache_dir) else 0
print(f"Existing cache entries: {existing_cache}")

# Calculate work
total_pairs = len(jobs) * len(prompts)
print(f"\nTotal (job, prompt) pairs: {total_pairs}")
print(f"Prompts: {[p['name'] for p in prompts]}")
print(f"\nThis will call Ollama {total_pairs - existing_cache} times (cached calls skip)")
print("Estimated time: ~0.5s per call\n")

input("Press Enter to start (Ctrl+C to cancel)...")

# Generate all results
correct = 0
total = 0

with tqdm(total=total_pairs, desc="Calling Ollama") as pbar:
    for job in jobs:
        ground_truth = job["label"]
        
        for prompt in prompts:
            # This will cache the result
            prediction = query_gemini(job["log"], prompt, instruction=prompt["name"])
            
            if prediction == ground_truth:
                correct += 1
            total += 1
            
            pbar.update(1)
            pbar.set_postfix({
                'accuracy': f'{correct/total:.1%}',
                'prompt': prompt['name'][:8]
            })

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

# Per-prompt accuracy
print("\nPer-Prompt Performance:")
for prompt in prompts:
    prompt_correct = 0
    prompt_total = 0
    
    for job in jobs:
        prediction = query_gemini(job["log"], prompt, instruction=prompt["name"])
        if prediction == job["label"]:
            prompt_correct += 1
        prompt_total += 1
    
    accuracy = prompt_correct / prompt_total
    print(f"  {prompt['name']:12s}: {accuracy:.1%} ({prompt_correct}/{prompt_total})")

print(f"\nOverall accuracy: {correct/total:.1%}")
print(f"Cache directory: {cache_dir}")
print(f"Total cached entries: {len([f for f in os.listdir(cache_dir) if f.endswith('.json')])}")

print("\n✓ All results cached! You can now run experiments with USE_REAL_LLM=True instantly.")
