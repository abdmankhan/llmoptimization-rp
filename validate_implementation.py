#!/usr/bin/env python3
"""
Quick validation script to verify paper implementation is correct.
Run this BEFORE full experiment to catch issues early.
"""

from jobs.loader import load_hdfs_jobs
from prompts.templates import PROMPT_TEMPLATES
from predictors.mlbp import create_training_data, train_predictor, predict_probabilities
import numpy as np

print("=" * 60)
print("PAPER IMPLEMENTATION VALIDATION")
print("=" * 60)

# Load small dataset for quick test
jobs = load_hdfs_jobs(
    "hdfs_data/HDFS.log",
    "hdfs_data/preprocessed/anomaly_label.csv",
    max_jobs=1000
)

prompts = PROMPT_TEMPLATES

# Train
X, y = create_training_data(jobs, prompts, use_real_llm=True)
model = train_predictor(X, y, predictor_type="xgb")

# Predict
probabilities = predict_probabilities(model, jobs, prompts)

# Compute baseline averages
print("\n" + "=" * 60)
print("BASELINE VALIDATION (should increase monotonically)")
print("=" * 60)

baseline_accs = []
for prompt in prompts:
    accs = [probabilities[(i, prompt["id"])] for i in range(len(jobs))]
    avg_acc = np.mean(accs)
    baseline_accs.append(avg_acc)
    print(f"{prompt['name']:12s} (${prompt['tokens']:3d} tokens): {avg_acc:.3f}")

# Validation checks
print("\n" + "=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)

checks_passed = 0
checks_total = 0

# Check 1: Monotonic increase
checks_total += 1
is_monotonic = all(baseline_accs[i] <= baseline_accs[i+1] for i in range(len(baseline_accs)-1))
if is_monotonic:
    print("✓ PASS: Accuracy increases with prompt complexity")
    checks_passed += 1
else:
    print("✗ FAIL: Accuracy NOT monotonically increasing")
    print(f"  Values: {[f'{x:.3f}' for x in baseline_accs]}")

# Check 2: Reasonable range
checks_total += 1
if 0.65 <= baseline_accs[0] <= 0.80 and 0.80 <= baseline_accs[-1] <= 0.95:
    print("✓ PASS: Accuracy values in reasonable range (65-80% to 80-95%)")
    checks_passed += 1
else:
    print("✗ FAIL: Accuracy values out of expected range")
    print(f"  Simple: {baseline_accs[0]:.3f} (expected: 0.65-0.80)")
    print(f"  Complex: {baseline_accs[-1]:.3f} (expected: 0.80-0.95)")

# Check 3: Clear separation
checks_total += 1
improvement = baseline_accs[-1] - baseline_accs[0]
if improvement >= 0.10:
    print(f"✓ PASS: Clear improvement from simple→complex: +{improvement:.1%}")
    checks_passed += 1
else:
    print(f"✗ FAIL: Insufficient separation: only +{improvement:.1%}")

# Check 4: Probability format
checks_total += 1
sample_prob = probabilities[(0, 0)]
if 0.0 <= sample_prob <= 1.0:
    print(f"✓ PASS: Probabilities in valid range [0,1] (sample: {sample_prob:.3f})")
    checks_passed += 1
else:
    print(f"✗ FAIL: Invalid probability value: {sample_prob}")

# Summary
print("\n" + "=" * 60)
print(f"RESULT: {checks_passed}/{checks_total} checks passed")
print("=" * 60)

if checks_passed == checks_total:
    print("✓ Implementation looks correct! Safe to run full experiment.")
    exit(0)
else:
    print("✗ Issues detected. Review IMPLEMENTATION_FIXES.md")
    exit(1)
