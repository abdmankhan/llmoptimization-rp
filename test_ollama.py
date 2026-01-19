#!/usr/bin/env python3
"""Test if Ollama API is working properly"""

from ollama import chat
import json

# Test 1: Simple anomaly detection
test_log_anomaly = "081109 204655 352 ERROR dfs.DataNode: DataNode shutdown started"
test_log_normal = "081109 203518 148 INFO dfs.DataNode: PacketResponder 2 terminating"

print("=" * 80)
print("OLLAMA API TEST")
print("=" * 80)

# Test anomaly log
print("\n[TEST 1] Anomaly log:")
print(f"LOG: {test_log_anomaly}")
print("\nCalling Ollama...")

try:
    response = chat(
        model="gemma3",
        messages=[
            {
                "role": "user",
                "content": f"""Classify this log as Anomaly or Normal.
Output JSON: {{"label": "Anomaly"}} or {{"label": "Normal"}}

LOG: {test_log_anomaly}"""
            }
        ]
    )
    
    print("\n✓ Ollama API call successful!")
    print(f"Raw response: {response}")
    print(f"\nMessage content: {response['message']['content']}")
    
    # Try to parse
    try:
        parsed = json.loads(response['message']['content'])
        print(f"Parsed JSON: {parsed}")
        label = parsed.get('label', 'UNKNOWN')
        print(f"Label: {label}")
    except Exception as e:
        print(f"✗ JSON parsing failed: {e}")
        print(f"Content was: {response['message']['content']}")
        
except Exception as e:
    print(f"\n✗ Ollama API call FAILED!")
    print(f"Error: {e}")
    print("\nIs Ollama running? Try: ollama serve")

# Test normal log
print("\n" + "-" * 80)
print("[TEST 2] Normal log:")
print(f"LOG: {test_log_normal}")
print("\nCalling Ollama...")

try:
    response = chat(
        model="gemma3",
        messages=[
            {
                "role": "user",
                "content": f"""Classify this log as Anomaly or Normal.
Output JSON: {{"label": "Anomaly"}} or {{"label": "Normal"}}

LOG: {test_log_normal}"""
            }
        ]
    )
    
    print("\n✓ Ollama API call successful!")
    print(f"\nMessage content: {response['message']['content']}")
    
    # Try to parse
    try:
        parsed = json.loads(response['message']['content'])
        print(f"Parsed JSON: {parsed}")
        label = parsed.get('label', 'UNKNOWN')
        print(f"Label: {label}")
    except Exception as e:
        print(f"✗ JSON parsing failed: {e}")
        content = response['message']['content']
        print(f"Content was: {content}")
        
        # Try fallback parsing
        if "anomaly" in content.lower():
            print("Fallback: Detected 'anomaly' in text → label = Anomaly")
        elif "normal" in content.lower():
            print("Fallback: Detected 'normal' in text → label = Normal")
        
except Exception as e:
    print(f"\n✗ Ollama API call FAILED!")
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
