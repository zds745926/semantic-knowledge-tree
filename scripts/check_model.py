#!/usr/bin/env python3
"""Check model files and their sizes"""
import os

# Check modelscope cache
model_path = "/data/semantic-knowledge-tree/models/qwen-model-cache"
if os.path.exists(model_path):
    total = 0
    for dp, dn, fn in os.walk(model_path):
        for f in fn:
            fp = os.path.join(dp, f)
            sz = os.path.getsize(fp)
            total += sz
    print(f"Modelscope cache total: {total/1024/1024:.0f} MB")
else:
    print("Modelscope cache not found")

# Check huggingface cache
hf_path = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
if os.path.exists(hf_path):
    total = 0
    for dp, dn, fn in os.walk(hf_path):
        for f in fn:
            fp = os.path.join(dp, f)
            sz = os.path.getsize(fp)
            total += sz
    print(f"HF cache total: {total/1024/1024:.0f} MB")
else:
    print("HF cache not found")

# Check for the snapshot files
for root, dirs, files in os.walk("/"):
    for d in dirs:
        if "Qwen" in d and "Qwen2.5-1.5B-Instruct" in d:
            p = os.path.join(root, d)
            print(f"Found: {p}")
            for dp, dn, fn in os.walk(p):
                for f in fn:
                    if f.endswith(".safetensors") or f.endswith(".json"):
                        fp = os.path.join(dp, f)
                        print(f"  {fp}  {os.path.getsize(fp)/1024/1024:.1f} MB")
            break
    else:
        continue
    break
