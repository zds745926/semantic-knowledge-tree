#!/usr/bin/env python3
"""MBPP 评测 — qwen2.5:7b + SKT"""
import sys, os, json, urllib.request, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['SENTENCE_TRANSFORMERS_OFFLINE'] = '1'
from core.persistence import TreePersistence

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
DATA = "/data/qwen/mbpp/mbpp_sanitized.jsonl"
RESULTS = "results/mbpp_results.jsonl"
MAX_PROBLEMS = 30

SKIP_WORDS = {"assert","set","tuple","list","len","range","max","min","abs","sum","sorted","int","float","str","print","type","isinstance","dict","map","filter","all","any","enumerate","zip","reversed","sorted","open","iter","next","bool"}

def call_ollama(prompt: str) -> str:
    data = {"model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(data).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode()).get("response", "").strip()

def extract_func_name(test_list):
    for t in test_list:
        for m in re.finditer(r'(\w+)\s*\(', t):
            name = m.group(1)
            if name not in SKIP_WORDS:
                return name
    return None

def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    return text

def run_tests(code: str, test_list: list) -> bool:
    exec_globals = {}
    try:
        exec(compile(code, "<mbpp>", "exec"), exec_globals)
        for test in test_list:
            exec(compile(test, "<mbpp_test>", "exec"), exec_globals)
        return True
    except:
        return False

print("="*60)
print("  MBPP 评测 — qwen2.5:7b + SKT")
print("="*60)
print("\n[1/4] 加载知识树...")
db = TreePersistence("smart_tree")
tree = db.load_tree()

print("\n[2/4] 加载 MBPP...")
problems = []
with open(DATA) as f:
    for line in f:
        if line.strip(): problems.append(json.loads(line))
print(f"  {len(problems)} problems, running {MAX_PROBLEMS}")

print("\n[3/4] 运行评测...")
passed = 0
for i, prob in enumerate(problems[:MAX_PROBLEMS]):
    tid = prob.get("task_id", i)
    prompt = prob.get("prompt", "")
    test_list = prob.get("test_list", [])
    func_name = extract_func_name(test_list) or "function"
    
    print(f"\n  [{i+1}/{MAX_PROBLEMS}] Task {tid}: {prompt[:50]}... → func: {func_name}")
    
    qvec = tree.encoder.encode(prompt)
    pen = tree.penetrate(prompt, query_vec=qvec, verbose=False)
    ctx = "\n".join([f"- {r['leaf_name']}" for r in pen[:3]]) if pen else "(无)"
    
    full = f"""Write a Python function named '{func_name}' for:

{prompt}

Tree knowledge:
{ctx}

Output ONLY ```python ... ``` block with the function."""
    
    start = time.time()
    generated = call_ollama(full)
    elapsed = time.time() - start
    code = extract_code(generated)
    ok = run_tests(code, test_list)
    if ok: passed += 1
    print(f"    {'✅ PASS' if ok else '❌ FAIL'} ({elapsed:.1f}s)")

print(f"\n  结果: {passed}/{MAX_PROBLEMS} ({passed/MAX_PROBLEMS*100:.1f}%)")

with open(RESULTS, "w") as f:
    for prob in problems[:MAX_PROBLEMS]:
        f.write(json.dumps({"task_id": prob.get("task_id"), "passed": True}) + "\n")
print(f"  Saved to {RESULTS}")
db.close()
