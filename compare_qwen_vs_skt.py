#!/usr/bin/env python3
"""裸 qwen2.5:7b vs qwen2.5:7b+SKT 对比测试"""
import sys, os, json, urllib.request, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['SENTENCE_TRANSFORMERS_OFFLINE'] = '1'
from core.persistence import TreePersistence

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

questions = [
    "Python 中 __slots__ 是如何工作的？使用它有什么利弊？",
    "Python 的 descriptor protocol（描述符协议）是什么？property 和 __getattr__ 有什么区别？",
    "Python 的 GIL（全局解释器锁）对多线程程序有什么影响？什么时候应该用 multiprocessing 而不是 threading？",
    "Python 中 metaclass 的 __new__ 和 __init__ 有什么区别？如何用 metaclass 实现单例模式？",
    "Python 的 async/await 底层是如何实现的？事件循环如何调度协程？",
]

def call_ollama(prompt: str, timeout=120) -> str:
    data = {"model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(data).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()).get("response", "").strip()

# Load tree
print("加载知识树...")
db = TreePersistence("smart_tree")
tree = db.load_tree()
print(f"编码器: {type(tree.encoder).__name__}")

results = []
for i, q in enumerate(questions, 1):
    print(f"\n{'='*60}")
    print(f"问题 {i}: {q}")
    print(f"{'='*60}")

    # 1. Bare qwen2.5:7b (no tree context)
    print(f"\n--- 裸 qwen2.5:7b ---")
    prompt_bare = f"你是一个Python技术专家。请用中文详细回答以下问题：\n\n{q}"
    start = time.time()
    bare = call_ollama(prompt_bare)
    t_bare = time.time() - start
    print(f"({t_bare:.1f}s)")
    # Truncate for display
    bare_preview = bare[:300] + "..." if len(bare) > 300 else bare
    print(bare_preview)

    # 2. qwen2.5:7b + SKT
    print(f"\n--- qwen2.5:7b + SKT ---")
    qvec = tree.encoder.encode(q)
    pen = tree.penetrate(q, query_vec=qvec, verbose=False)
    seen = set()
    ctx_items = []
    for r in pen[:5]:
        leaf = r.get("leaf_name", "")
        if leaf in seen: continue
        seen.add(leaf)
        path = " -> ".join(r.get("path", []))
        dp = r.get("data_pointers", [])
        c = dp[0].get("content_preview", "")[:150] if dp else ""
        ctx_items.append(f"- {path}: {c}")
    ctx = "\n".join(ctx_items) if ctx_items else "(无相关知识点)"
    
    prompt_skt = f"你是一个Python技术专家。请用中文详细回答以下问题：\n\n{q}\n\n相关知识树上下文:\n{ctx}"
    start = time.time()
    skt = call_ollama(prompt_skt)
    t_skt = time.time() - start
    print(f"({t_skt:.1f}s), Tree: {len(pen)} paths")
    skt_preview = skt[:300] + "..." if len(skt) > 300 else skt
    print(skt_preview)
    
    results.append({
        "q": q, "bare_preview": bare[:500], "skt_preview": skt[:500],
        "t_bare": t_bare, "t_skt": t_skt, "tree_paths": len(pen),
        "ctx": ctx
    })

print(f"\n\n{'='*60}")
print(f"对比总结")
print(f"{'='*60}")
for r in results:
    bare_len = len(r['bare_preview'])
    skt_len = len(r['skt_preview'])
    print(f"\nQ: {r['q'][:50]}...")
    print(f"  裸: {r['t_bare']:.1f}s ({bare_len}chars) | SKT: {r['t_skt']:.1f}s ({skt_len}chars, {r['tree_paths']}路径)")
    print(f"  知识树命中: {r['ctx'][:100]}...")

db.close()
