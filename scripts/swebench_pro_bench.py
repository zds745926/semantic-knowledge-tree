#!/usr/bin/env python3
"""SWE-Bench Pro 快速评测 — 知识树 + qwen2.5:7b"""
import sys, os, json, urllib.request, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import TreePersistence

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
RESULTS_FILE = "results/swebench_pro_results.jsonl"
MAX_TASKS = 5

def call_ollama(prompt: str) -> str:
    data = {"model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 4096}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(data).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode()).get("response", "").strip()

def fetch_file(repo: str, commit: str, path: str) -> str:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")

def extract_context(content: str, patch: str) -> str:
    lines = content.split("\n")
    hunk = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", patch)
    if hunk:
        ln = int(hunk.group(1))
        start, end = max(0, ln - 20), min(len(lines), ln + 30)
        return "\n".join(f"{i+1}:{lines[i]}" for i in range(start, end))
    return "\n".join(f"{i+1}:{lines[i]}" for i in range(min(60, len(lines))))

def main():
    os.makedirs("results", exist_ok=True)
    print("="*60)
    print("  SWE-Bench Pro 评测 — 知识树 + qwen2.5:7b")
    print("="*60)

    # 1. Tree
    print("\n[1/4] 加载知识树...")
    db = TreePersistence("smart_tree")
    tree = db.load_tree()
    print(f"  {tree.stats()['total_nodes']} nodes")

    # 2. Load SWE-Bench Pro (streaming)
    print("\n[2/4] 加载 SWE-Bench Pro...")
    from datasets import load_dataset
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test", streaming=True)
    tasks = []
    for i, task in enumerate(ds):
        if i >= MAX_TASKS: break
        tasks.append(task)
    print(f"  {len(tasks)} tasks loaded")

    # 3. Evaluate
    print(f"\n[3/4] 运行评测 ({len(tasks)} 题)...")
    results = []

    for i, task in enumerate(tasks):
        instance_id = task["instance_id"]
        repo = task["repo"]
        problem = task["problem_statement"]
        gold_patch = task["patch"]
        commit = task["base_commit"]
        lang = task.get("repo_language", "unknown")
        reqs = task.get("requirements", "")

        print(f"\n  [{i+1}/{len(tasks)}] {instance_id} ({repo}, {lang})")

        # Query tree
        qvec = tree.encoder.encode(problem)
        pen = tree.penetrate(problem, query_vec=qvec, verbose=False)
        seen = set()
        ctx_items = []
        for r in pen[:5]:
            leaf = r.get("leaf_name", "")
            if leaf in seen: continue
            seen.add(leaf)
            path = " -> ".join(r.get("path", []))
            dp = r.get("data_pointers", [])
            c = dp[0].get("content_preview", "")[:100] if dp else ""
            ctx_items.append(f"- {path}: {c}")
        ctx = "\n".join(ctx_items) if ctx_items else "(无相关知识点)"
        print(f"    Tree: {len(pen)} paths, {len(ctx_items)} unique")

        # Fetch code context
        files = re.findall(r"^\+\+\+ b/(.+)$", gold_patch, re.MULTILINE)
        code_parts = []
        for fpath in files[:3]:
            try:
                content = fetch_file(repo, commit, fpath)
                snippet = extract_context(content, gold_patch)
                code_parts.append(f"# {fpath}:\n{snippet}")
            except Exception as e:
                code_parts.append(f"# {fpath}: (fetch error: {e})")
        code_ctx = "\n\n".join(code_parts)

        # Generate patch
        prompt = f"""You are an expert software engineer. Fix the following GitHub issue by generating a complete git diff patch.

ISSUE:
{problem[:1200]}

REQUIREMENTS:
{reqs[:600]}

CODE CONTEXT ({', '.join(files[:3])}):
{code_ctx[:2500]}

KNOWLEDGE TREE:
{ctx}

Generate a complete git diff patch. Output ONLY the patch in ```diff ... ``` block."""

        print(f"    Generating...")
        start = time.time()
        try:
            generated = call_ollama(prompt)
        except Exception as e:
            print(f"    ❌ Generation failed: {e}")
            results.append({"instance_id": instance_id, "passed": False, "error": str(e), "time": time.time()-start})
            continue
        elapsed = time.time() - start

        # Evaluate
        m = re.search(r"```(?:diff)?\s*\n(.*?)```", generated, re.DOTALL)
        patch_out = m.group(1).strip() if m else generated
        has_diff = bool(re.search(r"^(diff --git|--- a/)", patch_out, re.MULTILINE))
        has_hunks = bool(re.search(r"^@@ ", patch_out, re.MULTILINE))
        targets_correct = any(f in patch_out for f in files)
        passed = has_diff and has_hunks and targets_correct

        status = "PASS" if passed else "FAIL"
        detail = []
        if not has_diff: detail.append("no_diff_format")
        if not has_hunks: detail.append("no_hunks")
        if not targets_correct: detail.append("wrong_files")
        print(f"    [{status}] diff={has_diff} hunks={has_hunks} files={'+'.join(files)} targets={targets_correct} ({elapsed:.1f}s)")
        if detail:
            print(f"    Issues: {', '.join(detail)}")

        results.append({
            "instance_id": instance_id, "repo": repo, "lang": lang,
            "passed": passed, "has_diff": has_diff, "has_hunks": has_hunks,
            "targets_correct_files": targets_correct, "time": elapsed,
            "tree_context": ctx, "gold_files": files,
            "generated_preview": generated[:500],
        })

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  SWE-Bench Pro Result: {passed}/{total} well-formed patches ({passed/total*100:.1f}%)")
    print(f"  Avg time: {sum(r['time'] for r in results)/total:.1f}s")
    print(f"{'='*60}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved to {RESULTS_FILE}")
    db.close()

if __name__ == "__main__":
    main()
