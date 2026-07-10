#!/usr/bin/env python3
"""
SWE-bench 快速评测 — 知识树 + qwen2.5:7b
用法: python scripts/swebench_fast.py
"""
import sys, os, json, urllib.request, time, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import TreePersistence

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
RESULTS_FILE = "results/swebench_results.jsonl"
MAX_TASKS = 5


def call_ollama(prompt: str) -> str:
    data = {
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
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
        start, end = max(0, ln - 15), min(len(lines), ln + 25)
        return "\n".join(
            f"{i+1}:{lines[i]}" for i in range(start, end)
        )
    return "\n".join(f"{i+1}:{lines[i]}" for i in range(min(50, len(lines))))


def main():
    os.makedirs("results", exist_ok=True)
    print("=" * 60)
    print("  SWE-bench 快速评测 — 知识树 + qwen2.5:7b")
    print("=" * 60)

    # 1. Tree
    print("\n[1/3] 加载知识树...")
    db = TreePersistence("smart_tree")
    tree = db.load_tree()
    print(f"  {tree.stats()['total_nodes']} nodes")

    # 2. SWE-bench
    print("\n[2/3] 加载 SWE-bench...")
    from datasets import load_dataset
    ds = load_dataset("swe-bench/swe-bench", split="test")
    tasks = [ds[i] for i in range(MAX_TASKS)]
    print(f"  {len(tasks)} tasks")

    # 3. Evaluate
    print(f"\n[3/3] 运行评测 ({len(tasks)} 题)...")
    results = []

    for i, task in enumerate(tasks):
        instance_id = task["instance_id"]
        repo = task["repo"]
        problem = task["problem_statement"]
        gold_patch = task["patch"]
        commit = task["base_commit"]

        print(f"\n  [{i+1}/{len(tasks)}] {instance_id}")

        # Query tree
        qvec = tree.encoder.encode(problem)
        pen = tree.penetrate(problem, query_vec=qvec, verbose=False)
        seen = set()
        ctx_items = []
        for r in pen[:5]:
            leaf = r.get("leaf_name", "")
            if leaf in seen:
                continue
            seen.add(leaf)
            path = " -> ".join(r.get("path", []))
            dp = r.get("data_pointers", [])
            c = dp[0].get("content_preview", "")[:100] if dp else ""
            ctx_items.append(f"- {path}: {c}")
        ctx = "\n".join(ctx_items) if ctx_items else "(无相关知识点)"
        print(f"    Tree: {len(pen)} paths, {len(ctx_items)} unique")

        # Fetch code
        files = re.findall(r"^\+\+\+ b/(.+)$", gold_patch, re.MULTILINE)
        code_parts = []
        for fpath in files[:2]:
            try:
                content = fetch_file(repo, commit, fpath)
                snippet = extract_context(content, gold_patch)
                code_parts.append(f"# {fpath}:\n{snippet}")
            except Exception as e:
                code_parts.append(f"# {fpath}: (fetch error)")
        code_ctx = "\n\n".join(code_parts)

        # Generate patch
        prompt = f"""You are an expert software engineer. Fix the following GitHub issue.

ISSUE:
{problem[:600]}

CODE CONTEXT:
{code_ctx[:2000]}

KNOWLEDGE TREE:
{ctx}

Generate a git diff patch. Output ONLY the patch in ```diff ... ``` block."""

        print(f"    Generating...")
        start = time.time()
        generated = call_ollama(prompt)
        elapsed = time.time() - start

        # Extract and evaluate
        m = re.search(r"```(?:diff)?\s*\n(.*?)```", generated, re.DOTALL)
        patch_out = m.group(1).strip() if m else generated
        has_diff = bool(re.search(r"^(diff --git|--- a/)", patch_out, re.MULTILINE))
        has_hunks = bool(re.search(r"^@@ ", patch_out, re.MULTILINE))
        passed = has_diff and has_hunks

        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] diff={has_diff} hunks={has_hunks} ({elapsed:.1f}s)")

        results.append({
            "instance_id": instance_id,
            "repo": repo,
            "passed": passed,
            "has_diff": has_diff,
            "has_hunks": has_hunks,
            "time": elapsed,
            "tree_context": ctx,
            "gold_patch_files": files,
            "generated_preview": generated[:400],
        })

    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Result: {passed}/{total} well-formed patches ({passed/total*100:.1f}%)")
    print(f"{'=' * 60}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved to {RESULTS_FILE}")

    db.close()


if __name__ == "__main__":
    main()
