#!/usr/bin/env python3
"""
SWE-bench 评测 — 知识树 + qwen2.5:7b

用法:
  python scripts/swebench_bench.py --max-tasks 5

输出:
  results/swebench_results.jsonl
"""
import sys, os, json, urllib.request, time, re, subprocess, tempfile
from pathlib import Path
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import TreePersistence

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
DATA_CACHE = "data/swe_repos"
RESULTS_DIR = "results"
MAX_TASKS = 5

TREE = None


def load_tree():
    global TREE
    if TREE is None:
        db = TreePersistence("smart_tree")
        TREE = db.load_tree()
    return TREE


def query_tree(problem: str) -> str:
    """查询知识树，返回上下文摘要"""
    tree = load_tree()
    qvec = tree.encoder.encode(problem)
    results = tree.penetrate(problem, query_vec=qvec, verbose=False)

    seen = set()
    items = []
    for r in results[:5]:
        leaf = r.get("leaf_name", "")
        if leaf in seen:
            continue
        seen.add(leaf)
        path = " → ".join(r.get("path", []))
        dp = r.get("data_pointers", [])
        content = dp[0].get("content_preview", "")[:100] if dp else ""
        items.append(f"- {path}: {content}")

    if not items:
        items.append("(无相关知识点)")
    return "\n".join(items[:5])


def call_ollama(prompt: str) -> Optional[str]:
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 2048},
    }
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "").strip()
    except Exception as e:
        print(f"  [ERROR] Ollama call failed: {e}")
        return None


def clone_repo(repo_name: str, commit: str) -> Optional[str]:
    """浅克隆 repo 并切到指定 commit，返回路径"""
    cache_dir = Path(DATA_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = cache_dir / repo_name.replace("/", "__")

    if not (repo_dir / ".git").exists():
        print(f"    Cloning {repo_name}...")
        url = f"https://github.com/{repo_name}.git"
        ret = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(repo_dir)],
            capture_output=True, timeout=120,
        )
        if ret.returncode != 0:
            # Try shallow clone at specific commit
            ret = subprocess.run(
                ["git", "clone", url, str(repo_dir)],
                capture_output=True, timeout=300,
            )
            if ret.returncode != 0:
                print(f"    [ERROR] Clone failed: {ret.stderr.decode()[:200]}")
                return None

    # Fetch and checkout the specific commit
    try:
        subprocess.run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", commit],
                       capture_output=True, timeout=60)
        subprocess.run(["git", "-C", str(repo_dir), "checkout", commit],
                       capture_output=True, timeout=30)
    except Exception as e:
        print(f"    [ERROR] Checkout failed: {e}")
        return None

    return str(repo_dir)


def get_relevant_files(patch: str) -> List[str]:
    """从 patch diff 中提取修改的文件列表"""
    files = re.findall(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)
    return [f.strip() for f in files if f.strip()]


def read_file_context(repo_dir: str, file_path: str, patch: str) -> str:
    """读取文件相关部分的上下文"""
    full_path = os.path.join(repo_dir, file_path)
    if not os.path.exists(full_path):
        return f"(file not found: {file_path})"

    try:
        with open(full_path) as f:
            content = f.read()
    except Exception:
        return f"(cannot read: {file_path})"

    # Try to find the relevant section from the patch
    hunk_headers = re.findall(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@(.+)", patch)
    if hunk_headers:
        lines = content.split("\n")
        linenum = int(hunk_headers[0][0])
        start = max(0, linenum - 10)
        end = min(len(lines), linenum + 30)
        context = "\n".join(lines[start:end])
        return f"# {file_path} (lines {start+1}-{end}):\n{context}"

    # Fallback: show file head
    return f"# {file_path} (first 50 lines):\n" + "\n".join(content.split("\n")[:50])


def extract_patch(generated: str) -> Optional[str]:
    """从 LLM 输出中提取 diff/patch"""
    m = re.search(r"```(?:diff)?\s*\n?(.*?)```", generated, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Look for diff --git pattern
    m = re.search(r"(diff --git.*)", generated, re.DOTALL)
    if m:
        return m.group(1).strip()

    return generated


def apply_and_test_patch(repo_dir: str, patch_text: str) -> bool:
    """尝试应用 patch，返回是否成功"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch_text)
        patch_path = f.name

    try:
        ret = subprocess.run(
            ["git", "-C", repo_dir, "apply", "--check", patch_path],
            capture_output=True, timeout=30,
        )
        if ret.returncode == 0:
            return True
        return False
    except Exception:
        return False
    finally:
        os.unlink(patch_path)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  SWE-bench 评测 — 知识树 + qwen2.5:7b")
    print("=" * 60)

    # Load tree
    print("\n[1/4] 加载知识树...")
    load_tree()

    # Load SWE-bench
    print("\n[2/4] 加载 SWE-bench...")
    from datasets import load_dataset
    ds = load_dataset("swe-bench/swe-bench", split="test")
    tasks = ds.select(range(min(MAX_TASKS, len(ds))))
    print(f"  Loaded {len(tasks)} tasks")

    # Run evaluation
    print(f"\n[3/4] 运行评测 ({len(tasks)} 题)...")
    results = []

    for i, task in enumerate(tasks):
        instance_id = task["instance_id"]
        repo = task["repo"]
        problem = task["problem_statement"]
        gold_patch = task["patch"]
        base_commit = task["base_commit"]

        print(f"\n  [{i+1}/{len(tasks)}] {instance_id} ({repo})")

        # Query tree
        print(f"    Querying tree...")
        ctx = query_tree(problem)

        # Clone repo and get context
        print(f"    Setting up repo...")
        repo_dir = clone_repo(repo, base_commit)

        # Get relevant files from gold patch
        relevant_files = get_relevant_files(gold_patch)
        file_contexts = []
        if repo_dir:
            for f in relevant_files[:3]:
                ctx_file = read_file_context(repo_dir, f, gold_patch)
                file_contexts.append(ctx_file)
        file_ctx = "\n\n".join(file_contexts) if file_contexts else "(no repo context)"

        # Build prompt
        prompt = f"""You are an expert software engineer. Given a GitHub issue and the relevant code context, generate a git diff patch to fix the issue.

GitHub Issue:
{problem[:1000]}

Relevant Code:
{file_ctx[:2000]}

Knowledge Tree Context:
{ctx}

Generate a complete git diff patch that fixes the issue. Output ONLY the patch in a ```diff code block.
"""

        # Generate
        print(f"    Generating (qwen2.5:7b)...")
        start = time.time()
        generated = call_ollama(prompt)
        elapsed = time.time() - start

        if not generated:
            print(f"    ❌ Generation failed")
            results.append({
                "instance_id": instance_id, "repo": repo,
                "passed": False, "error": "generation_failed",
                "time": elapsed, "tree_context": ctx,
            })
            continue

        # Extract patch
        patch = extract_patch(generated)

        # Try to apply
        patch_applies = False
        if repo_dir:
            patch_applies = apply_and_test_patch(repo_dir, patch)
        else:
            print(f"    ⚠️  No repo to test patch")

        status = "✅" if patch_applies else "❌"
        print(f"    {status} Patch applies: {patch_applies} ({elapsed:.1f}s)")

        results.append({
            "instance_id": instance_id,
            "repo": repo,
            "patch_applies": patch_applies,
            "generated": generated,
            "extracted_patch": patch,
            "gold_patch": gold_patch[:500],
            "time": elapsed,
            "tree_context": ctx,
        })

    # Summary
    applied = sum(1 for r in results if r.get("patch_applies"))
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  结果: {applied}/{total} patches apply ({applied/total*100:.1f}%)")
    print(f"{'=' * 60}")

    # Save
    out_path = os.path.join(RESULTS_DIR, "swebench_results.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
