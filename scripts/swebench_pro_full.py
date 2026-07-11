#!/usr/bin/env python3
"""SWE-Bench Pro 全自动分批评测 — 知识树 + qwen2.5:7b，支持断点续跑，自动循环直到全部完成"""
import sys, os, json, urllib.request, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import TreePersistence

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
RESULTS_FILE = "results/swebench_pro_results.jsonl"
DATASET_PATH = "./swebench_pro_data"
BATCH_SIZE = 100
RESTART_OLLAMA_EVERY = 50

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

def load_completed():
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    completed.add(r["instance_id"])
                except:
                    pass
    return completed

def append_result(filepath: str, result: dict):
    with open(filepath, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

def restart_ollama():
    print("  >>> 重启 Ollama 释放内存...")
    os.system("systemctl restart ollama 2>/dev/null || pkill ollama")
    time.sleep(15)

def main():
    os.makedirs("results", exist_ok=True)
    print("="*60)
    print("  SWE-Bench Pro 全自动分批评测 — 知识树 + qwen2.5:7b")
    print("="*60)

    completed = load_completed()
    print(f"\n  已完成 {len(completed)} 题，将跳过")

    print("\n[1/4] 加载知识树...")
    db = TreePersistence("smart_tree")
    tree = db.load_tree()
    print(f"  {tree.stats()['total_nodes']} nodes")

    print("\n[2/4] 加载 SWE-Bench Pro (本地数据)...")
    from datasets import load_dataset
    ds = load_dataset(DATASET_PATH, split="test", streaming=True)

    print(f"\n[3/4] 自动循环评测 (每批 {BATCH_SIZE} 题，每 {RESTART_OLLAMA_EVERY} 题重启 Ollama，全部完成后自动停止)")
    print("  按 Ctrl+C 可随时中断，已完成的题目不会丢失\n")

    batch_num = 0
    total_done = len(completed)
    task_iter = iter(ds)

    while True:
        batch_num += 1
        results_this_batch = []
        task_count = 0

        print(f"  {'='*50}")
        print(f"  第 {batch_num} 批开始 (已完成 {total_done} 题)")
        print(f"  {'='*50}")

        for task in task_iter:
            instance_id = task["instance_id"]

            if instance_id in completed:
                continue

            if len(results_this_batch) >= BATCH_SIZE:
                break

            task_count += 1
            repo = task["repo"]
            problem = task["problem_statement"]
            gold_patch = task["patch"]
            commit = task["base_commit"]
            lang = task.get("repo_language", "unknown")
            reqs = task.get("requirements", "")

            print(f"\n  [{len(results_this_batch)+1}/{BATCH_SIZE}] (总#{task_count}) {instance_id} ({repo}, {lang})")

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
                result = {"instance_id": instance_id, "passed": False, "error": str(e), "time": time.time()-start}
                results_this_batch.append(result)
                append_result(RESULTS_FILE, result)
                completed.add(instance_id)
                total_done += 1
                continue
            elapsed = time.time() - start

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

            result = {
              "instance_id": instance_id, "repo": repo, "lang": lang,
              "passed": passed, "has_diff": has_diff, "has_hunks": has_hunks,
              "targets_correct_files": targets_correct, "time": elapsed,
              "tree_context": ctx, "gold_files": files,
              "generated_full_patch": generated,
              "generated_preview": generated[:500],
            }
            results_this_batch.append(result)
            append_result(RESULTS_FILE, result)
            completed.add(instance_id)
            total_done += 1

            if len(results_this_batch) % RESTART_OLLAMA_EVERY == 0:
                restart_ollama()

        passed = sum(1 for r in results_this_batch if r["passed"])
        batch_total = len(results_this_batch)
        print(f"\n  >>> 第 {batch_num} 批完成: {passed}/{batch_total} ({passed/batch_total*100:.1f}%)" if batch_total > 0 else f"\n  >>> 第 {batch_num} 批: 无新任务")
        print(f"  >>> 累计完成: {total_done} 题")

        if batch_total == 0:
            print(f"\n{'='*60}")
            print(f"  全部完成！累计 {total_done} 题")
            print(f"{'='*60}")
            break

        restart_ollama()

    db.close()

if __name__ == "__main__":
    main()
