#!/usr/bin/env python3
"""
HumanEval 评测 — 知识树 + qwen2.5:7b (修复版)
- 增量保存结果（每完成一题立即写入）
- 短超时 + 重试（30s→60s）
- 跳过卡住的题目
"""
import sys, os, json, urllib.request, time, re
from pathlib import Path
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.persistence import TreePersistence

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
DATA_PATH = "data/HumanEval.jsonl"
RESULTS_FILE = "results/humaneval_results.jsonl"


def load_tree():
    db = TreePersistence("smart_tree")
    tree = db.load_tree()
    return tree, db


def load_humaneval(path: str) -> List[Dict]:
    problems = []
    with open(path) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


def query_tree(tree, prompt: str) -> str:
    qvec = tree.encoder.encode(prompt)
    results = tree.penetrate(prompt, query_vec=qvec, verbose=False)
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
    return "\n".join(items[:5]) if items else "(无相关知识点)"


def call_ollama(prompt: str) -> Optional[str]:
    """调用 Ollama，短超时 + 重试"""
    timeouts = [45, 60]  # 首次45s, 重试60s
    for attempt, timeout in enumerate(timeouts):
        data = {
            "model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }
        req = urllib.request.Request(
            OLLAMA, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode())
                return result.get("response", "").strip()
        except Exception as e:
            if attempt < len(timeouts) - 1:
                print(f"    ⏳ 重试 ({timeout}s超时): {e}")
                time.sleep(3)
            else:
                print(f"    ❌ Ollama失败 ({timeout}s超时): {e}")
                return None
    return None


def extract_code(generated: str) -> str:
    m = re.search(r"```(?:python)?\s*\n?(.*?)```", generated, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code.startswith("def ") or code.startswith("from ") or code.startswith("import "):
            return code
    lines = generated.split("\n")
    code_lines = []
    in_func = False
    for line in lines:
        if line.startswith("def "):
            in_func = True
            code_lines.append(line)
        elif in_func:
            if line.startswith("def ") or line.startswith("class "):
                break
            code_lines.append(line)
    return "\n".join(code_lines) if code_lines else generated


def run_test(problem: Dict, generated_code: str) -> bool:
    prompt = problem["prompt"]
    test = problem["test"]
    exec_globals = {"__builtins__": __builtins__}
    try:
        prompt_lines = prompt.split("\n")
        imports = []
        in_docstring = False
        for line in prompt_lines:
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                in_docstring = True
                continue
            if in_docstring:
                if line.strip().endswith('"""') or line.strip().endswith("'''"):
                    in_docstring = False
                continue
            if line.startswith("from ") or line.startswith("import "):
                imports.append(line)
        import_block = "\n".join(imports)
        entry_point = problem['entry_point']
        full_code = import_block + "\n\n" + generated_code + "\n\n" + test + f"\n\ncheck({entry_point})"
        exec(compile(full_code, "<humaneval>", "exec"), exec_globals)
        return True
    except AssertionError:
        return False
    except Exception as e:
        print(f"    [TEST ERROR] {type(e).__name__}: {e}")
        return False


def load_completed_ids():
    """加载已完成的任务ID（断点续传）"""
    done = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["task_id"])
                    except:
                        pass
    return done


def save_result(result: dict):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main():
    os.makedirs("results", exist_ok=True)

    completed = load_completed_ids()
    print(f"  已有 {len(completed)} 条完成结果（断点续传）")

    print("=" * 60)
    print("  HumanEval 评测 — 知识树 + qwen2.5:7b")
    print("=" * 60)

    print("\n[1/3] 加载知识树...")
    tree, db = load_tree()
    print(f"  编码器: {type(tree.encoder).__name__}, 节点: {tree.stats()['total_nodes']}")

    print("\n[2/3] 加载 HumanEval...")
    problems = load_humaneval(DATA_PATH)
    print(f"  Loaded {len(problems)} problems")

    print(f"\n[3/3] 运行评测 ({len(problems)} 题)...")
    passed = 0
    skipped = 0
    total = len(problems)

    for i, prob in enumerate(problems):
        task_id = prob["task_id"]
        entry_point = prob["entry_point"]

        if task_id in completed:
            skipped += 1
            if skipped % 20 == 1:
                print(f"  ...已跳过 {skipped} 题（断点续传）")
            continue

        print(f"\n  [{i+1}/{total}] {task_id} ({entry_point})")

        # Tree context
        print(f"    Querying tree...", end=" ", flush=True)
        ctx = query_tree(tree, prob["prompt"])
        print(f"OK", flush=True)

        # Generate
        print(f"    Generating...", end=" ", flush=True)
        start = time.time()
        generated = call_ollama(prob["prompt"] + f"\n\nRelevant knowledge:\n{ctx}\n\nComplete the function:")
        elapsed = time.time() - start

        if not generated:
            print(f"❌ (generation failed)")
            result = {
                "task_id": task_id, "entry_point": entry_point,
                "passed": False, "error": "generation_failed",
                "time": elapsed, "tree_context": ctx,
            }
            save_result(result)
            continue

        print(f"({elapsed:.1f}s)", flush=True)

        # Test
        code = extract_code(generated)
        ok = run_test(prob, code)
        if ok:
            passed += 1
            print(f"    ✅ PASS")
        else:
            print(f"    ❌ FAIL")

        save_result({
            "task_id": task_id, "entry_point": entry_point,
            "passed": ok, "generated": generated, "code": code,
            "time": elapsed, "tree_context": ctx,
        })

        # 每10题保存进度
        if (i + 1) % 10 == 0:
            current_pass = passed
            current_total = i + 1 - skipped
            print(f"\n  >>> 进度: {current_total} 题完成, {current_pass}/{current_total} ({current_pass/current_total*100:.1f}%)")

    # Summary
    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except:
                    pass
    final_passed = sum(1 for r in results if r.get("passed"))
    print(f"\n{'=' * 60}")
    print(f"  结果: {final_passed}/{len(results)} passed ({final_passed/len(results)*100:.1f}%)")
    print(f"{'=' * 60}")
    print(f"\n  Results saved to {RESULTS_FILE}")

    db.close()


if __name__ == "__main__":
    main()
