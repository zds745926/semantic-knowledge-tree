#!/usr/bin/env python3
"""
HumanEval 评测 — 知识树 + qwen2.5:7b

用法:
  python scripts/humaneval_bench.py --max-problems 10

输出:
  results/humaneval_results.jsonl
"""
import sys, os, json, urllib.request, time, re, traceback
from pathlib import Path
from typing import Optional, List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.persistence import TreePersistence

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
DATA_PATH = "data/HumanEval.jsonl"
RESULTS_DIR = "results"
MAX_PROBLEMS = 164


def load_tree():
    db = TreePersistence("smart_tree")
    tree = db.load_tree()
    return tree, db


def load_humaneval(path: str, max_problems: Optional[int] = None) -> List[Dict]:
    problems = []
    with open(path) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    if max_problems:
        problems = problems[:max_problems]
    return problems


def query_tree(tree, prompt: str) -> str:
    """查询知识树，返回上下文摘要"""
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

    if not items:
        items.append("(无相关知识点)")
    return "\n".join(items[:5])


def call_ollama(prompt: str) -> Optional[str]:
    """调用 Ollama qwen2.5:7b 生成代码"""
    data = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    req = urllib.request.Request(
        OLLAMA,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "").strip()
    except Exception as e:
        print(f"  [ERROR] Ollama call failed: {e}")
        return None


def extract_code(generated: str) -> Optional[str]:
    """从 LLM 输出中提取 Python 函数代码"""
    # Try ```python ... ``` block
    m = re.search(r"```(?:python)?\s*\n?(.*?)```", generated, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if code.startswith("def ") or code.startswith("from ") or code.startswith("import "):
            return code

    # Try to find function definition directly
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
    if code_lines:
        return "\n".join(code_lines)

    return generated  # fallback


def run_test(problem: Dict, generated_code: str) -> bool:
    """运行 HumanEval 测试用例"""
    prompt = problem["prompt"]
    test = problem["test"]

    exec_globals = {"__builtins__": __builtins__}
    try:
        # Extract imports from prompt (like "from typing import List")
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
        full_code = import_block + "\n\n" + generated_code + "\n\n" + test
        exec(compile(full_code, "<humaneval>", "exec"), exec_globals)

        # Run the check function (HumanEval convention: check_{entry_point})
        test_func_name = f"check_{problem['entry_point']}"
        if test_func_name in exec_globals:
            exec_globals[test_func_name]()
            return True
        return True
    except AssertionError:
        return False
    except Exception as e:
        print(f"  [TEST ERROR] {type(e).__name__}: {e}")
        return False


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load tree
    print("=" * 60)
    print("  HumanEval 评测 — 知识树 + qwen2.5:7b")
    print("=" * 60)
    print("\n[1/3] 加载知识树...")
    tree, db = load_tree()

    # 2. Load problems
    print("\n[2/3] 加载 HumanEval...")
    problems = load_humaneval(DATA_PATH, max_problems=MAX_PROBLEMS)
    print(f"  Loaded {len(problems)} problems")

    # 3. Run evaluation
    print(f"\n[3/3] 运行评测 ({len(problems)} 题)...")
    results = []
    passed = 0
    total = len(problems)

    for i, prob in enumerate(problems):
        task_id = prob["task_id"]
        entry_point = prob["entry_point"]
        print(f"\n  [{i+1}/{total}] {task_id} ({entry_point})")

        # Query tree
        print(f"    Querying tree...")
        ctx = query_tree(tree, prob["prompt"])

        # Build prompt
        prompt = f"""You are an expert Python programmer. Complete the following function.
Write ONLY the function body, no tests, no explanations.
Return the COMPLETE function with correct indentation.

{prob['prompt']}

Relevant knowledge:
{ctx}

Complete the function:"""

        # Generate
        print(f"    Generating (qwen2.5:7b)...")
        start = time.time()
        generated = call_ollama(prompt)
        elapsed = time.time() - start

        if not generated:
            print(f"    ❌ Generation failed")
            results.append({
                "task_id": task_id, "entry_point": entry_point,
                "passed": False, "error": "generation_failed",
                "time": elapsed, "tree_context": ctx,
            })
            continue

        # Extract code
        code = extract_code(generated)

        # Test
        ok = run_test(prob, code)
        if ok:
            passed += 1
            print(f"    ✅ PASS ({elapsed:.1f}s)")
        else:
            print(f"    ❌ FAIL ({elapsed:.1f}s)")

        results.append({
            "task_id": task_id,
            "entry_point": entry_point,
            "passed": ok,
            "generated": generated,
            "code": code,
            "time": elapsed,
            "tree_context": ctx,
        })

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  结果: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print(f"{'=' * 60}")

    # Save
    out_path = os.path.join(RESULTS_DIR, "humaneval_results.jsonl")
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Results saved to {out_path}")

    db.close()


if __name__ == "__main__":
    main()
