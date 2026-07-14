#!/usr/bin/env python3
"""
SKT Knowledge Tree Content Generator — Phase 2
===============================================
Batch-generates content for 570 nodes via DeepSeek API in 5 batches.

Usage:
  export DEEPSEEK_API_KEY="sk-xxx"
  .venv/bin/python phase2_deepseek.py --batch A     # Python leaves
  .venv/bin/python phase2_deepseek.py --batch D     # Internal nodes (parallel with A)
  .venv/bin/python phase2_deepseek.py --batch B     # Other language leaves
  .venv/bin/python phase2_deepseek.py --batch C     # Non-programming leaves
  .venv/bin/python phase2_deepseek.py --batch E     # Related-node extraction (LAST)
  .venv/bin/python phase2_deepseek.py --verify      # Verify results
  .venv/bin/python phase2_deepseek.py --all         # Run all batches in sequence

Requirements: openai, DEEPSEEK_API_KEY in environment
"""

import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TREE_ROOT = Path(__file__).parent / "smart_tree" / "根节点"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
    sys.exit(77)

# The date used for quality.last_updated
TODAY = date.today().isoformat()  # e.g. "2026-07-14"

# ---------------------------------------------------------------------------
# DeepSeek client setup
# ---------------------------------------------------------------------------

import openai
client = openai.OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/v1")


def llm_call(system: str, user: str, model="deepseek-chat",
             temperature=0.4, max_tokens=2000, retries=3) -> str | None:
    """Call DeepSeek with retry logic."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except openai.RateLimitError as e:
            wait = 30 * attempt
            print(f"    ⏳ Rate limited, waiting {wait}s (attempt {attempt}/{retries})")
            time.sleep(wait)
            last_exc = e
        except (openai.APIStatusError, openai.APITimeoutError, openai.APIConnectionError) as e:
            wait = 10 * attempt
            print(f"    ⏳ API error: {e.__class__.__name__}, waiting {wait}s (attempt {attempt}/{retries})")
            time.sleep(wait)
            last_exc = e
        except Exception as e:
            print(f"    ❌ Unexpected error: {e}")
            last_exc = e
            if attempt < retries:
                time.sleep(5)
    print(f"    ❌ All {retries} retries exhausted.")
    return None


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def parse_json_llm(text: str) -> dict | list | None:
    """Extract JSON from LLM output (handles ```json fences and bare {/[/)."""
    if not text:
        return None
    text = text.strip()
    # Try markdown code fences first
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Detect outermost structure by first non-whitespace char
    # Try array first if text starts with [ (batch E), object first if { (batches A-D)
    if text.startswith('['):
        m = re.search(r'\[[\s\S]*\]', text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    else:
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        m = re.search(r'\[[\s\S]*\]', text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Node loading / saving
# ---------------------------------------------------------------------------

def load_all_nodes() -> list[dict]:
    """Load every _node.json, attaching ephemeral _path and _file keys."""
    nodes = []
    for nj in sorted(TREE_ROOT.rglob("_node.json")):
        d = json.loads(nj.read_text(encoding="utf-8"))
        d["_path"] = str(nj.relative_to(TREE_ROOT).parent)
        d["_file"] = nj
        nodes.append(d)
    return nodes


def save_node(node: dict) -> None:
    """Persist a node (stripping ephemeral underscore keys)."""
    f = node["_file"]
    save = {k: v for k, v in node.items() if not k.startswith("_")}
    f.write_text(json.dumps(save, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Batch filters
# ---------------------------------------------------------------------------

def is_python_leaf(node: dict) -> bool:
    return node.get("is_leaf") and "Python" in node["_path"]

def is_other_lang_leaf(node: dict) -> bool:
    langs = ["JavaScript", "Java", "C／C++", "Go", "Rust"]
    return node.get("is_leaf") and any(l in node["_path"] for l in langs)

def is_non_prog_leaf(node: dict) -> bool:
    return node.get("is_leaf") and "工程技术/计算机科学" not in node["_path"]

def is_internal(node: dict) -> bool:
    return not node.get("is_leaf")

# ---------------------------------------------------------------------------
# Prompt 1 — Programming leaf content (batches A & B)
# ---------------------------------------------------------------------------

SYSTEM_PROG = "你是资深技术文档工程师。严格按 JSON 格式输出，不要其他文字。"

def prompt_prog_leaf(node: dict) -> str:
    name = node["name"]
    path = node["_path"]
    preview = node.get("content", {}).get("preview", "")
    # Determine language from path
    if "Python" in path:
        lang = "Python"
    elif "JavaScript" in path:
        lang = "JavaScript"
    elif "Java" in path:
        lang = "Java"
    elif "C／C++" in path:
        lang = "C/C++"
    elif "Go" in path:
        lang = "Go"
    elif "Rust" in path:
        lang = "Rust"
    else:
        lang = "Python"  # fallback

    return f"""知识点: {name}
知识路径: {path}
现有简述: {preview}

请生成该知识点的详细技术文档。输出严格 JSON：

{{
  "full": "## 定义\\n（一句话定义）\\n\\n## 何时使用\\n（适用场景，2-3 句）\\n\\n## 示例\\n```{lang}\\n（最小可运行示例）\\n```\\n\\n## 反例\\n（错误用法，1-2 句）\\n\\n## 边界条件\\n（注意事项，1-2 句）",
  "code_pattern": "（最小可复用代码模板，5-15 行，不要 markdown 围栏）",
  "common_pitfalls": ["（常见坑 1，具体描述）", "（常见坑 2）", "（常见坑 3）"]
}}

要求:
1. full 总长 400-800 字，5 个部分都要有
2. code_pattern 是可直接复用的模板，不含 markdown 围栏
3. common_pitfalls 2-4 个，要具体不是空话
4. 代码用 {lang}
5. 只输出 JSON，不要前后说明"""


def handle_batch_ab(node: dict) -> bool:
    """Process one programming leaf (batch A or B)."""
    # Ensure v5 fields exist
    if "content" not in node:
        node["content"] = {"preview": "", "full": "", "code_pattern": "", "common_pitfalls": []}
    if "quality" not in node:
        node["quality"] = {"confidence": 0.0, "version": 1, "reviewers": [], "last_updated": ""}
    raw = llm_call(SYSTEM_PROG, prompt_prog_leaf(node), max_tokens=2000)
    if not raw:
        return False

    data = parse_json_llm(raw)
    if not data or not isinstance(data, dict):
        # Degenerate fallback
        node["content"]["full"] = raw[:2500]
        node["content"]["code_pattern"] = ""
        node["content"]["common_pitfalls"] = []
        node["quality"]["confidence"] = 0.3
        node["quality"]["last_updated"] = TODAY
        return True

    node["content"]["full"] = str(data.get("full", raw))[:3000]
    node["content"]["code_pattern"] = str(data.get("code_pattern", ""))[:1500]
    pitfalls = data.get("common_pitfalls", [])
    if isinstance(pitfalls, list):
        node["content"]["common_pitfalls"] = [str(p)[:200] for p in pitfalls[:5]]
    else:
        node["content"]["common_pitfalls"] = []
    node["quality"]["confidence"] = 0.8
    node["quality"]["last_updated"] = TODAY
    return True


# ---------------------------------------------------------------------------
# Prompt 2 — Non-programming leaf content (batch C)
# ---------------------------------------------------------------------------

SYSTEM_NONPROG = "你是知识百科专家。严格按 JSON 格式输出。"

def prompt_nonprog_leaf(node: dict) -> str:
    name = node["name"]
    path = node["_path"]
    preview = node.get("content", {}).get("preview", "")
    return f"""知识点: {name}
知识路径: {path}
现有简述: {preview}

请生成该知识点的详细解释。输出严格 JSON：

{{
  "full": "## 定义\\n（核心定义，2-3 句）\\n\\n## 核心要点\\n（3-5 个要点，每个 1-2 句）\\n\\n## 应用场景\\n（实际用途，2-3 句）\\n\\n## 相关概念\\n（关联概念，2-3 句）"
}}

要求:
1. 总长 300-600 字
2. 4 个部分都要有
3. 全中文
4. 只输出 JSON"""


def handle_batch_c(node: dict) -> bool:
    """Process one non-programming leaf."""
    # Ensure v5 fields exist
    if "content" not in node:
        node["content"] = {"preview": "", "full": "", "code_pattern": "", "common_pitfalls": []}
    if "quality" not in node:
        node["quality"] = {"confidence": 0.0, "version": 1, "reviewers": [], "last_updated": ""}
    raw = llm_call(SYSTEM_NONPROG, prompt_nonprog_leaf(node), max_tokens=1200)
    if not raw:
        return False

    data = parse_json_llm(raw)
    if not data or not isinstance(data, dict):
        node["content"]["full"] = raw[:2500]
        node["quality"]["confidence"] = 0.3
        node["quality"]["last_updated"] = TODAY
        return True

    node["content"]["full"] = str(data.get("full", raw))[:3000]
    node["quality"]["confidence"] = 0.7
    node["quality"]["last_updated"] = TODAY
    return True


# ---------------------------------------------------------------------------
# Prompt 3 — Internal node summary (batch D)
# ---------------------------------------------------------------------------

SYSTEM_INTERNAL = "你是知识树架构师。严格按 JSON 格式输出。"

def prompt_internal(node: dict) -> str:
    name = node["name"]
    path = node["_path"]
    children = node.get("children", [])
    children_str = ", ".join(children[:8])
    return f"""节点: {name}
路径: {path}
子节点: {children_str}

请生成该分支的综述。输出严格 JSON：

{{
  "summary": "（200-400 字综述，说明本分支涵盖什么、子主题之间的关系、学习路径）",
  "example_queries": ["（典型问题 1）", "（典型问题 2）", "（典型问题 3）"]
}}

要求:
1. summary 200-400 字
2. example_queries 3-5 个该分支能回答的典型问题
3. 全中文
4. 只输出 JSON"""


def handle_batch_d(node: dict) -> bool:
    """Process one internal node."""
    raw = llm_call(SYSTEM_INTERNAL, prompt_internal(node), max_tokens=1000)
    if not raw:
        return False

    data = parse_json_llm(raw)
    if not data or not isinstance(data, dict):
        node["summary"] = raw[:1500]
        node["example_queries"] = []
        return True

    node["summary"] = str(data.get("summary", raw))[:1500]
    queries = data.get("example_queries", [])
    node["example_queries"] = [str(q)[:200] for q in queries[:5]] if isinstance(queries, list) else []
    return True


# ---------------------------------------------------------------------------
# Prompt 4 — Related-node extraction (batch E)
# ---------------------------------------------------------------------------

SYSTEM_RELATED = "你是知识关联分析师。严格按 JSON 数组输出。"

RELATION_TYPES = {"prerequisite", "related_pattern", "cross_lang_equivalent", "contrast", "application", "related"}

def build_candidates(node: dict, all_nodes: list[dict]) -> list[str]:
    """Build candidate related-node list per rules in section 4."""
    path = node["_path"]
    top = path.split("/")[0] if "/" in path else path
    node_id = node["node_id"]

    candidates_set: set[str] = set()
    # 1. Same top-level classification leaves
    for n in all_nodes:
        if n.get("is_leaf") and n["_path"].split("/")[0] == top and n["node_id"] != node_id:
            # Include path context for disambiguation (name alone can collide)
            candidates_set.add(f"{n['name']} ({n['_path']})")

    # 2. Same-name nodes (cross-language candidates)
    for n in all_nodes:
        if n.get("is_leaf") and n["name"] == node["name"] and n["node_id"] != node_id:
            candidates_set.add(f"{n['name']} ({n['_path']})")

    # 3. Limit to 40
    return list(candidates_set)[:40]


def prompt_related(node: dict, candidates: list[str]) -> str:
    name = node["name"]
    path = node["_path"]
    cand_str = ", ".join(candidates)
    return f"""当前知识点: {name}
路径: {path}

候选关联节点: {cand_str}

从中选出 0-5 个真正相关的节点，并标注关系类型。输出 JSON 数组：

[
  {{"id": "节点名", "type": "prerequisite"}},
  {{"id": "节点名", "type": "related_pattern"}}
]

关系类型（必须是这 5 种之一）:
- prerequisite: 前置知识（学这个之前要先学那个）
- related_pattern: 相关模式（类似但不同）
- cross_lang_equivalent: 跨语言等价（如 Python 闭包 ↔ JS 闭包）
- contrast: 对比关系（常被混淆）
- application: 应用场景

要求:
1. 只选真正相关的，宁缺毋滥
2. 只输出 JSON 数组，不要其他文字"""


def handle_batch_e(node: dict, all_nodes: list[dict], name_to_id: dict[str, str]) -> bool:
    """Process related-node extraction for one leaf."""
    candidates = build_candidates(node, all_nodes)
    if not candidates:
        node["related_nodes"] = []
        return True  # No candidates, nothing to do

    raw = llm_call(SYSTEM_RELATED, prompt_related(node, candidates),
                   model="deepseek-reasoner", temperature=0.2, max_tokens=600, retries=2)
    if not raw:
        # Empty response — treat as no related nodes
        node["related_nodes"] = [{"id": "__none__", "type": "related"}]
        return True

    data = parse_json_llm(raw)
    if data is None:
        # Parse failed — treat as no related nodes
        node["related_nodes"] = [{"id": "__none__", "type": "related"}]
        return True
    # Normalize: single dict -> list of one
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return False

    # Map node names → node_ids, validate types, deduplicate
    # Build a name→[node_id] map to handle duplicate names across languages
    name_to_ids: dict[str, list[str]] = {}
    for n in all_nodes:
        name_to_ids.setdefault(n["name"], []).append(n["node_id"])

    typed: list[dict] = []
    seen: set[str] = set()
    for r in data:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        rtype = r.get("type", "related")
        if not rid:
            continue
        if rtype not in RELATION_TYPES:
            rtype = "related"
        # Try direct name lookup first, then fall back
        nid = name_to_id.get(rid)
        if not nid:
            # Check all nodes with this name (for cross-lang duplicates)
            candidates = name_to_ids.get(rid, [])
            # Pick the one in the same classification if possible
            for cid in candidates:
                if cid != node["node_id"]:
                    nid = cid
                    break
        if nid and nid != node["node_id"] and nid not in seen:
            seen.add(nid)
            typed.append({"id": nid, "type": rtype})

    node["related_nodes"] = typed[:5]
    node["related_nodes"] = [r for r in node["related_nodes"] if r.get("id") != "__none__"]
    return True


# ---------------------------------------------------------------------------
# Batch runners
# ---------------------------------------------------------------------------

def run_batch(nodes: list[dict], handler, batch_label: str, skip_check=None):
    """Generic batch runner with skip logic."""
    # Sort by level ascending so shallow nodes process first
    targets = sorted(nodes, key=lambda n: n.get("level", 99))

    # Skip already-done
    if skip_check:
        before = len(targets)
        targets = [n for n in targets if skip_check(n)]
        print(f"  跳过 {before - len(targets)} 个已完成节点")

    print(f"\n{'='*60}")
    print(f"批次 {batch_label}: 待处理 {len(targets)} 个节点")
    print(f"{'='*60}")

    success = 0
    fail = 0
    for i, node in enumerate(targets, 1):
        name = node["name"]
        path_short = "/".join(node["_path"].split("/")[-3:])
        print(f"  [{i}/{len(targets)}] ({path_short}) {name} ...", end=" ", flush=True)
        try:
            ok = handler(node)
            if ok:
                save_node(node)
                print("✅")
                success += 1
            else:
                print("❌ (handler returned False)")
                fail += 1
        except Exception as e:
            print(f"❌ {e}")
            fail += 1

    print(f"\n批次 {batch_label} 完成: ✅ {success}   ❌ {fail}")
    return success, fail


def skip_confidence_ge_07(n: dict) -> bool:
    return n.get("quality", {}).get("confidence", 0) < 0.7

def skip_has_summary(n: dict) -> bool:
    return not n.get("summary")

def skip_has_related(n: dict) -> bool:
    # Skip only if related_nodes is explicitly a non-empty list
    rn = n.get("related_nodes")
    return rn is None or (isinstance(rn, list) and len(rn) == 0)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify():
    """Check completeness of the tree."""
    all_nodes = load_all_nodes()
    leaf_total = sum(1 for n in all_nodes if n.get("is_leaf"))
    internal_total = sum(1 for n in all_nodes if not n.get("is_leaf"))

    stats = {
        "叶子有full": 0,
        "叶子有code_pattern": 0,
        "叶子有common_pitfalls": 0,
        "中间节点有summary": 0,
        "中间节点有example_queries": 0,
        "叶子有related_nodes": 0,
        "高质量(confidence>=0.7)": 0,
        "有confidence>0": 0,
    }

    for n in all_nodes:
        if n.get("is_leaf"):
            c = n.get("content", {})
            if c.get("full"):         stats["叶子有full"] += 1
            if c.get("code_pattern"): stats["叶子有code_pattern"] += 1
            if c.get("common_pitfalls"): stats["叶子有common_pitfalls"] += 1
            if n.get("related_nodes"): stats["叶子有related_nodes"] += 1
        else:
            if n.get("summary"):            stats["中间节点有summary"] += 1
            if n.get("example_queries"):   stats["中间节点有example_queries"] += 1

        conf = n.get("quality", {}).get("confidence", 0)
        if conf >= 0.7:
            stats["高质量(confidence>=0.7)"] += 1
        if conf > 0:
            stats["有confidence>0"] += 1

    print(f"\n{'='*60}")
    print(f"验证结果 (总节点: {len(all_nodes)})")
    print(f"{'='*60}")
    print(f"  叶子总数: {leaf_total}")
    print(f"  中间节点总数: {internal_total}")
    print()
    for k, v in stats.items():
        if "叶子" in k:
            print(f"  {k}: {v}/{leaf_total} ({v*100//leaf_total}%)" if leaf_total else f"  {k}: {v}/0")
        elif "中间" in k:
            print(f"  {k}: {v}/{internal_total} ({v*100//internal_total}%)" if internal_total else f"  {k}: {v}/0")
        elif "高质量" in k or "confidence" in k:
            print(f"  {k}: {v}/{len(all_nodes)} ({v*100//len(all_nodes)}%)")
        else:
            print(f"  {k}: {v}")

    # Also show sample full content length
    has_full = [n for n in all_nodes if n.get("is_leaf") and n.get("content", {}).get("full")]
    if has_full:
        avg_len = sum(len(n["content"]["full"]) for n in has_full) // len(has_full)
        print(f"\n  content.full 平均长度: {avg_len} 字 (样本 {len(has_full)} 个)")
    has_code = [n for n in all_nodes if n.get("is_leaf") and n.get("content", {}).get("code_pattern")]
    if has_code:
        avg_code = sum(len(n["content"]["code_pattern"]) for n in has_code) // len(has_code)
        print(f"  code_pattern 平均长度: {avg_code} 字 (样本 {len(has_code)} 个)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SKT Phase 2 — DeepSeek content generation")
    parser.add_argument("--batch", choices=["A", "B", "C", "D", "E", "F"],
                        help="Run a single batch")
    parser.add_argument("--all", action="store_true",
                        help="Run all batches in sequence: A → D → B → C → E")
    parser.add_argument("--verify", action="store_true",
                        help="Verify node content completeness")
    args = parser.parse_args()

    if args.verify:
        verify()
        return

    all_nodes = load_all_nodes()

    if args.all:
        print("运行全部批次: A → D → B → C → E\n")

        # A: Python leaves
        targets = [n for n in all_nodes if is_python_leaf(n)]
        run_batch(targets, handle_batch_ab, "A (Python 叶子)", skip_confidence_ge_07)

        # D: Internal nodes (can run in parallel with A, but we do sequential for simplicity)
        all_nodes = load_all_nodes()  # reload
        targets = [n for n in all_nodes if is_internal(n)]
        run_batch(targets, handle_batch_d, "D (中间节点)", skip_has_summary)

        # B: Other language leaves
        all_nodes = load_all_nodes()
        targets = [n for n in all_nodes if is_other_lang_leaf(n)]
        run_batch(targets, handle_batch_ab, "B (其他语言叶子)", skip_confidence_ge_07)

        # C: Non-programming leaves
        all_nodes = load_all_nodes()
        targets = [n for n in all_nodes if is_non_prog_leaf(n)]
        run_batch(targets, handle_batch_c, "C (非编程叶子)", skip_confidence_ge_07)

        # E: Related-node extraction (LAST)
        all_nodes = load_all_nodes()
        name_to_id = {n["name"]: n["node_id"] for n in all_nodes}
        targets = [n for n in all_nodes if n.get("is_leaf")]
        run_batch(targets, lambda n: handle_batch_e(n, all_nodes, name_to_id),
                  "E (关联抽取)", skip_has_related)

        # Final verification
        print("\n" + "="*60)
        print("全部批次执行完毕，运行验证...")
        print("="*60)
        load_all_nodes()  # fresh load
        verify()

    elif args.batch == "A":
        targets = [n for n in all_nodes if is_python_leaf(n)]
        run_batch(targets, handle_batch_ab, "A (Python 叶子)", skip_confidence_ge_07)
    elif args.batch == "B":
        targets = [n for n in all_nodes if is_other_lang_leaf(n)]
        run_batch(targets, handle_batch_ab, "B (其他语言叶子)", skip_confidence_ge_07)
    elif args.batch == "C":
        targets = [n for n in all_nodes if is_non_prog_leaf(n)]
        run_batch(targets, handle_batch_c, "C (非编程叶子)", skip_confidence_ge_07)
    elif args.batch == "F":
        # Remaining CS leaves not covered by A or B (DevOps, OS, DB, networks, etc.)
        prog_langs = ["Python", "JavaScript", "Java", "C／C++", "Go", "Rust"]
        targets = [n for n in all_nodes if n.get("is_leaf") and "工程技术/计算机科学" in n["_path"]
                   and not any(l in n["_path"] for l in prog_langs)]
        run_batch(targets, handle_batch_c, "F (CS 非编程叶子)", skip_confidence_ge_07)
    elif args.batch == "D":
        targets = [n for n in all_nodes if is_internal(n)]
        run_batch(targets, handle_batch_d, "D (中间节点)", skip_has_summary)
    elif args.batch == "E":
        name_to_id = {n["name"]: n["node_id"] for n in all_nodes}
        targets = [n for n in all_nodes if n.get("is_leaf")]
        run_batch(targets, lambda n: handle_batch_e(n, all_nodes, name_to_id),
                  "E (关联抽取)", skip_has_related)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
