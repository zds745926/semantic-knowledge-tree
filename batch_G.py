#!/usr/bin/env python3
"""Batch G: 用 deepseek-chat 补全剩余的 related_nodes"""
import json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.insert(0, (Path(__file__).parent / ".venv" / "lib" / "python3.12" / "site-packages").as_posix())
import openai

client = openai.OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com/v1")
TREE_ROOT = Path(__file__).parent / "smart_tree" / "根节点"
SYSTEM = "你是知识关联分析师。严格按 JSON 数组输出。"


def load_nodes():
    nodes = []
    for nj in sorted(TREE_ROOT.rglob("_node.json")):
        d = json.loads(nj.read_text(encoding="utf-8"))
        if d.get("is_leaf"):
            d["_path"] = str(nj.relative_to(TREE_ROOT).parent)
            d["_file"] = nj
            nodes.append(d)
    return nodes


def save_node(d):
    f = d["_file"]
    save = {k: v for k, v in d.items() if not k.startswith("_")}
    f.write_text(json.dumps(save, ensure_ascii=False, indent=2), encoding="utf-8")


def llm_call(prompt, retries=2):
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=600,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⏳ API error: {e}")
            if attempt < retries:
                time.sleep(5)
    return None


def parse_json(text):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    if text.startswith("["):
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


nodes = load_nodes()
name_to_id = {n["name"]: n["node_id"] for n in nodes}
name_to_ids = defaultdict(list)
for n in nodes:
    name_to_ids[n["name"]].append(n["node_id"])

empty = [n for n in nodes if not n.get("related_nodes")]
print(f"需补全: {len(empty)} 个节点", flush=True)

success = 0
for i, n in enumerate(empty, 1):
    path = n["_path"]
    top = path.split("/")[0]

    # Build candidates: same top-level classification
    cand_set = set()
    for m in nodes:
        if m["_path"].split("/")[0] == top and m["node_id"] != n["node_id"]:
            cand_set.add(m["name"])
    # Same-name (cross-lang)
    for m in nodes:
        if m["name"] == n["name"] and m["node_id"] != n["node_id"]:
            cand_set.add(m["name"])
    cand = list(cand_set)[:30]

    if not cand:
        n["related_nodes"] = []
        save_node(n)
        success += 1
        continue

    prompt = f"""当前知识点: {n['name']}
路径: {path}

候选关联节点: {', '.join(cand)}

从中选出 0-5 个真正相关的节点并标注关系类型。输出 JSON 数组：
[{{"id": "节点名", "type": "prerequisite"}}]

关系类型(必须是这5种): prerequisite, related_pattern, cross_lang_equivalent, contrast, application

要求: 只选真正相关的，宁缺毋滥。只输出 JSON 数组。"""

    raw = llm_call(prompt)
    if not raw:
        n["related_nodes"] = []
        save_node(n)
        success += 1
        continue

    data = parse_json(raw)
    if not data or not isinstance(data, list):
        n["related_nodes"] = []
        save_node(n)
        success += 1
        continue

    typed = []
    seen = set()
    for r in data:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        rtype = r.get("type", "related")
        if not rid:
            continue
        if rtype not in ("prerequisite", "related_pattern", "cross_lang_equivalent", "contrast", "application"):
            rtype = "related"
        nid = name_to_id.get(rid)
        if not nid:
            ids = name_to_ids.get(rid, [])
            for cid in ids:
                if cid != n["node_id"]:
                    nid = cid
                    break
        if nid and nid != n["node_id"] and nid not in seen:
            seen.add(nid)
            typed.append({"id": nid, "type": rtype})

    n["related_nodes"] = typed[:5]
    save_node(n)
    success += 1
    print(f"  [{i}/{len(empty)}] {n['name']}: {len(typed)} related", flush=True)
    time.sleep(0.3)

print(f"\n完成: ✅ {success}/{len(empty)}", flush=True)
