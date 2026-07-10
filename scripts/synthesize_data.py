#!/usr/bin/env python3
"""
利用知识树自动合成训练数据，扩展数据集
用法: python scripts/synthesize_data.py
"""
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tree import SemanticKnowledgeTree
from core.persistence import TreePersistence
from core.encoder import SemanticEncoder, FallbackEncoder


def load_tree():
    db = TreePersistence("smart_tree")
    tree = db.load_tree()
    if tree is None:
        raise RuntimeError("知识树不存在，请先运行 demo.py 构建")
    try:
        tree.encoder = SemanticEncoder("all-MiniLM-L6-v2")
    except Exception:
        tree.encoder = FallbackEncoder(384)
    return tree, db


def generate_samples(tree, count=200):
    samples = []
    all_leaves = list(tree._all_leaves.values())
    random.seed(42)
    random.shuffle(all_leaves)

    query_templates = [
        "什么是{title}？",
        "介绍一下{title}",
        "讲讲{title}",
        "关于{title}",
        "说说{title}",
    ]

    for leaf in all_leaves:
        if len(samples) >= count:
            break

        title = leaf.name
        preview = ""
        if leaf.data_pointers:
            preview = leaf.data_pointers[0].get("content_preview", "")

        template = random.choice(query_templates)
        query = template.format(title=title)

        # 执行树检索
        query_vec = tree.encoder.encode(query)
        pen = tree.penetrate(query, query_vec=query_vec)
        short = tree.shortcut_search(query, query_vec=query_vec)

        # 格式化输入
        pen_lines = []
        for p in pen[:8]:
            path = ' → '.join(p.get("path", []))
            weight = p.get("total_weight", 0)
            pen_lines.append(f"  [{weight:.4f}] {path}")
            if p.get("data_pointers"):
                dp = p["data_pointers"][0]
                pen_lines.append(f"          内容: {dp.get('content_preview','')[:100]}")

        short_lines = []
        for s in short[:5]:
            spath = ' → '.join(s.get("path", []))
            sim = s.get("score", 0)
            short_lines.append(f"  [{sim:.4f}] {spath}")
            if s.get("data_pointers"):
                dp = s["data_pointers"][0]
                short_lines.append(f"          内容: {dp.get('content_preview','')[:100]}")

        user_input = (
            f"用户查询: {query}\n\n"
            f"渗透路径:\n" + "\n".join(pen_lines) + "\n\n"
            f"捷径叶子:\n" + "\n".join(short_lines)
        )

        # 构造期望输出
        path_parts = []
        for p in pen[:3]:
            pname = ' → '.join(p.get("path", []))
            path_parts.append(f"  - {pname}（权重 {p.get('total_weight',0):.4f}）")

        path_analysis = f"树通过以下路径定位到相关知识：\n" + "\n".join(path_parts)
        leaf_content = f"相关知识概要：{preview[:300] if preview else title}"
        conclusion = f"{title} 是 {preview[:80] if preview else title}。"

        output = (
            f"【路径分析】\n{path_analysis}\n\n"
            f"【知识汇总】\n{leaf_content}\n\n"
            f"【结论】\n{conclusion}"
        )

        samples.append({
            "input": user_input,
            "output": output,
            "metadata": {
                "source": "auto_synthetic",
                "query": query,
                "leaf_id": leaf.node_id,
            }
        })

        if (len(samples) % 50) == 0:
            print(f"  已生成 {len(samples)} 条...")

    return samples


def main():
    print("[1/2] 加载知识树...")
    tree, db = load_tree()
    stats = tree.stats()
    print(f"  ✅ {stats['total_nodes']} 节点, {stats['leaf_nodes']} 叶子")

    print("[2/2] 生成训练样本...")
    samples = generate_samples(tree, count=200)

    output_file = "data/train_data_synthetic.jsonl"
    os.makedirs("data", exist_ok=True)
    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✅ 生成 {len(samples)} 条合成数据 → {output_file}")
    db.close()


if __name__ == "__main__":
    main()
