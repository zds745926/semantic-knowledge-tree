#!/usr/bin/env python3
"""SKT 树完整性检查 — 验证所有 570 节点内容字段完整"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "smart_tree" / "根节点"
nodes = list(root.rglob("_node.json"))
errors = []

for nj in nodes:
    d = json.loads(nj.read_text(encoding="utf-8"))
    name = d.get("name", "?")
    if d.get("is_leaf"):
        c = d.get("content", {})
        if not c.get("full"):
            errors.append(f"叶子 {name}: content.full 缺失")
    else:
        if not d.get("summary"):
            errors.append(f"中间节点 {name}: summary 缺失")
        if not d.get("example_queries"):
            errors.append(f"中间节点 {name}: example_queries 缺失")

if errors:
    for e in errors:
        print(f"❌ {e}")
    print(f"\n共 {len(errors)} 个错误")
    exit(1)
else:
    print(f"✅ 全部 {len(nodes)} 个节点完整性检查通过")
