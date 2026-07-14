#!/usr/bin/env python3
"""SKT 树完整性测试 — pytest 方式"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "smart_tree" / "根节点"
PROG_LANG_DIRS = {"Python", "JavaScript", "Java", "C／C++", "Go", "Rust"}


@pytest.fixture(scope="session")
def all_nodes():
    """加载所有节点数据"""
    nodes = []
    errors = []
    for nj in sorted(ROOT.rglob("_node.json")):
        try:
            d = json.loads(nj.read_text(encoding="utf-8"))
            d["_name"] = d.get("name", str(nj.parent.name))
            d["_path"] = str(nj.relative_to(ROOT).parent)
            d["_path_parts"] = set(nj.relative_to(ROOT).parent.parts)
            nodes.append(d)
        except (json.JSONDecodeError, OSError) as e:
            errors.append(f"加载失败: {nj} -> {e}")
    if errors:
        raise RuntimeError("\n".join(errors))
    return nodes


def test_total_nodes(all_nodes):
    """报告节点总数"""
    leaf = sum(1 for d in all_nodes if d.get("is_leaf"))
    internal = sum(1 for d in all_nodes if not d.get("is_leaf"))
    print(f"叶子: {leaf}, 中间节点: {internal}, 总计: {len(all_nodes)}")
    assert len(all_nodes) > 0, "至少应有 1 个节点"


def test_every_leaf_has_full_content(all_nodes):
    """所有叶子节点应有 content.full"""
    missing = []
    for d in all_nodes:
        if d.get("is_leaf") and not d.get("content", {}).get("full"):
            missing.append(d["_name"])
    assert not missing, f"{len(missing)} 叶子节点缺少 content.full: {missing[:5]}"


def test_every_internal_has_summary(all_nodes):
    """所有中间节点应有 summary + example_queries"""
    missing_summary = []
    missing_queries = []
    for d in all_nodes:
        if not d.get("is_leaf"):
            if not d.get("summary"):
                missing_summary.append(d["_name"])
            if not d.get("example_queries"):
                missing_queries.append(d["_name"])
    assert not missing_summary, f"{len(missing_summary)} 中间节点缺少 summary"
    assert not missing_queries, f"{len(missing_queries)} 中间节点缺少 example_queries"


def test_programming_leaves_have_code(all_nodes):
    """编程叶子应有 code_pattern 和 common_pitfalls (允许低质量回退)"""
    missing = []
    for d in all_nodes:
        if d.get("is_leaf") and bool(d["_path_parts"] & PROG_LANG_DIRS):
            c = d.get("content", {})
            q = d.get("quality", {})
            if q.get("confidence", 0) <= 0.3:
                continue  # JSON 解析回退，不期待代码
            if not c.get("code_pattern"):
                missing.append(f"{d['_name']}: code_pattern 缺失")
            if not c.get("common_pitfalls"):
                missing.append(f"{d['_name']}: common_pitfalls 缺失")
    assert not missing, "\n".join(missing[:10])


def test_leaves_have_related_nodes(all_nodes):
    """超过 40% 的叶子应有 related_nodes"""
    leaves = [d for d in all_nodes if d.get("is_leaf")]
    with_rel = [d for d in leaves if d.get("related_nodes")]
    ratio = len(with_rel) / len(leaves) * 100 if leaves else 0
    print(f"related_nodes 覆盖率: {len(with_rel)}/{len(leaves)} ({ratio:.1f}%)")
    assert ratio > 40, f"related_nodes 覆盖率 ({ratio:.1f}%) 低于 40%"
