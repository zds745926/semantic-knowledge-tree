#!/usr/bin/env python3
"""
SKT 向量重算脚本 — Phase 3 (FallbackEncoder 版)
使用 FallbackEncoder (基于字符ngram, 支持中文) 重新编码所有 570 节点

用法: .venv/bin/python phase3_recompute_vectors.py
"""
import json, sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force FallbackEncoder — 不支持中文的 all-MiniLM-L6-v2 对中文查询质量很差
os.environ["SENTENCE_TRANSFORMERS_OFFLINE"] = "1"
# 阻止 SemanticEncoder 加载
import builtins
_original_import = builtins.__import__
def _block_sentence_transformers(name, *args, **kwargs):
    if name == "sentence_transformers":
        raise ImportError("使用 FallbackEncoder")
    return _original_import(name, *args, **kwargs)
builtins.__import__ = _block_sentence_transformers

from core.encoder import FallbackEncoder

TREE_ROOT = Path(__file__).parent / "smart_tree" / "根节点"


def get_node_text(d: dict) -> str:
    """生成用于编码的文本: 使用 data_pointer.content_preview (混合中英文短描述)
    与原始 build_v4 编码策略一致 — 短小精悍的中英混合文本"""
    name = d.get("name", "")
    # Leaf: use data_pointer.content_preview (original encoding source matching build_v4)
    dps = d.get("data_pointers", [])
    if dps:
        preview = dps[0].get("content_preview", "")
        if preview:
            return f"{name} {preview[:512]}"
    # Fallback
    if d.get("is_leaf"):
        full = d.get("content", {}).get("full", "").strip()
        return f"{name}: {full[:400]}" if full else name
    summary = d.get("summary", "").strip()
    if summary:
        return f"{name}: {summary[:400]}"
    meta = d.get("metadata", {}).get("description", "").strip()
    return f"{name}: {meta[:400]}" if meta else name


def save_vector(file_path: Path, vec: list):
    vec_str = ",".join(f"{v:.8f}" for v in vec)
    file_path.write_text(vec_str, encoding="utf-8")


def main():
    print("=" * 60)
    print("  SKT 向量重算 — Phase 3 (FallbackEncoder)")
    print("=" * 60)

    print("\n[1/3] 加载编码器...")
    encoder = FallbackEncoder(384)
    dim = encoder.dim
    print(f"  编码器: FallbackEncoder, 维度: {dim}")

    print("\n[2/3] 重算向量...")
    nodes = sorted(TREE_ROOT.rglob("_node.json"))
    total = len(nodes)
    errors = 0

    for i, nj in enumerate(nodes, 1):
        try:
            d = json.loads(nj.read_text(encoding="utf-8"))
            text = get_node_text(d)
            vec = encoder.encode(text)
            vec_str = ",".join(f"{v:.8f}" for v in vec)
            (nj.parent / "vector.txt").write_text(vec_str, encoding="utf-8")
            if i % 100 == 0 or i == total:
                print(f"  [{i}/{total}] {d.get('name', '?')}")
        except Exception as e:
            errors += 1
            print(f"  ❌ [{i}/{total}] {nj.parent.name}: {e}")

    print(f"\n[3/3] 验证向量文件...")
    vec_files = list(TREE_ROOT.rglob("vector.txt"))
    valid = 0
    for vf in vec_files:
        try:
            floats = [float(x) for x in vf.read_text(encoding="utf-8").strip().split(",") if x.strip()]
            if len(floats) == dim:
                valid += 1
        except:
            pass

    print(f"  有效向量: {valid}/{len(vec_files)}")
    print(f"  成功: {total - errors}/{total}")
    if errors:
        print(f"  失败: {errors}")

    print(f"\n{'=' * 60}")
    print(f"  向量重算完成 (FallbackEncoder)!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
