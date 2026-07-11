#!/usr/bin/env python3
"""修复知识树: 向量编码和存储文本截断不一致"""
import json, numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.encoder import SemanticEncoder
from core.tree import SemanticKnowledgeTree
from core.persistence import TreePersistence

TRUNCATE = 500  # 统一截断长度

def main():
    print("=" * 60)
    print("  修复知识树: 向量 ↔ 描述一致性")
    print("=" * 60)
    
    # 1. 加载翻译缓存
    print("\n[1/4] 加载翻译缓存...")
    cache = json.loads(Path("data/translations_cache.json").read_text())
    print(f"  缓存条目: {len(cache)}")
    
    # 2. 加载旧节点结构信息
    print("\n[2/4] 加载旧树结构...")
    old_root = Path("smart_tree")
    nodes = []
    for p in sorted(old_root.rglob('_node.json')):
        data = json.loads(p.read_text())
        nodes.append({
            'node_id': data['node_id'], 'name': data['name'],
            'level': data['level'], 'is_leaf': data.get('is_leaf', False),
            'parent_id': data.get('parent_id'),
        })
    print(f"  节点数: {len(nodes)}")
    
    # 3. 重建树（统一截断到 TRUNCATE 字符）
    print(f"\n[3/4] 重建树 (统一截断到 {TRUNCATE} 字符)...")
    encoder = SemanticEncoder("all-MiniLM-L6-v2")
    tree = SemanticKnowledgeTree(encoder=encoder)
    
    sorted_nodes = sorted(nodes, key=lambda n: n['level'])
    updated = 0
    
    for n in sorted_nodes:
        nid = n['node_id']
        is_leaf = n['is_leaf']
        
        # 从旧树获取原始中文文本（用于查缓存）
        old_path = old_root
        # 直接通过 node_id 查找（通过文件名反查太复杂，换个方式）
        # 实际上我们要用缓存中的文本来重建
        # 找到对应的中文文本
        if nid == 'root':
            cn_text = "全球知识体系，分为自然科学、工程技术和人文社科三大领域"
        else:
            # 查找旧 _node.json
            for p in old_root.rglob('_node.json'):
                d = json.loads(p.read_text())
                if d['node_id'] == nid:
                    if is_leaf:
                        dps = d.get('data_pointers', [])
                        cn_text = dps[0].get('content_preview', '') if dps else ''
                    else:
                        cn_text = d.get('metadata', {}).get('description', '')
                    break
            else:
                cn_text = n['name']
        
        # 获取英文翻译
        en_text = cache.get(cn_text, cn_text or n['name'])
        # 统一截断
        en_truncated = en_text[:TRUNCATE]
        source = f"{n['name']}: {en_truncated}"
        vector = encoder.encode(source)
        
        if nid == 'root':
            tree.root.vector = vector
        elif is_leaf:
            tree.add_leaf(
                parent_id=n['parent_id'], leaf_id=nid,
                title=n['name'], content=en_truncated, vector=vector,
                data_pointer={
                    "title": n['name'], "uri": f"knowledge://{nid}",
                    "content_preview": en_truncated,
                },
            )
        else:
            tree.add_node(
                parent_id=n['parent_id'], node_id=nid,
                name=n['name'], vector=vector,
                metadata={"description": en_truncated},
            )
        updated += 1
    
    print(f"  重建: {updated} 节点")
    
    # 4. 保存
    print("\n[4/4] 保存并验证...")
    import shutil
    backup = Path("smart_tree_v4_bad")
    if backup.exists():
        shutil.rmtree(backup)
    if Path("smart_tree").exists():
        Path("smart_tree").rename("smart_tree_v4_bad")
    
    db = TreePersistence("smart_tree")
    db.save_tree(tree)
    
    # 验证
    loaded = db.load_tree()
    
    # 验证向量一致性
    print("\n  验证向量一致性:")
    consistent = 0
    total = 0
    for p in Path("smart_tree").rglob('_node.json'):
        data = json.loads(p.read_text())
        nid = data['node_id']
        node = loaded._all_nodes.get(nid)
        if node and node.vector is not None:
            total += 1
            # Reconstruct source
            if nid == 'root':
                text = "全球知识体系，分为自然科学、工程技术和人文社科三大领域"
            elif data.get('is_leaf'):
                dps = data.get('data_pointers', [])
                text = dps[0].get('content_preview', '') if dps else ''
            else:
                text = data.get('metadata', {}).get('description', '')
            source = f"{data['name']}: {text}"
            recomputed = encoder.encode(source)
            if np.allclose(node.vector, recomputed):
                consistent += 1
    
    print(f"  ✅ 向量一致: {consistent}/{total}")
    
    # 测试检索
    print("\n  检索测试:")
    tests = [
        "sort a list of numbers in Python",
        "check if a number is prime",
        "find the maximum value in a list",
        "reverse a string",
    ]
    for q in tests:
        qvec = encoder.encode(q)
        res = loaded.penetrate(q, query_vec=qvec, verbose=False)
        if res:
            r = res[0]
            path = ' → '.join(r['path'][:4])
            print(f"    [{r['total_weight']:.1f}] {path}")
        else:
            print(f"    [无匹配] {q}")
    
    db.close()
    print(f"\n✅ 完成! 旧树备份在 smart_tree_v4_bad/")

if __name__ == "__main__":
    main()
