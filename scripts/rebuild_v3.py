#!/usr/bin/env python3
"""
知识树重建 v3 — 使用完整英文描述（不截断）
从 knowledge_builder.py 重建，确保中文→英文缓存匹配正确
"""
import sys, os, json, re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.encoder import SemanticEncoder
from core.tree import SemanticKnowledgeTree
from core.persistence import TreePersistence

def main():
    print("=" * 60)
    print("  知识树重建 v3 — 完整英文描述（不截断）")
    print("=" * 60)

    # 1. 加载翻译缓存
    print("\n[1/4] 加载翻译缓存...")
    cache = json.loads(Path("data/translations_cache.json").read_text())
    print(f"  缓存条目: {len(cache)}")

    # 2. 从 knowledge_builder 重建树
    print("\n[2/4] 从 knowledge_builder 构建...")
    # 直接 build_v4 会用中文描述创建树
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.encoder import SemanticEncoder
    encoder = SemanticEncoder("all-MiniLM-L6-v2")
    
    from core.knowledge_builder import build_v4
    tree = build_v4(encoder=encoder)
    
    print(f"  初始树: {len(tree._all_nodes)} 节点")
    
    # 3. 替换所有节点的文本和向量为英文完整版
    print("\n[3/4] 替换为英文描述...")
    
    # 收集当前所有节点的中文描述
    updated = 0
    skipped = 0
    for nid, node in list(tree._all_nodes.items()):
        # 获取当前节点的中文文本
        if nid == 'root':
            cn_text = "全球知识体系，分为自然科学、工程技术和人文社科三大领域"
        elif node.is_leaf():
            cn_text = ''
            if node.data_pointers:
                cn_text = node.data_pointers[0].get('content_preview', '')
            if not cn_text:
                cn_text = node.metadata.get('description', '')
        else:
            cn_text = node.metadata.get('description', '')
        
        if not cn_text:
            cn_text = node.name
        
        # 在缓存中查找中文→英文
        en_text = cache.get(cn_text)
        
        if en_text is None:
            # 尝试部分匹配
            for cn_key, en_val in cache.items():
                if cn_key and cn_text and (cn_key[:30] in cn_text or cn_text[:30] in cn_key):
                    en_text = en_val
                    break
        
        if en_text is None:
            # 就使用节点名作为后备
            en_text = node.name
            skipped += 1
        else:
            # 清理翻译残留
            en_text = re.sub(r'^Translation:\s*', '', en_text, flags=re.IGNORECASE).strip()
            en_text = re.sub(r'\s+', ' ', en_text).strip()
        
        # 完整英文文本作为编码源和存储内容
        source = f"{node.name}: {en_text}"
        vector = encoder.encode(source)
        node.vector = vector
        
        # 存储完整文本（不截断）
        if nid == 'root':
            node.metadata['description'] = source
        elif node.is_leaf():
            if node.data_pointers:
                node.data_pointers[0]['content_preview'] = source
            else:
                node.add_data_pointer(title=node.name, uri=f"knowledge://{nid}", content_preview=source)
            node.metadata['description'] = source
        else:
            node.metadata['description'] = source
        
        updated += 1
    
    print(f"  更新: {updated} 节点 (后备: {skipped})")
    
    # 4. 保存
    print("\n[4/4] 保存并验证...")
    import shutil
    for old in ["smart_tree_v4_bad", "smart_tree_v4_old"]:
        b = Path(old)
        if b.exists():
            shutil.rmtree(b)
    
    old_tree = Path("smart_tree")
    if old_tree.exists():
        old_tree.rename("smart_tree_v4_old")
    
    db = TreePersistence("smart_tree")
    db.save_tree(tree)
    
    # 验证
    print("\n  验证:")
    loaded = db.load_tree()
    
    # 向量一致性
    import numpy as np
    consistent = 0
    total = 0
    for p in Path("smart_tree").rglob('_node.json'):
        data = json.loads(p.read_text())
        nid = data['node_id']
        node = loaded._all_nodes.get(nid)
        if node and node.vector is not None:
            total += 1
            # Get stored text
            if data.get('is_leaf'):
                dps = data.get('data_pointers', [])
                text = dps[0].get('content_preview', '') if dps else ''
            else:
                text = data.get('metadata', {}).get('description', '')
            source = f"{data['name']}: {text}"
            recomputed = encoder.encode(source)
            if np.allclose(node.vector, recomputed, atol=1e-5):
                consistent += 1
    
    print(f"  向量一致: {consistent}/{total}")
    
    # 描述长度
    lengths = []
    for p in Path("smart_tree").rglob('_node.json'):
        data = json.loads(p.read_text())
        if data.get('is_leaf'):
            dps = data.get('data_pointers', [])
            t = dps[0].get('content_preview', '') if dps else ''
        else:
            t = data.get('metadata', {}).get('description', '')
        lengths.append(len(t))
    
    print(f"  描述长度: min={min(lengths)} avg={sum(lengths)/len(lengths):.0f} max={max(lengths)}")
    
    # 检索测试
    print("\n  检索测试:")
    tests = [
        "sort a list of numbers in Python",
        "check if a number is prime", 
        "find the maximum value in a list",
        "reverse a string",
        "calculate fibonacci sequence using recursion",
        "check if brackets are balanced in a string",
    ]
    for q in tests:
        qvec = encoder.encode(q)
        res = loaded.penetrate(q, query_vec=qvec, verbose=False)
        if res:
            r = res[0]
            path = ' → '.join(r['path'][:5])
            w = r['total_weight']
            print(f"  w={w:5.1f}  {path}")
        else:
            print(f"  [无匹配] {q}")
    
    db.close()
    print(f"\n✅ 完成! 旧树备份在 smart_tree_v4_old/")

if __name__ == "__main__":
    main()
