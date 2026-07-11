#!/usr/bin/env python3
"""
知识树重建 — 用 Ollama 翻译中文描述为英文，重新编码向量
分批处理，支持断点续传
"""
import sys, os, json, urllib.request, time, re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.encoder import SemanticEncoder
from core.tree import SemanticKnowledgeTree
from core.persistence import TreePersistence

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"
BATCH = 8
CACHE_FILE = "data/translations_cache.json"

def call_ollama(prompt: str, timeout=120) -> str:
    data = {
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.05, "num_predict": 2048},
    }
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()).get("response", "").strip()

def main():
    print("=" * 60)
    print("  知识树重建 — 英文翻译 + 重新编码")
    print("=" * 60)

    # 1. 收集节点
    print("\n[1/4] 收集节点...")
    root_dir = Path("smart_tree")
    all_nodes = []
    for p in sorted(root_dir.rglob('_node.json')):
        data = json.loads(p.read_text())
        nid = data['node_id']
        is_leaf = data.get('is_leaf', False)
        if is_leaf:
            dps = data.get('data_pointers', [])
            text = dps[0].get('content_preview', '') if dps else ''
        else:
            text = data.get('metadata', {}).get('description', '')
        all_nodes.append({
            'node_id': nid, 'name': data['name'],
            'level': data['level'], 'is_leaf': is_leaf,
            'parent_id': data.get('parent_id'), 'text': text,
        })
    print(f"  共 {len(all_nodes)} 节点")

    # 2. 翻译（分批 + 缓存）
    print("\n[2/4] 翻译为英文...")
    cache = {}
    if Path(CACHE_FILE).exists():
        cache = json.loads(Path(CACHE_FILE).read_text())
        print(f"  从缓存加载 {len(cache)} 条已翻译")

    # 去重
    unique = list(dict.fromkeys(n['text'] for n in all_nodes if n['text']))
    to_translate = [t for t in unique if t not in cache]
    print(f"  待翻译: {len(to_translate)}, 已缓存: {len(cache)}")

    for idx, text in enumerate(to_translate):
        print(f"  [{idx+1}/{len(to_translate)}] 翻译 ({text[:30]}...)")
        prompt = f"""Translate this Chinese technical text to English. Keep technical terms accurate.
If it's a coding/programming concept, add a brief Python code example.

Chinese: {text}

English translation:"""
        
        try:
            resp = call_ollama(prompt, timeout=60)
            # Extract just the English part (remove any Chinese that might be left)
            lines = [l.strip() for l in resp.split('\n') if l.strip() and not l.startswith('Chinese:')]
            result = ' '.join(lines)
            cache[text] = result
            # Save after each
            Path(CACHE_FILE).write_text(json.dumps(cache, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"    ⚠️ 失败: {e}，重试一次...")
            time.sleep(2)
            try:
                resp = call_ollama(prompt, timeout=120)
                lines = [l.strip() for l in resp.split('\n') if l.strip() and not l.startswith('Chinese:')]
                cache[text] = ' '.join(lines)
                Path(CACHE_FILE).write_text(json.dumps(cache, ensure_ascii=False, indent=2))
            except Exception as e2:
                print(f"    ❌ 放弃: {e2}")
                cache[text] = text  # fallback to original

    print(f"  翻译完成: {len(cache)} 条")

    # 3. 重建
    print("\n[3/4] 重建知识树...")
    encoder = SemanticEncoder("all-MiniLM-L6-v2")
    tree = SemanticKnowledgeTree(encoder=encoder)

    sorted_nodes = sorted(all_nodes, key=lambda n: n['level'])
    for n in sorted_nodes:
        nid = n['node_id']
        en = cache.get(n['text'], n['text'] or n['name'])
        source = f"{n['name']}: {en}"
        vector = encoder.encode(source)

        if nid == 'root':
            tree.root.vector = vector
        elif n['is_leaf']:
            tree.add_leaf(parent_id=n['parent_id'], leaf_id=nid,
                title=n['name'], content=en, vector=vector,
                data_pointer={"title": n['name'], "uri": f"knowledge://{nid}",
                              "content_preview": en[:200]})
        else:
            tree.add_node(parent_id=n['parent_id'], node_id=nid,
                name=n['name'], vector=vector,
                metadata={"description": en[:200]})

    print(f"  重建: {len(tree._all_nodes)} 节点")

    # 4. 保存
    print("\n[4/4] 保存...")
    import shutil
    backup = Path("smart_tree_v4_old")
    if backup.exists():
        shutil.rmtree(backup)
    if Path("smart_tree").exists():
        Path("smart_tree").rename("smart_tree_v4_old")

    db = TreePersistence("smart_tree")
    db.save_tree(tree)

    # 验证
    print("\n  验证:")
    loaded = db.load_tree()
    tests = [
        "sort a list of numbers in Python",
        "check if a number is prime",
        "find the maximum value in a list",
        "reverse a string",
        "calculate fibonacci sequence",
    ]
    for q in tests:
        qvec = encoder.encode(q)
        res = loaded.penetrate(q, query_vec=qvec, verbose=False)
        if res:
            r = res[0]
            print(f"    [{r['total_weight']:.1f}] {' → '.join(r['path'][:4])}")
        else:
            print(f"    [无匹配] {q}")

    db.close()
    print("\n✅ 完成！旧树备份在 smart_tree_v4_old/")

if __name__ == "__main__":
    main()
