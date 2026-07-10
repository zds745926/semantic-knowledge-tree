#!/usr/bin/env python3
"""
将 SQLite 知识树迁移到文件系统存储
用法: python scripts/migrate_to_fs.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 用旧的 SQLite persistence 加载
from core.persistence_old import TreePersistence as OldPersistence
# 用新的文件系统 persistence 保存
from core.persistence import TreePersistence as NewPersistence

def main():
    db_path = os.environ.get("DB_PATH", "data/knowledge_tree.db")
    tree_path = os.environ.get("TREE_PATH", "smart_tree")

    if not os.path.exists(db_path):
        print(f"❌ SQLite 数据库不存在: {db_path}")
        sys.exit(1)

    print(f"📂 从 SQLite 加载: {db_path}")
    old_db = OldPersistence(db_path)
    tree = old_db.load_tree()
    old_db.close()

    if tree is None:
        print("❌ 加载失败")
        sys.exit(1)

    stats = tree.stats()
    print(f"✅ 已加载 {stats['total_nodes']} 节点, {stats['leaf_nodes']} 叶子, {stats['depth']} 层")

    print(f"📂 写入文件系统: {tree_path}")
    new_db = NewPersistence(tree_path)
    new_db.save_tree(tree)
    new_db.close()

    # 验证
    print("🔍 验证写入结果...")
    verify_db = NewPersistence(tree_path)
    loaded = verify_db.load_tree()
    verify_db.close()

    if loaded is None:
        print("❌ 验证失败: 无法加载")
        sys.exit(1)

    vs = loaded.stats()
    print(f"✅ 验证通过: {vs['total_nodes']} 节点, {vs['leaf_nodes']} 叶子, {vs['depth']} 层")
    print(f"   tree_size: {vs.get('tree_size_bytes', 0) / 1024:.0f} KB")
    print("🎉 迁移完成!")

if __name__ == "__main__":
    main()
