"""
持久化层 — SQLite 存储语义知识树
"""
import sqlite3
import json
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from core.node import TreeNode
from core.tree import SemanticKnowledgeTree


class TreePersistence:
    """语义知识树 SQLite 持久化"""

    def __init__(self, db_path: str = "data/knowledge_tree.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def init_schema(self):
        """初始化表结构"""
        conn = self.connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id     TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                parent_id   TEXT,
                level       INTEGER NOT NULL DEFAULT 0,
                absorption_rate REAL NOT NULL DEFAULT 0.05,
                is_leaf     INTEGER NOT NULL DEFAULT 0,
                version     INTEGER NOT NULL DEFAULT 1,
                metadata    TEXT DEFAULT '{}',
                vector      BLOB,
                created_at  TEXT DEFAULT (datetime('now')),
                updated_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS data_pointers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id     TEXT NOT NULL REFERENCES nodes(node_id),
                title       TEXT NOT NULL,
                uri         TEXT NOT NULL,
                content_preview TEXT DEFAULT '',
                UNIQUE(node_id, uri)
            );

            CREATE TABLE IF NOT EXISTS related_nodes (
                node_id     TEXT NOT NULL REFERENCES nodes(node_id),
                related_id  TEXT NOT NULL REFERENCES nodes(node_id),
                PRIMARY KEY (node_id, related_id)
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_level ON nodes(level);
            CREATE INDEX IF NOT EXISTS idx_nodes_is_leaf ON nodes(is_leaf);
        """)
        conn.commit()

    def save_node(self, node: TreeNode):
        """保存单个节点"""
        conn = self.connect()
        vector_blob = node.vector.tobytes() if node.vector is not None else None
        metadata_json = json.dumps(node.metadata, ensure_ascii=False)

        conn.execute("""
            INSERT INTO nodes (node_id, name, parent_id, level, absorption_rate,
                               is_leaf, version, metadata, vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                name            = excluded.name,
                parent_id       = excluded.parent_id,
                level           = excluded.level,
                absorption_rate = excluded.absorption_rate,
                is_leaf         = excluded.is_leaf,
                version         = version + 1,
                metadata        = excluded.metadata,
                vector          = excluded.vector,
                updated_at      = datetime('now')
        """, (
            node.node_id, node.name, node.parent_id, node.level,
            node.absorption_rate, 1 if node.is_leaf() else 0,
            node.version, metadata_json, vector_blob,
        ))

        # 数据指针
        conn.execute("DELETE FROM data_pointers WHERE node_id = ?", (node.node_id,))
        for dp in node.data_pointers:
            conn.execute("""
                INSERT INTO data_pointers (node_id, title, uri, content_preview)
                VALUES (?, ?, ?, ?)
            """, (node.node_id, dp["title"], dp["uri"], dp.get("content_preview", "")))

        conn.commit()

    def save_tree(self, tree: SemanticKnowledgeTree):
        """保存整棵树"""
        self.init_schema()
        conn = self.connect()

        # 清空重建相关表（简单处理）
        conn.execute("DELETE FROM related_nodes")
        conn.execute("DELETE FROM data_pointers")
        conn.execute("DELETE FROM nodes")

        for node in tree._all_nodes.values():
            self.save_node(node)

        # 跨域引用
        for node in tree._all_nodes.values():
            for rid in node.related_nodes:
                conn.execute("""
                    INSERT OR IGNORE INTO related_nodes (node_id, related_id)
                    VALUES (?, ?)
                """, (node.node_id, rid))

        conn.commit()
        print(f"[持久化] 已保存 {len(tree._all_nodes)} 个节点到 {self.db_path}")

    def load_node(self, row: sqlite3.Row) -> TreeNode:
        """从数据库行加载节点"""
        vector = None
        if row["vector"] is not None:
            vector = np.frombuffer(row["vector"], dtype=np.float32)

        node = TreeNode(
            node_id=row["node_id"],
            name=row["name"],
            parent_id=row["parent_id"],
            vector=vector,
            absorption_rate=row["absorption_rate"],
            level=row["level"],
            metadata=json.loads(row["metadata"] or "{}"),
        )
        node.version = row["version"]
        return node

    def load_tree(self) -> Optional[SemanticKnowledgeTree]:
        """从数据库加载整棵树"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.encoder import SemanticEncoder, FallbackEncoder

        if not self.db_path.exists():
            print(f"[持久化] 数据库不存在: {self.db_path}")
            return None

        self.init_schema()
        conn = self.connect()
        conn.row_factory = sqlite3.Row

        # 加载所有节点
        rows = conn.execute("SELECT * FROM nodes ORDER BY level ASC").fetchall()
        if not rows:
            print("[持久化] 数据库为空")
            return None

        # 尝试恢复编码器信息
        try:
            encoder = SemanticEncoder("all-MiniLM-L6-v2")
        except Exception:
            encoder = FallbackEncoder(384)

        tree = SemanticKnowledgeTree(encoder=encoder)

        # 重建节点（按层级升序，确保父节点先创建）
        node_map = {}
        for row in rows:
            if row["node_id"] == "root":
                node = self.load_node(row)
                tree.root = node
                tree._all_nodes["root"] = node
            else:
                node = self.load_node(row)
                node_map[row["node_id"]] = node

        # 建立父子关系
        for row in rows:
            nid = row["node_id"]
            pid = row["parent_id"]
            if nid == "root" or pid is None:
                continue
            node = node_map[nid]
            parent = tree._all_nodes.get(pid)
            if parent:
                parent.add_child(node)

            # 加入索引
            tree._all_nodes[nid] = node
            if row["is_leaf"]:
                tree._all_leaves[nid] = node

        # 加载数据指针
        dp_rows = conn.execute("""
            SELECT node_id, title, uri, content_preview FROM data_pointers
        """).fetchall()
        for dpr in dp_rows:
            node = tree._all_nodes.get(dpr["node_id"])
            if node:
                node.add_data_pointer(
                    title=dpr["title"],
                    uri=dpr["uri"],
                    content_preview=dpr["content_preview"],
                )

        # 加载跨域引用
        rel_rows = conn.execute("""
            SELECT node_id, related_id FROM related_nodes
        """).fetchall()
        for rr in rel_rows:
            node = tree._all_nodes.get(rr["node_id"])
            if node:
                node.add_related(rr["related_id"])

        print(f"[持久化] 已加载 {len(tree._all_nodes)} 个节点从 {self.db_path}")
        return tree

    def upsert_leaf(self, tree: SemanticKnowledgeTree, parent_id: str,
                    leaf_id: str, title: str, content: str,
                    data_pointer: Optional[dict] = None):
        """新增或更新叶子（触发局部持久化）"""
        leaf = tree.add_leaf(
            parent_id=parent_id,
            leaf_id=leaf_id,
            title=title,
            content=content,
            data_pointer=data_pointer,
        )
        self.save_node(leaf)

        # 重新池化父节点链
        current = tree._all_nodes.get(parent_id)
        while current:
            current.pool_vector_from_children()
            self.save_node(current)
            current = tree._all_nodes.get(current.parent_id)

        return leaf

    def stats(self) -> Dict:
        """数据库统计"""
        self.init_schema()
        conn = self.connect()
        total = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        leaves = conn.execute("SELECT COUNT(*) FROM nodes WHERE is_leaf=1").fetchone()[0]
        refs = conn.execute("SELECT COUNT(*) FROM related_nodes").fetchone()[0]
        pointers = conn.execute("SELECT COUNT(*) FROM data_pointers").fetchone()[0]
        max_level = conn.execute("SELECT MAX(level) FROM nodes").fetchone()[0] or 0
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "total_nodes": total,
            "leaf_nodes": leaves,
            "depth": max_level,
            "cross_refs": refs,
            "data_pointers": pointers,
            "db_size_bytes": db_size,
            "db_path": str(self.db_path),
        }
