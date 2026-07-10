"""
持久化层 — 文件系统存储语义知识树
每个节点一个目录，内含 _node.json（元数据）+ vector.txt（384维向量）

结构:
  smart_tree/
  └── 根节点/
      ├── _node.json
      ├── vector.txt
      ├── 人文社科/
      │   ├── _node.json
      │   ├── vector.txt
      │   └── 历史/ ...
      └── 工程技术/ ...
"""
import json
import shutil
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path

from core.node import TreeNode
from core.tree import SemanticKnowledgeTree


class TreePersistence:
    """语义知识树文件系统持久化"""

    def __init__(self, tree_path: str = "smart_tree"):
        self.tree_path = Path(tree_path)

    # ─── 目录初始化 ─────────────────────────────────

    def init_schema(self):
        """确保根目录存在"""
        self.tree_path.mkdir(parents=True, exist_ok=True)

    def close(self):
        """文件系统版无需关闭连接"""
        pass

    # ─── 路径工具 ───────────────────────────────────

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """将节点名中的 / 替换为全角 ／，避免目录层级混淆"""
        return name.replace("/", "／")

    def _get_node_dir(self, node: TreeNode, all_nodes: Dict[str, TreeNode]) -> Path:
        """从根到节点构建目录路径，使用中文名作为目录名"""
        if node.node_id == "root":
            return self.tree_path / self._sanitize_name(node.name)
        names = []
        current = node
        while current:
            names.append(self._sanitize_name(current.name))
            current = all_nodes.get(current.parent_id) if current.parent_id else None
        names.reverse()
        return self.tree_path.joinpath(*names)

    # ─── 写入 ────────────────────────────────────────

    def _write_node(self, node: TreeNode, all_nodes: Dict[str, TreeNode]):
        """将单个节点写入文件系统"""
        node_dir = self._get_node_dir(node, all_nodes)
        node_dir.mkdir(parents=True, exist_ok=True)

        parent_name = None
        if node.parent_id and node.parent_id in all_nodes:
            parent_name = all_nodes[node.parent_id].name

        children_names = [c.name for c in node.children.values()]

        metadata = {
            "name": node.name,
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "level": node.level,
            "is_leaf": node.is_leaf(),
            "absorption_rate": node.absorption_rate,
            "version": node.version,
            "parent": parent_name,
            "children": children_names,
            "vector_file": "vector.txt" if node.vector is not None else None,
            "metadata": node.metadata or {},
            "data_pointers": node.data_pointers,
            "related_nodes": node.related_nodes,
        }

        with open(node_dir / "_node.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        if node.vector is not None:
            vec_str = ",".join(f"{v:.8f}" for v in node.vector)
            with open(node_dir / "vector.txt", "w", encoding="utf-8") as f:
                f.write(vec_str)

    def save_tree(self, tree: SemanticKnowledgeTree):
        """保存整棵树到文件系统"""
        self.init_schema()

        root_dir = self.tree_path / self._sanitize_name(tree.root.name)
        if root_dir.exists():
            shutil.rmtree(root_dir)

        all_nodes = tree._all_nodes
        sorted_nodes = sorted(all_nodes.values(), key=lambda n: n.level)

        for node in sorted_nodes:
            self._write_node(node, all_nodes)

        print(f"[持久化] 已保存 {len(all_nodes)} 个节点到 {self.tree_path}")

    # ─── 读取 ────────────────────────────────────────

    def _read_node_from_dir(self, node_dir: Path) -> Optional[dict]:
        meta_file = node_dir / "_node.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _read_vector(self, node_dir: Path) -> Optional[np.ndarray]:
        vec_file = node_dir / "vector.txt"
        if not vec_file.exists():
            return None
        with open(vec_file, "r") as f:
            text = f.read().strip()
        if not text:
            return None
        floats = [float(x) for x in text.split(",") if x.strip()]
        return np.array(floats, dtype=np.float32)

    def _walk_tree(self, current_dir: Path) -> List[dict]:
        """递归扫描目录，返回所有节点的元数据列表"""
        nodes_data = []
        meta = self._read_node_from_dir(current_dir)
        if meta is None:
            # 目录本身不是节点（例如 smart_tree/），扫描子目录
            for child_dir in sorted(current_dir.iterdir()):
                if child_dir.is_dir() and (child_dir / "_node.json").exists():
                    nodes_data.extend(self._walk_tree(child_dir))
            return nodes_data

        nodes_data.append({"dir": current_dir, "meta": meta})
        for child_dir in sorted(current_dir.iterdir()):
            if child_dir.is_dir() and child_dir.name != current_dir.name:
                sub = self._walk_tree(child_dir)
                nodes_data.extend(sub)

        return nodes_data

    def load_tree(self) -> Optional[SemanticKnowledgeTree]:
        """从文件系统加载整棵树"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.encoder import SemanticEncoder, FallbackEncoder

        root_node_dir = None
        for child in sorted(self.tree_path.iterdir()):
            if child.is_dir():
                meta = self._read_node_from_dir(child)
                if meta and meta.get("node_id") == "root":
                    root_node_dir = child
                    break

        if root_node_dir is None:
            print(f"[持久化] 树不存在: {self.tree_path}")
            return None

        all_data = self._walk_tree(root_node_dir)
        if not all_data:
            print("[持久化] 树为空")
            return None

        try:
            encoder = SemanticEncoder("all-MiniLM-L6-v2")
        except Exception:
            encoder = FallbackEncoder(384)

        tree = SemanticKnowledgeTree(encoder=encoder)

        # 按 level 升序创建节点，确保父节点先存在
        sorted_items = sorted(all_data, key=lambda x: x["meta"]["level"])

        for item in sorted_items:
            meta = item["meta"]
            nid = meta["node_id"]
            level = meta["level"]
            vector = self._read_vector(item["dir"])

            if nid == "root":
                tree.root.name = meta["name"]
                tree.root.level = level
                tree.root.absorption_rate = meta.get("absorption_rate", 0.01)
                tree.root.version = meta.get("version", 1)
                tree.root.metadata = meta.get("metadata", {})
                if vector is not None:
                    tree.root.vector = vector
                tree._all_nodes["root"] = tree.root
                for dp in meta.get("data_pointers", []):
                    tree.root.add_data_pointer(**dp)
                for rid in meta.get("related_nodes", []):
                    tree.root.add_related(rid)
            else:
                node = TreeNode(
                    node_id=nid,
                    name=meta["name"],
                    parent_id=meta.get("parent_id"),
                    vector=vector,
                    absorption_rate=meta.get("absorption_rate", 0.05),
                    level=level,
                    metadata=meta.get("metadata", {}),
                )
                node.version = meta.get("version", 1)
                for dp in meta.get("data_pointers", []):
                    node.add_data_pointer(**dp)
                for rid in meta.get("related_nodes", []):
                    node.add_related(rid)
                tree._all_nodes[nid] = node
                if meta.get("is_leaf", False):
                    tree._all_leaves[nid] = node

        # 建立父子关系
        for item in sorted_items:
            meta = item["meta"]
            nid = meta["node_id"]
            if nid == "root":
                continue
            node = tree._all_nodes.get(nid)
            pid = meta.get("parent_id")
            if pid and pid in tree._all_nodes:
                parent = tree._all_nodes[pid]
                parent.add_child(node)

        print(f"[持久化] 已加载 {len(tree._all_nodes)} 个节点从 {self.tree_path}")
        return tree

    # ─── 增量操作 ────────────────────────────────────

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
        all_nodes = tree._all_nodes
        self._write_node(leaf, all_nodes)

        # 重新池化并写入父节点链
        current = tree._all_nodes.get(parent_id)
        while current:
            current.pool_vector_from_children()
            self._write_node(current, all_nodes)
            current = tree._all_nodes.get(current.parent_id)

        return leaf

    # ─── 统计 ────────────────────────────────────────

    def stats(self) -> Dict:
        """文件系统统计"""
        self.init_schema()
        total = 0
        leaves = 0
        max_level = 0
        cross_refs = 0
        data_pointers = 0
        tree_size = 0

        def scan_dir(dir_path: Path):
            nonlocal total, leaves, max_level, cross_refs, data_pointers, tree_size
            meta_file = dir_path / "_node.json"
            if meta_file.exists():
                total += 1
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("is_leaf"):
                    leaves += 1
                lv = meta.get("level", 0)
                if lv > max_level:
                    max_level = lv
                cross_refs += len(meta.get("related_nodes", []))
                data_pointers += len(meta.get("data_pointers", []))
                tree_size += meta_file.stat().st_size
                vec_file = dir_path / "vector.txt"
                if vec_file.exists():
                    tree_size += vec_file.stat().st_size
            for child in sorted(dir_path.iterdir()):
                if child.is_dir():
                    scan_dir(child)

        for child in sorted(self.tree_path.iterdir()):
            if child.is_dir():
                meta = self._read_node_from_dir(child)
                if meta and meta.get("node_id") == "root":
                    scan_dir(child)
                    break

        return {
            "total_nodes": total,
            "leaf_nodes": leaves,
            "depth": max_level,
            "cross_refs": cross_refs,
            "data_pointers": data_pointers,
            "tree_size_bytes": tree_size,
            "tree_path": str(self.tree_path),
        }
