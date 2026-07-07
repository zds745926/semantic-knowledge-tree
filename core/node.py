"""
语义知识树节点模型
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
import numpy as np


class TreeNode:
    """知识树节点"""

    def __init__(
        self,
        node_id: str,
        name: str,
        parent_id: Optional[str] = None,
        vector: Optional[np.ndarray] = None,
        absorption_rate: float = 0.05,
        level: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.node_id = node_id
        self.name = name
        self.parent_id = parent_id
        self.vector = vector  # 384-dim semantic vector
        self.absorption_rate = absorption_rate  # 权重吸收率
        self.level = level
        self.metadata = metadata or {}

        # 子节点
        self.children: Dict[str, TreeNode] = {}

        # 叶子节点专属：数据指针
        self.data_pointers: List[Dict[str, Any]] = []

        # 跨域引用 (叶子节点可声明关联的其他节点)
        self.related_nodes: List[str] = []

        # 版本号
        self.version: int = 1

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, child: TreeNode):
        self.children[child.node_id] = child
        child.parent_id = self.node_id

    def add_data_pointer(self, title: str, uri: str, content_preview: str = ""):
        self.data_pointers.append({
            "title": title,
            "uri": uri,
            "content_preview": content_preview,
        })

    def add_related(self, node_id: str):
        if node_id not in self.related_nodes:
            self.related_nodes.append(node_id)

    def pool_vector_from_children(self):
        """从子节点向量实时池化，防止语义漂移"""
        if not self.children:
            return
        child_vecs = [c.vector for c in self.children.values() if c.vector is not None]
        if child_vecs:
            self.vector = np.mean(child_vecs, axis=0)
            self.vector = self.vector / (np.linalg.norm(self.vector) + 1e-12)

    def to_dict(self, include_vector: bool = False) -> Dict:
        d = {
            "node_id": self.node_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "absorption_rate": self.absorption_rate,
            "level": self.level,
            "is_leaf": self.is_leaf(),
            "version": self.version,
            "children_count": len(self.children),
            "data_pointers": self.data_pointers,
            "related_nodes": self.related_nodes,
            "metadata": self.metadata,
        }
        if include_vector and self.vector is not None:
            d["vector"] = self.vector.tolist()
        return d

    def __repr__(self):
        indent = "  " * self.level
        leaf_tag = " 📄" if self.is_leaf() else ""
        return f"{indent}├─ {self.name}{leaf_tag} (id={self.node_id}, abs={self.absorption_rate})"
