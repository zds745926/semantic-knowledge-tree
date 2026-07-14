"""
语义知识树 — 树状索引结构核心
"""
from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Callable
import numpy as np

from core.node import TreeNode
from core.encoder import SemanticEncoder, FallbackEncoder


class SemanticKnowledgeTree:
    """全球语义知识树（原型版）"""

    def __init__(self, encoder=None):
        self.root = TreeNode(
            node_id="root",
            name="根节点",
            level=0,
            absorption_rate=0.01,
        )
        self.encoder = encoder or FallbackEncoder(384)
        self._all_leaves: Dict[str, TreeNode] = {}  # 捷径用：所有叶子索引
        self._all_nodes: Dict[str, TreeNode] = {}  # 全局节点索引
        self._all_nodes["root"] = self.root

    # ─── 构造 ───────────────────────────────────────

    def add_node(
        self,
        parent_id: str,
        node_id: str,
        name: str,
        vector: Optional[np.ndarray] = None,
        absorption_rate: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> TreeNode:
        """添加非叶子节点"""
        parent = self._all_nodes.get(parent_id)
        if parent is None:
            raise ValueError(f"父节点 {parent_id} 不存在")

        level = parent.level + 1
        # 越深层吸收率越高
        if absorption_rate is None:
            absorption_rate = min(0.02 + level * 0.015, 0.20)

        node = TreeNode(
            node_id=node_id,
            name=name,
            parent_id=parent_id,
            vector=vector,
            absorption_rate=absorption_rate,
            level=level,
            metadata=metadata or {},
        )
        parent.add_child(node)
        self._all_nodes[node_id] = node
        return node

    def add_leaf(
        self,
        parent_id: str,
        leaf_id: str,
        title: str,
        content: str = "",
        vector: Optional[np.ndarray] = None,
        data_pointer: Optional[dict] = None,
        absorption_rate: float = 0.20,
        metadata: Optional[dict] = None,
    ) -> TreeNode:
        """添加叶子节点"""
        parent = self._all_nodes.get(parent_id)
        if parent is None:
            raise ValueError(f"父节点 {parent_id} 不存在")

        # 自动编码
        if vector is None and content:
            vector = self.encoder.encode(title + " " + content[:512])

        leaf = TreeNode(
            node_id=leaf_id,
            name=title,
            parent_id=parent_id,
            vector=vector,
            absorption_rate=absorption_rate,
            level=parent.level + 1,
            metadata=metadata or {},
        )
        if data_pointer:
            leaf.add_data_pointer(**data_pointer)
        elif content:
            leaf.add_data_pointer(
                title=title,
                uri=f"knowledge://{leaf_id}",
                content_preview=content[:200],
            )

        parent.add_child(leaf)
        self._all_nodes[leaf_id] = leaf
        self._all_leaves[leaf_id] = leaf
        return leaf

    def add_cross_reference(self, leaf_id: str, related_id: str):
        """添加跨域引用"""
        leaf = self._all_nodes.get(leaf_id)
        related = self._all_nodes.get(related_id)
        if leaf and related:
            leaf.add_related(related_id)

    def pool_node_vector(self, node_id: str):
        """触发节点从子节点池化向量"""
        node = self._all_nodes.get(node_id)
        if node:
            node.pool_vector_from_children()

    def pool_all(self):
        """从下到上池化所有非叶子节点"""
        # 按层级降序排列
        sorted_nodes = sorted(
            self._all_nodes.values(),
            key=lambda n: n.level,
            reverse=True,
        )
        for node in sorted_nodes:
            if not node.is_leaf():
                node.pool_vector_from_children()

    # ─── 查询 ───────────────────────────────────────

    def penetrate(
        self,
        query: str,
        query_vec: Optional[np.ndarray] = None,
        elimination_ratio: float = 0.5,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        逐层渗透算法 v2

        核心规则：
          - 共 10 层，每层分配 10 权重，总计 100
          - 同一层所有候选节点按语义相似度比例分配该层 10 权重
          - 动态淘汰：相似度 < 本层最高相似度 × elimination_ratio 的节点被淘汰
          - 某层全部淘汰则终止
          - 叶子节点记录累计权重，按权重降序返回

        Args:
            query: 查询文本
            query_vec: 预编码查询向量
            elimination_ratio: 淘汰比例（默认 0.5，即低于最高分 50% 的淘汰）
            verbose: 是否打印详细路径

        Returns:
            叶子激活路径列表，按 total_weight 降序
        """
        if query_vec is None:
            query_vec = self.encoder.encode(query)

        LAYER_WEIGHT = 10.0
        MAX_DEPTH = 10
        results: List[Dict] = []

        # 候选列表：(node, cumulative_weight, path_names, path_ids, layer_weights)
        candidates: List[Tuple] = [
            (child, 0.0, [self.root.name], [self.root.node_id], [])
            for child in self.root.children.values()
        ]

        for depth in range(MAX_DEPTH):
            if not candidates:
                break

            # 1. 计算所有候选节点的语义相似度
            scored = []
            for node, cum_w, p_names, p_ids, l_w in candidates:
                sim = self.encoder.similarity(query_vec, node.vector) if node.vector is not None else 0.0
                scored.append((node, sim, cum_w, p_names, p_ids, l_w))

            # 2. 动态淘汰：低于最高相似度 × elimination_ratio 的淘汰
            max_sim = max(s for _, s, _, _, _, _ in scored)
            threshold = max_sim * elimination_ratio

            survivors = [x for x in scored if x[1] >= threshold]

            if not survivors:
                break  # 全部淘汰，终止

            # 3. 按语义相似度比例分配 10 权重
            total_sim = sum(max(s, 0.0) for _, s, _, _, _, _ in survivors)
            next_candidates = []

            for node, sim, cum_w, p_names, p_ids, l_w in survivors:
                if total_sim > 0:
                    share = LAYER_WEIGHT * (max(sim, 0.0) / total_sim)
                else:
                    share = LAYER_WEIGHT / len(survivors)

                new_cum_w = cum_w + share
                new_p_names = p_names + [node.name]
                new_p_ids = p_ids + [node.node_id]
                new_l_w = l_w + [share]

                if node.is_leaf():
                    results.append({
                        "path": new_p_names,
                        "node_ids": new_p_ids,
                        "total_weight": new_cum_w,
                        "leaf_id": node.node_id,
                        "leaf_name": node.name,
                        "weights": new_l_w,
                        "data_pointers": node.data_pointers,
                        "related_nodes": node.related_nodes,
                    })
                else:
                    for child in node.children.values():
                        next_candidates.append(
                            (child, new_cum_w, new_p_names, new_p_ids, new_l_w)
                        )

            candidates = next_candidates

        # 按权重降序
        results.sort(key=lambda x: x["total_weight"], reverse=True)

        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 查询: {query}")
            print(f"{'='*60}")
            for i, p in enumerate(results):
                print(f"\n  [{i+1}] 路径权重: {p['total_weight']:.4f}")
                print(f"      路径: {' → '.join(p['path'])}")
                print(f"      叶子: {p['leaf_name']}")
                if p['data_pointers']:
                    dp = p['data_pointers'][0]
                    print(f"      指向: {dp['title']}")
                    print(f"      预览: {dp.get('content_preview', '')[:80]}...")
                if p['related_nodes']:
                    print(f"      跨域: {p['related_nodes']}")

        return results

    # ─── 内部方法 ───────────────────────────────────

    def _get_path_to(self, node: TreeNode) -> List[str]:
        """获取从根到指定节点的路径名称列表"""
        path = []
        current = node
        while current:
            path.append(current.name)
            current = self._all_nodes.get(current.parent_id)
        return list(reversed(path))

    def _get_path_ids_to(self, node: TreeNode) -> List[str]:
        path = []
        current = node
        while current:
            path.append(current.node_id)
            current = self._all_nodes.get(current.parent_id)
        return list(reversed(path))

    # ─── 统计 ───────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "total_nodes": len(self._all_nodes),
            "leaf_nodes": len(self._all_leaves),
            "depth": max((n.level for n in self._all_nodes.values()), default=0),
            "encoder_dim": self.encoder.dim,
        }

    def print_tree(self, node: Optional[TreeNode] = None, max_depth: int = 999):
        """打印树结构"""
        node = node or self.root
        if node == self.root or node.level > 0:
            print(node)
        if node.level >= max_depth:
            return
        for child in node.children.values():
            self.print_tree(child, max_depth)

    def print_node(self, node_id: str):
        node = self._all_nodes.get(node_id)
        if node:
            print(node)
            if node.data_pointers:
                for dp in node.data_pointers:
                    print(f"    📎 {dp['title']} → {dp['uri']}")
            if node.related_nodes:
                print(f"    🔗 跨域: {node.related_nodes}")
