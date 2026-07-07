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
        min_weight: float = 0.001,
        max_paths: int = 10,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        逐层渗透：从根节点向下传播权重，返回激活路径

        Args:
            query: 查询文本
            query_vec: 预编码查询向量（若为 None 则自动编码）
            min_weight: 剪枝权重阈值（低于此值的分支中断下传）
            max_paths: 返回最多路径数
            verbose: 是否打印详细路径

        Returns:
            激活路径列表，每条含：
              - path: 节点名称序列
              - node_ids: 节点ID序列
              - total_weight: 叶子累积吸收权重
              - leaf: 叶子节点信息
              - weights: 各层相似度分布
        """
        if query_vec is None:
            query_vec = self.encoder.encode(query)

        # BFS 逐层渗透
        active_paths: List[Dict] = []
        # (node, weight_in, path_names, path_ids, weight_list)
        # weight_in = 从上个节点下传进来的权重
        # path_names/path_ids 初始为空，进入节点时附加自身
        stack = [(self.root, 1.0, [], [], [])]

        while stack:
            node, weight_in, path_names, path_ids, weight_list = stack.pop(0)

            if node.vector is None:
                sim = 0.0
            else:
                sim = self.encoder.similarity(query_vec, node.vector)

            # 当前层的语义相似度（用于路径解释）
            layer_sim = sim
            current_weight_list = weight_list + [layer_sim]

            if node.is_leaf():
                # 叶子节点：始终记录（激活权重 = weight_in × sim）
                leaf_activation = weight_in * sim
                active_paths.append({
                    "path": path_names + [node.name],
                    "node_ids": path_ids + [node.node_id],
                    "total_weight": leaf_activation,
                    "leaf_id": node.node_id,
                    "leaf_name": node.name,
                    "weights": current_weight_list,
                    "data_pointers": node.data_pointers,
                    "related_nodes": node.related_nodes,
                })
            else:
                # 非叶子节点
                # 节点激活权重 = 接收权重 × 语义相似度
                activation = weight_in * sim

                # 吸收（保留在此层的语义贡献）
                absorbed = activation * node.absorption_rate

                # 下传给子节点
                pass_down = activation * (1 - node.absorption_rate)

                # 剪枝：下传权重过小时中断
                if pass_down > min_weight:
                    for child in node.children.values():
                        stack.append((
                            child, pass_down,
                            path_names + [node.name],
                            path_ids + [node.node_id],
                            current_weight_list,
                        ))

        # 按激活权重（叶子总贡献）排序
        active_paths.sort(key=lambda x: x["total_weight"], reverse=True)

        result = active_paths[:max_paths]

        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 查询: {query}")
            print(f"{'='*60}")
            for i, p in enumerate(result):
                print(f"\n  [{i+1}] 路径权重: {p['total_weight']:.4f}")
                print(f"      路径: {' → '.join(p['path'])}")
                print(f"      叶子: {p['leaf_name']}")
                if p['data_pointers']:
                    dp = p['data_pointers'][0]
                    print(f"      指向: {dp['title']}")
                    print(f"      预览: {dp.get('content_preview', '')[:80]}...")
                if p['related_nodes']:
                    print(f"      跨域: {p['related_nodes']}")

        return result

    def shortcut_search(
        self,
        query: str,
        query_vec: Optional[np.ndarray] = None,
        top_k: int = 10,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        捷径机制：全局叶子 Top-K 匹配，绕过逐层衰减

        Args:
            query: 查询文本
            query_vec: 预编码向量
            top_k: 返回 top-k 叶子
            verbose: 是否打印

        Returns:
            叶子匹配结果列表
        """
        if query_vec is None:
            query_vec = self.encoder.encode(query)

        if not self._all_leaves:
            return []

        leaf_ids = list(self._all_leaves.keys())
        leaf_vecs = np.array([
            self._all_leaves[lid].vector
            for lid in leaf_ids
            if self._all_leaves[lid].vector is not None
        ])

        if len(leaf_vecs) == 0:
            return []

        scores = np.dot(leaf_vecs, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            lid = leaf_ids[idx]
            leaf = self._all_leaves[lid]
            score = float(scores[idx])
            results.append({
                "leaf_id": lid,
                "leaf_name": leaf.name,
                "score": score,
                "path": self._get_path_to(leaf),
                "data_pointers": leaf.data_pointers,
                "related_nodes": leaf.related_nodes,
            })

        if verbose:
            print(f"\n{'='*60}")
            print(f"⚡ 捷径搜索: {query}")
            print(f"{'='*60}")
            for i, r in enumerate(results[:5]):
                print(f"  [{i+1}] 得分: {r['score']:.4f} | {r['leaf_name']}")
                print(f"      路径: {' → '.join(r['path'])}")

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        verbose: bool = False,
    ) -> Dict:
        """
        混合搜索：渗透 + 捷径，由调用方（AI推理层）融合

        Returns:
            {"penetration": [...], "shortcut": [...], "query": query}
        """
        query_vec = self.encoder.encode(query)

        penetration_results = self.penetrate(
            query, query_vec=query_vec, verbose=verbose
        )
        shortcut_results = self.shortcut_search(
            query, query_vec=query_vec, top_k=top_k, verbose=verbose
        )

        return {
            "query": query,
            "penetration": penetration_results,
            "shortcut": shortcut_results,
        }

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
