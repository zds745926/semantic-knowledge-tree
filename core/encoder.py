"""
语义编码器 — 将文本编码为 384 维语义向量
"""
import numpy as np
from typing import List, Optional


class SemanticEncoder:
    """语义编码器，使用 sentence-transformers 生成 384 维向量"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dim = 384

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import os
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            print(f"[编码器] 加载模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
            self._dim = self._model.get_sentence_embedding_dimension()
            print(f"[编码器] 向量维度: {self._dim}")
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def warmup(self):
        """预加载模型（启动时调用，避免首次查询延迟）"""
        import os
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        _ = self.model

    def encode(self, text: str) -> np.ndarray:
        """编码单条文本"""
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码"""
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度（向量已归一化则等价于点积）"""
        return float(np.dot(a, b))

    def top_k(self, query_vec: np.ndarray, candidates: List[np.ndarray], k: int = 10):
        """从候选向量列表中找 top-k 最相似的"""
        dots = np.dot(candidates, query_vec)
        indices = np.argsort(dots)[::-1][:k]
        return [(int(i), float(dots[i])) for i in indices]


class FallbackEncoder:
    """回退编码器 — 基于词重叠的Jaccard/TF相似度（无外部依赖）"""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self._cache = {}
        self._token_cache = {}
        print(f"[编码器] 使用回退编码器 (dim={dim}, 基于词重叠)")

    @property
    def dim(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> set:
        """分词并提取中英文词"""
        if text in self._token_cache:
            return self._token_cache[text]
        import re
        # 英文字母单词
        words = set(re.findall(r'[a-zA-Z_]+', text.lower()))
        # 中文单字和双字词
        chars = set()
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                chars.add(ch)
        # 双字组合
        bigrams = set()
        for i in range(len(text) - 1):
            if ('\u4e00' <= text[i] <= '\u9fff') and ('\u4e00' <= text[i+1] <= '\u9fff'):
                bigrams.add(text[i:i+2])
        result = words | chars | bigrams
        self._token_cache[text] = result
        return result

    def encode(self, text: str) -> np.ndarray:
        """将分词结果转为稀疏二进制特征向量"""
        if text in self._cache:
            return self._cache[text]
        tokens = self._tokenize(text)
        # 使用哈希将token映射到向量维度
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokens:
            idx = hash(token) % self._dim
            vec[idx] = 1.0
        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self._cache[text] = vec
        return vec

    def encode_batch(self, texts) -> np.ndarray:
        return np.array([self.encode(t) for t in texts])

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def top_k(self, query_vec, candidates, k=10):
        dots = np.dot(candidates, query_vec)
        indices = np.argsort(dots)[::-1][:k]
        return [(int(i), float(dots[i])) for i in indices]
