"""
AI 推理层 — 使用 Ollama phi4-mini
根据渗透路径+叶子内容生成答案
"""
import json
import os
import urllib.request
from typing import List, Dict, Optional


OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = "qwen2.5:7b"  # phi4 / phi4-mini / qwen2.5 / qwen2.5:0.5b 间切换


def _call_ollama(prompt: str, system: str = "",
                 temperature: float = 0.1,
                 max_tokens: int = 512) -> str:
    """调用 Ollama API"""
    data = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result.get("response", "").strip()
    except Exception as e:
        return f"[Ollama 调用失败: {e}]"


class ReasoningLayer:
    """AI 推理层 — 用 phi4-mini 根据知识树结果生成答案"""

    def classify_intent(self, query: str) -> str:
        """意图分类 — 纯规则，不调 LLM"""
        q = query.lower()
        if any(w in q for w in ["对比", "区别", "异同", "差异", "哪个好", "vs", "versus"]):
            return "对比"
        if any(w in q for w in ["有哪些", "关于", "概述", "介绍", "说说", "探索"]):
            return "探索"
        return "查找"


    def answer_query(self, query: str, merged: dict) -> str:
        """
        知识树作为数据提供者。把树检索到的所有结果发给 AI 参考，
        AI 自由回答，不限制是否使用训练知识。
        """
        def _zh(s): return set(c for c in s if '\u4e00' <= c <= '\u9fff')
        q_chars = _zh(query)

        # 汇总渗透路径（去重）
        seen = set()
        items = []
        for p in merged.get('_penetration_raw', [])[:20]:
            leaf = p.get('leaf_name','')
            if leaf in seen: continue
            seen.add(leaf)
            c = (p.get('data_pointers',[{}])[0].get('content_preview','')[:120]
                 if p.get('data_pointers') else '')
            path = ' → '.join(p.get('path',[]))
            ov = q_chars & _zh(leaf)
            tag = " [相关]" if ov else ""
            items.append(f"• {leaf}{tag}: {c}")

        context = '\n'.join(items) if items else '(知识树中未找到直接相关内容)'

        system = (
            "你是一个智能助手。下方是知识树检索到的相关内容供你参考。\n"
            "你可以自由结合自己的知识来回答。回答要自然有用。"
        )
        prompt = f"用户问题: {query}\n\n知识树参考数据:\n{context}\n\n回答："
        return _call_ollama(prompt, system, temperature=0.3, max_tokens=600)

    def merge_results(self, query: str, penetration: List[Dict]) -> Dict:
        """
        把渗透结果传给 AI 自行判断。
        """
        intent = self.classify_intent(query)

        merged = {
            "intent": intent,
            "query": query,
            "results": [],
            "penetration_count": len(penetration),
            "_penetration_raw": penetration,
        }

        merged["answer"] = self.answer_query(query, merged)
        return merged

    def format_answer(self, merged: Dict) -> str:
        return merged.get("answer", "(无回答)")
