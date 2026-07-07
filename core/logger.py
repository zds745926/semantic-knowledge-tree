"""
查询日志记录器 — 将权重、路径等结构化数据写入 JSONL 日志文件
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class QueryLogger:
    """查询日志记录器，每条查询记录为一行 JSON"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._session_file = None
        self._session_start = datetime.now()
        self._counter = 0

    def _get_session_file(self) -> Path:
        if self._session_file is None:
            timestamp = self._session_start.strftime("%Y%m%d-%H%M%S")
            self._session_file = self.log_dir / f"session-{timestamp}.jsonl"
        return self._session_file

    def log_query(self, query: str, result: Dict[str, Any]):
        """记录一次完整查询的所有技术细节"""
        self._counter += 1
        record = {
            "seq": self._counter,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "intent": result.get("intent", "unknown"),
            # 渗透路径详情
            "penetration": [
                {
                    "leaf": p.get("leaf_name"),
                    "path": " → ".join(p.get("path", [])),
                    "total_weight": round(p.get("total_weight", 0), 6),
                    "weights_per_layer": [round(w, 4) for w in p.get("weights", [])],
                    "data_count": len(p.get("data_pointers", [])),
                    "cross_refs": p.get("related_nodes", []),
                }
                for p in result.get("_penetration_raw", [])
            ],
            # 捷径搜索详情
            "shortcut": [
                {
                    "leaf": s.get("leaf_name"),
                    "path": " → ".join(s.get("path", [])),
                    "similarity": round(s.get("score", 0), 6),
                    "data_count": len(s.get("data_pointers", [])),
                    "cross_refs": s.get("related_nodes", []),
                }
                for s in result.get("_shortcut_raw", [])
            ],
            # 融合结果
            "fused": [
                {
                    "rank": i + 1,
                    "source": r.get("source"),
                    "leaf": r.get("leaf"),
                    "weight": round(r.get("weight", 0), 6),
                    "path": r.get("path"),
                }
                for i, r in enumerate(result.get("results", []))
            ],
        }

        log_path = self._get_session_file()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return log_path, self._counter

    def get_session_path(self) -> Optional[Path]:
        return self._get_session_file() if self._session_file else None

    def summary(self) -> Dict:
        return {
            "session_file": str(self._get_session_file()),
            "queries_logged": self._counter,
        }
