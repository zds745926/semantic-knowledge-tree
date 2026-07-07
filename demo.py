"""
语义知识树 — 最终版演示
启动即加载模型，AI 根据渗透路径+叶子内容生成答案
每个枝干节点自带语义向量，支持 10 层深度渗透
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.tree import SemanticKnowledgeTree
from core.knowledge_builder import build_v4
from core.reasoning import ReasoningLayer, MODEL
from core.encoder import SemanticEncoder, FallbackEncoder
from core.persistence import TreePersistence
from core.logger import QueryLogger


def get_or_build_tree() -> SemanticKnowledgeTree:
    """从持久化加载，不存在则构建并保存"""
    db = TreePersistence("data/knowledge_tree.db")
    tree = db.load_tree()
    if tree is not None:
        return tree, db

    print("  🏗️  构建新知识树 v4...")
    try:
        encoder = SemanticEncoder("all-MiniLM-L6-v2")
    except Exception:
        encoder = FallbackEncoder(384)

    tree = build_v4(encoder=encoder)
    db.save_tree(tree)
    return tree, db


def run_query(tree, query, reasoner, logger):
    """执行查询：全量数据写日志，控制台只展示 AI 答案"""
    qvec = tree.encoder.encode(query)
    pen = tree.penetrate(query, query_vec=qvec, verbose=False)
    short = tree.shortcut_search(query, query_vec=qvec, verbose=False)

    # AI 融合 + 生成答案
    merged = reasoner.merge_results(query, pen, short)

    # 写日志（全量技术数据）
    log_path, seq = logger.log_query(query, merged)

    # 控制台输出
    print(f"\n{'─' * 55}")
    tag = {"查找": "🔍", "探索": "🧭", "对比": "⚖️"}.get(merged["intent"], "❓")
    print(f"  {tag} {merged['intent']}  |  {query}")
    print(f"{'─' * 55}")

    answer = merged.get("answer", "")
    print(f"\n{answer}\n")

    results = merged.get("results", [])
    if results:
        sources = [f"{r['leaf']}({r['source']})" for r in results[:5]]
        print(f"  来源: {' | '.join(sources)}")

    print(f"  📋 日志: {log_path}")


def interactive(tree, reasoner, logger):
    print("\n输入查询 (支持多行，两次回车发送，q 退出)")
    while True:
        lines = []
        try:
            while True:
                line = input(">>> " if not lines else "... ")
                if not line.strip() and lines:
                    break
                if not line.strip() and not lines:
                    continue
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print(); break

        q = ' '.join(lines).strip()
        if q.lower() in ("q", "exit", "quit"):
            break
        if q:
            run_query(tree, q, reasoner, logger)


def batch(tree, reasoner, logger):
    queries = [
        "用Python写一个解一元二次方程的程序",
        "什么是二分搜索",
        "Python 函数定义",
        "关于线性代数",
        "对比快排和归并",
        "说说有哪些排序算法",
        "TCP三次握手原理",
        "Docker和Kubernetes的关系",
    ]
    for q in queries:
        run_query(tree, q, reasoner, logger)


def main():
    print("=" * 55)
    print("  🌐 全球语义知识树系统  v4")
    print("  枝干自带语义向量  |  10层深度  |  Ollama推理")
    print("=" * 55)

    # 1. 树
    print("\n[1/3] 加载知识树...")
    tree, db = get_or_build_tree()
    stats = tree.stats()
    print(f"  ✅ {stats['total_nodes']} 节点, {stats['leaf_nodes']} 叶子, {stats['depth']} 层")

    # 2. 编码器预热
    print("\n[2/3] 预热编码器...")
    if hasattr(tree.encoder, 'warmup'):
        tree.encoder.warmup()
        print("  ✅ 模型已加载，首次查询无延迟")
    else:
        print("  ✅ 编码器就绪")

    # 3. 推理层
    print("\n[3/3] 推理层 + 日志...")
    reasoner = ReasoningLayer()
    logger = QueryLogger()
    print(f"  ✅ {MODEL} 就绪  |  日志 → {logger.get_session_path()}")
    ds = db.stats()
    print(f"  📦 {ds['db_path']} ({ds['db_size_bytes']/1024:.0f} KB)")

    print(f"\n{'─' * 55}")
    print("  1 — 批量演示")
    print("  2 — 交互模式")
    print(f"{'─' * 55}")
    choice = input("\n选择 [1/2]: ").strip()

    if choice == "2":
        interactive(tree, reasoner, logger)
    else:
        batch(tree, reasoner, logger)

    print(f"\n📋 日志: {logger.get_session_path()} ({logger.summary()['queries_logged']} 条)")
    db.close()


if __name__ == "__main__":
    main()
