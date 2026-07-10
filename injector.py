#!/usr/bin/env python3
"""
自动知识注入器 v2
- 遍历所有非叶子节点，AI 分析缺什么叶子
- 自动生成内容并注入
- 自动创建中间节点
- 持续迭代直到树丰满
"""
import sys, json, urllib.request, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.persistence import TreePersistence
from core.tree import SemanticKnowledgeTree
from core.encoder import SemanticEncoder

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "phi4-mini:latest"
MAX_TOKENS = 800


def llm(prompt: str, system: str = "", temp: float = 0.2, max_tok: int = 512) -> str:
    data = {
        "model": MODEL, "prompt": prompt, "system": system,
        "stream": False, "options": {"temperature": temp, "num_predict": max_tok},
    }
    req = urllib.request.Request(OLLAMA, data=json.dumps(data).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode()).get("response", "").strip()
    except Exception as e:
        return ""


def get_subtree_summary(tree, node_id, depth=2):
    """获取节点下子树的文本摘要（用于 AI 分析上下文）"""
    node = tree._all_nodes.get(node_id)
    if not node:
        return ""
    lines = [f"当前节点: {node.name} (ID={node_id})"]
    # 直接子节点
    children = list(node.children.values())
    if children:
        leaf_names = [c.name for c in children if c.is_leaf()]
        sub_names = [c.name for c in children if not c.is_leaf()]
        if leaf_names:
            lines.append(f"  已有叶子({len(leaf_names)}): {'、'.join(leaf_names[:15])}")
        if sub_names:
            lines.append(f"  子分支({len(sub_names)}): {'、'.join(sub_names[:10])}")
    return "\n".join(lines)


def analyze_node(tree, node_id):
    """AI 分析节点，推荐应新增的主题及是否需新建子分支"""
    summary = get_subtree_summary(tree, node_id)
    prompt = (
        f"你是一名知识树架构师。请分析以下树节点。\n\n"
        f"{summary}\n\n"
        f"任务：列出该节点下应该补充的知识点。格式要求：\n"
        f"1. 每行一个知识点主题（2-6个字）。\n"
        f"2. 如果需要新建子分类节点，用 [子分类:xxx] 开头。\n"
        f"3. 直接子分类下的主题缩进写在其后。\n"
        f"4. 每行只输出主题，不要编号。\n"
        f"5. 列出 3-8 个新增项。"
    )
    result = llm(prompt, max_tok=400)
    return result


def parse_recommendations(text):
    """解析 AI 推荐结果"""
    items = []
    current_sub = None
    for line in text.split('\n'):
        line = line.strip().strip('-').strip('0123456789. ').strip()
        if not line or len(line) < 2:
            continue
        if line.startswith('[子分类:') or line.startswith('[分类:'):
            current_sub = line.split(':', 1)[1].rstrip(']').strip()
            continue
        if current_sub:
            items.append((current_sub, line))
        else:
            items.append((None, line))
    return items


def find_best_parent(tree, topic, context_node_id):
    """在指定节点下找到或创建插入位置"""
    parent = tree._all_nodes.get(context_node_id)
    if not parent:
        return None

    # 检查是否有匹配的子分支
    for cid, child in parent.children.items():
        if not child.is_leaf():
            # AI 判断 topic 是否属于该子分支
            prompt = (
                f"父分类: {parent.name}\n"
                f"子分类: {child.name} (ID={cid})\n"
                f"新主题: {topic}\n"
                f"这个主题是否属于子分类「{child.name}」？只回答 是 或 否。"
            )
            result = llm(prompt, max_tok=10)
            if '是' in result:
                return find_best_parent(tree, topic, cid)

    return parent


def create_missing_node(tree, parent_id, sub_name):
    """创建不存在的子分支节点"""
    node = tree._all_nodes.get(parent_id)
    if not node:
        return None

    # 检查是否已有同名子节点
    for child in node.children.values():
        if child.name == sub_name:
            return child.node_id

    # 创建
    nid = sub_name.lower().replace(' ', '_').replace('-', '_')
    # 确保 ID 唯一
    base_nid = nid
    counter = 1
    while nid in tree._all_nodes:
        nid = f"{base_nid}_{counter}"
        counter += 1

    new_node = tree.add_node(parent_id, nid, sub_name)
    print(f"  🆕 创建子分类: {sub_name} ({nid})")
    return nid


def inject_topic(tree, topic, parent_id):
    """注入单个知识点"""
    leaf_id = topic.lower().replace(' ', '_').replace('-', '_')[:40]
    if leaf_id in tree._all_leaves:
        return False

    parent = tree._all_nodes.get(parent_id)
    if not parent:
        return False

    # 生成内容
    system = "你是一名技术文档作者。用一段通顺的中文详细解释这个知识点，150-250字。"
    prompt = f"知识点: {topic}\n\n解释："
    content = llm(prompt, system, max_tok=512)
    if not content or len(content) < 30:
        content = f"{topic} 是计算机科学领域的一个重要概念，广泛应用于软件开发和系统设计中。"

    # 添加叶子
    tree.add_leaf(
        parent_id=parent_id,
        leaf_id=leaf_id,
        title=topic,
        content=content,
        data_pointer={"title": topic, "uri": f"knowledge://{leaf_id}", "content_preview": content[:200]},
    )

    # 局部池化
    cur = parent
    while cur:
        cur.pool_vector_from_children()
        cur = tree._all_nodes.get(cur.parent_id)

    print(f"  ✅ 注入: {topic} → {parent.name}")
    return True


def expand_node(tree, node_id, depth=0):
    """递归扩展一个节点"""
    node = tree._all_nodes.get(node_id)
    if not node or node.is_leaf() or depth > 4:
        return 0

    indent = "  " * depth
    print(f"\n{indent}📂 {node.name} ({node_id})")

    # AI 分析该节点缺什么
    recs = analyze_node(tree, node_id)
    items = parse_recommendations(recs)

    if not items:
        print(f"{indent}   ⏭️  无推荐")
        return 0

    count = 0
    for sub_name, topic in items:
        if sub_name:
            # 需要子分类
            sub_id = create_missing_node(tree, node_id, sub_name)
            if sub_id:
                # 先注入该分类下的主题
                if topic:
                    if inject_topic(tree, topic, sub_id):
                        count += 1
                # 递归扩展子分类
                count += expand_node(tree, sub_id, depth + 1)
        else:
            # 直接主题
            if topic and inject_topic(tree, topic, node_id):
                count += 1

    return count


def auto_build(tree, max_rounds=3):
    """自动迭代扩展整棵树"""
    print("=" * 55)
    print("  自动知识树构建")
    print(f"  模型: {MODEL}")
    print("=" * 55)

    total = 0
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*55}")
        print(f"  第 {round_num} 轮迭代")
        print(f"{'='*55}")

        round_count = 0
        # 按深度遍历所有非叶子节点
        nodes_by_level = sorted(
            [n for n in tree._all_nodes.values() if not n.is_leaf() and n.level >= 1],
            key=lambda n: n.level
        )

        for node in nodes_by_level:
            c = expand_node(tree, node.node_id)
            round_count += c

        total += round_count
        stats = tree.stats()
        print(f"\n  本轮新增: {round_count} 个叶子")
        print(f"  当前规模: {stats['total_nodes']} 节点, {stats['leaf_nodes']} 叶子")

        if round_count == 0:
            print("  ✅ 树已丰满，无需继续")
            break

    stats = tree.stats()
    print(f"\n{'='*55}")
    print(f"  ✅ 构建完成")
    print(f"  总节点: {stats['total_nodes']} | 叶子: {stats['leaf_nodes']}")
    print(f"  深度: {stats['depth']} 层")
    return stats


def main():
    print("=" * 55)
    print("  知识树自动注入器 v2")
    print(f"  模型: {MODEL}")
    print("=" * 55)

    db = TreePersistence("smart_tree")
    tree = db.load_tree()
    if not tree:
        print("❌ 知识树不存在，请先 run.sh 构建")
        return

    try:
        enc = SemanticEncoder("all-MiniLM-L6-v2")
        tree.encoder = enc
    except:
        pass

    print(f"\n当前: {tree.stats()['total_nodes']} 节点, {tree.stats()['leaf_nodes']} 叶子")
    print()

    print("模式:")
    print("  1 — 全自动构建（多轮迭代，直到树丰满）")
    print("  2 — 扩展特定领域")
    print("  3 — 交互注入")
    print()

    choice = input("选择 [1/2/3]: ").strip()

    if choice == "2":
        nodes = [(nid, n.name) for nid, n in tree._all_nodes.items()
                 if not n.is_leaf() and n.level >= 1 and nid != "root"]
        print("\n可选领域:")
        for i, (nid, name) in enumerate(nodes, 1):
            child_count = len(tree._all_nodes.get(nid).children) if tree._all_nodes.get(nid) else 0
            print(f"  {i:2d}. {name} ({nid})  [{child_count} 子节点]")
        picks = input("\n输入序号（逗号分隔, 或 all）: ").strip()
        if picks.lower() == 'all':
            targets = [n[0] for n in nodes]
        else:
            targets = []
            for p in picks.split(','):
                p = p.strip()
                if p.isdigit():
                    idx = int(p) - 1
                    if 0 <= idx < len(nodes):
                        targets.append(nodes[idx][0])

        total = 0
        for pid in targets:
            c = expand_node(tree, pid)
            total += c
        print(f"\n✅ 新增 {total} 个叶子")

    elif choice == "3":
        print("输入主题，输入 q 退出")
        while True:
            topic = input("\n>>> 主题: ").strip()
            if not topic or topic.lower() in ('q', 'quit'):
                break

            # 找应该归到哪个父节点
            parent_id = None
            nodes = sorted(
                [n for n in tree._all_nodes.values()
                 if not n.is_leaf() and n.level >= 1],
                key=lambda n: n.level, reverse=True
            )
            for node in nodes:
                prompt = f"父分类: {node.name}\n新主题: {topic}\n这个主题是否应该属于父分类「{node.name}」？只回答 是 或 否。"
                r = llm(prompt, max_tok=10)
                if '是' in r:
                    parent_id = find_best_parent(tree, topic, node.node_id)
                    break

            if not parent_id:
                parent_id = "cs"

            inject_topic(tree, topic, parent_id)
    else:
        auto_build(tree, max_rounds=2)

    # 保存
    db.save_tree(tree)
    s = db.stats()
    print(f"\n📦 保存: {s['total_nodes']} 节点, {s['leaf_nodes']} 叶子, {s['tree_path']}")
    db.close()


if __name__ == "__main__":
    main()
