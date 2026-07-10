#!/usr/bin/env python3
"""
从 SKT 日志提取训练数据，调用 7B 模型改写为标准三段式输出
用法: python scripts/prepare_data_from_logs.py
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLLAMA = "http://localhost:11434/api/generate"
REWRITE_MODEL = "phi4-mini:latest"


def llm(prompt, system="", temp=0.1, max_tok=512):
    data = {
        "model": REWRITE_MODEL, "prompt": prompt, "system": system,
        "stream": False,
        "options": {"temperature": temp, "num_predict": max_tok},
    }
    req = urllib.request.Request(OLLAMA, data=json.dumps(data).encode(),
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode()).get("response", "").strip()
    except Exception as e:
        print(f"  [LLM error] {e}")
        return ""


def format_penetration(pen_list):
    lines = []
    for p in pen_list[:8]:
        leaf = p.get("leaf", "")
        path = p.get("path", "")
        weight = p.get("total_weight", 0)
        wl = p.get("weights_per_layer", [])
        wl_str = ", ".join(f"{w:.3f}" for w in wl)
        lines.append(f"  [{weight:.4f}] {path}")
        lines.append(f"          每层权重: {wl_str}")
        if p.get("data_pointers"):
            dp = p.get("data_pointers", [{}])[0]
            lines.append(f"          内容: {dp.get('content_preview','')[:100]}")
    return "\n".join(lines)


def format_shortcut(short_list):
    lines = []
    for s in short_list[:5]:
        leaf = s.get("leaf", "")
        path = s.get("path", "")
        sim = s.get("similarity", 0)
        lines.append(f"  [{sim:.4f}] {path}")
        if s.get("data_pointers"):
            dp = s.get("data_pointers", [{}])[0]
            lines.append(f"          内容: {dp.get('content_preview','')[:100]}")
    return "\n".join(lines)


def rewrite_to_standard(query, penetration, shortcut):
    pen_text = format_penetration(penetration)
    short_text = format_shortcut(shortcut)

    system = (
        "你是语义知识树的推理引擎。你的工作方式：\n"
        "1. 分析渗透路径的权重分布，解释为什么树选择了这些节点\n"
        "2. 汇总各路径叶子节点的知识内容\n"
        "3. 给出对用户查询的完整回答\n"
        "你的回答必须包含三段：【路径分析】【知识汇总】【结论】。\n"
        "路径分析要具体到权重值，不要泛泛而谈。"
    )

    prompt = (
        f"用户查询: {query}\n\n"
        f"渗透路径（带每层权重）:\n{pen_text}\n\n"
        f"捷径命中的叶子:\n{short_text}\n\n"
        f"请按照三段式输出：\n"
        f"【路径分析】\n"
        f"【知识汇总】\n"
        f"【结论】"
    )

    return llm(prompt, system, temp=0.1, max_tok=1024)


def main():
    logs_dir = "logs"
    output_file = "data/train_data.jsonl"
    os.makedirs("data", exist_ok=True)

    records = []
    log_files = sorted(os.listdir(logs_dir))

    for fname in log_files:
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(logs_dir, fname)
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                query = r.get("query", "")
                penetration = r.get("penetration", [])
                shortcut = r.get("shortcut", [])
                if not query or not penetration:
                    continue

                print(f"  [{len(records)+1}] {query[:50]}...")
                new_answer = rewrite_to_standard(query, penetration, shortcut)
                if not new_answer or len(new_answer) < 20:
                    print(f"    改写结果过短，跳过")
                    continue

                pen_text = format_penetration(penetration)
                short_text = format_shortcut(shortcut)
                user_input = (
                    f"用户查询: {query}\n\n"
                    f"渗透路径:\n{pen_text}\n\n"
                    f"捷径叶子:\n{short_text}"
                )

                records.append({
                    "input": user_input,
                    "output": new_answer,
                    "metadata": {
                        "source": fname,
                        "query": query,
                        "intent": r.get("intent", ""),
                    }
                })

    with open(output_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ 完成! 生成 {len(records)} 条训练数据 → {output_file}")


if __name__ == "__main__":
    main()
