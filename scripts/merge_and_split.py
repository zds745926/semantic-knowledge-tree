#!/usr/bin/env python3
"""
合并日志改写 + 合成数据，分割为 90% 训练 / 10% 验证，
并转换为 ChatML 格式用于 SFT 训练。
用法: python scripts/merge_and_split.py
"""
import json
import os
import random


def load_jsonl(path):
    records = []
    if not os.path.exists(path):
        print(f"  ⚠️  文件不存在: {path}")
        return records
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"  📥 加载 {len(records)} 条 <- {path}")
    return records


def to_chatml(input_text, output_text):
    return {
        "conversations": [
            {
                "role": "system",
                "content": (
                    "你是语义知识树系统的推理引擎。你的任务是：\n"
                    "1. 分析渗透路径——解释树为什么选择了这些路径，每层权重代表什么\n"
                    "2. 汇总路径上叶子节点的知识内容\n"
                    "3. 基于知识给出对用户查询的最终回答\n"
                    "输出格式：三段式（【路径分析】【知识汇总】【结论】）"
                ),
            },
            {
                "role": "user",
                "content": input_text,
            },
            {
                "role": "assistant",
                "content": output_text,
            },
        ]
    }


def main():
    random.seed(42)
    os.makedirs("data", exist_ok=True)

    # 1. 加载所有数据
    all_records = []
    all_records += load_jsonl("data/train_data.jsonl")          # 日志改写
    all_records += load_jsonl("data/train_data_synthetic.jsonl") # 合成数据

    if not all_records:
        print("❌ 无数据可处理")
        return

    # 2. 打乱
    random.shuffle(all_records)

    # 3. 分割 90/10
    split = int(len(all_records) * 0.9)
    train_records = all_records[:split]
    eval_records = all_records[split:]

    # 4. 保存原始格式
    with open("data/train.jsonl", "w") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("data/eval.jsonl", "w") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n📦 原始格式: train={len(train_records)}, eval={len(eval_records)}")

    # 5. 转为 ChatML
    train_chatml = [to_chatml(r["input"], r["output"]) for r in train_records]
    eval_chatml = [to_chatml(r["input"], r["output"]) for r in eval_records]

    with open("data/train_chatml.jsonl", "w") as f:
        for r in train_chatml:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open("data/eval_chatml.jsonl", "w") as f:
        for r in eval_chatml:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"📦 ChatML格式: train={len(train_chatml)}, eval={len(eval_chatml)}")
    print(f"\n✅ 完成!")


if __name__ == "__main__":
    main()
