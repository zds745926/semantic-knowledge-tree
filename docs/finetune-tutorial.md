# 语义知识树 — 1.5B 推理模型 LoRA 微调教程

> 为 SKT 系统定制一个专用于渗透路径推理 + 叶子知识汇总的 1.5B 模型

---

## 目录

1. [概览](#1-概览)
2. [环境搭建](#2-环境搭建)
3. [训练数据准备](#3-训练数据准备)
4. [LoRA 微调](#4-lora-微调)
5. [模型评估](#5-模型评估)
6. [导出与部署](#6-导出与部署)
7. [集成到 SKT 系统](#7-集成到-skt-系统)
8. [迭代优化](#8-迭代优化)

---

## 1. 概览

### 为什么要微调一个专用模型？

当前 `reasoning.py` 用的是通用 LLM（qwen2.5:7b），它的能力没有被充分聚焦：

| 维度 | 通用 7B | 专用 1.5B (微调后) |
|------|---------|-------------------|
| 参数量 | 7B | **1.5B** |
| VRAM 占用 | ~8GB | **~3GB (4-bit)** |
| 推理速度 | ~50 tok/s | **~120 tok/s** |
| 路径理解 | 需 prompt 引导 | **原生理解渗透路径** |
| 叶子融合 | 通用摘要能力 | **结构化路径+叶子汇总** |
| 幻觉率 | 通用模型通病 | **可训至 <2%** |

### 模型做什么？

输入是**结构化数据**（不是自然语言对话）：
- 用户的查询
- 渗透路径（节点序列 + 每层权重）
- 捷径命中的叶子（名称 + 内容预览）

输出是：
1. **路径分析** — 为什么走这些分支，权重分布说明了什么
2. **叶子知识汇总** — 把各路径找到的知识点融合成答案
3. **结论** — 直接回答用户问题

### 数据流

```
用户查询
    │
    ▼
SKT 树检索 ──→ 渗透路径 + 捷径叶子 ──→ 格式化输入
                                              │
                                              ▼
                                    微调后的 1.5B 模型
                                              │
                                              ▼
                                    路径分析 + 知识汇总
```

---

## 2. 环境搭建

### 2.1 基础环境

项目已使用 Python 3.10+ 和虚拟环境 `.venv`，我们在此基础上安装训练依赖。

```bash
cd /data/semantic-knowledge-tree
source .venv/bin/activate
```

### 2.2 安装 PyTorch

根据 CUDA 版本选择对应 PyTorch。当前环境 CUDA 13.2，安装 nightly 或兼容版本：

```bash
# CUDA 13.2 需要 PyTorch nightly（或降级 CUDA 兼容版）
pip install --upgrade --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu128
```

> 如果 Nightly 不稳定，备选方案：安装 PyTorch 2.6+ 的 CUDA 12.8 版本也能在 CUDA 13.2 驱动下运行（驱动向下兼容）。

验证安装：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

预期输出：
```
PyTorch: 2.6.0+cu128
CUDA available: True
Device: NVIDIA GeForce RTX 2070 with Max-Q Design
```

### 2.3 安装训练工具链

```bash
pip install transformers datasets accelerate peft bitsandbytes trl
```

可选加速库（推荐）：

```bash
pip install unsloth  # 大幅加速训练，减少显存
# 如果 unsloth 装不上，用标准工具链也够
```

各库的作用：

| 库 | 用途 |
|----|------|
| `torch` | 深度学习框架 |
| `transformers` | HuggingFace 模型库、Tokenizer、Trainer |
| `datasets` | 数据集加载与预处理 |
| `accelerate` | 多 GPU/混合精度训练 |
| `peft` | LoRA/QLoRA 参数高效微调 |
| `bitsandbytes` | 4-bit/8-bit 量化 |
| `trl` | SFTTrainer（监督微调 Trainer） |
| `unsloth` | 优化版训练（速度 2x，显存降 50%） |

### 2.4 验证 GPU 训练能力

```bash
python -c "
import torch
# 检查 CUDA 可用性
assert torch.cuda.is_available(), 'CUDA not available'
# 检查显存
gpu = torch.cuda.get_device_properties(0)
print(f'GPU: {gpu.name}')
print(f'VRAM: {gpu.total_memory / 1024**3:.1f} GB')
print(f'Compute Capability: {gpu.major}.{gpu.minor}')
# 简单 tensor 操作验证
x = torch.randn(1000, 1000).cuda()
y = torch.mm(x, x)
print(f'Tensor test OK: {y.shape}')
"
```

---

## 3. 训练数据准备

### 3.1 数据格式

每条训练样本是一个 JSON 对象，包含 input 和 output 字段。input 是树检索结果的结构化文本，output 是期望的推理+汇总回答。

```
样本格式（用于 SFT，基于 ChatML）：

{
  "conversations": [
    {
      "role": "system",
      "content": "你是语义知识树的推理引擎..."
    },
    {
      "role": "user",
      "content": "用户查询: ...\n\n渗透路径:\n...\n\n捷径叶子:\n..."
    },
    {
      "role": "assistant",
      "content": "路径分析:\n...\n\n知识汇总:\n...\n\n结论:\n..."
    }
  ]
}
```

### 3.2 数据生成策略

我们有三条途径生成训练数据，建议按优先级使用：

#### 途径 A：从历史日志提取并改写（最高优先级）

现有 75 条日志包含完整渗透路径。用以下脚本从日志提取并重写 answer 为微调风格：

**`scripts/prepare_data_from_logs.py`**：

```python
"""
从 SKT 日志提取训练数据，并调用 7B 模型改写为标准格式
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLLAMA = "http://localhost:11434/api/generate"
REWRITE_MODEL = "qwen2.5:7b"  # 用现有 7B 模型改写

def llm(prompt, system="", temp=0.1, max_tok=1024):
    data = {
        "model": REWRITE_MODEL, "prompt": prompt, "system": system,
        "stream": False,
        "options": {"temperature": temp, "num_predict": max_tok},
    }
    req = urllib.request.Request(OLLAMA, data=json.dumps(data).encode(),
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode()).get("response", "").strip()
    except Exception as e:
        print(f"  [LLM error] {e}")
        return ""


def format_penetration(pen_list):
    lines = []
    for p in pen_list[:8]:  # 取 top-8
        leaf = p.get("leaf", "")
        path = p.get("path", "")
        weight = p.get("total_weight", 0)
        wl = p.get("weights_per_layer", [])
        wl_str = ", ".join(f"{w:.3f}" for w in wl)
        lines.append(f"  [{weight:.4f}] {path}")
        lines.append(f"          每层权重: {wl_str}")
    return "\n".join(lines)


def format_shortcut(short_list):
    lines = []
    for s in short_list[:8]:
        leaf = s.get("leaf", "")
        path = s.get("path", "")
        sim = s.get("similarity", 0)
        lines.append(f"  [{sim:.4f}] {path}")
    return "\n".join(lines)


def rewrite_to_standard(query, penetration, shortcut, old_answer):
    """用 7B 模型将旧 answer 改写成标准推理格式"""
    pen_text = format_penetration(penetration)
    short_text = format_shortcut(shortcut)

    system = (
        "你是语义知识树的推理引擎。你的工作方式：\n"
        "1. 分析渗透路径的权重分布，解释为什么树选择了这些节点\n"
        "2. 汇总各路径叶子节点的知识内容\n"
        "3. 给出对用户查询的完整回答\n"
        "你的回答必须包含三段：路径分析、知识汇总、结论。\n"
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

    records = []

    for fname in sorted(os.listdir(logs_dir)):
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
                answer = r.get("answer", "")

                if not query or not penetration:
                    continue

                print(f"  改写: {query[:40]}...")
                new_answer = rewrite_to_standard(query, penetration, shortcut, answer)
                if not new_answer:
                    print(f"    改写失败，跳过")
                    continue

                # 构造 input
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

    # 写文件
    with open(output_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ 完成! 生成 {len(records)} 条训练数据 → {output_file}")


if __name__ == "__main__":
    main()
```

#### 途径 B：利用知识树自动合成（中等优先级）

用知识树已有的 473 个节点，自动生成查询→检索→回答的样本：

```python
"""
利用 SKT 知识树自动合成训练数据
"""
import json
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tree import SemanticKnowledgeTree
from core.persistence import TreePersistence
from core.encoder import SemanticEncoder, FallbackEncoder


def load_tree():
    """加载已有知识树"""
    db = TreePersistence("data/knowledge_tree.db")
    tree = db.load_tree()
    if tree is None:
        raise RuntimeError("知识树不存在，请先运行 demo.py 构建")
    try:
        tree.encoder = SemanticEncoder("all-MiniLM-L6-v2")
    except Exception:
        tree.encoder = FallbackEncoder(384)
    return tree, db


def generate_leaf_query(leaf):
    """根据叶子内容生成一个自然查询"""
    title = leaf.name
    preview = ""
    if leaf.data_pointers:
        preview = leaf.data_pointers[0].get("content_preview", "")
    return title, preview


def generate_samples(tree, count=200):
    """
    从知识树自动生成训练样本
    策略：
    - 直接问叶子标题（70%）："什么是XXX？"
    - 跨领域组合（20%）："对比XXX和YYY"
    - 领域探索（10%）："关于XXX有哪些内容"
    """
    samples = []
    all_leaves = list(tree._all_leaves.values())
    random.shuffle(all_leaves)

    for i, leaf in enumerate(all_leaves):
        if len(samples) >= count:
            break

        title, preview = generate_leaf_query(leaf)

        # 1. 单叶子查询
        if random.random() < 0.7:
            query = f"什么是{title}？"
        else:
            prefixes = ["介绍一下", "讲讲", "关于", "说说"]
            query = f"{random.choice(prefixes)}{title}"

        # 执行树检索
        query_vec = tree.encoder.encode(query)
        pen = tree.penetrate(query, query_vec=query_vec)
        short = tree.shortcut_search(query, query_vec=query_vec)

        # 格式化输入
        pen_lines = []
        for p in pen[:8]:
            path = p.get("path", [])
            weight = p.get("total_weight", 0)
            pen_lines.append(f"  [{weight:.4f}] {' → '.join(path)}")
            if p.get("data_pointers"):
                dp = p["data_pointers"][0]
                pen_lines.append(f"          内容: {dp.get('content_preview','')[:100]}")

        short_lines = []
        for s in short[:5]:
            spath = s.get("path", [])
            sim = s.get("score", 0)
            short_lines.append(f"  [{sim:.4f}] {' → '.join(spath)}")
            if s.get("data_pointers"):
                dp = s["data_pointers"][0]
                short_lines.append(f"          内容: {dp.get('content_preview','')[:100]}")

        user_input = (
            f"用户查询: {query}\n\n"
            f"渗透路径:\n" + "\n".join(pen_lines) + "\n\n"
            f"捷径叶子:\n" + "\n".join(short_lines)
        )

        # 构造期望输出（基于预览内容）
        preview_content = preview[:300] if preview else title
        parts = []
        for p in pen[:3]:
            pname = ' → '.join(p.get("path", []))
            parts.append(f"  - {pname}（权重 {p.get('total_weight',0):.4f}）")
        path_analysis = f"树通过以下路径定位到相关知识：\n" + "\n".join(parts)

        leaf_content = f"相关知识概要：{preview_content}"
        conclusion = f"{title} 是 {preview_content[:100]}..."

        output = (
            f"【路径分析】\n{path_analysis}\n\n"
            f"【知识汇总】\n{leaf_content}\n\n"
            f"【结论】\n{conclusion}"
        )

        samples.append({
            "input": user_input,
            "output": output,
            "metadata": {
                "source": "auto_synthetic",
                "query": query,
                "leaf_id": leaf.node_id,
            }
        })

    return samples


def main():
    print("加载知识树...")
    tree, db = load_tree()
    print(f"树状态: {tree.stats()['total_nodes']} 节点, {tree.stats()['leaf_nodes']} 叶子")

    print("自动生成训练样本...")
    samples = generate_samples(tree, count=200)

    output_file = "data/train_data_synthetic.jsonl"
    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ 生成 {len(samples)} 条合成数据 → {output_file}")
    db.close()


if __name__ == "__main__":
    main()
```

#### 途径 C：CRAG 等公开数据集蒸馏（补充）

从 CRAG（Comprehensive RAG Benchmark）或 HotpotQA 等数据集，筛选需要多步推理的问题，改写为 SKT 风格。这部分可按需执行，初期先用途径 A+B 就够了。

### 3.3 数据集合并与分割

```python
"""
合并并分割训练/验证集
"""
import json
import random

random.seed(42)

all_records = []

# 加载日志改写数据
with open("data/train_data.jsonl") as f:
    for line in f:
        if line.strip():
            all_records.append(json.loads(line))

# 加载合成数据
with open("data/train_data_synthetic.jsonl") as f:
    for line in f:
        if line.strip():
            all_records.append(json.loads(line))

random.shuffle(all_records)

# 90% 训练 / 10% 验证
split = int(len(all_records) * 0.9)
train = all_records[:split]
eval_data = all_records[split:]

with open("data/train.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open("data/eval.jsonl", "w") as f:
    for r in eval_data:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"训练集: {len(train)} 条")
print(f"验证集: {len(eval_data)} 条")
```

### 3.4 数据格式转换（ChatML）

SFT 训练需要 ChatML 格式的对话。从 JSONL 转换：

```python
"""
将 (input, output) 格式转为 ChatML conversations 格式
"""
import json

def convert_to_chatml(input_text, output_text):
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

# 读取 data/train.jsonl 并转换
records = []
with open("data/train.jsonl") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            records.append(convert_to_chatml(r["input"], r["output"]))

with open("data/train_chatml.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"转换完成，共 {len(records)} 条 ChatML 格式样本")
```

---

## 4. LoRA 微调

### 4.1 训练脚本

以下脚本使用 HuggingFace Transformers + PEFT + TRL 进行 QLoRA 微调。

**`scripts/train_lora.py`**：

```python
#!/usr/bin/env python3
"""
SKT 推理模型 LoRA 微调
基座: Qwen2.5-1.5B-Instruct
方法: QLoRA (4-bit NormalFloat) + LoRA rank=32
显存: ~6-7GB (RTX 2070 8GB 可跑)
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer

# ── 配置 ──────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "models/skt-reasoner-lora"

# QLoRA 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA 配置
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,     # 有效 batch_size = 8
    gradient_checkpointing=True,       # 节省显存
    optim="paged_adamw_8bit",         # 8-bit 优化器节省显存
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    evaluation_strategy="steps",
    fp16=True,                         # 半精度训练
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
    ddp_find_unused_parameters=False,
)


# ── 加载模型与 tokenizer ───────────────────────────────

print(f"[1/5] 加载模型: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Qwen2.5 需要设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# 为 k-bit 训练准备
model = prepare_model_for_kbit_training(model)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 预期输出: trainable params: ~42M / 1.5B = ~2.8%


# ── 加载数据 ──────────────────────────────────────────

print("[2/5] 加载训练数据")

def load_data(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def format_func(example):
    """将 conversations 转为模型输入文本"""
    conv = example["conversations"]
    # ChatML 格式
    system = conv[0]["content"]
    user = conv[1]["content"]
    assistant = conv[2]["content"]

    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )
    return {"text": text}


train_records = load_data("data/train_chatml.jsonl")
eval_records = load_data("data/eval_chatml.jsonl") if os.path.exists("data/eval_chatml.jsonl") else []

train_dataset = Dataset.from_list(train_records)
if eval_records:
    eval_dataset = Dataset.from_list(eval_records)
else:
    eval_dataset = None

print(f"  训练: {len(train_dataset)} 条")
print(f"  验证: {len(eval_dataset) if eval_dataset else 0} 条")


# ── SFT Trainer ────────────────────────────────────────

print("[3/5] 初始化 SFT Trainer")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=format_func,
    max_seq_length=2048,          # 渗透路径 + 叶子内容 一般不超过 2K
    dataset_text_field="text",    # 使用 format_func 返回的字段
)


# ── 训练 ──────────────────────────────────────────────

print("[4/5] 开始训练...")
trainer.train()


# ── 保存 ──────────────────────────────────────────────

print("[5/5] 保存模型")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 同时保存一份合并的 adapter 配置
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump({
        "base_model": MODEL_NAME,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs": training_args.num_train_epochs,
    }, f, indent=2)

print(f"✅ 模型已保存到 {OUTPUT_DIR}")
```

### 4.2 运行训练

```bash
cd /data/semantic-knowledge-tree
source .venv/bin/activate

# 确保数据已经准备好
ls -la data/train_chatml.jsonl

# 创建 scripts 目录（如果还没有）
mkdir -p scripts

# 将上面的训练脚本写入
# （或用 vim/nano 创建 scripts/train_lora.py）

# 跑训练
python scripts/train_lora.py
```

预期训练时间（RTX 2070 8GB）：

| 数据量 | Epochs | 耗时 |
|--------|--------|------|
| ~250 条 | 3 | ~15-25 分钟 |
| ~500 条 | 3 | ~30-45 分钟 |
| ~1000 条 | 3 | ~60-90 分钟 |

### 4.3 关于显存的说明

RTX 2070 Max-Q（8GB）跑 QLoRA 训练的实际显存占用：

| 组件 | 显存 |
|------|------|
| 模型 (4-bit Qwen2.5-1.5B) | ~2.5 GB |
| LoRA 梯度 | ~0.5 GB |
| 优化器状态 (8-bit Adam) | ~0.8 GB |
| 激活值 (batch_size=2, seq=2048) | ~2.0 GB |
| 其他开销 | ~0.5 GB |
| **合计** | **~6.3 GB** |

留有余量，不会 OOM。如果遇到 OOM：

```
# 降低 batch_size
per_device_train_batch_size=1
gradient_accumulation_steps=8  # 保持有效 batch_size=8

# 或缩短 max_seq_length
max_seq_length=1536
```

---

## 5. 模型评估

### 5.1 快速验证

训练完成后，用脚本测试模型在验证集上的表现：

**`scripts/evaluate_lora.py`**：

```python
#!/usr/bin/env python3
"""
评估微调后的 LoRA 模型
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "models/skt-reasoner-lora"

# 加载基座模型 + LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()


def generate(input_text, max_new=512):
    messages = [
        {"role": "system",
         "content": "你是语义知识树系统的推理引擎..."},
        {"role": "user", "content": input_text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


# 读取验证集前 5 条做演示
with open("data/eval_chatml.jsonl") as f:
    eval_samples = [json.loads(line) for line in f if line.strip()][:5]

for i, sample in enumerate(eval_samples):
    user_content = sample["conversations"][1]["content"]
    expected = sample["conversations"][2]["content"]

    print(f"\n{'='*60}")
    print(f"[样本 {i+1}]")
    print(f"输入: {user_content[:80]}...")
    print(f"\n→ 期望输出 ({len(expected)} chars)")
    print(f"→ 模型输出:")
    output = generate(user_content)
    print(output)
    print()
```

### 5.2 评估指标

| 指标 | 计算方式 | 目标值 |
|------|---------|--------|
| 路径准确率 | 人工评分路径分析是否合理 | >85% |
| 叶子召回率 | 答案是否覆盖 top-3 路径 | >90% |
| 幻觉率 | 答案是否包含树未提供的知识 | <5% |
| 答案完整性 | 是否包含三段式结构 | >95% |

---

## 6. 导出与部署

### 6.1 方案 A：合并 LoRA 权重 + Ollama 部署（推荐）

将 LoRA 权重合并回基座模型，导出为 GGUF 格式，直接作为 Ollama 模型使用：

```bash
# 1. 合并 LoRA 权重
python scripts/merge_lora.py
```

**`scripts/merge_lora.py`**：

```python
#!/usr/bin/env python3
"""
将 LoRA adapter 合并回基座模型，导出 safetensors + Ollama Modelfile
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "models/skt-reasoner-lora"
OUTPUT_PATH = "models/skt-reasoner-merged"

# 加载
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, LORA_PATH)

# 合并
print("合并 LoRA 权重...")
merged = model.merge_and_unload()
print("保存合并模型...")
merged.save_pretrained(OUTPUT_PATH, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"✅ 合并模型已保存到 {OUTPUT_PATH}")
```

```bash
# 2. 导出为 Ollama GGUF 格式
# 方式一：使用 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j

# 将 HuggingFace 模型转为 GGUF
python3 convert-hf-to-gguf.py \
  /data/semantic-knowledge-tree/models/skt-reasoner-merged \
  --outfile /data/semantic-knowledge-tree/models/skt-reasoner.gguf \
  --outtype q4_0  # 4-bit 量化

# 方式二：使用 ollama 直接创建（推荐，更简单）
cd /data/semantic-knowledge-tree

cat > Modelfile << 'EOF'
FROM ./models/skt-reasoner-merged
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 1024
EOF

ollama create skt-reasoner -f Modelfile
```

### 6.2 方案 B：直接加载 LoRA adapter（开发用）

不合并权重，推理时动态加载 adapter，适合调试和迭代：

```python
# 在 reasoning.py 中直接使用
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LORA = "models/skt-reasoner-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto", torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(model, LORA)
```

---

## 7. 集成到 SKT 系统

微调完成后，修改 `core/reasoning.py` 让它使用新模型。

### 7.1 修改 reasoning.py

核心改动：替换 Ollama 调用为本地模型推理，并处理格式化的三段式输出。

**`core/reasoning.py` 新版本核心部分**：

```python
"""
AI 推理层 — 使用微调的 1.5B SKT-Reasoner 模型
"""
import json
import os
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── 配置 ──
USE_LOCAL = True  # True = 用本地微调模型，False = 回退到 Ollama
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = os.environ.get(
    "SKT_LORA_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "models/skt-reasoner-lora")
)
OLLAMA_FALLBACK = "qwen2.5:7b"  # 本地模型不可用时的回退


class SKTReasoner:
    """微调后的轻量推理引擎"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载微调模型"""
        print(f"[推理] 加载模型: {BASE_MODEL} + LoRA: {LORA_PATH}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            if os.path.exists(LORA_PATH):
                self.model = PeftModel.from_pretrained(model, LORA_PATH)
                print(f"[推理] ✅ LoRA adapter 加载成功")
            else:
                self.model = model
                print(f"[推理] ⚠️ 未找到 LoRA adapter，使用基座模型")

            self.model.eval()
        except Exception as e:
            print(f"[推理] ❌ 加载失败: {e}")
            print(f"[推理] 将使用 Ollama fallback")
            self.model = None
            self.tokenizer = None

    def _generate(self, prompt: str, max_new: int = 512) -> str:
        if self.model is None:
            return self._ollama_fallback(prompt)

        messages = [
            {"role": "system",
             "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return response.strip()

    SYSTEM_PROMPT = (
        "你是语义知识树系统的推理引擎。你的任务是：\n"
        "1. 分析渗透路径——解释树为什么选择了这些路径，每层权重代表什么\n"
        "2. 汇总路径上叶子节点的知识内容\n"
        "3. 基于知识给出对用户查询的最终回答\n"
        "输出格式：三段式（【路径分析】【知识汇总】【结论】）"
    )

    def answer_query(self, query: str, merged: dict) -> str:
        """主入口：根据检索结果生成回答"""
        def _zh(s):
            return set(c for c in s if '\u4e00' <= c <= '\u9fff')
        q_chars = _zh(query)

        # 格式化渗透路径
        seen = set()
        path_lines = []
        for p in merged.get('_penetration_raw', [])[:8]:
            leaf = p.get('leaf_name', '')
            if leaf in seen:
                continue
            seen.add(leaf)
            path_str = ' → '.join(p.get('path', []))
            weight = p.get('total_weight', 0)
            weights_str = ', '.join(f"{w:.3f}" for w in p.get('weights', []))
            preview = ''
            if p.get('data_pointers'):
                preview = p['data_pointers'][0].get('content_preview', '')[:150]
            tag = " [相关]" if (q_chars & _zh(leaf)) else ""
            path_lines.append(
                f"  [{weight:.4f}] {path_str}{tag}\n"
                f"          每层权重: [{weights_str}]\n"
                f"          知识内容: {preview}"
            )

        # 格式化捷径叶子
        short_lines = []
        for s in merged.get('_shortcut_raw', [])[:5]:
            leaf = s.get('leaf_name', '')
            if leaf in seen:
                continue
            seen.add(leaf)
            path_str = ' → '.join(s.get('path', []))
            score = s.get('score', 0)
            preview = ''
            if s.get('data_pointers'):
                preview = s['data_pointers'][0].get('content_preview', '')[:150]
            tag = " [相关]" if (q_chars & _zh(leaf)) else ""
            short_lines.append(
                f"  [{score:.4f}] {path_str}{tag}\n"
                f"          知识内容: {preview}"
            )

        user_input = (
            f"用户查询: {query}\n\n"
            f"渗透路径:\n" + "\n".join(path_lines) + "\n\n"
            f"捷径叶子:\n" + "\n".join(short_lines)
        )

        return self._generate(user_input)

    def _ollama_fallback(self, prompt: str) -> str:
        """Ollama 回退"""
        import urllib.request
        OLLAMA = "http://localhost:11434/api/generate"
        data = {
            "model": OLLAMA_FALLBACK,
            "prompt": f"{self.SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 600},
        }
        req = urllib.request.Request(
            OLLAMA, data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode()).get("response", "").strip()
        except Exception as e:
            return f"[推理失败: {e}]"

    def merge_results(self, query: str, penetration: List[Dict],
                      shortcut: List[Dict]) -> Dict:
        intent = self.classify_intent(query)
        merged = {
            "intent": intent,
            "query": query,
            "results": [],
            "penetration_count": len(penetration),
            "shortcut_count": len(shortcut),
            "_penetration_raw": penetration,
            "_shortcut_raw": shortcut,
        }
        merged["answer"] = self.answer_query(query, merged)
        return merged

    def classify_intent(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["对比", "区别", "异同", "差异", "哪个好"]):
            return "对比"
        if any(w in q for w in ["有哪些", "关于", "概述", "介绍", "说说", "探索"]):
            return "探索"
        return "查找"

    def format_answer(self, merged: Dict) -> str:
        return merged.get("answer", "(无回答)")
```

### 7.2 配置文件

在项目根目录创建 `config.yaml` 方便切换模型：

```yaml
# config.yaml
reasoner:
  mode: local  # local | ollama
  local:
    base_model: Qwen/Qwen2.5-1.5B-Instruct
    lora_path: models/skt-reasoner-lora
  ollama:
    model: qwen2.5:7b
    host: http://localhost:11434
```

---

## 8. 迭代优化

### 8.1 数据增强方向

经过第一轮微调后，可以通过以下方式持续改善：

1. **难例挖掘**：找出模型回答不好的查询，人工标注后加入训练集
2. **对抗样本**：构造"题目相似但路径不同"的查询，训练模型区分细微语义差异
3. **多轮对话**：支持追问场景，如"再详细解释一下某条路径"
4. **多语言混合**：中英文查询混合训练

### 8.2 训练参数调优

| 参数 | 推荐值 | 调优方向 |
|------|--------|----------|
| LoRA rank | 32 | 更大(64)→更强拟合，更小(16)→抗过拟合 |
| learning_rate | 2e-4 | 数据量大→降低，量小→提高 |
| num_epochs | 3 | 监控 eval loss，过拟合则降 |
| max_seq_length | 2048 | 路径深度大则增加到 3072 |
| batch_size | 8 (有效) | 越大越稳定，受显存限制 |

### 8.3 性能监控

每次迭代记录：

```
迭代日志:
  日期: 2026-07-08
  数据量: 300 条（日志改写 + 200 条合成）
  训练耗时: 22 分钟
  Eval loss: 0.87
  路径分析准确率: 86%
  幻觉率: 3.2%
  推理延迟: 28ms
  模型大小: ~2.1GB (4-bit)
```

---

## 附录

### A. 完整的端到端工作流

```bash
# 0. 环境
cd /data/semantic-knowledge-tree
source .venv/bin/activate

# 1. 准备数据
python scripts/prepare_data_from_logs.py   # 从日志改写
python scripts/synthesize_data.py          # 合成数据
python scripts/merge_and_split.py          # 合并分割

# 2. 转换格式
python scripts/convert_to_chatml.py

# 3. 训练
python scripts/train_lora.py

# 4. 评估
python scripts/evaluate_lora.py

# 5. 导出
python scripts/merge_lora.py
ollama create skt-reasoner -f Modelfile

# 6. 验证集成
python demo.py
```

### B. 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| CUDA OOM | 显存不足 | 减小 batch_size 或 max_seq_length |
| bitsandbytes 报错 | CUDA 版本不匹配 | `pip install --upgrade bitsandbytes` |
| Unsloth 安装失败 | 无二进制包 | 跳过，用标准 TRL |
| LoRA adapter 加载慢 | 首次 | 后续加载有缓存 |
| 模型输出胡言乱语 | 过拟合/欠拟合 | 调整学习率或 epoch |

### C. 推荐资料

- [HuggingFace PEFT 文档](https://huggingface.co/docs/peft)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [TRL SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer)
- [Qwen2.5 官方](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)

---

> **文档版本**: v1.0
> **最后更新**: 2026-07-07
> **适用项目**: 全球语义知识树 (Semantic Knowledge Tree) v4
