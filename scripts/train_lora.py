#!/usr/bin/env python3
"""
SKT 推理模型 LoRA 微调训练脚本
基座: Qwen2.5-1.5B-Instruct
方法: QLoRA (4-bit NF4) + LoRA rank=32
显存: ~6-7GB (RTX 2070 8GB 适用)
"""

import os, json, torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# ── 配置 ──
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "models/skt-reasoner-lora"
DATA_DIR = "data"

# 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)

# LoRA
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,        # 有效 batch_size = 8
    gradient_checkpointing=True,          # 省显存
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=2,
)

print("[1/5] 加载模型和 tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("[2/5] 加载数据")
def load_jsonl(path):
    recs = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip(): recs.append(json.loads(line))
    return recs

train_recs = load_jsonl(os.path.join(DATA_DIR, "train_chatml.jsonl"))
eval_recs = load_jsonl(os.path.join(DATA_DIR, "eval_chatml.jsonl"))

def fmt(example):
    conv = example["conversations"]
    sys_m = conv[0]["content"]
    usr_m = conv[1]["content"]
    asst_m = conv[2]["content"]
    return {
        "text": f"<|im_start|>system\n{sys_m}<|im_end|>\n"
                f"<|im_start|>user\n{usr_m}<|im_end|>\n"
                f"<|im_start|>assistant\n{asst_m}<|im_end|>"
    }

train_ds = Dataset.from_list(train_recs).map(fmt)
eval_ds = Dataset.from_list(eval_recs).map(fmt) if eval_recs else None
print(f"  训练: {len(train_ds)} 条 | 验证: {len(eval_ds) if eval_ds else 0} 条")

print("[3/5] 初始化 SFTTrainer")
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, args=training_args,
    train_dataset=train_ds, eval_dataset=eval_ds,
    formatting_func=lambda x: x["text"],
    max_seq_length=2048,
)

print("[4/5] 开始训练")
trainer.train()

print("[5/5] 保存模型")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump({
        "base_model": MODEL_NAME,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.l_alpha if hasattr(lora_config, 'l_alpha') else lora_config.lora_alpha,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs": training_args.num_train_epochs,
    }, f, indent=2)

print(f"✅ 模型已保存到 {OUTPUT_DIR}")
