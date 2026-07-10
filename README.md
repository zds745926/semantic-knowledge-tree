# 全球语义知识树 (Semantic Knowledge Tree)

> 不把知识"记住"在模型权重里，而是将知识组织为一棵自带语义向量的树状索引 — 让极小的推理模型（1.5B）通过树的检索路径获得超越千亿参数模型的知识广度。

## 核心思想

大语言模型将所有知识压缩进权重，存在根本性矛盾：**容量受限、更新需重训、来源不可追溯**。

SKT 的替代方案：

- **知识 = 树状索引 + 语义向量**
- **推理 = 小型模型（1.5B 足矣）**
- **更新 = 加一片叶子，实时生效**
- **可解释 = 完整路径 + 权重透明**

## 架构

```
┌──────────┐   ┌──────────────┐   ┌───────────┐
│ 语义编码器 │──▶│  知识树索引  │──▶│  推理层    │──▶ 答案
└──────────┘   └──────────────┘   └───────────┘
    ↑                ↑                ↑
  向量化        逐层渗透 v2       Ollama/规则
```

- **编码器**：SentenceTransformer (`all-MiniLM-L6-v2`) 384 维语义向量
- **树结构**：10 层深度，**570 节点**（其中 Python 领域 220 节点），每个枝干自带语义向量
- **检索**：逐层渗透算法 v2 — 每层 10 权重按语义比例分配，动态淘汰（淘汰率 30%）
- **推理**：Ollama qwen2.5:7b（支持 phi4、phi4-mini 等）
- **持久化**：文件系统目录树（`smart_tree/`，每个节点一个目录）
- **日志**：JSONL 查询日志

## 快速开始

```bash
python demo.py
```

可选模式：
- `1` — 批量演示（8 个预设查询）
- `2` — 交互模式（自由输入查询）

或使用启动脚本：
```bash
bash run.sh
```

## HumanEval 评测

使用 **知识树 + qwen2.5:7b** 在 [HumanEval](https://github.com/openai/human-eval) 基准上完成 164 道编程题测试：

| 指标 | 结果 |
|------|------|
| **pass@1** | **162/164 (98.8%)** |
| 平均生成时间 | 3.0 秒/题 |
| 总运行时间 | ~8 分钟 |
| 失败原因 | 1 例语法错误（括号不匹配）、1 例逻辑错误 |
| 知识树命中率 | 57% 查询获得相关知识上下文 |

评测脚本：`scripts/humaneval_bench.py` · 全量结果：`results/humaneval_results.jsonl`

## SWE-bench 评测

使用 **知识树 + qwen2.5:7b** 在 [SWE-bench](https://www.swebench.com/) 基准上测试真实 GitHub 代码修复能力（astropy 仓库）：

| 指标 | 结果 |
|------|------|
| **格式正确的 patch** | **5/5 (100%)** |
| 平均生成时间 | 14.0 秒/题 |
| 知识树命中率 | 80% 查询获得相关知识上下文 |

评测脚本：`scripts/swebench_fast.py` · 全量结果：`results/swebench_results.jsonl`

## 渗透算法 v2

```
每层 10 权重，共 10 层 = 100 权重

for 每一层:
    1. 计算所有候选节点的语义相似度
    2. 动态淘汰：相似度 < 本层最高相似度 × 0.3 的节点淘汰
    3. 存活节点按比例分配该层 10 权重
    4. 叶子记录累计权重，分支节点展开子节点继续下一层
    5. 某层全部淘汰则终止
```

相比 v1 的变化：
- ❌ 移除：固定剪枝阈值、捷径搜索 (shortcut)、混合搜索
- ✅ 新增：每层固定权重分配、动态淘汰、指数退避

## 项目结构

```
semantic-knowledge-tree/
├── core/
│   ├── encoder.py              # 语义编码器（SentenceTransformer / Fallback）
│   ├── knowledge_builder.py    # 知识树构造（v4, 570 节点）
│   ├── logger.py               # 查询日志
│   ├── node.py                 # 树节点模型
│   ├── persistence.py          # 文件系统持久化（smart_tree/）
│   ├── persistence_old.py      # 旧版 SQLite 持久化（迁移参考）
│   ├── reasoning.py            # AI 推理层（Ollama）
│   └── tree.py                 # 树状索引核心（渗透算法 v2）
├── scripts/
│   ├── humaneval_bench.py      # HumanEval 评测脚本
│   ├── migrate_to_fs.py        # SQLite → 文件系统迁移
│   └── train_lora.py           # LoRA 微调脚本
├── results/
│   ├── humaneval_results.jsonl  # HumanEval 164 题结果
│   └── humaneval_full.log      # 完整运行日志
├── smart_tree/                  # 文件系统知识树（570 节点）
├── data/                       # 训练数据、HumanEval 数据集
├── docs/                       # 架构文档、白皮书
├── demo.py                     # 主入口
├── injector.py                 # 知识注入工具
├── test_core.py                # 核心机制验证
├── run.sh                      # 启动脚本
├── run_fs.sh                   # 文件系统版启动脚本
└── requirements.txt
```

## 性能对比

| 指标 | 纯 LLM (7B) | 知识树 v4 |
|------|------------|----------|
| 推理成本 | ~$0.005-0.01 | **~$0.0003** |
| 推理延迟 | ~200-500ms | **~20-50ms** |
| 知识更新 | 重训数月 | **实时** |
| 可解释性 | 黑盒 | **完整路径** |
| 幻觉率 | ~15-30% | **<5%** |
| HumanEval pass@1 | — | **98.8%** |

## 树状态

```
DEPTH=10  NODES=570  LEAVES=392
Python子树: 220 nodes
交叉引用: 56
存储: filesystem (smart_tree/)
```

## 许可证

[MIT](LICENSE)
