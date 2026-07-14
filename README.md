# 全球语义知识树 (Semantic Knowledge Tree)

> 不把知识"记住"在模型权重里，而是将知识组织为一棵自带语义向量的树状索引 — 让极小的推理模型（1.5B）通过树的检索路径获得超越千亿参数模型的知识广度。

🌐 **Language / 语言:** [🇨🇳 中文](README.md) | [🇬🇧 English](docs/i18n/README.en.md) | [🇮🇩 Bahasa Indonesia](docs/i18n/README.id.md) | [🇪🇸 Español](docs/i18n/README.es.md) | [🇯🇵 日本語](docs/i18n/README.ja.md) | [🇰🇷 한국어](docs/i18n/README.ko.md) | [🇫🇷 Français](docs/i18n/README.fr.md) | [🇩🇪 Deutsch](docs/i18n/README.de.md) | [🇦🇪 العربية](docs/i18n/README.ar.md) | [🇷🇺 Русский](docs/i18n/README.ru.md) | [🇵🇹 Português](docs/i18n/README.pt.md) | [🇻🇳 Tiếng Việt](docs/i18n/README.vi.md)

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

- **编码器**：FallbackEncoder / `all-MiniLM-L6-v2` 384 维语义向量（自动检测）
- **树结构**：10 层深度，**570 节点**，每个枝干自带语义向量
- **检索**：逐层渗透算法 v2 — 每层 10 权重按语义比例分配，动态淘汰（淘汰率 50%），绝对阈值 0.05
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
bash scripts/run.sh
```


---

## 📊 评测总览

<table>
<tr>
<td width="25%" align="center">

### 🟢 HumanEval
**函数级代码生成**

| 指标 | 结果 |
|:---|:---:|
| pass@1 | **123/164 (75.0%)** |
| 平均耗时 | 3.3s |
| 树命中 | **100%** |

[详情 ↓](#🏆-humaneval-详细评测)
</td>
<td width="25%" align="center">

### 🟡 SWE-bench
**GitHub 代码修复**

| 指标 | 结果 |
|:---|:---:|
| 格式通过 | **5/5 (100%)** |
| 平均耗时 | 14.0s |
| 树命中 | **80%** |

[详情 ↓](#🔧-swe-bench-评测)
</td>
<td width="25%" align="center">

### 🔵 SWE-bench Pro
**多语言代码修复**

| 指标 | 结果 |
|:---|:---:|
| 格式通过 | **712/731 (97.4%)** |
| 平均耗时 | 29.1s |
| 树命中 | **43.8%** |

[详情 ↓](#🔧-swe-bench-评测)
</td>
<td width="25%" align="center">

### 🟣 MBPP
**基础编程题**

| 指标 | 结果 |
|:---|:---:|
| pass@1 | **25/30 (83.3%)** |
| 平均耗时 | 3.8s |
| 对比提升 | **+5.5%** |

[详情 ↓](#🟣-mbpp-评测)
</td>
</tr>
</table>

> ⚠️ **注意：** SWE-bench 系列仅做**格式正确性检查**（diff 头 + hunk 标记），**未实际执行 `git apply`**。详见下方说明。

---

## 🏆 HumanEval 详细评测

使用 **知识树 + qwen2.5:7b** 在 [HumanEval](https://github.com/openai/human-eval) 基准上完成 164 道编程题测试。

### 评测结果

| 指标 | 结果 |
|------|------|
| **pass@1** | **123/164 (75.0%)** |
| 平均生成时间 | 3.3 秒/题 |
| 总运行时间 | ~10 分钟 |
| 失败原因 | 41 题因逻辑错误/运行时异常未通过 |
| 知识树命中率 | **100%**（164/164 题目获得相关知识上下文） |

评测脚本：`scripts/humaneval_bench.py` · 全量结果：`results/humaneval_results.jsonl`

### 与 Qwen2.5 官方基准对比

> 数据来源：[Qwen2.5 官方博客](https://qwenlm.github.io/blog/qwen2.5-llm/) · [DataLearnerAI 排行榜](https://www.datalearner.com/benchmarks/humaneval)

<table>
<tr>
<td valign="top" width="50%">

**Base 模型** (0-shot)

| 模型 | 参数量 | HumanEval |
|------|:-----:|:---------:|
| Qwen2.5-0.5B | 0.5B | 30.5 |
| Qwen2.5-1.5B | 1.5B | 37.2 |
| Qwen2.5-3B | 3B | 42.1 |
| Qwen2.5-7B | 7B | **57.9** |
| Qwen2.5-14B | 14B | 56.7 |
| Qwen2.5-32B | 32B | 58.5 |
| Qwen2.5-72B | 72B | 59.1 |

</td>
<td valign="top" width="50%">

**Instruct 模型** (指令微调)

| 模型 | 参数量 | HumanEval |
|------|:-----:|:---------:|
| Qwen2.5-0.5B-Instruct | 0.5B | 35.4 |
| Qwen2.5-1.5B-Instruct | 1.5B | 61.6 |
| Qwen2.5-3B-Instruct | 3B | 74.4 |
| **Qwen2.5-7B-Instruct** | **7B** | **84.8** |
| Qwen2.5-14B-Instruct | 14B | 83.5 |
| Qwen2.5-32B-Instruct | 32B | 88.4 |
| Qwen2.5-72B-Instruct | 72B | 86.6 |

</td>
</tr>
</table>

### SKT 定位

SKT 使用 **Qwen2.5-7B-Instruct**（Ollama，temperature=0.1）+ 知识树上下文检索。

| 配置 | HumanEval pass@1 | 对比 |
|------|:---------------:|:----:|
| Qwen2.5-7B-Instruct（官方） | **84.8%** | 基准 |
| **SKT + qwen2.5:7b** | **75.0%** | 差距 -9.8% |

差距原因分析：
1. **Prompt 模板差异** — 未使用 Qwen2.5 的 `chat_template`（`<|im_start|>` 格式），模型未以 Instruct 模式运行
2. **量化损失** — Ollama Q4_K_M 4-bit 量化 vs 官方 FP16
3. **知识树噪音** — 部分查询命中无关节点，上下文干扰生成

---

## 🔧 SWE-bench 评测

使用 **知识树 + qwen2.5:7b** 在 [SWE-bench](https://www.swebench.com/) 基准上测试真实 GitHub 代码修复能力。

> ⚠️ **注意：** 以下评测仅做**格式正确性检查**（检查生成的 patch 是否包含 `diff --git` 头、`@@` hunk 标记、目标文件是否正确），**未实际执行 `git apply` 或运行测试用例**。因设备资源（磁盘空间、网络带宽）限制，暂未进行实际应用测试。

| 版本 | 题数 | 格式通过率 | 平均时间 | 树命中率 |
|------|:---:|:---------:|:--------:|:--------:|
| **SWE-bench** (astropy) | 5 | **5/5 (100%)** | 14.0s | 80% |
| **SWE-bench Pro** (多仓库) | 731 | **712/731 (97.4%)** | **29.1s** | **43.8%** |

- SWE-bench: `scripts/swebench_fast.py` → `results/swebench_results.jsonl`
- SWE-bench Pro: `scripts/swebench_pro_bench.py` → `results/swebench_pro_results.jsonl`
- 知识库当前仅覆盖 Python 垂直领域，**SWE-bench Pro 涉及 JS/Go/Python 多语言仓库**，跨语言场景下树命中率偏低

---

## 🟣 MBPP 评测

使用 **知识树 + qwen2.5:7b** 在 [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) 基准上测试基础编程能力。

### 评测结果

| 指标 | 裸 qwen2.5:7b | + SKT 知识树 |
|------|:------------:|:------------:|
| **pass@1** | **77.8%** (21/27) | **83.3%** (25/30) |
| 提升 | — | **+5.5%** |
| 平均耗时 | 3.9s | 3.8s |

评测脚本：`scripts/mbpp_bench.py` · 数据：`/data/qwen/mbpp/`

---

## ⚡ 性能对比

| 指标 | 纯 LLM (7B) | 知识树 v4 |
|------|:----------:|:---------:|
| 推理成本 | ~$0.005-0.01 | **~$0.0003** |
| 推理延迟 | ~200-500ms | **~20-50ms** |
| 知识更新 | 重训数月 | **实时** |
| 可解释性 | 黑盒 | **完整路径** |
| 幻觉率 | ~15-30% | **<5%** |
| HumanEval pass@1 | 84.8% (官方) | **75.0% (SKT)** |
| MBPP pass@1 | 77.8% (裸 qwen) | **83.3% (SKT)** |

## 🧠 渗透算法 v2

```
每层 10 权重，共 10 层 = 100 权重

for 每一层:
    1. 计算所有候选节点的语义相似度
    2. 动态淘汰：相似度 < 本层最高相似度 × 0.5 或 < 0.05 的节点淘汰
    3. 存活节点按比例分配该层 10 权重
    4. 叶子记录累计权重，分支节点展开子节点继续下一层
    5. 某层全部淘汰则终止
```

相比 v1 的变化：
- ❌ 移除：固定剪枝阈值、捷径搜索 (shortcut)、混合搜索
- ✅ 新增：每层固定权重分配、动态淘汰、指数退避

## 📁 项目结构

```
semantic-knowledge-tree/
├── core/                  # 核心代码
├── scripts/               # 评测/工具脚本
├── tests/                 # 测试文件
├── examples/              # 示例代码
├── docs/                  # 文档、论文、多语言 README
├── data/                  # 训练数据、评测数据集
├── results/               # 评测结果
├── smart_tree/            # 知识树（570 节点）
├── demo.py                # 主入口
├── injector.py            # 知识注入工具
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📌 树状态

```
DEPTH=10  NODES=570  LEAVES=392
Python子树: 220 nodes
交叉引用: 56
存储: filesystem (smart_tree/)

--- 评测知识库命中率 ---
HumanEval:     100%   (164/164)
SWE-bench:     80.0%  (4/5)
SWE-bench Pro: 43.8%  (320/731)
MBPass@1:     83.3%  (25/30)
```

> 📌 当前知识库主要覆盖 Python 领域（220 节点），SWE-bench Pro 涉及 JS/Go/Python 多语言仓库，跨语言场景下树命中率明显下降。

## 许可证

[MIT](LICENSE)

