# CIMA: Causal Invariant Memory Agent

基于因果记忆（CMem）与事实记忆（FMem）融合的对抗推广检测方法。

## 方法概述

CIMA 通过两种互补的记忆机制增强 LLM 在跨对抗手段场景下的推广检测能力：

- **CMem (Causal Memory)**：通过 COAT 框架在 Seen Set 上迭代发现因果不变因子，利用 GSQ 条件独立性检验和 FCI 因果发现算法筛选 Markov Blanket 因子，注入 prompt 提供跨分布不变的判断依据。

- **FMem (Fact Memory)**：基于 embedding 相似度从 Seen Set 中检索最相似的正/负样本作为 few-shot 示例，提供感知层面的锚定参考。

## 项目结构

```
CIMA/
├── cima/                       # 核心方法实现
│   ├── config.py              # 统一配置（模型、参数、版本定义）
│   ├── prompts.yaml           # System Prompt 集中管理
│   ├── llm_factory.py         # LLM 客户端构建
│   ├── inference.py           # 推理引擎（4种模式）
│   ├── cmem/                  # Causal Memory 模块
│   │   ├── coat_discovery.py  # COAT 因子发现
│   │   └── prompt_builders.py # CMem Prompt 构建
│   └── fmem/                  # Fact Memory 模块
│       ├── retriever.py       # 相似度检索器
│       └── precompute_embeddings.py  # Embedding 预计算
├── data/                       # 数据集与划分
│   └── split.py               # v1-v8 Seen/Unseen 数据划分
├── baselines/                  # 对比方法
│   ├── train_bert_erm.py      # BERT-ERM
│   ├── train_bert_vrex.py     # BERT-VREx
│   └── train_qwen_lora.py     # Qwen-LoRA
├── scripts/                    # 运行脚本
│   └── run_all.sh             # 全流程脚本
└── README.md
```

## 环境配置

```bash
# 依赖安装
pip install langchain-core langchain-openai openai numpy pandas scipy scikit-learn \
    causal-learn json-repair pyyaml openpyxl

# 环境变量
export CIMA_API_KEY="your-api-key"
export CIMA_API_BASE_URL="https://your-api-endpoint/v2"
export CIMA_INFERENCE_MODEL="qwen3.5-35b"      # 推理模型
export CIMA_PROPOSE_MODEL="qwen3.5-35b"        # 因子提取模型
export CIMA_ANNOTATION_MODEL="qwen3.5-35b"     # 因子标注模型
export CIMA_EMBEDDING_MODEL="qwen3-embedding-4b"  # Embedding 模型
```

## 使用方法

### 1. 数据划分

将标注数据集按 8 种 Seen/Unseen 对抗手段组合划分：

```bash
python data/split.py --dataset path/to/dataset.xlsx --output-dir data/splits
```

数据集格式要求：xlsx 文件，包含以下列：
- `content`: 文本内容
- `is_leadgen`: 标签（"是"/"否"）
- `对抗手段`: 对抗手段标签（逗号分隔，如"语义诱导,圈层黑话"）

### 2. 预计算 Embedding（FMem）

```bash
python -m cima.fmem.precompute_embeddings --dataset path/to/dataset.xlsx --output-dir data
```

### 3. COAT 因子发现（CMem）

```bash
python -m cima.cmem.coat_discovery --seen-set data/splits/v3/seen_set.xlsx --output-dir results/v3
```

### 4. 推理实验

支持四种推理模式：

```bash
# Zero-shot（无额外知识）
python -m cima.inference --mode zero_shot \
    --seen-set data/splits/v3/seen_set.xlsx \
    --unseen-set data/splits/v3/unseen_test_set.xlsx

# FMem-only（相似度检索 few-shot）
python -m cima.inference --mode fmem_only \
    --seen-set data/splits/v3/seen_set.xlsx \
    --unseen-set data/splits/v3/unseen_test_set.xlsx \
    --embeddings data/global_embeddings.npy \
    --embed-meta data/global_embed_meta.json

# CMem-only（因果因子注入）
python -m cima.inference --mode cmem_only \
    --seen-set data/splits/v3/seen_set.xlsx \
    --unseen-set data/splits/v3/unseen_test_set.xlsx \
    --coat-factors results/v3/coat_factors.json

# CMem+FMem（融合模式，完整 CIMA）
python -m cima.inference --mode cmem_fmem \
    --seen-set data/splits/v3/seen_set.xlsx \
    --unseen-set data/splits/v3/unseen_test_set.xlsx \
    --coat-factors results/v3/coat_factors.json \
    --embeddings data/global_embeddings.npy \
    --embed-meta data/global_embed_meta.json
```

### 5. Baselines

```bash
# BERT-ERM
python baselines/train_bert_erm.py --data_dir data/splits --versions v1 v2 v3

# BERT-VREx
python baselines/train_bert_vrex.py --data_dir data/splits --versions v1 v2 v3

# Qwen-LoRA
python baselines/train_qwen_lora.py --data_dir data/splits --versions v1 v2 v3
```

## 实验设置

### 数据集

VPNAD（Versatile Promotion with Natural Adversarial Diversity）数据集包含 10,530 条标注样本（正样本 1,755 条），覆盖 8 种对抗手段：

| 类别 | 对抗手段 |
|------|---------|
| 表层/结构型 | 字符变体、文本结构扰动、特殊编码字符、长文稀释 |
| 语义/行为型 | 间接指引、语义诱导、标签劫持、圈层黑话 |

### 评估协议

8 个版本（v1-v8）的 Seen/Unseen 划分，每个版本将 2-3 种对抗手段设为 Unseen，在 Unseen Test Set 上评估模型的跨对抗手段泛化能力。

## 方法对应关系

| 论文方法名 | 代码模块 | 推理模式 |
|-----------|---------|---------|
| Zero-shot (B0) | `cima/inference.py` | `zero_shot` |
| FMem (B1) | `cima/fmem/` | `fmem_only` |
| CMem (B2) | `cima/cmem/` | `cmem_only` |
| CMem+FMem (B3) | `cima/inference.py` | `cmem_fmem` |
