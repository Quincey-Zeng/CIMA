# -*- coding: utf-8 -*-
"""
CIMA 配置中心：模型配置、版本配置、实验参数。

使用前请设置环境变量：
  export CIMA_API_KEY="your-api-key"
  export CIMA_API_BASE_URL="https://your-api-endpoint/v2"
"""

import os

# ========== API 配置 ==========
API_KEY = os.environ.get("CIMA_API_KEY", "")
API_BASE_URL = os.environ.get("CIMA_API_BASE_URL", "")

# ========== 模型配置 ==========
# 推理模型（用于最终分类判断）
INFERENCE_MODEL = os.environ.get("CIMA_INFERENCE_MODEL", "qwen3.5-35b")
# 因子提取模型（用于 COAT 因子发现阶段的 propose）
PROPOSE_MODEL = os.environ.get("CIMA_PROPOSE_MODEL", "qwen3.5-35b")
# 标注模型（用于 COAT 因子标注阶段）
ANNOTATION_MODEL = os.environ.get("CIMA_ANNOTATION_MODEL", "qwen3.5-35b")
# Embedding 模型（用于 FMem 相似度检索）
EMBEDDING_MODEL = os.environ.get("CIMA_EMBEDDING_MODEL", "qwen3-embedding-4b")

# ========== 实验参数 ==========
RANDOM_SEED = 42

# COAT 因子发现参数
COAT_N_ITERATIONS = 5          # 迭代轮数
COAT_N_SAMPLES_PER_GROUP = 100 # 每组采样数
COAT_ALPHA = 0.05              # 条件独立性检验显著性水平
COAT_CI_METHOD = 'gsq'         # 条件独立性检验方法 (gsq/kci)
COAT_CORR_THRESHOLD = 0.9      # 去重高相关因子阈值

# FMem 检索参数
FMEM_N_POS = 3                 # 检索正样本数
FMEM_N_NEG = 2                 # 检索负样本数
FMEM_MAX_INPUT_CHARS = 4000    # embedding 输入最大字符数

# 推理参数
MAX_CONCURRENCY = 60
MAX_RETRIES = 5
RETRY_BASE_WAIT = 3

# ========== 数据列名 ==========
TEXT_COL = "content"
LABEL_COL = "is_leadgen"
TACTIC_COL = "对抗手段"

# ========== 版本定义 ==========
ALL_TACTICS = {'间接指引', '文本结构扰动', '字符变体', '语义诱导',
               '长文稀释', '特殊编码字符', '圈层黑话', '标签劫持'}

VERSIONS = {
    'v1': {'unseen': {'语义诱导', '圈层黑话', '标签劫持'}, 'label': '原始划分'},
    'v2': {'unseen': {'文本结构扰动', '字符变体', '特殊编码字符'}, 'label': '字符/结构扰动'},
    'v3': {'unseen': {'文本结构扰动', '长文稀释'}, 'label': '结构/格式扰动'},
    'v4': {'unseen': {'语义诱导', '圈层黑话'}, 'label': '语义层对抗'},
    'v5': {'unseen': {'字符变体', '特殊编码字符'}, 'label': '字符级混淆'},
    'v6': {'unseen': {'间接指引', '标签劫持'}, 'label': '引导/劫持类'},
    'v7': {'unseen': {'文本结构扰动', '语义诱导'}, 'label': '跨类型混合(表层+语义)'},
    'v8': {'unseen': {'长文稀释', '圈层黑话'}, 'label': '跨类型混合(格式+黑话)'},
}
