# -*- coding: utf-8 -*-
"""
CMem Prompt 构建函数：用于因子发现阶段（propose/annotate）和推理阶段（因子注入分类）。
"""

import json
import os

import numpy as np
import yaml


# 加载 prompts.yaml
_PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prompts.yaml')
with open(_PROMPTS_PATH, encoding='utf-8') as _f:
    _P = yaml.safe_load(_f)

SYSTEM_CLASSIFY = _P['system']['classify']
SYSTEM_PROPOSE = _P['system']['propose']
SYSTEM_ANNOTATE = _P['system']['annotate']
SYSTEM_EXPLAIN = _P['system']['explain']


def _prompt_payload(system_instruct, content):
    return {"system_instruct": system_instruct, "content": content}


# ---- 因子发现阶段 prompt ----

def _build_data_example(texts, labels, n_per_group=100):
    """从正/负样本中各采样 n_per_group 条，拼接为数据展示文本。"""
    data = ''
    for group_val, group_name in [(1, '推广'), (0, '正常')]:
        indices = np.where(labels == group_val)[0]
        sampled = np.random.choice(indices, min(len(indices), n_per_group), replace=False)
        data += f"\n## 分组：'is_leadgen' = {group_name}\n\n"
        for i in sampled:
            data += f"- {texts[i].replace(chr(10), ' ').strip()[:300]}\n"
    return data


def build_propose_prompt(texts, labels, used_factors=None, n_per_group=100):
    """构建 COAT 因子提取阶段的 prompt payload。"""
    example = _build_data_example(texts, labels, n_per_group)
    used_factors_hint = ''
    if used_factors:
        used_factors_hint = (
            "\n\n# 已有因子（避免重叠）\n\n- "
            + '\n- '.join(used_factors)
        )
    content = f'''# 数据

{example}

# 任务

请观察上述已标注样本，抽象出能够区分"推广"和"正常"的高层语义因子。请先进行思考与筛选，再给出最终采用的因子。{used_factors_hint}
'''
    return _prompt_payload(SYSTEM_PROPOSE, content)


def build_annotation_prompt(text, factor_name, criteria):
    """构建单条文本的因子标注 prompt payload。"""
    content = (
        f"请阅读以下文本：\n\"\"\"{text}\"\"\"\n\n"
        f"请根据以下评判标准，对因子 **\"{factor_name}\"** 进行评估：\n"
        f"{criteria}"
    )
    return _prompt_payload(SYSTEM_ANNOTATE, content)


# ---- 推理阶段 prompt (CMem 因子注入) ----

def build_cmem_prompt(text, causal_factors_text):
    """CMem-only: 注入因果因子定义进行分类。"""
    content = causal_factors_text
    content += f"\n\n请综合以上因子分析以下文本，判断是否为推广。\n文本：{text}"
    return _prompt_payload(SYSTEM_CLASSIFY, content)


def build_cmem_fmem_prompt(text, causal_factors_text, n_shot, examples):
    """CMem+FMem: 因果因子 + few-shot 示例联合注入。"""
    content = causal_factors_text
    if n_shot > 0:
        content += "\n以下是一些示例：\n"
        for i, (ex_text, ex_label) in enumerate(examples[:n_shot], 1):
            ex_json = json.dumps({"reasoning": "略", "is_leadgen": ex_label}, ensure_ascii=False)
            content += f"\n示例{i}：\n文本：{ex_text}\n输出：{ex_json}\n"
    content += f"\n请综合以上因子和示例分析以下文本，判断是否为推广。\n文本：{text}"
    return _prompt_payload(SYSTEM_CLASSIFY, content)


def build_explain_prompt(ex_text, ex_label, causal_factors_text):
    """为 few-shot 示例生成基于因子的可解释性推理。"""
    label_str = "推广" if ex_label else "正常"
    content = (
        f"请根据以下因果因子，分析这段文本为什么属于「{label_str}」内容。\n\n"
        f"{causal_factors_text}"
        f"文本：{ex_text}\n"
    )
    return _prompt_payload(SYSTEM_EXPLAIN, content)


def format_causal_factors_text(mb_factors, mb_factor_defs):
    """将因子定义格式化为 prompt 注入文本。"""
    text = "以下是经过因果分析发现的、与推广判定相关的关键因子及其评判标准：\n\n"
    for i, f_name in enumerate(mb_factors, 1):
        if f_name in mb_factor_defs:
            full_def = mb_factor_defs[f_name]
            text += f"{i}. {full_def}\n\n"
        else:
            text += f"{i}. {f_name}\n\n"
    return text
