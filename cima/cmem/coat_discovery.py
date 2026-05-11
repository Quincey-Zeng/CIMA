# -*- coding: utf-8 -*-
"""
CMem COAT 因子发现引擎

在 Seen Set 上运行迭代因子发现流程：
1. 采样 → 2. LLM Propose → 3. LLM Annotate → 4. GSQ 检验 → 5. FCI 因果发现

输出：Markov Blanket 因子定义（供推理阶段注入 prompt）。

用法:
    python -m cima.cmem.coat_discovery --version v3 --seen-set path/to/seen_set.xlsx --output-dir ./results
"""

import argparse
import asyncio
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy
from sklearn.cluster import KMeans
from json_repair import repair_json

from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import CIT

from ..config import (
    COAT_N_ITERATIONS, COAT_N_SAMPLES_PER_GROUP, COAT_ALPHA,
    COAT_CI_METHOD, COAT_CORR_THRESHOLD, MAX_CONCURRENCY, RANDOM_SEED,
    TEXT_COL, LABEL_COL,
)
from ..llm_factory import build_propose_chain, build_annotation_chain
from .prompt_builders import build_propose_prompt, build_annotation_prompt


# ========== 工具函数 ==========

def parse_annotation_value(response):
    """从 JSON 格式的标注响应中解析因子取值。"""
    if response is None:
        return '?'
    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
        val = parsed.get('value')
        if val in (-1, 0, 1):
            return val
    except Exception:
        pass
    try:
        parsed = repair_json(text, return_objects=True)
        if isinstance(parsed, dict):
            val = parsed.get('value')
            if val in (-1, 0, 1):
                return val
    except Exception:
        pass
    return '?'


def get_entropy_from_samples(y):
    _, sk = np.unique(y, return_counts=True)
    pk = sk / sk.sum()
    return entropy(pk)


def get_factor_name(this_factor):
    this_factor = this_factor.split('\n')[0].lower()
    for i in range(20):
        if f'{i}.' in this_factor:
            this_factor = this_factor.replace(f'{i}.', '').strip()
    for s in ":":
        if s in this_factor:
            this_factor = this_factor.replace(s, '').strip()
    return this_factor


def check_factor_list(factor_list):
    factor_list = [f for f in factor_list if len(f) > 10]
    factor_list = [f for f in factor_list if (" 1:" in f) and (" -1:" in f) and (" 0:" in f)]
    for outer_idx, each_str in enumerate(factor_list):
        for idx, ch in enumerate(each_str):
            if ch.isalpha():
                break
        each_str_list = each_str[idx:].split('\n')
        each_str = each_str_list[0].lower()
        each_str_2 = '\n'.join(each_str_list[1:])
        for i in range(100):
            if f'{i}' in each_str:
                each_str = each_str.replace(f'{i}', '').strip()
        for s in ['factor', '*', 'name', ':', '.']:
            if s in each_str:
                each_str = each_str.replace(s, '').strip()
        if len(each_str) > 0:
            factor_list[outer_idx] = each_str + '\n' + each_str_2
        else:
            factor_list[outer_idx] = ''
    factor_list = [f for f in factor_list if len(f) > 10]
    return factor_list


def GetMB(G, node_name, y_node=0):
    mbset = set([y_node])
    d = G.shape[0]
    get_direct_set = lambda x: set([idx for idx in range(d) if np.abs(G[x, idx]) + np.abs(G[idx, x]) > 0])
    direct_set = get_direct_set(y_node)
    mbset = mbset.union(direct_set)
    for idx in direct_set:
        if G[idx, y_node] == -1:
            continue
        for each_secondary in get_direct_set(idx):
            if G[idx, each_secondary] == -1:
                continue
            mbset.add(each_secondary)
    return set([node_name[i] for i in mbset])


def get_possible_ancestors(G, annotated_name):
    d = G.shape[0]
    A = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if (G[i, j] == -1) and (G[j, i] == 1):
                A[j, i] = 1
            if (G[i, j] == 2) and (G[j, i] == 1):
                A[j, i] = 1
            if (G[i, j] == 2) and (G[j, i] == 2):
                A[j, i] = 1
                A[i, j] = 1
    possible_path = A > 0
    for _ in range(d):
        possible_path |= (possible_path @ A) > 0
    return [annotated_name[idx] for idx in range(d) if possible_path[0, idx]]


def get_possible_parents(g, annotated_name):
    possible_parents = np.array([False] * len(g.graph[:, 0]))
    possible_parents |= (g.graph[:, 0] == 2) & (g.graph[0, :] == 1)
    possible_parents |= (g.graph[:, 0] == -1) & (g.graph[0, :] == 1)
    possible_parents[0] = False
    return [annotated_name[idx] for idx in range(len(annotated_name)) if possible_parents[idx]]


# ========== 标注函数 ==========

async def annotate_factors(texts, factor_list, chain, max_concurrency=MAX_CONCURRENCY):
    """并发标注所有文本的所有因子取值。"""
    parsed_factors = []
    for f in factor_list:
        lines = f.strip().split('\n')
        name = get_factor_name(lines[0])
        criteria = '\n'.join(lines[1:]) if len(lines) > 1 else ''
        parsed_factors.append((name, criteria))

    n_texts = len(texts)
    n_factors = len(parsed_factors)
    result_matrix = np.zeros((n_texts, n_factors), dtype=int)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _call(t_idx, f_idx, text, name, criteria):
        prompt_payload = build_annotation_prompt(text, name, criteria)
        async with semaphore:
            try:
                response = await chain.ainvoke(prompt_payload)
                val = parse_annotation_value(response)
                return (t_idx, f_idx, val if val in (-1, 0, 1) else 0)
            except Exception as e:
                print(f"  标注出错 [{name}]: {e}")
                return (t_idx, f_idx, 0)

    tasks = [
        _call(t_idx, f_idx, text, name, criteria)
        for t_idx, text in enumerate(texts)
        for f_idx, (name, criteria) in enumerate(parsed_factors)
    ]

    print(f"  并发标注中 (共 {len(tasks)} 次调用, max_concurrency={max_concurrency})...")
    results_list = await asyncio.gather(*tasks)

    for t_idx, f_idx, val in results_list:
        result_matrix[t_idx, f_idx] = val

    return pd.DataFrame(result_matrix, columns=[name for name, _ in parsed_factors])


# ========== 主流程 ==========

async def run_coat_discovery(seen_set_path, output_dir, max_samples=None):
    """执行 COAT 因子发现流程。

    Args:
        seen_set_path: Seen Set 数据路径 (xlsx)
        output_dir: 输出目录
        max_samples: 最大样本数（用于快速测试）

    Returns:
        dict: 包含 mb_factors, mb_factor_definitions, causal_parents 等
    """
    TARGET_NAME = "is_leadgen"

    # 数据加载
    meta = pd.read_excel(seen_set_path)
    if max_samples:
        meta = meta.head(max_samples)
    print(f"数据集: {seen_set_path}")

    label_map = {'是': 1, '否': 0, True: 1, False: 0, 1: 1, 0: 0}
    meta['_label_int'] = [label_map.get(v, int(v) if str(v).isdigit() else 0) for v in meta[LABEL_COL].values]

    # 1:1 正负样本比例
    pos_df = meta[meta['_label_int'] == 1]
    neg_df = meta[meta['_label_int'] == 0]
    n_pos = len(pos_df)
    if len(neg_df) > n_pos:
        neg_df = neg_df.sample(n=n_pos, random_state=RANDOM_SEED)
        meta = pd.concat([pos_df, neg_df], ignore_index=True).sample(
            frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"COAT 1:1 降采样: 负样本 {len(neg_df)} -> {n_pos}")

    texts = meta[TEXT_COL].astype(str).values
    y = meta['_label_int'].values
    print(f"数据集形状: {meta.shape}, 标签分布: 引流={sum(y==1)}, 正常={sum(y==0)}")

    # 构建 LLM 客户端
    propose_chain = build_propose_chain()
    annotation_chain = build_annotation_chain()

    # 初始化
    V = set([TARGET_NAME])
    data_interface = pd.DataFrame({TARGET_NAME: y})
    used_factor_names = []
    factor_definitions = {}

    # COAT 迭代循环
    for iteration in range(1, COAT_N_ITERATIONS + 1):
        print(f"\n{'='*50} COAT 第 {iteration} 轮迭代 {'='*50}")

        # Step 1: 采样
        if iteration == 1 or data_interface.shape[1] <= 1:
            sample_texts = texts
            sample_labels = y
        else:
            feature_cols = [c for c in V if c != TARGET_NAME]
            if len(feature_cols) > 0:
                cond_vectors = data_interface[feature_cols].values.astype(float)
                n_clusters = min(len(feature_cols), 4)
                if n_clusters < 2:
                    n_clusters = 2
                km_labels = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(cond_vectors).labels_
                entropy_values = []
                set_g = list(set(km_labels))
                for g_idx in set_g:
                    mask = km_labels == g_idx
                    if mask.sum() < 6:
                        entropy_values.append(-1)
                        continue
                    qk = np.unique(y[mask], return_counts=True)[1]
                    qk = qk / qk.sum()
                    entropy_values.append(stats.entropy(qk))
                best_cluster = set_g[np.argmax(entropy_values)]
                hard_mask = km_labels == best_cluster
                sample_texts = texts[hard_mask]
                sample_labels = y[hard_mask]
                print(f"  使用高熵子集（簇 {best_cluster}），共 {hard_mask.sum()} 条样本")
            else:
                sample_texts = texts
                sample_labels = y

        # Step 2: LLM 提出候选因子
        print(f"  [Step 2] LLM 提出候选因子...")
        prompt_payload = build_propose_prompt(
            sample_texts, sample_labels,
            used_factors=used_factor_names if used_factor_names else None,
            n_per_group=COAT_N_SAMPLES_PER_GROUP,
        )
        llm_response = propose_chain.invoke(prompt_payload)

        # Step 3: 解析因子
        raw_text = llm_response.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").removeprefix("json").strip()

        parsed_json = None
        try:
            parsed_json = json.loads(raw_text)
        except Exception:
            try:
                parsed_json = repair_json(raw_text, return_objects=True)
            except Exception:
                pass

        raw_factors = []
        if parsed_json and isinstance(parsed_json, dict) and 'factors' in parsed_json:
            for f_obj in parsed_json['factors']:
                name = f_obj.get('name', '').strip()
                criteria_pos = f_obj.get('criteria_pos', '')
                criteria_zero = f_obj.get('criteria_zero', '其他情况；或未提及')
                criteria_neg = f_obj.get('criteria_neg', '')
                if name:
                    raw_factors.append(
                        f"{name}\n- 1: {criteria_pos}\n- 0: {criteria_zero}\n- -1: {criteria_neg}"
                    )
        else:
            print("  警告：JSON 解析失败，尝试正则回退解析...")
            factor_pattern = re.compile(r'\*\*([^*]+)\*\*\s*\n((?:[-•].*\n?)+)', re.MULTILINE)
            for m in factor_pattern.finditer(llm_response):
                factor_name = m.group(1).strip()
                factor_body = m.group(2).strip()
                raw_factors.append(f"{factor_name}\n{factor_body}")

        candidate_factors = check_factor_list(raw_factors)
        print(f"  解析到 {len(candidate_factors)} 个有效候选因子")
        for f in candidate_factors:
            fname = get_factor_name(f)
            print(f"    - {fname}")
            factor_definitions[fname] = f

        if not candidate_factors:
            print("  警告：未解析到有效因子，跳过本轮")
            continue

        # Step 4: 自动标注
        print(f"  [Step 4] 自动标注（{len(texts)} 条 × {len(candidate_factors)} 个因子）...")
        annotation_df = await annotate_factors(texts, candidate_factors, annotation_chain)

        for col in annotation_df.columns:
            data_interface[col] = annotation_df[col].values

        # Step 5: 条件独立性检验
        print(f"  [Step 5] 条件独立性检验 ({COAT_CI_METHOD})...")
        V_cols = [c for c in data_interface.columns if c != TARGET_NAME]
        new_factor_names = list(annotation_df.columns)

        ci_matrix = data_interface[[TARGET_NAME] + V_cols].values.astype(float)
        ci_test = CIT(ci_matrix, COAT_CI_METHOD)

        target_idx = 0
        existing_idxs = [i+1 for i, c in enumerate(V_cols) if c not in new_factor_names]
        new_idxs = [i+1 for i, c in enumerate(V_cols) if c in new_factor_names]

        new_factors_passed = []
        for idx in new_idxs:
            p_val = ci_test(target_idx, idx, existing_idxs)
            factor_name = V_cols[idx - 1]
            status = '通过' if p_val < COAT_ALPHA else '拒绝'
            print(f"    {factor_name}: p={p_val:.4f} {status}")
            if p_val < COAT_ALPHA:
                new_factors_passed.append(idx)

        accepted_factor_names = [V_cols[i-1] for i in new_factors_passed]
        print(f"  本轮接受因子: {accepted_factor_names}")

        if not new_factors_passed:
            continue

        # Step 6: 去重 + FCI
        print(f"  [Step 6] 去重 + FCI 因果发现...")
        all_valid_cols = [TARGET_NAME] + [c for c in V if c != TARGET_NAME] + accepted_factor_names
        all_valid_cols = list(dict.fromkeys(all_valid_cols))

        fci_matrix = data_interface[all_valid_cols].values.astype(float)
        corr_matrix = np.corrcoef(fci_matrix.T)
        to_remove = set()
        n_cols = len(all_valid_cols)
        for i in range(1, n_cols):
            for j in range(i+1, n_cols):
                if abs(corr_matrix[i, j]) > COAT_CORR_THRESHOLD:
                    to_remove.add(all_valid_cols[j])

        surviving_cols = [c for c in all_valid_cols if c not in to_remove]
        fci_data = data_interface[surviving_cols].values.astype(float)

        try:
            g, edges = fci(fci_data, alpha=COAT_ALPHA, independence_test_method='gsq', verbose=False)
            target_node_idx = surviving_cols.index(TARGET_NAME)
            V = GetMB(g.graph, surviving_cols, y_node=target_node_idx)
            V.add(TARGET_NAME)
            print(f"  更新后 V (Markov Blanket) = {V}")
        except Exception as e:
            print(f"  FCI 运行出错: {e}")
            V = V.union(set(accepted_factor_names))

        used_factor_names.extend([get_factor_name(f) for f in candidate_factors])

    # 结果汇总
    final_mb_factors = [c for c in V if c != TARGET_NAME]
    print(f"\n{'='*50} COAT 因子发现完成 {'='*50}")
    print(f"MB 因子数: {len(final_mb_factors)}")
    for f in final_mb_factors:
        print(f"  - {f}")

    # 最终因果图
    causal_parents = []
    if final_mb_factors:
        final_cols = [TARGET_NAME] + final_mb_factors
        final_data = data_interface[final_cols].values.astype(float)
        try:
            g_final, _ = fci(final_data, alpha=COAT_ALPHA, independence_test_method='gsq', verbose=False)
            causal_parents = get_possible_parents(g_final, final_cols)
        except Exception:
            pass

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    mb_factor_defs = {}
    for f in final_mb_factors:
        mb_factor_defs[f] = factor_definitions.get(f, f)

    result = {
        'mb_factors': final_mb_factors,
        'mb_factor_definitions': mb_factor_defs,
        'all_factors': [c for c in data_interface.columns if c != TARGET_NAME],
        'all_factor_definitions': factor_definitions,
        'causal_parents': causal_parents,
    }

    factor_json_path = os.path.join(output_dir, 'coat_factors.json')
    with open(factor_json_path, 'w', encoding='utf-8') as fp:
        json.dump(result, fp, ensure_ascii=False, indent=2)
    print(f"因子定义已保存: {factor_json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="COAT 因子发现")
    parser.add_argument('--seen-set', required=True, help='Seen Set 数据路径 (xlsx)')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--max-samples', type=int, default=None, help='最大样本数（快速测试用）')
    args = parser.parse_args()

    asyncio.run(run_coat_discovery(args.seen_set, args.output_dir, args.max_samples))


if __name__ == '__main__':
    main()
