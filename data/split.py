# -*- coding: utf-8 -*-
"""
数据划分脚本：将标注数据集按对抗手段划分为 Seen Set 和 Unseen Test Set。

版本说明（v1-v8 对应 8 种 Seen/Unseen 对抗手段组合）：
  v1: Unseen = {语义诱导, 圈层黑话, 标签劫持}
  v2: Unseen = {文本结构扰动, 字符变体, 特殊编码字符}
  v3: Unseen = {文本结构扰动, 长文稀释}
  v4: Unseen = {语义诱导, 圈层黑话}
  v5: Unseen = {字符变体, 特殊编码字符}
  v6: Unseen = {间接指引, 标签劫持}
  v7: Unseen = {文本结构扰动, 语义诱导}
  v8: Unseen = {长文稀释, 圈层黑话}

划分规则：
- 含任一 unseen 手段的引流样本 → Unseen Test Set
- 纯 seen 引流样本 + 无对抗手段引流样本 → Seen Set
- 正常样本按比例分配，Unseen Test Set 保持正负例平衡

用法: python split.py --dataset path/to/dataset.xlsx --output-dir ./splits
"""

import os
import argparse

import numpy as np
import pandas as pd
from collections import Counter


# ========== 配置 ==========
RANDOM_SEED = 42

ALL_TACTICS = {'间接指引', '文本结构扰动', '字符变体', '语义诱导',
               '长文稀释', '特殊编码字符', '圈层黑话', '标签劫持'}

VERSIONS = {
    'v1': {
        'unseen': {'语义诱导', '圈层黑话', '标签劫持'},
        'label': '原始划分',
    },
    'v2': {
        'unseen': {'文本结构扰动', '字符变体', '特殊编码字符'},
        'label': '字符/结构扰动',
    },
    'v3': {
        'unseen': {'文本结构扰动', '长文稀释'},
        'label': '结构/格式扰动',
    },
    'v4': {
        'unseen': {'语义诱导', '圈层黑话'},
        'label': '语义层对抗',
    },
    'v5': {
        'unseen': {'字符变体', '特殊编码字符'},
        'label': '字符级混淆',
    },
    'v6': {
        'unseen': {'间接指引', '标签劫持'},
        'label': '引导/劫持类',
    },
    'v7': {
        'unseen': {'文本结构扰动', '语义诱导'},
        'label': '跨类型混合(表层+语义)',
    },
    'v8': {
        'unseen': {'长文稀释', '圈层黑话'},
        'label': '跨类型混合(格式+黑话)',
    },
}


def get_tactics(s):
    """解析对抗手段字段，返回 set。支持逗号和顿号分隔。"""
    if pd.isna(s) or str(s).strip() == '无':
        return set()
    import re
    parts = set(x.strip() for x in re.split(r'[,、]', str(s)) if x.strip())
    # 归一化：特殊字符 → 特殊编码字符
    if '特殊字符' in parts:
        parts.discard('特殊字符')
        parts.add('特殊编码字符')
    return parts


def build_version(df, version_name, unseen_tactics, label, base_dir):
    """为一个版本生成 seen/unseen 数据集。"""
    np.random.seed(RANDOM_SEED)
    seen_tactics = ALL_TACTICS - unseen_tactics

    print(f"\n{'='*60}")
    print(f"  {version_name}: {label}")
    print(f"  Unseen 手段: {sorted(unseen_tactics)}")
    print(f"  Seen 手段:   {sorted(seen_tactics)}")
    print(f"{'='*60}")

    df['tactic_set'] = df['对抗手段'].apply(get_tactics)

    df_pos = df[df['is_leadgen'] == '是'].copy()
    df_neg = df[df['is_leadgen'] == '否'].copy()

    has_unseen = df_pos['tactic_set'].apply(lambda s: len(s) > 0 and len(s & unseen_tactics) > 0)
    pure_seen = df_pos['tactic_set'].apply(lambda s: len(s) > 0 and s.issubset(seen_tactics))
    no_tactic = df_pos['tactic_set'].apply(lambda s: len(s) == 0)

    pos_unseen = df_pos[has_unseen]
    pos_seen = df_pos[pure_seen | no_tactic]

    print(f"  Unseen 引流: {len(pos_unseen)},  Seen 引流: {len(pos_seen)}")

    # 正常样本按 1:5 比例分配
    neg_ratio = 5
    n_neg_for_unseen = min(len(pos_unseen) * neg_ratio, len(df_neg))
    n_neg_for_seen = min(len(pos_seen) * neg_ratio, len(df_neg) - n_neg_for_unseen)
    neg_indices = np.random.permutation(len(df_neg))
    neg_unseen = df_neg.iloc[neg_indices[:n_neg_for_unseen]]
    neg_seen = df_neg.iloc[neg_indices[n_neg_for_unseen:n_neg_for_unseen + n_neg_for_seen]]

    seen_set = pd.concat([pos_seen, neg_seen], ignore_index=True)
    unseen_set = pd.concat([pos_unseen, neg_unseen], ignore_index=True)

    seen_set = seen_set.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    unseen_set = unseen_set.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    for ds in [seen_set, unseen_set]:
        if 'tactic_set' in ds.columns:
            ds.drop(columns=['tactic_set'], inplace=True)

    # 保存
    out_dir = os.path.join(base_dir, version_name)
    os.makedirs(out_dir, exist_ok=True)
    seen_path = os.path.join(out_dir, 'seen_set.xlsx')
    unseen_path = os.path.join(out_dir, 'unseen_test_set.xlsx')
    seen_set.to_excel(seen_path, index=False)
    unseen_set.to_excel(unseen_path, index=False)

    n_seen_pos = (seen_set['is_leadgen'] == '是').sum()
    n_seen_neg = (seen_set['is_leadgen'] == '否').sum()
    n_unseen_pos = (unseen_set['is_leadgen'] == '是').sum()
    n_unseen_neg = (unseen_set['is_leadgen'] == '否').sum()

    print(f"  Seen Set:   {len(seen_set):4d} 条 (引流 {n_seen_pos}, 正常 {n_seen_neg})")
    print(f"  Unseen Set: {len(unseen_set):4d} 条 (引流 {n_unseen_pos}, 正常 {n_unseen_neg})")

    # 无泄漏验证
    seen_tactics_found = set()
    for s in seen_set[seen_set['is_leadgen'] == '是']['对抗手段'].apply(get_tactics):
        seen_tactics_found.update(s)
    leak = seen_tactics_found & unseen_tactics
    if leak:
        print(f"  *** 警告: Seen 中出现 unseen 手段: {leak} ***")
    else:
        print(f"  无泄漏验证通过")

    return {
        'version': version_name,
        'label': label,
        'unseen_tactics': sorted(unseen_tactics),
        'seen_size': len(seen_set),
        'unseen_size': len(unseen_set),
    }


def main():
    parser = argparse.ArgumentParser(description="VPNAD 数据集 Seen/Unseen 划分")
    parser.add_argument('--dataset', required=True, help='标注数据集路径 (xlsx)')
    parser.add_argument('--output-dir', default='./splits', help='输出目录')
    parser.add_argument('--versions', nargs='+', default=list(VERSIONS.keys()),
                        help='要生成的版本列表，默认全部 v1-v8')
    args = parser.parse_args()

    df = pd.read_excel(args.dataset)

    # 兼容 is_leadgen 为 1/0 整数的情况
    if df['is_leadgen'].dtype != object:
        df['is_leadgen'] = df['is_leadgen'].map({1: '是', 0: '否'})

    print(f"原始数据: {len(df)} 条 (引流 {(df['is_leadgen']=='是').sum()}, "
          f"正常 {(df['is_leadgen']=='否').sum()})")

    summaries = []
    for ver in args.versions:
        cfg = VERSIONS[ver]
        info = build_version(df.copy(), ver, cfg['unseen'], cfg['label'], args.output_dir)
        summaries.append(info)

    # 汇总表
    print(f"\n{'='*60}")
    print(f"  所有版本汇总")
    print(f"{'='*60}")
    sum_df = pd.DataFrame(summaries)
    print(sum_df.to_string(index=False))


if __name__ == '__main__':
    main()
