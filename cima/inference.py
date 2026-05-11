# -*- coding: utf-8 -*-
"""
CIMA 推理引擎：支持四种推理模式的统一入口。

推理模式：
- zero_shot:  无额外知识，直接分类
- fmem_only:  FMem 相似度检索 few-shot (B1)
- cmem_only:  CMem 因果因子注入 (B2)
- cmem_fmem:  CMem + FMem 融合 (B3)

用法:
    python -m cima.inference --version v3 --mode cmem_fmem \\
        --seen-set data/v3/seen_set.xlsx \\
        --unseen-set data/v3/unseen_test_set.xlsx \\
        --coat-factors results/v3/coat_factors.json \\
        --embeddings data/global_embeddings.npy \\
        --embed-meta data/global_embed_meta.json
"""

import argparse
import asyncio
import json
import os
import random

import numpy as np
import pandas as pd
from json_repair import repair_json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .config import RANDOM_SEED, MAX_CONCURRENCY, MAX_RETRIES, RETRY_BASE_WAIT, TEXT_COL, LABEL_COL
from .llm_factory import build_inference_chain
from .cmem.prompt_builders import (
    build_cmem_prompt, build_cmem_fmem_prompt, format_causal_factors_text,
    SYSTEM_CLASSIFY, _prompt_payload,
)
from .fmem.retriever import SimilarityRetriever


def parse_prediction(response):
    """解析 LLM 分类结果。"""
    if response is None:
        return 0, "", "failed"
    text = response.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        parsed = json.loads(text)
        is_lg = parsed.get('is_leadgen', False)
        reasoning = parsed.get('reasoning', '')
        return (1 if is_lg else 0), reasoning, "ok"
    except Exception:
        pass
    try:
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            is_lg = repaired.get('is_leadgen', False)
            reasoning = repaired.get('reasoning', '')
            return (1 if is_lg else 0), reasoning, "json_repair"
    except Exception:
        pass
    return 0, text, "failed"


def build_zero_shot_prompt(text):
    """Zero-shot: 无额外知识。"""
    content = f"请判断以下文本：\n文本：{text}"
    return _prompt_payload(SYSTEM_CLASSIFY, content)


def build_fmem_prompt(text, n_shot, examples):
    """FMem-only: 相似度检索 few-shot。"""
    content = ""
    if n_shot > 0:
        content += "以下是一些示例：\n"
        for i, (ex_text, ex_label) in enumerate(examples[:n_shot], 1):
            ex_json = json.dumps({"reasoning": "略", "is_leadgen": ex_label}, ensure_ascii=False)
            content += f"\n示例{i}：\n文本：{ex_text}\n输出：{ex_json}\n"
    content += f"\n请判断以下文本：\n文本：{text}"
    return _prompt_payload(SYSTEM_CLASSIFY, content)


async def run_inference(texts, prompt_builder, chain, max_concurrency=MAX_CONCURRENCY):
    """并发推理。"""
    semaphore = asyncio.Semaphore(max_concurrency)
    predictions = [0] * len(texts)
    details = [None] * len(texts)

    async def _call(idx, text):
        prompt_payload = prompt_builder(text)
        async with semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    raw_response = await chain.ainvoke(prompt_payload)
                    pred, reasoning, status = parse_prediction(raw_response)
                    return (idx, pred, status)
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_BASE_WAIT * (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(wait)
                    else:
                        return (idx, 0, "error")
            return (idx, 0, "retry_exhausted")

    tasks = [_call(i, t) for i, t in enumerate(texts)]
    print(f"  推理中，共 {len(tasks)} 条...")
    results = await asyncio.gather(*tasks)

    for idx, pred, status in results:
        predictions[idx] = pred

    return np.array(predictions)


async def run_experiment(mode, seen_set_path, unseen_set_path,
                         coat_factors_path=None, embeddings_path=None,
                         embed_meta_path=None, output_dir=None, max_samples=None):
    """执行指定模式的推理实验。

    Args:
        mode: 推理模式 (zero_shot / fmem_only / cmem_only / cmem_fmem)
        seen_set_path: Seen Set 路径
        unseen_set_path: Unseen Test Set 路径
        coat_factors_path: COAT 因子文件路径 (cmem_only/cmem_fmem 需要)
        embeddings_path: 全量 embedding 路径 (fmem_only/cmem_fmem 需要)
        embed_meta_path: embedding meta 路径 (fmem_only/cmem_fmem 需要)
        output_dir: 结果输出目录
        max_samples: 最大测试样本数
    """
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # 加载数据
    seen_df = pd.read_excel(seen_set_path)
    unseen_df = pd.read_excel(unseen_set_path)
    if max_samples:
        unseen_df = unseen_df.head(max_samples)

    label_map = {'是': 1, '否': 0, True: 1, False: 0, 1: 1, 0: 0}
    seen_texts = seen_df[TEXT_COL].astype(str).values
    unseen_texts = unseen_df[TEXT_COL].astype(str).values
    unseen_y = np.array([label_map.get(v, 0) for v in unseen_df[LABEL_COL].values])

    print(f"Seen Set: {len(seen_texts)} 条, Unseen Set: {len(unseen_texts)} 条")

    # 构建推理 chain
    chain = build_inference_chain()

    # 根据模式构建 prompt builder
    if mode == 'zero_shot':
        prompt_builder = build_zero_shot_prompt

    elif mode == 'fmem_only':
        assert embeddings_path and embed_meta_path, "FMem 模式需要 embedding 文件"
        retriever = SimilarityRetriever(embeddings_path, embed_meta_path, seen_texts.tolist())
        await retriever.precompute_queries(unseen_texts.tolist())
        _text_to_idx = {t: i for i, t in enumerate(unseen_texts)}

        def prompt_builder(text):
            idx = _text_to_idx.get(text, 0)
            examples = retriever.retrieve_by_index(idx)
            return build_fmem_prompt(text, 5, examples)

    elif mode == 'cmem_only':
        assert coat_factors_path, "CMem 模式需要 COAT 因子文件"
        with open(coat_factors_path, 'r', encoding='utf-8') as f:
            coat_info = json.load(f)
        causal_factors_text = format_causal_factors_text(
            coat_info['mb_factors'], coat_info['mb_factor_definitions'])

        def prompt_builder(text):
            return build_cmem_prompt(text, causal_factors_text)

    elif mode == 'cmem_fmem':
        assert coat_factors_path, "CMem+FMem 模式需要 COAT 因子文件"
        assert embeddings_path and embed_meta_path, "CMem+FMem 模式需要 embedding 文件"

        with open(coat_factors_path, 'r', encoding='utf-8') as f:
            coat_info = json.load(f)
        causal_factors_text = format_causal_factors_text(
            coat_info['mb_factors'], coat_info['mb_factor_definitions'])

        retriever = SimilarityRetriever(embeddings_path, embed_meta_path, seen_texts.tolist())
        await retriever.precompute_queries(unseen_texts.tolist())
        _text_to_idx = {t: i for i, t in enumerate(unseen_texts)}

        def prompt_builder(text):
            idx = _text_to_idx.get(text, 0)
            examples = retriever.retrieve_by_index(idx)
            return build_cmem_fmem_prompt(text, causal_factors_text, 5, examples)

    else:
        raise ValueError(f"未知推理模式: {mode}")

    # 执行推理
    print(f"\n{'='*50} {mode} {'='*50}")
    preds = await run_inference(unseen_texts, prompt_builder, chain)

    # 计算指标
    acc = accuracy_score(unseen_y, preds)
    prec = precision_score(unseen_y, preds, zero_division=0)
    rec = recall_score(unseen_y, preds, zero_division=0)
    f1 = f1_score(unseen_y, preds, zero_division=0)

    print(f"\n结果 [{mode}]:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")

    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result = {
            'mode': mode,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'n_samples': len(unseen_texts),
        }
        result_path = os.path.join(output_dir, f'{mode}_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存: {result_path}")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def main():
    parser = argparse.ArgumentParser(description="CIMA 推理")
    parser.add_argument('--mode', required=True,
                        choices=['zero_shot', 'fmem_only', 'cmem_only', 'cmem_fmem'],
                        help='推理模式')
    parser.add_argument('--seen-set', required=True, help='Seen Set 路径')
    parser.add_argument('--unseen-set', required=True, help='Unseen Test Set 路径')
    parser.add_argument('--coat-factors', default=None, help='COAT 因子文件路径')
    parser.add_argument('--embeddings', default=None, help='全量 embedding 路径')
    parser.add_argument('--embed-meta', default=None, help='embedding meta 路径')
    parser.add_argument('--output-dir', default='./results', help='输出目录')
    parser.add_argument('--max-samples', type=int, default=None, help='最大测试样本数')
    args = parser.parse_args()

    asyncio.run(run_experiment(
        mode=args.mode,
        seen_set_path=args.seen_set,
        unseen_set_path=args.unseen_set,
        coat_factors_path=args.coat_factors,
        embeddings_path=args.embeddings,
        embed_meta_path=args.embed_meta,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    ))


if __name__ == '__main__':
    main()
