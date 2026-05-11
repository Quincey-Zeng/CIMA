# -*- coding: utf-8 -*-
"""
预计算全量数据集的 embedding 向量，供 FMem 相似度检索使用。
只需运行一次，所有版本(v1-v8)共享同一份 embedding。

用法:
    python -m cima.fmem.precompute_embeddings --dataset path/to/dataset.xlsx --output-dir ./data
"""

import argparse
import asyncio
import json
import os

import numpy as np
import pandas as pd
from openai import OpenAI

from ..config import API_KEY, API_BASE_URL, EMBEDDING_MODEL, FMEM_MAX_INPUT_CHARS

MAX_CONCURRENCY = 8
MAX_RETRIES = 3

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    return _client


async def _embed_one(semaphore, index, text):
    async with semaphore:
        client = _get_client()
        text = text[:FMEM_MAX_INPUT_CHARS]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await asyncio.to_thread(
                    client.embeddings.create, model=EMBEDDING_MODEL, input=text
                )
                return index, response.data[0].embedding
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                await asyncio.sleep(2 ** attempt)
                print(f"[Retry {attempt}/{MAX_RETRIES}] index={index}: {e}")


async def get_all_embeddings(texts):
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [_embed_one(semaphore, i, t) for i, t in enumerate(texts)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])
    return np.array([vec for _, vec in results], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="预计算全量数据集 embedding（只需运行一次，所有版本共享）"
    )
    parser.add_argument('--dataset', required=True, help='数据集路径 (xlsx)')
    parser.add_argument('--output-dir', default='./data', help='输出目录')
    args = parser.parse_args()

    print(f"数据集: {args.dataset}")
    df = pd.read_excel(args.dataset)
    texts = df['content'].fillna("").astype(str).tolist()
    labels = df['is_leadgen'].tolist()
    label_map = {'是': 1, '否': 0, True: 1, False: 0, 1: 1, 0: 0}
    labels = [label_map.get(v, 0) for v in labels]

    print(f"共 {len(texts)} 条文本，开始获取 Embedding...")
    embeddings = asyncio.run(get_all_embeddings(texts))

    # L2 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    embeddings_normed = embeddings / norms

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    emb_path = os.path.join(args.output_dir, 'global_embeddings.npy')
    meta_path = os.path.join(args.output_dir, 'global_embed_meta.json')

    np.save(emb_path, embeddings_normed)

    meta = {
        'embed_model': EMBEDDING_MODEL,
        'n_samples': len(texts),
        'embed_dim': int(embeddings_normed.shape[1]),
        'contents': texts,
        'labels': labels,
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Embedding 已保存: {emb_path} (shape={embeddings_normed.shape})")
    print(f"Meta 已保存: {meta_path}")


if __name__ == '__main__':
    main()
