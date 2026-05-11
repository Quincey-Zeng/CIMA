# -*- coding: utf-8 -*-
"""
FMem 相似度检索器：基于预计算 embedding，从 Seen Set 中检索最相似的正/负样本。

用于 FMem-only 和 CMem+FMem 推理模式，提供感知锚定的 few-shot 示例。
"""

import asyncio
import json
import os

import numpy as np
from openai import OpenAI

from ..config import (
    API_KEY, API_BASE_URL, EMBEDDING_MODEL,
    FMEM_N_POS, FMEM_N_NEG, FMEM_MAX_INPUT_CHARS,
)

MAX_CONCURRENCY = 8
MAX_RETRIES = 8


class SimilarityRetriever:
    """加载预计算 embedding，按 Seen Set 过滤后支持批量 query 检索。"""

    def __init__(self, embeddings_path, meta_path, seen_contents):
        """
        Args:
            embeddings_path: 全量 embedding .npy 文件路径
            meta_path: 全量 meta .json 文件路径
            seen_contents: Seen Set 的 content 列表，用于从全量中过滤出 seen 子集
        """
        all_embeddings = np.load(embeddings_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        all_contents = meta['contents']
        all_labels = meta['labels']

        # 建立 content -> global index 映射
        content_to_idx = {}
        for i, c in enumerate(all_contents):
            content_to_idx[c] = i

        # 过滤出 seen set 对应的子集
        seen_indices = [content_to_idx[c] for c in seen_contents if c in content_to_idx]

        self.embeddings = all_embeddings[seen_indices]
        self.texts = [all_contents[i] for i in seen_indices]
        self.labels = np.array([all_labels[i] for i in seen_indices], dtype=int)
        self.pos_mask = self.labels == 1
        self.neg_mask = self.labels == 0

        # 保存全量数据用于 query embedding 查找
        self._all_embeddings = all_embeddings
        self._all_content_to_idx = content_to_idx

        self._client = None
        self._query_embeddings = None

    def _get_client(self):
        if self._client is None:
            self._client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        return self._client

    async def _embed_one(self, semaphore, index, text):
        async with semaphore:
            client = self._get_client()
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
                    backoff = min(2 ** attempt, 30)
                    await asyncio.sleep(backoff)

    async def precompute_queries(self, query_texts):
        """批量预计算 query 文本的 embedding。

        优先从全量 embedding 中查找，找不到的才调 API 计算。
        """
        n = len(query_texts)
        dim = self.embeddings.shape[1]
        result_vecs = np.zeros((n, dim), dtype=np.float32)
        need_embed_indices = []
        need_embed_texts = []

        for i, text in enumerate(query_texts):
            global_idx = self._all_content_to_idx.get(text)
            if global_idx is not None:
                result_vecs[i] = self._all_embeddings[global_idx]
            else:
                need_embed_indices.append(i)
                need_embed_texts.append(text)

        if need_embed_texts:
            semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
            tasks = [self._embed_one(semaphore, i, t) for i, t in enumerate(need_embed_texts)]
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda x: x[0])
            for local_i, (_, vec) in enumerate(results):
                vec_arr = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(vec_arr)
                if norm > 0:
                    vec_arr /= norm
                result_vecs[need_embed_indices[local_i]] = vec_arr

        self._query_embeddings = result_vecs

    def retrieve_by_index(self, query_idx, n_pos=None, n_neg=None):
        """根据预计算的 query embedding index 检索最相似的正负样本。

        Returns:
            list of (text, label) tuples，与 prompt builder 的 examples 格式兼容。
        """
        if n_pos is None:
            n_pos = FMEM_N_POS
        if n_neg is None:
            n_neg = FMEM_N_NEG

        q_vec = self._query_embeddings[query_idx]
        sims = self.embeddings @ q_vec

        # 正样本：取相似度最高的 n_pos 个
        pos_sims = np.where(self.pos_mask, sims, -2.0)
        pos_top_idx = np.argsort(pos_sims)[::-1][:n_pos]

        # 负样本：取相似度最高的 n_neg 个
        neg_sims = np.where(self.neg_mask, sims, -2.0)
        neg_top_idx = np.argsort(neg_sims)[::-1][:n_neg]

        # 交替排列正负样本
        examples = []
        pos_list = [(self.texts[i], True) for i in pos_top_idx]
        neg_list = [(self.texts[i], False) for i in neg_top_idx]
        for p, n in zip(pos_list, neg_list):
            examples.append(p)
            examples.append(n)
        remaining_pos = pos_list[len(neg_list):]
        remaining_neg = neg_list[len(pos_list):]
        examples.extend(remaining_pos)
        examples.extend(remaining_neg)
        return examples
