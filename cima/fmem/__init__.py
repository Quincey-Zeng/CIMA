# -*- coding: utf-8 -*-
"""
FMem (Fact Memory) - 基于相似度检索的 In-Context Learning 模块

通过 embedding 相似度从 Seen Set 中检索最相似的正/负样本，
作为 few-shot 示例注入 prompt，提供感知锚定。
"""
