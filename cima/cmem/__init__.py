# -*- coding: utf-8 -*-
"""
CMem (Causal Memory) - COAT 因子发现模块

通过 COAT 框架，
在 Seen Set 上迭代发现与推广判定因果相关的不变因子。

流程：
1. 采样高熵子集
2. LLM 提出候选因子
3. LLM 标注因子取值
4. GSQ 条件独立性检验筛选显著因子
5. FCI 因果发现确定 Markov Blanket
"""
