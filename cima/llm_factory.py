# -*- coding: utf-8 -*-
"""
LLM 工厂：统一管理 LLM 客户端构建。

支持 OpenAI-compatible API (Qwen, DeepSeek 等) 和 Anthropic API。
通过环境变量配置 API 密钥和端点。
"""

import os
from typing import List, Tuple, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import API_KEY, API_BASE_URL, INFERENCE_MODEL, PROPOSE_MODEL, ANNOTATION_MODEL


def build_llm(model: str, api_key: str = None, base_url: str = None,
              temperature: float = 0.0, max_tokens: int = 2048, **kwargs):
    """构建 LLM 实例。"""
    return ChatOpenAI(
        model=model,
        api_key=api_key or API_KEY,
        base_url=base_url or API_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def build_chain(model: str, api_key: str = None, base_url: str = None,
                callbacks=None, **kwargs):
    """构建 LCEL chain: ChatPromptTemplate | LLM | StrOutputParser。"""
    llm = build_llm(model, api_key, base_url, **kwargs)
    if callbacks:
        llm = llm.with_config({"callbacks": callbacks})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_instruct}"),
        ("human", "{content}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain


def build_inference_chain(callbacks=None, **kwargs):
    """构建推理用 chain。"""
    return build_chain(INFERENCE_MODEL, callbacks=callbacks, **kwargs)


def build_propose_chain(callbacks=None, **kwargs):
    """构建因子提取用 chain。"""
    return build_chain(PROPOSE_MODEL, callbacks=callbacks, **kwargs)


def build_annotation_chain(callbacks=None, **kwargs):
    """构建因子标注用 chain。"""
    return build_chain(ANNOTATION_MODEL, callbacks=callbacks, **kwargs)
