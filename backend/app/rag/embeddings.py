"""Embedding 模型:智谱 embedding-3(OpenAI 兼容端点)与本地伪向量(干跑用)。"""

import asyncio
import hashlib
import math
import re
from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI
from pydantic import PrivateAttr


class ZhipuEmbedding(BaseEmbedding):
    """通过智谱 OpenAI 兼容端点调用 embedding-3。

    文档:https://docs.bigmodel.cn/cn/guide/platform/model-migration
    """

    _client: Any = PrivateAttr()
    _dim: int = PrivateAttr()

    def __init__(self, api_key: str, base_url: str, model_name: str = "embedding-3", dim: int = 1024, **kwargs):
        kwargs.setdefault("embed_batch_size", 32)  # 智谱 embedding 单请求批量稳妥值
        super().__init__(model_name=model_name, **kwargs)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._dim = dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=self.model_name, input=texts, dimensions=self._dim
        )
        return [d.embedding for d in resp.data]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)


class FakeEmbedding(BaseEmbedding):
    """确定性词袋伪向量,仅用于无 API Key 时干跑整条链路。"""

    _dim: int = PrivateAttr()

    def __init__(self, dim: int = 1024, **kwargs):
        super().__init__(model_name="fake-bow", **kwargs)
        self._dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        # 单汉字 + ASCII 词,保证中文语义有区分度
        for tok in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed_one(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_one(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed_one(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed_one(text)


def build_embed_model(fake: bool = False):
    """按配置构建 embedding 模型;fake=True 或 EMBED_FAKE=1 时用伪向量。"""
    from app.config import (
        EMBEDDING_DIM,
        EMBEDDING_MODEL,
        ZHIPU_API_KEY,
        ZHIPU_BASE_URL,
        EMBED_FAKE,
    )

    if fake or EMBED_FAKE:
        return FakeEmbedding(dim=EMBEDDING_DIM)
    if not ZHIPU_API_KEY:
        raise RuntimeError("ZHIPU_API_KEY 未配置:请在 backend/.env 填入,或置 EMBED_FAKE=1 干跑")
    return ZhipuEmbedding(
        api_key=ZHIPU_API_KEY,
        base_url=ZHIPU_BASE_URL,
        model_name=EMBEDDING_MODEL,
        dim=EMBEDDING_DIM,
    )
