"""混合检索:BM25(稀疏) × 向量(稠密) 双路召回 → 融合(RRF/加权) → 交叉编码器重排。

链路:
  1. 稠密路:query → embedding-3 → Chroma 近邻(top DENSE_TOPK)
  2. 稀疏路:query → jieba 分词 → BM25Okapi 打分(top BM25_TOPK);
     BM25 索引启动时从 Chroma 全量语料内存构建(数千块毫秒级),无需持久化
  3. 融合:RRF(名次型备选)或 min-max 归一加权(默认)
  4. 重排:融合后 top RERANK_CANDIDATES 交给交叉编码器 rerank,取 top HYBRID_FINAL_K
  5. 分点检索 retrieve_multi():多个检索点各自走 1-4,原句锚定 top-K + 各点最佳
     新块追加合并(agent.py 的拆分子 agent 提供检索点)

对外接口与 LlamaIndex retriever 对齐(retrieve/aretrieve → list[NodeWithScore]),
超参数评测/调优见 scripts/eval_rag.py 与 scripts/tune_rag.py,
设计文档见 docs/rag-hybrid-design.md。
"""

import asyncio
import threading
from dataclasses import dataclass

import chromadb
import jieba
from llama_index.core.schema import NodeWithScore, TextNode

from app.config import (
    BM25_TOPK,
    BM25_WEIGHT,
    CHROMA_PATH,
    COLLECTION_NAME,
    DENSE_TOPK,
    DENSE_WEIGHT,
    FUSION,
    HYBRID_ALPHA,
    HYBRID_FINAL_K,
    RERANK_CANDIDATES,
    RERANK_MODEL,
    RERANK_PROVIDER,
    RETRIEVAL_MODE,
    RRF_K,
)
from app.rag.embeddings import build_embed_model

# 发给 rerank API 的单文档长度上限(API 硬限 4096 字,留余量)
_RERANK_TEXT_MAX_CHARS = 2000


# ---------- 语料与索引(进程级缓存) ----------


@dataclass
class CorpusChunk:
    chunk_id: str
    text: str
    metadata: dict


class Corpus:
    """从 Chroma 全量拉取语料块,并构建 BM25 索引(一次性,进程内复用)。"""

    def __init__(self, collection):
        got = collection.get(include=["documents", "metadatas"])
        self.chunks: list[CorpusChunk] = [
            CorpusChunk(chunk_id=cid, text=doc or "", metadata=meta or {})
            for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
        ]
        self.by_id: dict[str, CorpusChunk] = {c.chunk_id: c for c in self.chunks}

        jieba.initialize()  # 预加载词典,避免并发首次分词的竞态
        tokenized = [self._tokenize(c.text) for c in self.chunks]
        from rank_bm25 import BM25Okapi

        # 语料含空块时 BM25Okapi 会除零,过滤后需要保留原下标映射
        self._valid = [i for i, toks in enumerate(tokenized) if toks]
        self._bm25 = BM25Okapi([tokenized[i] for i in self._valid]) if self._valid else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in jieba.lcut_for_search(text) if t.strip()]

    def bm25_scores(self, query: str) -> list[tuple[str, float]]:
        """返回全量块的 BM25 分数列表 [(chunk_id, score)],按分数降序。"""
        if self._bm25 is None:
            return []
        toks = self._tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)  # 仅覆盖非空块
        pairs = [
            (self.chunks[self._valid[i]].chunk_id, float(scores[i]))
            for i in range(len(self._valid))
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs


# 可重入锁:get_hybrid_retriever 持锁建单例时,HybridRetriever.__init__ 内的
# get_corpus 会再次加锁,threading.Lock 会同线程自死锁,必须用 RLock
_resource_lock = threading.RLock()
_corpus: Corpus | None = None


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(COLLECTION_NAME)


def get_corpus() -> Corpus:
    global _corpus
    with _resource_lock:
        if _corpus is None:
            _corpus = Corpus(_get_collection())
        return _corpus


def reset_corpus_cache() -> None:
    """重建索引(ingest --reset)后调用,强制下次访问重新拉取语料。"""
    global _corpus
    with _resource_lock:
        _corpus = None


# ---------- 双路召回 ----------


def _dense_ranking(query: str, top_k: int) -> list[tuple[str, float]]:
    """稠密路:返回 [(chunk_id, -distance)],距离越小分数越大。"""
    if top_k <= 0:
        return []
    col = _get_collection()
    n = min(top_k, col.count())
    if n <= 0:
        return []
    embed = build_embed_model()
    vec = embed.get_query_embedding(query)
    res = col.query(query_embeddings=[vec], n_results=n, include=["distances"])
    ids = res["ids"][0]
    dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
    return [(cid, -float(d)) for cid, d in zip(ids, dists)]


def _sparse_ranking(corpus: Corpus, query: str, top_k: int) -> list[tuple[str, float]]:
    if top_k <= 0:
        return []
    # 过滤零分类:与 query 无词面重叠的块对两路融合都是噪声
    return [(cid, s) for cid, s in corpus.bm25_scores(query)[:top_k] if s > 0.0]


# ---------- 融合 ----------


def _minmax(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    if not pairs:
        return []
    vals = [s for _, s in pairs]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        # 分数无区分度:正分(挤在榜首)记 1,非正分(BM25 全零)记 0,避免污染融合
        return [(cid, 1.0 if hi > 0 else 0.0) for cid, _ in pairs]
    return [(cid, (s - lo) / (hi - lo)) for cid, s in pairs]


def fuse_rankings(
    legs: list[tuple[float, list[tuple[str, float]]]],
    method: str = "rrf",
    rrf_k: int = 10,
) -> list[tuple[str, float]]:
    """多路融合。legs = [(路权重, [(chunk_id, 原始分), ...]), ...],返回融合分降序。

    rrf:     score = Σ w_i / (rrf_k + rank_i),只用名次,两路分数不可比也稳;
    weighted:各路分数先 min-max 归一,再 Σ w_i * norm_i,对分数分布敏感。
    """
    if method == "weighted":
        merged: dict[str, float] = {}
        for w, pairs in legs:
            for cid, s in _minmax(pairs):
                merged[cid] = merged.get(cid, 0.0) + w * s
    else:  # rrf(默认)
        merged = {}
        for w, pairs in legs:
            for rank, (cid, _) in enumerate(pairs):
                merged[cid] = merged.get(cid, 0.0) + w / (rrf_k + rank + 1)
    # 并列时按最佳单路名次稳定排序,避免同分抖动
    best_rank: dict[str, int] = {}
    for _, pairs in legs:
        for rank, (cid, _) in enumerate(pairs):
            best_rank[cid] = min(best_rank.get(cid, rank + 1), rank + 1)
    return sorted(merged.items(), key=lambda kv: (-kv[1], best_rank[kv[0]]))


# ---------- 交叉编码器重排 ----------


class ZhipuReranker:
    """智谱 rerank API(交叉编码器):POST /paas/v4/rerank。

    单请求 ≤128 条文档、每条 ≤4096 字;按 64 条一批切分稳妥。
    """

    def __init__(self, model: str = "rerank", batch_size: int = 64):
        import os

        from app.config import ZHIPU_API_KEY, ZHIPU_BASE_URL

        import httpx

        if not ZHIPU_API_KEY:
            raise RuntimeError("ZHIPU_API_KEY 未配置,无法使用 zhipu 重排")
        self._model = model
        self._batch = batch_size
        self._url = ZHIPU_BASE_URL.rstrip("/") + "/rerank"
        self._headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}"}
        self._client = httpx.Client(timeout=60)

    def score(self, query: str, texts: list[str]) -> list[float]:
        texts = [t[:_RERANK_TEXT_MAX_CHARS] for t in texts]
        out: dict[int, float] = {}
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            resp = self._client.post(
                self._url,
                headers=self._headers,
                json={
                    "model": self._model,
                    "query": query[:4096],
                    "documents": batch,
                    "top_n": len(batch),
                    "return_documents": False,
                },
            )
            resp.raise_for_status()
            for item in resp.json().get("results", []):
                out[start + item["index"]] = float(item["relevance_score"])
        return [out.get(i, -1e9) for i in range(len(texts))]


class LocalCrossEncoderReranker:
    """本地交叉编码器(可选 extra:uv sync --extra rerank-local)。

    默认 BAAI/bge-reranker-base(中英双语),首次运行经 HF 下载 ONNX 权重;
    国内网络可置 HF_ENDPOINT=https://hf-mirror.com 加速。
    """

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        texts = [t[:_RERANK_TEXT_MAX_CHARS] for t in texts]
        return [float(s) for s in self._model.rerank(query, texts)]


def build_reranker(provider: str, model: str):
    if provider == "zhipu":
        return ZhipuReranker(model=model)
    if provider == "local":
        return LocalCrossEncoderReranker(model=model)
    return None


# ---------- 混合检索器 ----------


class HybridRetriever:
    """参数显式传入(评测/调优脚本按网格覆盖);线上经 get_hybrid_retriever() 单例使用。"""

    def __init__(
        self,
        dense_topk: int = DENSE_TOPK,
        bm25_topk: int = BM25_TOPK,
        dense_weight: float = DENSE_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        fusion: str = FUSION,
        rrf_k: int = RRF_K,
        alpha: float = HYBRID_ALPHA,
        rerank_provider: str = RERANK_PROVIDER,
        rerank_model: str = RERANK_MODEL,
        rerank_candidates: int = RERANK_CANDIDATES,
        final_k: int = HYBRID_FINAL_K,
    ):
        self.dense_topk = dense_topk
        self.bm25_topk = bm25_topk
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.alpha = alpha  # weighted 融合时稠密路权重(替代 dense_weight)
        self.rerank_candidates = rerank_candidates
        self.final_k = final_k
        self._corpus = get_corpus()
        self._reranker = build_reranker(rerank_provider, rerank_model) if rerank_provider != "off" else None

    def _fuse_legs(self, query: str) -> list[tuple[str, float]]:
        """单查询双路召回 + 融合,返回融合分降序的完整名次。"""
        dense = _dense_ranking(query, self.dense_topk)
        sparse = _sparse_ranking(self._corpus, query, self.bm25_topk)
        if not dense and not sparse:
            return []
        if self.fusion == "weighted":
            dw, bw = self.alpha, 1.0 - self.alpha
        else:
            dw, bw = self.dense_weight, self.bm25_weight
        return fuse_rankings([(dw, dense), (bw, sparse)], method=self.fusion, rrf_k=self.rrf_k)

    def _finalize(self, anchor_query: str, cand: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """以 anchor_query 为锚做交叉编码器重排并截取 final_k;无重排器时保持融合序。"""
        if not cand:
            return []
        cand_ids = [cid for cid, _ in cand]
        if self._reranker is not None:
            texts = [self._corpus.by_id[cid].text for cid in cand_ids]
            scores = self._reranker.score(anchor_query, texts)
            order = sorted(range(len(cand_ids)), key=lambda i: -scores[i])
            return [(cand_ids[i], scores[i]) for i in order[: self.final_k]]
        score_map = dict(cand)
        return [(cid, score_map[cid]) for cid in cand_ids[: self.final_k]]

    def _to_nodes(self, scored: list[tuple[str, float]]) -> list[NodeWithScore]:
        return [
            NodeWithScore(
                node=TextNode(
                    id_=cid,
                    text=self._corpus.by_id[cid].text,
                    metadata=self._corpus.by_id[cid].metadata,
                ),
                score=float(score),
            )
            for cid, score in scored
        ]

    def retrieve(self, query: str) -> list[NodeWithScore]:
        fused = self._fuse_legs(query)
        cap = self.rerank_candidates if self._reranker is not None else self.final_k
        return self._to_nodes(self._finalize(query, fused[:cap]))

    def retrieve_multi(self, queries: list[str]) -> list[NodeWithScore]:
        """分点检索:queries[0] 为原始问题,其余为拆分出的检索点。

        合并策略(实测三种方案后的取舍,见 docs/rag-hybrid-design.md §8):
          1. 前 final_k 名**完全由原始问题锚定**(与 retrieve() 同序)——语料
             gold 匹配依赖与原句的逐字重合,任何给拆分点让位的重排都会把
             单意图问题的 gold 挤出头部(实测 Recall@5 0.96→0.94);
          2. 每个检索点以自己为锚独立重排,其最佳"新"块(原句 top final_k
             之外的)**追加**到结果尾部进入 LLM 上下文——子意图覆盖靠上下文
             变宽实现,总数 ≤ final_k + 拆分点数。
        """
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            return []
        base = self._finalize(queries[0], self._fuse_legs(queries[0]))
        # 单意图(无拆分点)直接等价 retrieve()
        if len(queries) == 1:
            return self._to_nodes(base)
        base_ids = [cid for cid, _ in base]
        extra: list[tuple[str, float]] = []
        seen = set(base_ids)
        for q in queries[1:]:
            fused = self._fuse_legs(q)[: self.rerank_candidates]
            if not fused:
                continue
            cand_ids = [cid for cid, _ in fused]
            if self._reranker is not None:
                texts = [self._corpus.by_id[cid].text for cid in cand_ids]
                scores = self._reranker.score(q, texts)
            else:
                scores = [s for _, s in fused]
            for i in sorted(range(len(cand_ids)), key=lambda j: -scores[j]):
                cid = cand_ids[i]
                if cid not in seen:  # 该检索点的最佳新块,追加进上下文
                    seen.add(cid)
                    extra.append((cid, scores[i]))
                    break
        return self._to_nodes((base + extra)[: self.final_k + len(queries) - 1])

    async def aretrieve(self, query: str) -> list[NodeWithScore]:
        return await asyncio.to_thread(self.retrieve, query)

    async def aretrieve_multi(self, queries: list[str]) -> list[NodeWithScore]:
        return await asyncio.to_thread(self.retrieve_multi, queries)


_retriever: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    """进程级单例:BM25 索引/重排客户端只建一次,供 API 每请求复用。"""
    global _retriever
    with _resource_lock:
        if _retriever is None:
            _retriever = HybridRetriever()
        return _retriever


def mode_enabled() -> bool:
    return RETRIEVAL_MODE == "hybrid"
