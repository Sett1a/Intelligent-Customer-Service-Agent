"""检索评测(摸底/回归):从 kb.jsonl 固定种子抽样 N 题,计算召回率/准确率/MRR。

用法:
  uv run python scripts/eval_rag.py                       # 按 .env 当前配置评测
  uv run python scripts/eval_rag.py --mode dense --topk 4 # 纯向量基线(改造前现状)
  uv run python scripts/eval_rag.py --tag baseline_dense --json 落盘
  覆盖超参:--dense-topk --bm25-topk --fusion --rrf-k --alpha
           --rerank-provider --rerank-candidates --final-k --no-rerank

指标定义(gold = 知识库中"问题文本相同"的全部块,绝大多数问题对应 1 块):
  Recall@k    召回率 = gold 与 top-k 有交集的查询占比(单 gold 块时即命中率)
  Precision@k 准确率 = top-k 中 gold 块的平均占比
  MRR         首个 gold 块名次倒数的均值
评测集与调参脚本共用(tune_rag.py 引用本模块的 load_eval_set/build_gold_map)。
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

KB_PATH = BACKEND_DIR / "data" / "raw" / "kb.jsonl"
RESULTS_DIR = BACKEND_DIR / "eval" / "results"
DEFAULT_KS = (1, 3, 5, 10)


def norm_q(q: str) -> str:
    """问题文本归一(去空白、转小写),用于去重与 gold 匹配。"""
    return re.sub(r"\s+", "", q or "").lower()


def load_kb_items() -> list[dict]:
    if not KB_PATH.exists():
        sys.exit(f"知识库源文件不存在:{KB_PATH}(先运行 scripts/prepare_dataset.py)")
    return [json.loads(line) for line in KB_PATH.open(encoding="utf-8")]


def load_eval_set(n: int, seed: int) -> list[dict]:
    """固定种子抽 n 个去重后的问题(过短问题剔除),保证评测集可复现。"""
    seen: set[str] = set()
    pool: list[dict] = []
    for item in load_kb_items():
        q = (item.get("question") or "").strip()
        key = norm_q(q)
        if len(key) < 4 or key in seen:
            continue
        seen.add(key)
        pool.append(item)
    rng = random.Random(seed)
    idx = list(range(len(pool)))
    rng.shuffle(idx)
    return [pool[i] for i in idx[:n]]


def build_gold_map(corpus) -> dict[str, set[str]]:
    """归一问题文本 → 含该问题的全部 chunk id(ingest 时元数据随块继承)。"""
    gold: dict[str, set[str]] = {}
    for c in corpus.chunks:
        q = norm_q(c.metadata.get("question", ""))
        if q:
            gold.setdefault(q, set()).add(c.chunk_id)
    return gold


def build_retriever(args, cfg):
    """按解析后的 cfg 构造检索器(cfg 缺省值已回落到 config.py/.env)。"""
    from app.rag.hybrid import HybridRetriever

    if cfg["mode"] == "dense":
        return HybridRetriever(
            dense_topk=cfg["topk"],
            bm25_topk=0,
            rerank_provider="off",
            final_k=cfg["topk"],
        )
    return HybridRetriever(
        dense_topk=cfg["dense_topk"],
        bm25_topk=cfg["bm25_topk"],
        fusion=cfg["fusion"],
        rrf_k=cfg["rrf_k"],
        alpha=cfg["alpha"],
        rerank_provider=cfg["rerank_provider"],
        rerank_candidates=cfg["rerank_candidates"],
        final_k=cfg["final_k"],
    )


def run_eval(retrieve_fn, eval_set, gold_map, ks) -> dict:
    """retrieve_fn(query) -> (ranked_chunk_ids, extras dict),extras 并入 per_query 行。"""
    agg = {k: {"recall": 0.0, "precision": 0.0} for k in ks}
    mrr_sum = 0.0
    misses: list[dict] = []
    per_query: list[dict] = []
    skipped = 0

    for i, item in enumerate(eval_set, 1):
        q = item["question"].strip()
        gold = gold_map.get(norm_q(q), set())
        if not gold:  # 语料中无对应块(理论不发生),剔除并计数
            skipped += 1
            continue
        for attempt in range(3):
            try:
                ranked, extra = retrieve_fn(q)
                break
            except Exception as e:  # noqa: BLE001 - 限流/网络抖动退避重试
                if attempt == 2:
                    raise
                print(f"  [retry {attempt + 1}] qid={item.get('id')} {type(e).__name__}: {e}")
                time.sleep(2 * (attempt + 1))
        first_hit = next((r for r, cid in enumerate(ranked, 1) if cid in gold), None)
        row = {"qid": item.get("id"), "question": q, "gold": sorted(gold),
               "first_hit_rank": first_hit, "top": ranked[: max(ks)], **extra}
        per_query.append(row)
        mrr_sum += 1.0 / first_hit if first_hit else 0.0
        if first_hit is None:
            misses.append({"qid": item.get("id"), "question": q, "gold": sorted(gold)})
        for k in ks:
            inter = len(set(ranked[:k]) & gold)
            agg[k]["recall"] += inter / len(gold)
            agg[k]["precision"] += inter / k
        print(f"  [{i:>3}/{len(eval_set)}] rank={first_hit or '-'} qid={item.get('id')}")

    n = len(per_query)
    summary = {
        "n": n,
        "skipped_no_gold": skipped,
        "mrr": round(mrr_sum / n, 4) if n else 0.0,
        "metrics": {
            str(k): {
                "recall": round(v["recall"] / n, 4) if n else 0.0,
                "precision": round(v["precision"] / n, 4) if n else 0.0,
            }
            for k, v in agg.items()
        },
    }
    return {"summary": summary, "per_query": per_query, "misses": misses}


def main() -> int:
    p = argparse.ArgumentParser(description="RAG 检索评测(召回率/准确率/MRR)")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default=None, help="结果文件名标识")
    p.add_argument("--json", action="store_true", help="落盘 JSON 到 backend/eval/results/")
    p.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    # 模式与超参覆盖(缺省读 .env / config.py)
    p.add_argument("--mode", choices=["dense", "hybrid"], default=None)
    p.add_argument("--topk", type=int, default=4, help="dense 模式的 top-k")
    p.add_argument("--dense-topk", type=int, default=None)
    p.add_argument("--bm25-topk", type=int, default=None)
    p.add_argument("--fusion", choices=["rrf", "weighted"], default=None)
    p.add_argument("--rrf-k", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--rerank-provider", default=None)
    p.add_argument("--rerank-candidates", type=int, default=None)
    p.add_argument("--final-k", type=int, default=None)
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--decompose", action="store_true",
                   help="分点检索:LLM 拆分子 agent 先拆查询,逐点召回后跨查询融合")
    args = p.parse_args()

    from app.config import (  # 延迟导入,便于 --help 快速响应
        BM25_TOPK,
        DENSE_TOPK,
        FUSION,
        HYBRID_ALPHA,
        HYBRID_FINAL_K,
        RERANK_CANDIDATES,
        RERANK_PROVIDER,
        RETRIEVAL_MODE,
        RRF_K,
    )

    cfg = {
        "mode": args.mode or RETRIEVAL_MODE,
        "decompose": args.decompose,
        "dense_topk": args.dense_topk or DENSE_TOPK,
        "bm25_topk": args.bm25_topk or BM25_TOPK,
        "fusion": args.fusion or FUSION,
        "rrf_k": args.rrf_k or RRF_K,
        "alpha": args.alpha if args.alpha is not None else HYBRID_ALPHA,
        "rerank_provider": args.rerank_provider or RERANK_PROVIDER,
        "rerank_candidates": args.rerank_candidates or RERANK_CANDIDATES,
        "final_k": args.final_k or HYBRID_FINAL_K,
    }
    if args.mode == "dense":
        cfg = {"mode": "dense", "topk": args.topk}
    elif args.no_rerank:
        cfg["rerank_provider"] = "off"
    args.mode = args.mode or cfg["mode"]

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    eval_set = load_eval_set(args.n, args.seed)
    print(f"[eval] n={len(eval_set)} seed={args.seed} ks={ks} tag={args.tag or '-'}")
    print(f"[config] {json.dumps(cfg, ensure_ascii=False)}")

    from app.rag.hybrid import get_corpus

    corpus = get_corpus()
    gold_map = build_gold_map(corpus)
    retriever = build_retriever(args, cfg)

    if args.decompose:
        import asyncio

        from app.agent import decompose_queries

        def retrieve_fn(q: str):
            points = asyncio.run(decompose_queries(q))
            nodes = retriever.retrieve_multi([q, *points])
            return [n.node.id_ for n in nodes], {"sub_queries": points}
    else:
        def retrieve_fn(q: str):
            return [n.node.id_ for n in retriever.retrieve(q)], {}

    result = run_eval(retrieve_fn, eval_set, gold_map, ks)

    s = result["summary"]
    print("\n========== 摘要 ==========")
    print(f"查询数={s['n']}  跳过(无gold)={s['skipped_no_gold']}  MRR={s['mrr']}")
    print(f"{'k':>4} {'Recall@k':>10} {'Precision@k':>12}")
    for k in ks:
        m = s["metrics"][str(k)]
        print(f"{k:>4} {m['recall']:>10.4f} {m['precision']:>12.4f}")
    if result["misses"]:
        print(f"未命中 {len(result['misses'])} 题, qid: {[m['qid'] for m in result['misses']]}")

    if args.json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
        out = RESULTS_DIR / f"eval_{tag}.json"
        out.write_text(
            json.dumps({"config": cfg, "seed": args.seed, **result}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
