"""RAG 检索超参网格搜索:双路召回深度 × 融合方式 × 重排候选数 → 目标 Recall@5 ≥ 95%。

做法(省 API、可复现):
  1. 固定评测集:scripts/eval_rag.load_eval_set(n=50, seed=42);
  2. 每个查询只做一次稠密召回(top DEPTH)与一次 BM25 召回(top DEPTH),缓存名次与分数;
  3. 每对 (query, 候选块) 只调一次交叉编码器重排,分数按 (qid, chunk_id) 缓存;
  4. 网格里的每个超参组合都是纯内存的列表运算,秒级完成;
  5. 输出 top 配置表 + 达标(Recall@5≥0.95)的最低成本配置,JSON 落盘 backend/eval/results/。

用法:
  uv run python scripts/tune_rag.py                     # 默认网格(智谱 rerank)
  uv run python scripts/tune_rag.py --n 50 --seed 42    # 显式规模
  uv run python scripts/tune_rag.py --rerank-provider local   # 本地 cross-encoder
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

KS = (1, 3, 5, 8, 10)
DEPTH = 100  # 双路候选缓存深度(网格里最大深度的超参不能超过它)
RC_MAX = 50  # 重排候选缓存覆盖的融合池上限(同上)

RESULTS_DIR = BACKEND_DIR / "eval" / "results"


def metrics(ranked: list[str], gold: set[str]) -> dict:
    first_hit = next((r for r, cid in enumerate(ranked, 1) if cid in gold), None)
    out = {"mrr": 1.0 / first_hit if first_hit else 0.0}
    for k in KS:
        inter = len(set(ranked[:k]) & gold)
        out[f"r@{k}"] = inter / len(gold)
        out[f"p@{k}"] = inter / k
    return out


def avg(rows: list[dict]) -> dict:
    n = len(rows)
    keys = rows[0].keys() if rows else []
    return {k: round(sum(r[k] for r in rows) / n, 4) for k in keys}


def main() -> int:
    p = argparse.ArgumentParser(description="RAG 检索超参网格搜索")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rerank-provider", default="zhipu", choices=["zhipu", "local"])
    p.add_argument("--rerank-model", default=None)
    p.add_argument("--skip-rerank", action="store_true", help="跳过重排(只调融合参数)")
    args = p.parse_args()

    from app.rag.hybrid import _dense_ranking, _sparse_ranking, build_reranker, fuse_rankings, get_corpus
    from eval_rag import build_gold_map, load_eval_set, norm_q

    eval_set = load_eval_set(args.n, args.seed)
    corpus = get_corpus()
    gold_map = build_gold_map(corpus)

    # ---------- 1. 双路候选缓存 ----------
    print(f"[cache] 稠密 top-{DEPTH} + BM25 top-{DEPTH},n={len(eval_set)} ...")
    dense_cache: dict[int, list[tuple[str, float]]] = {}
    sparse_cache: dict[int, list[tuple[str, float]]] = {}
    gold_sets: dict[int, set[str]] = {}
    for i, item in enumerate(eval_set, 1):
        qid = item["id"]
        q = item["question"].strip()
        dense_cache[qid] = _dense_ranking(q, DEPTH)
        sparse_cache[qid] = _sparse_ranking(corpus, q, DEPTH)
        gold_sets[qid] = gold_map.get(norm_q(q), set())
        print(f"  [{i:>3}/{len(eval_set)}] qid={qid} dense={len(dense_cache[qid])} sparse={len(sparse_cache[qid])}")

    # ---------- 2. 重排分数缓存 ----------
    rerank_cache: dict[tuple[int, str], float] = {}
    if not args.skip_rerank:
        reranker = build_reranker(args.rerank_provider, args.rerank_model or ("rerank" if args.rerank_provider == "zhipu" else "BAAI/bge-reranker-base"))
        print(f"[cache] 重排分数 provider={args.rerank_provider},候选≤{RC_MAX} ...")
        for i, item in enumerate(eval_set, 1):
            qid = item["id"]
            q = item["question"].strip()
            union_ids: list[str] = []
            seen = set()
            for _, pairs in ((1.0, dense_cache[qid][:RC_MAX]), (1.0, sparse_cache[qid][:RC_MAX])):
                for cid, _s in pairs:
                    if cid not in seen:
                        seen.add(cid)
                        union_ids.append(cid)
            scores = reranker.score(q, [corpus.by_id[cid].text for cid in union_ids])
            for cid, s in zip(union_ids, scores):
                rerank_cache[(qid, cid)] = s
            print(f"  [{i:>3}/{len(eval_set)}] qid={qid} reranked={len(union_ids)}")

    # ---------- 3. 网格评估(纯内存) ----------
    def rank_one(qid: int, d: int, s: int, method: str, rrf_k: int, alpha: float,
                 rc: int, use_rr: bool) -> list[str]:
        dense = dense_cache[qid][:d]
        sparse = sparse_cache[qid][:s]
        dw, bw = (alpha, 1.0 - alpha) if method == "weighted" else (1.0, 1.0)
        fused = fuse_rankings([(dw, dense), (bw, sparse)], method=method, rrf_k=rrf_k)
        if use_rr:
            cand = fused[:rc]
            scored = sorted(((cid, rerank_cache[(qid, cid)]) for cid, _ in cand),
                            key=lambda x: -x[1])
            return [cid for cid, _ in scored]
        return [cid for cid, _ in fused]

    configs = []
    depths = [(10, 10), (20, 20), (50, 50), (50, 10), (10, 50)]
    fusions = [("rrf", k, 0.5) for k in (5, 10, 30, 60)] + [
        ("weighted", 60, a) for a in (0.3, 0.4, 0.5, 0.6, 0.7)
    ]
    rr_opts = [(False, 0)] if args.skip_rerank else [(False, 0), (True, 10), (True, 20), (True, 50)]

    for (d, s), (method, rrf_k, alpha), (use_rr, rc) in itertools.product(depths, fusions, rr_opts):
        rows = [metrics(rank_one(it["id"], d, s, method, rrf_k, alpha, rc, use_rr), gold_sets[it["id"]])
                for it in eval_set]
        a = avg(rows)
        configs.append({
            "dense_topk": d, "bm25_topk": s, "fusion": method, "rrf_k": rrf_k,
            "alpha": alpha, "rerank": use_rr, "rerank_candidates": rc, **a,
        })

    # 基线参照:单路(不融合)
    for leg, name in (("dense", "dense_only"), ("bm25", "bm25_only")):
        for d in (4, 10, 20, 50):
            rows = []
            for it in eval_set:
                ranked = [cid for cid, _ in (dense_cache if leg == "dense" else sparse_cache)[it["id"]][:d]]
                rows.append(metrics(ranked, gold_sets[it["id"]]))
            configs.append({"dense_topk": d if leg == "dense" else 0,
                            "bm25_topk": d if leg == "bm25" else 0,
                            "fusion": name, "rrf_k": 0, "alpha": 0.0,
                            "rerank": False, "rerank_candidates": 0, **avg(rows)})

    configs.sort(key=lambda c: (-c["r@5"], -c["mrr"], -c["p@1"]))

    print("\n========== TOP 15(按 Recall@5) ==========")
    hdr = f"{'dense':>5} {'bm25':>5} {'fusion':>9} {'k':>3} {'alpha':>5} {'rr':>4} {'rc':>3} | {'r@1':>6} {'r@3':>6} {'r@5':>6} {'r@8':>6} {'p@1':>6} {'mrr':>6}"
    print(hdr)
    for c in configs[:15]:
        print(f"{c['dense_topk']:>5} {c['bm25_topk']:>5} {c['fusion']:>9} {c['rrf_k']:>3} "
              f"{c['alpha']:>5.2f} {str(c['rerank']):>4} {c['rerank_candidates']:>3} | "
              f"{c['r@1']:>6.3f} {c['r@3']:>6.3f} {c['r@5']:>6.3f} {c['r@8']:>6.3f} {c['p@1']:>6.3f} {c['mrr']:>6.3f}")

    ok = [c for c in configs if c["r@5"] >= 0.95]
    if ok:
        # 达标前提下选成本最低,且优先带重排(交叉编码器显著改善 top-1/MRR,
        # 也是本项目的目标架构);同权重下再比重排候选数与召回深度。
        best = min(ok, key=lambda c: (
            not c["rerank"], c["rerank_candidates"],
            c["dense_topk"] + c["bm25_topk"], -c["mrr"],
        ))
        print("\n>>> 达标配置(Recall@5 >= 0.95,共 %d 个;选成本最低者):" % len(ok))
        print(json.dumps(best, ensure_ascii=False, indent=2))
    else:
        best = configs[0]
        print("\n>>> 无配置达到 Recall@5 >= 0.95;当前最优:")
        print(json.dumps(best, ensure_ascii=False, indent=2))
        pool = []
        for it in eval_set:
            gold = gold_sets[it["id"]]
            in_pool = any(gold & {cid for cid, _ in (dense_cache[it["id"]][:50] + sparse_cache[it["id"]][:50])})
            pool.append(in_pool)
        print(f">>> 诊断:融合池(top50+top50)覆盖 gold 的查询占比 = {sum(pool) / len(pool):.2%}"
              f"(该值是重排链路的召回上限)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"tune_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "n": args.n, "seed": args.seed, "ks": KS, "depth": DEPTH,
        "rerank_provider": None if args.skip_rerank else args.rerank_provider,
        "best": best, "configs": configs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
