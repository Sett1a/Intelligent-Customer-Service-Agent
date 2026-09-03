"""复合问题压力测试:验证「分点检索」对多意图查询的价值。

做法:把 50 题评测集的问题两两拼成复合问题(gold = 两题 gold 的并集),
对比两种检索方式的子意图覆盖:
  single    整句一次检索(HybridRetriever.retrieve)
  decompose LLM 拆分 + 逐点召回 + 跨查询融合(HybridRetriever.retrieve_multi)

用法:
  uv run python scripts/eval_compound.py            # 默认 20 个复合问题,seed=42
  uv run python scripts/eval_compound.py --n 30
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "scripts"))

RESULTS_DIR = BACKEND_DIR / "eval" / "results"
KS = (3, 5, 8)


def avg(rows: list[dict]) -> dict:
    metric_keys = [k for k in (rows[0] if rows else {}) if k in ("cov", "both", "mrr")]
    n = len(rows)
    return {k: round(sum(r[k] for r in rows) / n, 4) for k in metric_keys} if rows else {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="复合问题数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    from app.agent import decompose_queries
    from app.rag.hybrid import HybridRetriever, get_corpus
    from app.rag.retriever import get_retriever
    from eval_rag import build_gold_map, load_eval_set, norm_q

    # 拆分点取材池放宽到 150 题,保证信息型问题两两配对能凑够 --n 对
    eval_set = load_eval_set(150, args.seed)
    corpus = get_corpus()
    gold_map = build_gold_map(corpus)

    # 两两拼对:gold 不同才拼(同 gold 无意义),固定种子保证可复现。
    # 只用"信息型问题"拼对:寒暄/确认类语句("好的 谢谢")拆分器本就该返回空,
    # 混进来只会稀释对比(实验发现)。
    INTENT_RE = re.compile(r"什么|怎么|如何|为什么|多久|多少|哪|谁|吗|能不能|可以|是否|怎样")
    items = [
        it for it in eval_set
        if gold_map.get(norm_q(it["question"].strip()))
        and len(norm_q(it["question"])) >= 6
        and INTENT_RE.search(it["question"])
    ]
    print(f"[compound] 信息型问题池: {len(items)}")
    rng = random.Random(args.seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    compounds = []
    used: set[int] = set()
    for a, b in zip(idx[::2], idx[1::2]):
        if len(compounds) >= args.n:
            break
        ga = gold_map[norm_q(items[a]["question"].strip())]
        gb = gold_map[norm_q(items[b]["question"].strip())]
        if ga == gb:
            continue
        used |= {a, b}
        compounds.append({
            "question": f"{items[a]['question'].strip()}；另外，{items[b]['question'].strip()}",
            "gold": sorted(ga | gb),
            "parts": [items[a]["id"], items[b]["id"]],
        })
    print(f"[compound] n={len(compounds)} seed={args.seed}")

    retriever = get_retriever()
    # 同上下文预算对照:单句检索直接放大 final_k 到 8(分点模式为 5 + 拆分点数 ≤ 8)
    deep_single = HybridRetriever(final_k=8)

    def metrics(ranked: list[str], gold: list[str]) -> dict:
        """覆盖率按返回给 LLM 的完整上下文计(位置不影响模型阅读)。"""
        gold_set = set(gold)
        first = next((r for r, c in enumerate(ranked, 1) if c in gold_set), None)
        return {
            "cov": len(gold_set & set(ranked)) / len(gold_set),
            "both": 1.0 if gold_set <= set(ranked) else 0.0,
            "mrr": 1.0 / first if first else 0.0,
        }

    per_mode: dict[str, list[dict]] = {"single@5": [], "single@8": [], "decompose": []}

    def with_retry(fn, *a, **kw):  # embedding/LLM 偶发抖动退避重试
        for attempt in range(3):
            try:
                return fn(*a, **kw)
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"    [retry {attempt + 1}] {type(e).__name__}", flush=True)
                time.sleep(2 * (attempt + 1))

    for i, c in enumerate(compounds, 1):
        row = {"question": c["question"], "gold": c["gold"], "parts": c["parts"]}
        s5 = [n.node.id_ for n in with_retry(retriever.retrieve, c["question"])]
        per_mode["single@5"].append({**row, **metrics(s5, c["gold"])})
        s8 = [n.node.id_ for n in with_retry(deep_single.retrieve, c["question"])]
        per_mode["single@8"].append({**row, **metrics(s8, c["gold"])})
        points = with_retry(asyncio.run, decompose_queries(c["question"]))
        multi = [
            n.node.id_
            for n in with_retry(retriever.retrieve_multi, [c["question"], *points])
        ]
        per_mode["decompose"].append({
            **row, "sub_queries": points, "ctx_len": len(multi), **metrics(multi, c["gold"]),
        })
        print(f"  [{i:>3}/{len(compounds)}] points={len(points)} ctx={len(multi)}", flush=True)

    print("\n====== 复合问题(多意图)检索对比(上下文覆盖率) ======")
    for mode, rows in per_mode.items():
        a = avg(rows)
        ctx = sum(r.get("ctx_len", 5) for r in rows) / len(rows)
        print(f"{mode:>10}: cov={a['cov']:.3f} both={a['both']:.3f} mrr={a['mrr']:.3f} ctx≈{ctx:.1f}")
    win = sum(
        1 for s, d in zip(per_mode["single@8"], per_mode["decompose"]) if d["cov"] > s["cov"]
    )
    lose = sum(
        1 for s, d in zip(per_mode["single@8"], per_mode["decompose"]) if d["cov"] < s["cov"]
    )
    print(f"\nper-question 覆盖率: decompose 胜 {win} / 平 {len(compounds) - win - lose} / 负 {lose}(对照 single@8)")

    if args.json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"compound_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(
            {"n": len(compounds), "seed": args.seed, "modes": per_mode},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")
        print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
