"""下载并清洗 JDDC 京东客服语料,抽取 QA 对构建轻量知识库。

数据源:JDDC 官方基线仓库 https://github.com/SimonJYang/JDDC-Baseline-Seq2Seq
        data/chat.txt(京东客服真实会话,官方脱敏子集,约 1 万组会话,20MB,TSV 格式)
用途限制:该语料仅用于学习/研究演示,不得商用。

用法:uv run python scripts/prepare_dataset.py
输出:data/raw/kb.jsonl,每行 {"id", "question", "answer", "source", "session_id"}
"""

import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import httpx

RAW_URL = (
    "https://raw.githubusercontent.com/SimonJYang/JDDC-Baseline-Seq2Seq/master/data/chat.txt"
)
BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BACKEND_DIR / "data" / "raw" / "jddc_chat.txt"
OUT_PATH = BACKEND_DIR / "data" / "raw" / "kb.jsonl"

MAX_PAIRS = 2500  # 轻量上限
MIN_Q_LEN = 4     # 问题最少字符数
MIN_A_LEN = 8     # 回答最少字符数
PLACEHOLDER = re.compile(r"\[[^\]]{1,6}x\]")  # [数字x] [姓名x] 等脱敏占位符
ARTIFACT = re.compile(r"#E-\S{0,8}")  # 京东语料的表情/分隔符残留,如 #E-s[数字x]
# 纯寒暄/无信息量回答(按去空格精确匹配)
TRIVIAL_ANSWERS = {"好的", "嗯嗯", "是的", "对", "对的", "好的呢", "ok", "OK", "亲", "在的", "您好"}


def download() -> Path:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 10_000_000:
        print(f"[skip] 已存在 {RAW_PATH} ({RAW_PATH.stat().st_size / 1e6:.1f} MB)")
        return RAW_PATH
    print(f"[down] {RAW_URL}")
    with httpx.stream("GET", RAW_URL, timeout=120, follow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with RAW_PATH.open("wb") as f:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\n[ok]   已保存 {RAW_PATH}")
    return RAW_PATH


def load_sessions(path: Path) -> dict[str, list[dict]]:
    """TSV → {session_id: [按序 turn]},turn = {is_user, content}。

    注意:该文件表头错位——索引 5 是空的匿名列,实际文本在索引 6 起
    (content 内含 tab 时会占多列),因此文本统一从 row[6:] 取。
    """
    sessions: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: session_id user_id waiter_send is_transfer is_repeat content ''
        for row in reader:
            if len(row) < 6:
                continue
            content = ("\t".join(row[6:]) if len(row) > 6 else row[5]).strip()
            if not content:
                continue
            if row[4].strip() == "1":
                continue  # 用户重发消息,跳过
            sessions[row[0].strip()].append(
                {
                    "is_user": row[2].strip() == "0",
                    "content": content,
                }
            )
    return sessions


def extract_pairs(sessions: dict[str, list[dict]]):
    """状态机抽取 (question, answer, session_id):连续用户消息合并为问,连续客服消息合并为答。"""
    pairs = []
    for sid, turns in sessions.items():
        cur_q: list[str] = []
        cur_a: list[str] = []
        for t in turns:
            if t["is_user"]:
                if cur_a:
                    pairs.append((" ".join(cur_q), "\n".join(cur_a), sid))
                    cur_q, cur_a = [], []
                cur_q.append(t["content"])
            else:
                if cur_q:
                    cur_a.append(t["content"])
        if cur_q and cur_a:
            pairs.append((" ".join(cur_q), "\n".join(cur_a), sid))
    return pairs


def is_trivial(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return compact in TRIVIAL_ANSWERS or len(PLACEHOLDER.findall(compact)) > 2


def clean(pairs):
    seen_q: dict[str, tuple[str, str]] = {}
    for q, a, sid in pairs:
        q = ARTIFACT.sub("", q).strip()
        a = ARTIFACT.sub("", a).strip()
        if len(q) < MIN_Q_LEN or len(a) < MIN_A_LEN:
            continue
        if is_trivial(q) or is_trivial(a):
            continue
        if q == a:
            continue
        # 同一问题保留信息量最大的回答
        if q not in seen_q or len(a) > len(seen_q[q][0]):
            seen_q[q] = (a, sid)
    # 轻量化:按回答信息量取前 MAX_PAIRS
    ranked = sorted(seen_q.items(), key=lambda kv: len(kv[1][0]), reverse=True)
    return ranked[:MAX_PAIRS]


def main() -> None:
    path = download()
    print("[parse] 解析 TSV ...")
    sessions = load_sessions(path)
    n_turns = sum(len(v) for v in sessions.values())
    print(f"  {len(sessions)} 组会话,{n_turns} 条消息")
    pairs = extract_pairs(sessions)
    print(f"[clean] 原始问答对 {len(pairs)},清洗后 ...")
    final = clean(pairs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, (q, (a, sid)) in enumerate(final):
            f.write(
                json.dumps(
                    {
                        "id": i,
                        "question": q,
                        "answer": a,
                        "source": f"JDDC:{sid[:12]}",
                        "session_id": sid,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[done] {OUT_PATH} 共 {len(final)} 条 QA 对")


if __name__ == "__main__":
    sys.exit(main())
