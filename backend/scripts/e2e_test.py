"""端到端测试:RAG/订单工具/上下文连续/转人工/持久化(需后端已启动)。"""

import json
import uuid

import httpx

BASE = "http://127.0.0.1:8000/api"


def chat(payload):
    deltas, text, citations, done, errors = 0, "", 0, None, []
    with httpx.stream("POST", f"{BASE}/chat", json=payload, timeout=180) as r:
        ev = "message"
        for line in r.iter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:") and line[5:].strip():
                p = json.loads(line[5:].strip())
                if ev == "delta":
                    deltas += 1
                    text += p.get("text", "")
                elif ev == "citations":
                    citations = len(p.get("citations", []))
                elif ev == "done":
                    done = p
                elif ev == "error":
                    errors.append(p.get("message"))
    return deltas, text, citations, done, errors


def main():
    sid = httpx.post(f"{BASE}/sessions", json={}, timeout=30).json()["id"]
    print(f"[session] {sid}")

    # 1. RAG 知识库问答
    d, t, c, done, errs = chat({"session_id": sid, "content": "退换货政策是怎样的?"})
    print(f"\n[1 RAG] deltas={d} citations={c} errors={errs}")
    print(f"  回答({len(t)}字): {t[:180]}...")

    # 2. 订单工具
    d, t, c, done, errs = chat({"session_id": sid, "content": "帮我查一下订单 JD202608120001 的物流到哪了"})
    print(f"\n[2 订单工具] deltas={d} citations={c} errors={errs}")
    print(f"  回答({len(t)}字): {t[:180]}...")

    # 3. 上下文连续(指代"这个订单")
    d, t, c, done, errs = chat({"session_id": sid, "content": "那这个订单现在是什么状态?花了多少钱?"})
    print(f"\n[3 上下文连续] deltas={d} errors={errs}")
    print(f"  回答({len(t)}字): {t[:180]}...")

    # 4. 转人工
    d, t, c, done, errs = chat({"session_id": sid, "content": "我要投诉,给我转人工客服"})
    print(f"\n[4 转人工] deltas={d} errors={errs}")
    print(f"  回答({len(t)}字): {t[:150]}...")
    print(f"  done.escalated={done.get('escalated') if done else None} handoff={done.get('handoff') if done else None}")

    # 5. 持久化验证
    msgs = httpx.get(f"{BASE}/sessions/{sid}/messages", timeout=30).json()
    sess = [s for s in httpx.get(f"{BASE}/sessions", timeout=30).json() if s["id"] == sid][0]
    print(f"\n[5 持久化] 消息条数={len(msgs)} 会话标题={sess['title']!r} handoff={sess['handoff']}")
    cites = [m for m in msgs if m["role"] == "assistant" and m.get("citations")]
    print(f"  带引用的助手消息={len(cites)} 首条引用={json.dumps(cites[0]['citations'][0], ensure_ascii=False)[:120] if cites else '无'}")

    httpx.delete(f"{BASE}/sessions/{sid}", timeout=30)


if __name__ == "__main__":
    main()
