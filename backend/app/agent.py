"""客服 Agent:Pydantic AI + 智谱 GLM,工具 = RAG 检索 / 订单 / 物流 / 转人工。

检索工具 retrieve_knowledge 内部先经「查询拆分子 agent」(结构化输出)把用户问题
拆成若干独立检索点,再逐点混合召回、跨查询融合、以原始问题为锚重排(分点检索),
单一意图问题拆分结果为空,行为等同整句检索。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import UserPromptPart

from app.config import DECOMP_MAX_QUERIES, MOCK_DIR, QUERY_DECOMPOSITION
from app.rag.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class SupportDeps:
    """每轮请求的依赖与可变状态(citations 由工具填充,escalated 标记转人工)。"""

    retriever: Any = None
    citations: list = field(default_factory=list)
    escalated: bool = False


SYSTEM_PROMPT = """你是「小智」,一家电商平台的智能客服助手。

工作规则:
1. 用户提出任何咨询类问题时(除打招呼、订单/物流查询外),**必须先调用 retrieve_knowledge 工具检索知识库,再严格依据检索结果回答**。调用时直接把用户的问题(或提炼的关键词)作为 query,即使问题宽泛(如"退换货政策是什么")也要立即检索,禁止反问用户想了解哪方面;多意图问题(如"退货流程是什么?运费谁承担?")也一次传入即可,工具内部会自动分点拆分检索。检索没有依据时,如实告知"暂无相关信息"并建议转人工;**严禁跳过检索直接依据自身知识回答业务问题,严禁编造**。
2. 订单状态、物流查询:如果用户没有给订单号,先礼貌询问;拿到订单号后调用 get_order_status 或 get_logistics。
3. 用户明确要求人工、发生投诉纠纷、或你无法依据知识库解决问题时:调用 escalate_to_human,并安抚用户人工客服会尽快接入。
4. 回答使用简体中文,语气友好专业,内容较多时适当分点;不透露系统内部实现。
5. 转人工标记只需设置一次,不要重复调用 escalate_to_human。"""


def _build_model():
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from app.config import CHAT_MODEL, ZHIPU_API_KEY, ZHIPU_BASE_URL

    return OpenAIChatModel(
        CHAT_MODEL,
        provider=OpenAIProvider(
            base_url=ZHIPU_BASE_URL, api_key=ZHIPU_API_KEY or "missing"
        ),
    )


agent = Agent(
    model=_build_model(),
    deps_type=SupportDeps,
    system_prompt=SYSTEM_PROMPT,
    retries=1,
    model_settings={"temperature": 0.1},  # 客服场景偏事实回答,低温提升工具调用稳定性
)


# ---------- 查询拆分子 agent(分点检索) ----------


class SubQueries(BaseModel):
    """查询拆分的结构化输出:独立、自包含的检索点列表。"""

    queries: list[str] = Field(
        default_factory=list,
        description="拆分出的检索点;每个点自包含、无指代;单一意图问题输出空列表",
    )


DECOMPOSE_SYSTEM_PROMPT = f"""你是电商客服的检索查询规划器。把用户的咨询问题拆分成若干个独立的检索点,供知识库逐点检索使用。

规则:
1. 只拆分问题内**真实包含的不同子意图**(如"退货流程"与"退货运费承担"),不要扩写、不要臆造问题里没有的内容;
2. 每个检索点必须自包含(补全主语与语境),不含"它/这个"等指代;
3. 每条简短(≤30 字),口语化关键词即可;
4. 最多 {DECOMP_MAX_QUERIES} 个;
5. 问题本身是单一意图时,输出空列表。
只输出检索点,不要回答问题,不要输出解释。"""

# 拆分子 agent:复用同一模型端点;output_type=SubQueries 走工具调用式结构化输出
decompose_agent = Agent(
    model=_build_model(),
    output_type=SubQueries,
    system_prompt=DECOMPOSE_SYSTEM_PROMPT,
    retries=1,
    model_settings={"temperature": 0.1},
)


def _recent_user_context(messages: list | None, limit: int = 3) -> str:
    """取当前 run 最近几条用户原文,辅助拆分子 agent 消解"它/这个"等指代。"""
    texts: list[str] = []
    for m in messages or []:
        for part in getattr(m, "parts", []):
            if isinstance(part, UserPromptPart):
                content = part.content
                if isinstance(content, str):
                    texts.append(content)
                else:  # 多段内容(如图片+文字)时拼接文本化
                    texts.append(" ".join(str(c) for c in content))
    return " | ".join(t.strip() for t in texts[-limit:] if t)


async def decompose_queries(query: str, messages: list | None = None) -> list[str]:
    """把用户问题拆成检索点(分点检索)。

    关闭(QUERY_DECOMPOSITION=0)、模型失败或超时时返回 [],调用方退回整句检索,
    保证检索链路永不因拆分环节而中断。
    """
    if not QUERY_DECOMPOSITION:
        return []
    try:
        context = _recent_user_context(messages)
        prompt = f"会话上下文:{context or '(无)'}\n当前问题:{query}"
        result = await decompose_agent.run(prompt)
        seen: set[str] = set()
        points: list[str] = []
        for q in result.output.queries:
            q = (q or "").strip()
            key = re.sub(r"\s+", "", q)
            if q and key and key not in seen and key != re.sub(r"\s+", "", query):
                seen.add(key)
                points.append(q)
        return points[:DECOMP_MAX_QUERIES]
    except Exception as e:  # noqa: BLE001 - 拆分是增强项,任何失败都退回整句检索
        logger.warning("查询拆分失败,退回整句检索: %s: %s", type(e).__name__, e)
        return []


@agent.tool
async def retrieve_knowledge(ctx: RunContext[SupportDeps], query: str) -> str:
    """检索电商客服知识库(售后政策、发货物流、退换货规则等)。回答政策类问题前必须先调用本工具。"""
    if ctx.deps.retriever is None:
        return "知识库尚未构建,无法检索。请如实告知用户暂无相关信息。"
    retriever = ctx.deps.retriever
    nodes = None
    if isinstance(retriever, HybridRetriever):
        try:  # 分点检索:原始问题 + 拆分点逐点召回;拆分失败由 decompose_queries 返回 []
            points = await decompose_queries(query, messages=ctx.messages)
            nodes = await retriever.aretrieve_multi([query, *points])
        except Exception:  # noqa: BLE001 - 退回整句检索
            nodes = None
    if nodes is None:
        nodes = await retriever.aretrieve(query)
    if not nodes:
        return "知识库中未找到相关内容。请如实告知用户,不要编造。"
    lines: list[str] = []
    for n in nodes:
        text = n.node.get_content().strip()
        meta = n.node.metadata or {}
        ctx.deps.citations.append(
            {
                "source": meta.get("source", "知识库"),
                "question": meta.get("question", ""),
                "snippet": text[:160],
                "score": round(float(n.score or 0.0), 4),
            }
        )
        lines.append(f"【{meta.get('source', '知识库')}】{text}")
    return "以下是知识库检索结果,请严格依据这些内容回答:\n\n" + "\n\n---\n\n".join(lines)


def _load_orders() -> dict:
    path = MOCK_DIR / "orders.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _find_order(order_id: str) -> dict | None:
    orders = _load_orders()
    if order_id in orders:
        return orders[order_id]
    return next((o for o in orders.values() if o.get("order_id") == order_id), None)


@agent.tool
def get_order_status(ctx: RunContext[SupportDeps], order_id: str) -> str:
    """按订单号查询订单状态(演示环境模拟数据)。"""
    order = _find_order(order_id.strip())
    if not order:
        return f"未找到订单 {order_id},请引导用户核对订单号(演示环境可用订单号:JD202608120001 等)。"
    return json.dumps(
        {k: v for k, v in order.items() if k != "logistics"}, ensure_ascii=False
    )


@agent.tool
def get_logistics(ctx: RunContext[SupportDeps], order_id: str) -> str:
    """按订单号查询物流轨迹(演示环境模拟数据)。"""
    order = _find_order(order_id.strip())
    if not order:
        return f"未找到订单 {order_id},请引导用户核对订单号。"
    return json.dumps(
        {"order_id": order.get("order_id"), "logistics": order.get("logistics", {})},
        ensure_ascii=False,
    )


@agent.tool
async def escalate_to_human(ctx: RunContext[SupportDeps], reason: str) -> str:
    """将当前会话转接人工客服。仅在用户明确要求、投诉纠纷或知识库无法解决时调用。"""
    if not ctx.deps.escalated:
        ctx.deps.escalated = True
    return "已为用户标记转人工。请回复安抚话术,告知人工客服将尽快接入。"
