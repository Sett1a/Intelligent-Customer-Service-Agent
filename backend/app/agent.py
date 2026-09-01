"""客服 Agent:Pydantic AI + 智谱 GLM,工具 = RAG 检索 / 订单 / 物流 / 转人工。"""

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext

from app.config import MOCK_DIR


@dataclass
class SupportDeps:
    """每轮请求的依赖与可变状态(citations 由工具填充,escalated 标记转人工)。"""

    retriever: Any = None
    citations: list = field(default_factory=list)
    escalated: bool = False


SYSTEM_PROMPT = """你是「小智」,一家电商平台的智能客服助手。

工作规则:
1. 用户提出任何咨询类问题时(除打招呼、订单/物流查询外),**必须先调用 retrieve_knowledge 工具检索知识库,再严格依据检索结果回答**。调用时直接把用户的问题(或提炼的关键词)作为 query;即使问题宽泛(如"退换货政策是什么")也要立即检索,禁止反问用户想了解哪方面。检索没有依据时,如实告知"暂无相关信息"并建议转人工;**严禁跳过检索直接依据自身知识回答业务问题,严禁编造**。
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


@agent.tool
async def retrieve_knowledge(ctx: RunContext[SupportDeps], query: str) -> str:
    """检索电商客服知识库(售后政策、发货物流、退换货规则等)。回答政策类问题前必须先调用本工具。"""
    if ctx.deps.retriever is None:
        return "知识库尚未构建,无法检索。请如实告知用户暂无相关信息。"
    nodes = await ctx.deps.retriever.aretrieve(query)
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
