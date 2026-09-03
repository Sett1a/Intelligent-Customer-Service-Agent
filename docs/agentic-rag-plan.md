# 客服 Agent「Agentic RAG」升级方案（2026-09-02）

> 配套调研：[agentic-rag-research.md](agentic-rag-research.md)（全部论断有一手引用，下文用 `[n]` 指向其参考文献）。
> 本文档只回答三件事：**现状差在哪、目标架构长什么样、分几步落地**。

---

## 0. 结论摘要（TL;DR）

1. **现状定性**：本项目目前是"工具化 RAG"——检索的**时机**由 agent 决定（Agentic），但检索**策略**
   是单次稠密向量召回、无改写/无重排/无自检（Naive）。调研文档 §1。
2. **关于"pi agent + RAG"**：调研确认 "pi" 最可能指 badlogic 的极简编码 agent pi-mono（TypeScript，
   详见调研 §7）。**不建议为此换栈**：它是编码 CLI + Node 库，采用即重写整个 Python 后端；而它的
   "极简 agent loop"理念，Python 侧的等价物就是项目现有且已验证的 Pydantic AI。若 "pi" 指 Pydantic AI，
   则本方案正是其深化。
3. **路线**：保持 **Pydantic AI（编排）× LlamaIndex（检索组件）× 智谱（模型 + rerank API）** 不变，
   把"一次检索"升级为"改写 → 混合召回 → 重排 → 阈值 → 自检重试"的 agentic 检索链。
4. **做六件事**（调研 §6 判定高性价比）：混合检索 RRF、查询改写、重排 + 阈值过滤、CRAG-lite 自检
   重试、引用约束（RIG-lite）+ output validator、（可选）真 HITL 转人工。
   **不做**：多跳/SubQuestion、Self-RAG 训练、LangGraph/OpenAI SDK 迁移、多专精 agent、
   auto-merging/sentence-window、Temporal（理由见调研 §6）。
5. **两个平台级顺手升级**：`CHAT_MODEL` 免费升到 `glm-4.7-flash`（200K 上下文、工具调用更强，
   调研 §8 [46]）；接入智谱 rerank API（`POST /paas/v4/rerank`，调研 §8 [45]）。
6. **一件地基**：先建 RAGAS 小评测集（20-50 条）拿 baseline，之后每次检索改动都跑回归——
   没有 baseline 的"增强"无法证明有效（调研 §5.3）。

---

## 1. 现状诊断（对照代码）

| # | 现状（文件） | 问题 | 对应 Agentic RAG 模式 |
|---|---|---|---|
| 1 | `rag/retriever.py`：`index.as_retriever(similarity_top_k=4)` 单路稠密召回 | 型号/订单号/专有名词等词面匹配召回差；QA 短文本上 embedding-3 单路不稳 | 混合检索（BM25+RRF）[1][8] |
| 2 | `agent.py: retrieve_knowledge`：用户原话直接当 query | 多轮指代（"它什么时候到"）直接检索，召回必然差 | 查询改写/扩展 [1][7][8] |
| 3 | 检索结果直接拼接进上下文，无相关性门槛 | 无关 chunk 混入 → 幻觉率上升（Chroma RIG 结论，调研 §5.2） | 重排 + 阈值过滤 [18][1] |
| 4 | 检索失败时无重试机制，直接"告知暂无信息" | 明明可答的问题因一次检索失败而答非所问 | CRAG-lite（评估→改写→重试一次）[17][6] |
| 5 | 系统提示约束"仅依据检索结果回答"，但无程序化校验 | 模型偶发越出检索内容 | RIG-lite 引用约束 + `output_validator` [35][24] |
| 6 | `escalate_to_human` 是普通工具，返回一段文本 | "转人工"只是标记，无真实人工确认生命周期 | （可选）deferred tools 真 HITL [23] |
| 7 | 无任何评测 | 每次改动好坏全凭感觉 | RAGAS + 固定回归集 [33][34] |

---

## 2. 目标架构

```
用户消息 → POST /api/chat (SSE)
  └─ Pydantic AI 单 Agent（glm-4.7-flash，temperature 0.1）
       │
       ├─ 工具 retrieve_knowledge(query)          ← 改造重点
       │    ① 查询改写/扩展: 多轮上下文消解指代 + 生成变体（num_queries≈3）
       │    ② 混合召回: Chroma 稠密 top-20 × BM25 稀疏 top-20（jieba 分词）
       │       → QueryFusionRetriever RRF 融合（k=60，可 retriever_weights 偏稀疏）
       │    ③ 重排: 智谱 /paas/v4/rerank 取 top-4（备选: 本地 bge-reranker-v2-m3）
       │    ④ 阈值过滤: relevance_score < 阈值剔除
       │    ⑤ CRAG-lite: 全部低于阈值 → 改写 query 重试一次 → 仍失败
       │       → 返回显式"未找到"信号（模型如实告知，不编造）
       │    citations 含 rerank 分数，经 api.py 现有去重逻辑下发 SSE
       │
       ├─ 工具 get_order_status / get_logistics   ← 不变
       │
       └─ 工具 escalate_to_human                  ← 两种形态二选一
            A. 维持普通工具（产品语义 = 标记 + 安抚，坐席系统另行接入）→ 现状已够
            B. requires_approval=True 真 HITL（DeferredToolRequests/Results，
               run 暂停、坐席批准后带 message_history 恢复）[23]
       │
       └─ 输出侧: @agent.output_validator 抽查"回答是否可由 citations 支持"（RIG-lite）[24]
```

设计原则沿用调研两条硬结论：

- **单 Agent + 少工具，不拆多智能体**——Anthropic"最简方案"，4 工具单知识库下多专精 agent 无收益 [4][15][27]；
- **自纠错只做"plug-and-play"档**——CRAG 定位即评估器 + 分支动作，不训练任何模型 [17]。

---

## 3. 分期实施（每期独立可交付、可验证）

### Phase 0 · 评测地基（约 0.5 天）——先做，其他都是它的分支

- 新增 `backend/scripts/eval_rag.py`：从 kb.jsonl 抽 20-50 条验证问题（人工筛掉过偏的），
  跑"检索→生成"全链路，输出 RAGAS 四指标：faithfulness / answer relevancy / context precision /
  context recall [33][34]；工具链指标 ToolCallAccuracy 可选。
- RAGAS 依赖较重（含 langchain 一族），用 `uv` 依赖组（`[dependency-groups] dev`）安装，不进运行时依赖。
- 产出：**baseline 数字**写进本文档 §6，后续每期跑对比。
- 无 Key 干跑：沿用 EMBED_FAKE 路线不可行（fake 向量无语义），评测必须有真实 Key。

### Phase 1 · 检索质量（约 1-2 天，性价比最高）

1. `rag/retriever.py` 改造：`get_fusion_retriever()` =
   `QueryFusionRetriever([ChromaVectorStoreRetriever, BM25Retriever], mode="reciprocal_rerank",
   num_queries=3~4, use_async=True)` [8][49][50][52]。
   - BM25 节点来源：语料量级小（数千块），**启动时从 Chroma 全量拉文本内存构建**即可，无需持久化
     BM25 索引、无需改 ingest.py；
   - 中文分词：BM25 需自定义 jieba tokenize 函数（0.14 的参数名实现时核对，调研 §9）。
2. 新增 `rag/rerank.py`：`ZhipuRerank(BaseNodePostprocessor)` 调 `POST /paas/v4/rerank`
   （model=`rerank`，query + candidates 文本 + top_n；128 条/4K 字上限，送 top-20 重排足够）[45]；
   配置 `RERANK_PROVIDER=zhipu|local|off`，`local` 走 `FlagEmbeddingReranker(bge-reranker-v2-m3)`
   （CPU 可跑）[53][54]，便于断网/比价 A/B。
3. `SimilarityPostprocessor` / rerank 分数阈值过滤（阈值用 Phase 0 集合调出来）[18]。
4. 验收：Phase 0 指标对比，context precision / recall 至少不回退、faithfulness 提升。

### Phase 2 · Agent 自治（约 1-2 天）

1. **查询改写**：优先用 QueryFusionRetriever 自带 query generation（零额外组件）；多轮指代消解
   在 `retrieve_knowledge` 内拼最近 2-3 轮会话做一次改写调用（免费档模型承担）[7][8]。
2. **CRAG-lite**：检索全空/低于阈值 → 改写 query 重试一次（可用 `ModelRetry` 或工具内循环）→
   仍失败返回"未找到"信号；**至多重试一次，显式 stopping condition**（Anthropic 原则）[4][17][6]。
3. **RIG-lite**：系统提示升级（仅依据检索内容 + 标注引用 + 说不清就说没有）+ `@agent.output_validator`
   抽查回答是否被 citations 支持，不通过 `ModelRetry` 重答 [24][35]。
4. **模型升级**：`.env` 默认 `CHAT_MODEL=glm-4.7-flash`（免费、200K、工具调用更强 [46]）；
   按仓库约定换模型后**必须重跑冒烟**（uvicorn + /api/health + 会话 CRUD + 工具链用例）。
5. SSE 可选增强：新增 `step` 事件（tool_call / retrieval retry 可见性），前端可选展示"正在检索…"。

### Phase 3 · 可选增强（视产品需求，不默认做）

- **真 HITL 转人工**：形态 B（§2）。前提是存在真实的坐席确认环节；若产品语义就是"标记 + 安抚"，
  现状形态 A 已正确——deferred tools 不是必须的仪式感 [23]。
- **联网兜底**：CRAG 的 web-search 分支用智谱 Web Search API 补中文检索 [47][17]。
- **多知识源**：接入新品类知识时再考虑 RouterQueryEngine / 子问题引擎（当前判定为过度设计）[12][10]。
- **pi 技术栈**：仅当未来整体转向 TS 时评估 pi-agent-core + pi-ai 作 Node 侧 harness（调研 §7）。

---

## 4. 关键改动点（文件级清单）

| 文件 | 改动 |
|---|---|
| `backend/app/rag/retriever.py` | `get_fusion_retriever()`（稠密+BM25+RRF）、retriever 实例缓存 |
| `backend/app/rag/rerank.py` | 新增：`ZhipuRerank` / 本地 rerank 后处理器，`RERANK_PROVIDER` 分派 |
| `backend/app/agent.py` | `retrieve_knowledge` 内部换 fusion 链 + CRAG-lite；系统提示升级；`output_validator` |
| `backend/app/config.py` | 新增 `RERANK_PROVIDER`、`RERANK_TOP_N`、`FUSION_NUM_QUERIES`、`SIMILARITY_CUTOFF` 等；`CHAT_MODEL` 默认改 `glm-4.7-flash` |
| `backend/scripts/eval_rag.py` | 新增：RAGAS 评测脚本（读 kb.jsonl 验证集） |
| `backend/.env.example` | 同步新配置项 |
| `backend/scripts/ingest.py` | **不动**（embedding 未换，无需 `--reset`；BM25 从 Chroma 现量构建） |
| `frontend/` | **不动**（citations 渲染已有；`step` 事件为可选增量） |

风险与注意：

- 换 `CHAT_MODEL` 后工具调用行为可能变化——冒烟必跑（AGENTS.md 既有约定）。
- 智谱 rerank 可用模型名以控制台实测为准（schema 枚举 `rerank`，计费页列 GLM-rerank/-pro）[45][48]。
- BM25 中文 tokenize 的具体参数名在 0.14 实现时核对（调研 §9 已标注）。
- 延迟预算：改写（1 次轻量调用）+ 重排（1 次 API）+ 重试（至多 1 次）；免费模型 + 0.8 元/千 token
  的重排，单轮成本可忽略，但要盯首 token 延迟——阈值别设太紧、`step` 事件缓解感知。
- Windows + Chroma 中文路径 bug 的规避（相对路径 + chdir）继续遵守，勿动 `config.py` 现有逻辑。

---

## 5. 验收口径

1. Phase 0 后：评测脚本产出 baseline（4 项 RAGAS 指标 + 延迟 + 每轮 token 成本）。
2. Phase 1 后：context precision/recall 提升，faithfulness 不回退；人工抽 10 条对比检索质量。
3. Phase 2 后：多轮指代问题（"它什么时候到"）检索命中正确政策；检索失败问题如实答"没有依据"
   而非编造；工具链 TestModel 离线用例全绿。
4. 每期结束跑一次评测脚本，数字写回本文档 §6 形成台账。

## 6. 评测台账（随实施更新）

| 日期 | 配置 | faithfulness | relevancy | ctx precision | ctx recall | 备注 |
|---|---|---|---|---|---|---|
| 2026-09-02 | 现状（dense top-4, glm-4-flash） | 待测 | 待测 | 待测 | 待测 | baseline |
