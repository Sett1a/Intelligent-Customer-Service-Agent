# Agentic RAG 技术调研（2026-09-02）

> 调研方式：全部论断对照一手来源（arXiv 论文原文/摘要、官方文档、官方工程博客、源码仓库），
> 内联标注 `[n]`，文末为编号参考文献。个别一手来源已下线或本环境不可达的，均在正文明确说明。
>
> 调研环境说明（影响引用方式，不影响结论）：
> - Pydantic AI 官方文档已整体迁移：`ai.pydantic.dev` 现在 301 重定向到 `pydantic.dev/docs/ai/`，本文引用以新域名为准 [41]。
> - LlamaIndex 官方文档已整体迁移：`docs.llamaindex.ai` 301 重定向到 `developers.llamaindex.ai`，本文引用以新域名为准 [2]。
> - LangChain/LangGraph 官方教程在 2026 年的文档整合中大量合并：旧的 `langgraph_crag` / `langgraph_self_rag` /
>   `langgraph_adaptive_rag` / `customer-support` 教程 URL 在官方仓库 `redirects.json` 中全部指向新的
>   "Build a custom RAG agent" 教程 [6][30]。
> - Chroma 的 RIG 研究笔记（RAG 检索质量评估）已从官网下线（官网 research 栏目现仅列 2024-05 之后的 5 篇 [37]），
>   本环境亦无法访问 web.archive.org 存档，相关结论标注为"未能重新逐字核对"，见第 5 节。

## 摘要

1. RAG 的演进是"从固定流水线到动态决策"：Naive RAG（一次索引 + 一次稠密检索 + 一次生成）→
   Advanced/Modular RAG（在检索前/检索中/检索后插入确定性优化模块）→ Agentic RAG（由 LLM agent
   决定是否检索、检索什么、检索几轮、何时停下）[1][3][4][5]。
2. 核心模式可归纳为七类：查询改写与扩展、查询分解（子问题）、路由、迭代/多跳检索、自我纠错与反思
   （Self-RAG / CRAG / Adaptive-RAG）、检索后处理（重排 / 句子窗口 / 自动合并）、记忆与会话上下文。
   每种模式都有成熟的一手来源与开源实现。
3. 框架层面：LlamaIndex 提供的 QueryFusionRetriever（RRF 混合检索）、Router、SubQuestion、
   node postprocessors、Memory 等组件可以作为 Pydantic AI 工具的"内部实现"直接复用，编排权仍在
   Pydantic AI；LangGraph 的官方 RAG 教程（grade → rewrite 循环）是模式的最佳参考实现，但引入它意味着
   换编排栈；OpenAI Agents SDK 的 handoff/guardrail 思想可以在 Pydantic AI 中以工具/output validator/
   deferred tools 平价实现 [8][6][23][24][31][32]。
4. 对本项目（客服 Agent：Pydantic AI 单 Agent + 4 工具 + 单次稠密检索 + JDDC QA 对）性价比最高的是：
   查询改写、混合检索 RRF、重排 + 阈值过滤、CRAG-lite 自检重试、RIG-lite 引用约束、deferred-tools 转人工、
   RAGAS 小评测集；过度设计的是：多跳/迭代检索、Self-RAG 训练类方案、复杂多智能体编排、Durable Execution、
   Auto-merging 层级索引。

---

## 1. RAG → Advanced RAG → Agentic RAG：演进与分类

**Naive RAG（朴素 RAG）**：最短链路"索引 → 检索 → 生成"。LlamaIndex 官方把标准 RAG 分为五个阶段：
Loading → Indexing → Storing → Querying → Evaluation，其中 Querying 又细分为 Retrieval + Postprocessing +
Synthesis [2]。综述论文指出 Naive RAG 的三类典型问题：检索质量低（幻觉背景）、增强过程噪声大、
生成过程可能过拟合或无视检索内容 [1]。

**Advanced RAG / Modular RAG**：Gao 等人的综述（arXiv 2312.10997）明确"we categorize it into three
stages: Naive RAG, Advanced RAG, and Modular RAG" [1]。Advanced RAG 在 Naive 流水线周围加确定性优化模块
[1]：

- **检索前（pre-retrieval）**：Query Expansion（Multi-Query、Sub-Query、CoVe）、Query Transformation
  （Query Rewrite 如 RRR、HyDE、Step-back Prompting）、Query Routing（Metadata/Semantic Router），
  以及索引侧优化（数据粒度、元数据、混合检索）[1]。
- **检索中**：Mix/hybrid Retrieval——"Sparse and dense embedding approaches capture different relevance
  features"，BM25 类稀疏检索与稠密检索互补，尤其改善零样本与罕见实体召回 [1]。
- **检索后（post-retrieval）**："The main methods in post-retrieval process include rerank chunks and
  context compressing"（重排 + 上下文压缩/筛选）[1]。

Modular RAG 则进一步把 RAG 拆成可自由编排的模块与交换模式（允许循环、条件跳转等非固定拓扑）[1]。

**Agentic RAG**：把"检索前/中/后"的确定性模块升级为"由 agent 动态决策"。Agentic RAG 综述（arXiv
2501.09136）的定义：通过"embedding autonomous AI agents into the RAG pipeline"克服传统 RAG
"static workflows and lack the adaptability required for multi-step reasoning"的缺陷，核心是让 agent
"dynamically manage retrieval strategies, iteratively refine contextual understanding" [3]。
LlamaIndex 官方博客将 Agentic RAG 描述为"how agents can be incorporated into existing RAG pipelines for
enhanced, conversational search and retrieval"，强调多文档 agent 编排 [5]。

Anthropic《Building effective agents》提供了更基础的分类学，也是判断"要不要上 agent"的标尺 [4]：

- **Workflow**："systems where LLMs and tools are orchestrated through predefined code paths"；
- **Agent**："systems where LLMs dynamically direct their own processes and tool usage"；
- 基础构件是 **Augmented LLM**（检索 + 工具 + 记忆增强的 LLM），五种 workflow 模式：prompt chaining、
  **routing**（"classifies an input and directs it to a specialized followup task"，典型场景就是客服）、
  parallelization、orchestrator-workers、evaluator-optimizer；agent 本质上是
  "LLMs using tools based on environmental feedback in a loop" [4]；
- 指导原则："finding the simplest solution possible, and only increasing complexity when needed"，
  并显式设置"stopping conditions (such as a maximum number of iterations)" [4]。

**分类口径**：当前主流的"工具化 RAG"（把检索包成工具给单 agent 调用一次）介于 Naive RAG 与 Agentic RAG
之间——检索时机由 agent 决定（这是 Agentic 的），但检索策略是单次、单向、无自检的（这又是 Naive/Advanced
的）。本项目的现状即属此类。

---

## 2. Agentic RAG 核心模式

### 2.1 查询理解与改写（query rewriting / expansion / HyDE）

- **定义**：在检索前用 LLM 或规则把用户原始 query 转换成更适合检索的形式：多查询扩展、改写（消解指代、
  口语→关键词）、假设文档（HyDE）[1]。
- **代表实现**：
  - Query Rewrite 的 RRR（Rewrite-Retrieve-Read）与 HyDE 均被 RAG 综述列为代表性检索前优化 [1]；
  - **HyDE**（arXiv 2212.10496）：让模型先生成"hypothetical document"，用该假设文档的 embedding 去查
    相似真实文档；编码器充当"dense bottleneck filtering out the incorrect details"，零标注即可逼近
    有微调的检索器 [7]。适用场景：问答语料与 query 的表述差异大、稠密检索召回差。
  - **LlamaIndex QueryFusionRetriever**（源码级核实）：默认 prompt 要求模型"generates multiple search
    queries based on a single input query"，生成 `num_queries-1` 条变体（默认 4，含原始 query），
    多路检索后用融合模式（RECIPROCAL_RANK / RELATIVE_SCORE / DIST_BASED_SCORE / SIMPLE）合并 [8]。
- **适用场景**：多轮客服对话里"它什么时候到"这类指代/省略严重的 query，改写几乎是必选项；成本是每轮
  多一次 LLM 调用。

### 2.2 查询分解 / 子问题（sub-question / decomposition）

- **定义**：把复合问题拆成多个可独立回答的子问题，分别检索回答后合成 [1][9]。
- **代表论文**：
  - **Self-Ask**（arXiv 2210.03350）：模型"explicitly asks itself (and answers) follow-up questions
    before answering the initial question"；结构化格式使"we can easily plug in a search engine to answer
    the follow-up questions"。该文还测出"compositionality gap"——GPT-3 家族模型单跳问答随规模提升快于
    多跳，说明组合推理不会免费获得 [9]。
  - **IRCoT**（arXiv 2212.10509）：多跳问题"what to retrieve depends on what has already been derived"，
    因此把检索与 CoT 交错进行，GPT-3 上检索质量提升"up to 21 points"、QA 提升"up to 15 points"，并
    "reduces model hallucination" [11]。
- **代表实现**：LlamaIndex **SubQuestionQueryEngine**——"breaks down the complex query into sub questions
  for each relevant data source"，收集中间答案后"synthesizes a final response"，构造用
  `SubQuestionQueryEngine.from_defaults(query_engine_tools=[QueryEngineTool(...)], use_async=True)` [10]。
- **适用场景**：问题天然由多个独立子问题组成（如跨多个知识源/多实体比较）。客服单实体 QA 对场景收益低。

### 2.3 路由（router）

- **定义**：Anthropic 定义 routing 为"classifies an input and directs it to a specialized followup task"，
  价值在"separation of concerns"，客服是文中典型例子 [4]。
- **代表实现**：
  - **LlamaIndex Routers**："Routers are modules that take in a user query and a set of 'choices'
    (defined by metadata), and returns one or more selected choices"；LLM selector（文本补全）与
    Pydantic selector（函数调用）两类，`LLMSingleSelector` / `LLMMultiSelector` /
    `PydanticSingleSelector` / `PydanticMultiSelector`，组合形态 `RouterQueryEngine`（choices 为
    `QueryEngineTool`）与 `RouterRetriever`（choices 为 `RetrieverTool`）[12]。
  - **Adaptive-RAG**（arXiv 2403.14403）：用一个小分类器预测 query 复杂度，在"no-retrieval /
    single-step retrieval / iterative retrieval"三档间动态路由，标签来自"actual predicted outcomes of
    models and inherent inductive biases in datasets"自动收集，目标是准确率与成本的平衡 [14]。
  - **Pydantic AI 路线**：单 Agent 的多工具选择本身就是路由——LLM 在"查知识库 / 查订单 / 查物流 /
    转人工"间选择，无需额外 Router 模块 [13]。
- **适用场景**：多数据源/多业务线；单知识库 + 少量工具的项目，框架级 Router 是冗余的。

### 2.4 迭代检索 / 多跳（multi-hop）

- **定义**：检索-推理交错，第二轮检索依赖第一轮结果 [11][9]。
- **代表工作**：IRCoT [11]、Self-Ask [9]（见 2.2）；Anthropic 多智能体研究系统则从工程角度描述：
  subagent 的搜索"Unlike static RAG retrieval, search is multi-step and adapts to findings" [15]。
- **适用场景**：答案需要链式事实（HotpotQA 类多跳、深挖式研究）。客服 QA 对（一问一答）基本是单跳，
  完整多跳管线在本场景属于过度设计；"至多重试一次"的受控迭代（见 2.5 CRAG）性价比远高于自由多跳。

### 2.5 自我纠错与反思（Self-RAG / CRAG / Adaptive-RAG）

- **Self-RAG**（arXiv 2310.11511，Asai et al., UW/Allen AI 等）：批评传统 RAG"indiscriminately
  retrieving and incorporating a fixed number of retrieved passages"；训练单个 LM
  "adaptively retrieves passages on-demand"并用"reflection tokens"对检索段落与自身生成打分，
  反思 token 同时让生成行为在推理期可控；7B/13B 模型在开放域 QA、推理与事实核查上超过 ChatGPT 与
  检索增强的 Llama2-chat [16]。**代价**：需要专门训练模型，与"调用 API 模型（如 glm-4-flash）"的项目
  不兼容。
- **CRAG**（arXiv 2401.15884，Yan et al.）：先问"how the model behaves if retrieval goes wrong"。
  用"lightweight retrieval evaluator"评估检索结果质量，产生"confidence degree"并触发不同知识检索动作
  （correct/incorrect/ambiguous 三分支，不正确时用"large-scale web searches"补充）；配合
  "decompose-then-recompose"过滤无关信息；整条方案"plug-and-play"，可叠加到任意 RAG 管线 [17]。
  **这是对现有项目最友好的自纠错方案**：不训练模型、不改模型，只在管线里加一个评估器 + 一次重试。
- **Adaptive-RAG**（arXiv 2403.14403）：见 2.3，本质是"按 query 复杂度决定用多重的检索管线"的路由式
  自适应 [14]。分类器可以训练（原文做法），也可以用提示词近似。
- **LangGraph 的官方落地**：现行 "Build a custom RAG agent" 教程把纠错做成了显式图：`retrieve` 之后
  `grade_documents` 用 `GradeDocuments` 结构化输出判相关性，失败走 `rewrite_question`（"rewrite the
  original user question into a clearer search query"）再回到 `generate_query_or_respond`，官方明确称
  这是 agent "recovers from a weak first retrieval" 的方式 [6]。旧的 corrective_rag / self_rag /
  adaptive_rag 教程已统一重定向到该教程 [30]。

### 2.6 检索后处理（rerank / sentence-window / auto-merging）

- **重排（rerank）**：综述："The main methods in post-retrieval process include rerank chunks and
  context compressing" [1]。LlamaIndex 提供一族 node postprocessor [18]：
  - 跨编码器/模型重排：`SentenceTransformerRerank`（本地 cross-encoder，如
    `cross-encoder/ms-marco-MiniLM-L-2-v2`）、`CohereRerank`、`ColbertRerank`、`LLMRerank`、
    `JinaRerank`、`RankGPTRerank` 等 [18]；
  - 过滤类：`SimilarityPostprocessor`（分数阈值剔除）、`KeywordNodePostprocessor`（必须/排除关键词）、
    `SentenceEmbeddingOptimizer`（句子级剪枝）[18]；
  - 上下文工程：`LongContextReorder`（大 top-k 时把关键信息挪到上下文首尾，抗 lost-in-the-middle）[18]。
- **句子窗口（sentence-window）**：`SentenceWindowNodeParser` 把文档切成单句节点、每个节点在 metadata
  里带周边句子窗口；检索命中的是"细粒度、 Embedding 精准"的句子，回答前用
  `MetadataReplacementPostProcessor(target_metadata_key="window")` 把单句换回窗口上下文，兼顾召回精度
  与合成上下文完整性 [19]。适用前提：语料是**长文档**。
- **自动合并（auto-merging）**：`HierarchicalNodeParser` 按 2048/512/128 三层切块，只索引叶子节点；
  `AutoMergingRetriever` "looks at a set of leaf nodes and recursively 'merges' subsets of leaf nodes
  that reference a parent node beyond a given threshold"，命中多个小节点就整体替换为大父节点，给 LLM
  更连贯的上下文 [20]。官方示例里 auto-merging 与 base 检索的评测差距"roughly the same"（GPT-4 偏好
  0.525，接近持平）[20]——收益并不总是显著，需要按语料评估。
- **混合检索 RRF**：LlamaIndex `QueryFusionRetriever` 的 `RECIPROCAL_RANK` 模式实现 reciprocal rank
  fusion，源码注明"The original paper uses k=60 for best results"（Cormack et al. SIGIR 2009），
  并支持 `retriever_weights` 加权多路检索器 [8][39]。

### 2.7 记忆与会话上下文（agentic memory）

- **Pydantic AI**：多轮对话通过 `message_history` 传入先前的 `ModelRequest`/`ModelResponse` 列表；
  传入非空历史时"new system prompt is not generated"；结果用 `result.all_messages()` /
  `new_messages()`（及 `_json` 变体）取回；入库/回载用 `ModelMessagesTypeAdapter` +
  `to_jsonable_python()` / `validate_python()` 往返；新引入 `run_id`（单次运行）与 `conversation_id`
  （跨轮会话）标识，`ProcessHistory` 能力可在每轮请求前拦截修改历史（截断/摘要），`sanitize_messages`
  可清洗不受信历史 [22]。**与本项目现状完全对口**：本项目"每轮从 SQLite 加载全部消息映射为
  message_history"的方案就是官方推荐姿势，60 条上限可用 `ProcessHistory` 实现。
- **LlamaIndex**：agent 默认 `ChatMemoryBuffer`（token 限额）；新版推荐 `Memory` 类——短期 FIFO 队列 +
  可选长期记忆 block（`StaticMemoryBlock` / `FactExtractionMemoryBlock`（LLM 抽事实，`max_facts` 上限）/
  `VectorMemoryBlock`（向量库存取历史片段）），参数 `token_limit`（默认 30000）、
  `chat_history_token_ratio`（0.7）、`token_flush_size`（3000）；自定义 block 继承 `BaseMemoryBlock`；
  支持远程库（`async_database_uri`，默认内存 SQLite）[21]。
- **Agentic memory 的意义**：Anthropic 多智能体系统的核心经验之一——上下文有限时把计划/中间产物外置
  （lead agent 把计划写入 Memory，200k token 截断），并"spawn fresh subagents with clean contexts" [15]。

---

## 3. 框架能力对照

### 3.1 Pydantic AI 2.x（ai.pydantic.dev → pydantic.dev/docs/ai [41]）

| 能力 | 关键 API | 说明 |
|---|---|---|
| 工具 | `@agent.tool`（带 `RunContext`）、`@agent.tool_plain`、`Tool(takes_ctx=...)`、toolsets（`FunctionToolset`、MCP）、`prepare` 动态工具 | docstring 经 griffe 解析生成 schema；工具可返回任何可 JSON 化内容；`ModelRetry` 驱动工具级重试 [13] |
| 人工确认 / 延迟工具 | `requires_approval=True`、`ApprovalRequired`、`CallDeferred`、`DeferredToolRequests`（`approvals`/`calls`）、`DeferredToolResults`（`ToolApproved`/`ToolDenied`/`ToolReturn`）、`output_type=[..., DeferredToolRequests]`、`HandleDeferredToolCalls` | 两条路径：内联 handler 或"停机-恢复"（run 以 `DeferredToolRequests` 结束，人工处理后带 `message_history + deferred_tool_results` 重启）；官方警示 approval"不是对抗不可信客户端的授权边界" [23] |
| 输出类型 | `output_type=PydanticModel`、列表/联合（`[A, B]` 每个成员注册为独立 output tool）、`ToolOutput` / `NativeOutput` / `PromptedOutput` / `TextOutput`、`@agent.output_validator`（`ModelRetry` 重试） | 默认用模型工具调用承载结构化输出；output validator 是挂"结果自检（faithfulness/引用检查）"的官方位置 [24] |
| 多轮 / 历史 | `message_history`、`result.all_messages()`、`ModelMessagesTypeAdapter`、`run_id` / `conversation_id`、`ProcessHistory`、`sanitize_messages`、`ReinjectSystemPrompt` | 历史可持久化到数据库再回载，正是本项目 SSE 会话所需 [22] |
| 流式 | `run_stream()`、`stream_text(delta=...)`、`stream_output()`、`iter()`、`run_stream_events()`、`StreamedRunResult.cancel()` | 与 FastAPI SSE 对接的官方路径 [24] |
| 图/状态机 | `pydantic-graph`（"async graph and state machine library... nodes and edges are defined using type hints"）、`GraphBuilder`、decision nodes；agent 内部即用它管理执行流 | 供复杂控制流使用，agent 本身已覆盖大多数场景 [26] |
| 多智能体 | 官方指南五级复杂度：单 agent → agent delegation（在 `@tool` 里 `await other_agent.run(..., usage=ctx.usage)`）→ programmatic hand-off（应用代码在 agent 间传 `message_history`）→ graph 控制流 → Deep Agents | delegation 聚合 usage；hand-off 用 `result.all_messages(output_tool_return_content=...)` 回写历史 [27] |
| Durable Execution | `pydantic_ai.durable_exec.temporal`：`TemporalDurability()` capability、`PydanticAIPlugin`、`PydanticAIWorkflow`；DBOS/Prefect 有对应页；`TemporalAgent` 已弃用（v3 移除） | 长事务 agent 的断点重放；限制多（仅 async、payload 2MB、流式被缓冲）[25] |

### 3.2 LlamaIndex 0.14（developers.llamaindex.ai [2]）

| 能力 | 关键 API | 说明 |
|---|---|---|
| Agent | `FunctionAgent`（`llama_index.core.agent.workflow`，基于函数调用，"streaming is on by default"）、`ReActAgent`、`CodeActAgent`、手写循环（`llm.chat_with_tools`） | 0.14 的主力是 workflow 系 agent；AgentWorker/AgentRunner 为旧机制 [28] |
| 多智能体 | `AgentWorkflow(agents=[...], root_agent=..., initial_state=...)`、`can_handoff_to=[...]`；agent 间"handoff control to another agent"；也支持 orchestrator 模式（把专精 agent 包成 `call_xxx_agent` 工具，"tools always return back to the orchestrator"） | 三种模式对比：AgentWorkflow（最少代码、内置流式）/ orchestrator / 自定义 planner [29] |
| 转人工 | 工具内 `ctx.wait_for_event(HumanResponseEvent, waiter_event=InputRequiredEvent(...), requirements=...)` 暂停等人工回复，外部 `handler.ctx.send_event(HumanResponseEvent(...))` 续跑；可序列化 Context 稍后恢复 | 事件驱动 HITL [40] |
| 路由 | `RouterQueryEngine` / `RouterRetriever` + `LLMSingleSelector` / `PydanticSingleSelector` 等 [12] | 见 2.3 |
| 子问题 | `SubQuestionQueryEngine.from_defaults(query_engine_tools=..., use_async=True)` [10] | 见 2.2 |
| 混合检索/融合 | `QueryFusionRetriever(retrievers=[...], mode=..., num_queries=4, use_async=True, retriever_weights=...)`，RRF k=60；官方 RAG 文档列有 RRF / Relative Score Fusion / Auto-Merging / BM25 等检索器 | 可同时做"查询扩展 + 多路检索 + 融合"，是本项目改造检索工具的首选组件 [8][2] |
| 后处理 | `SimilarityPostprocessor`、`SentenceTransformerRerank`、`CohereRerank`、`LLMRerank`、`LongContextReorder`、`MetadataReplacementPostProcessor` 等（`llama_index.core.postprocessor`） [18] | 见 2.6 |
| 句子窗口 | `SentenceWindowNodeParser.from_defaults(window_size=...)` + `MetadataReplacementPostProcessor(target_metadata_key="window")` [19] | 长文档场景 |
| 自动合并 | `HierarchicalNodeParser` + `AutoMergingRetriever` + `SimpleDocumentStore` + `RetrieverQueryEngine.from_args(...)` [20] | 需要层级 docstore |
| 记忆 | `Memory`（+ 三种内置 block）、`ChatMemoryBuffer`（将废弃）、`VectorMemory`、`SimpleComposableMemory` [21] | 见 2.7 |
| RAG 五阶段 | Loading / Indexing / Storing / Querying / Evaluation [2] | 官方对标准 RAG 的模块化定义 |

### 3.3 LangGraph（docs.langchain.com）

- 现行官方 Agentic RAG 教程：`MessagesState` + `StateGraph` 节点 `generate_query_or_respond` →
  （有 tool_calls?）→ `ToolNode([retriever_tool])` → `grade_documents`（`GradeDocuments` 结构化判相关）→
  `generate_answer` 或 `rewrite_question` → 回环；检索是 `@tool` + 向量库 retriever；官方明确该教程让 agent
  可选地检索、并在"weak first retrieval"后自纠错 [6]。关键 API：`StateGraph.add_conditional_edges`、
  `ToolNode`、`with_structured_output`、`create_react_agent` [6]。
- **模板迁移事实**：官方仓库 redirects.json 显示 `/tutorials/rag/langgraph_crag`、
  `/tutorials/rag/langgraph_self_rag`、`/tutorials/rag/langgraph_adaptive_rag`（及 local 变体）、
  `/tutorials/customer-support/customer-support` 全部重定向到新 agentic-rag 教程 [30]。即：LangChain 官方
  现在只维护一条 Agentic RAG 入门路径，旧 CRAG/Self-RAG 模板不再是一等公民（模式本身仍以论文形式存在
  [16][17][14]）。

### 3.4 OpenAI Agents SDK（openai.github.io/openai-agents-python）

- **Handoffs**："Handoffs allow an agent to delegate tasks to another agent"，客服例子正是
  "order status, refunds, FAQs in a support app"；handoff"represented as tools to the LLM"
  （`transfer_to_refund_agent`）；`handoff()` 支持 `tool_name_override` / `on_handoff` 回调 /
  `input_type`（Pydantic 参数）/ `input_filter`（`agents.extensions.handoff_filters.remove_all_tools`
  等预置过滤器）/ `is_enabled`；`handoff_description` 提示何时该交给谁；只想借专家输出而不转移对话时用
  `Agent.as_tool(...)`；官方推荐提示词 `handoff_prompt.RECOMMENDED_PROMPT_PREFIX` [31]。
- **Guardrails**："checks and validations of user input and agent output"，官方示例就是
  "Customer support agent + 数学作业 guardrail"；`@input_guardrail` / `@output_guardrail` 返回
  `GuardrailFunctionOutput(output_info, tripwire_triggered)`，触发即抛
  `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered` 中止；input guardrail 只对链首
  agent 生效、默认与主 agent 并行跑（`run_in_parallel`），常用"快而便宜的模型"当守门员 [32]。
- 对本项目的意义：handoff ≈ Pydantic AI 的 delegation/hand-off 模式 [27]；guardrail ≈
  Pydantic AI 的 output validator 或一个独立小工具 [24]。**采用 SDK 本身需要换掉编排层，收益不抵成本。**

### 3.5 在"已有 Pydantic AI + LlamaIndex"项目里做 Agentic RAG 的增量成本

| 方案 | 增量成本 | 结论 |
|---|---|---|
| **LlamaIndex 组件作为 Pydantic AI 工具的内部实现**（推荐） | 只改 `retrieve_knowledge` 工具内部：检索器换成 `QueryFusionRetriever`（稠密 Chroma + BM25 + RRF）[8]、后处理加 `SimilarityPostprocessor` + reranker [18]；编排不动 | 零编排迁移；LlamaIndex 0.14 与 pydantic-ai-slim 无冲突，依赖已在本项目 |
| **Pydantic AI 原生扩展** | 加查询改写子 agent（delegation）[27]、`output_validator` 做 faithfulness/引用自检 [24]、`requires_approval` 转人工 [23] | 改动集中在 agent.py 与提示词 |
| **引入 LangGraph** | 第二套状态图运行时 + 自建消息桥接（LangGraph 消息 ↔ Pydantic AI message_history），与 AGENTS.md 既定技术栈冲突 | 不划算；其教程当"模式参考实现"读即可 [6] |
| **引入 OpenAI Agents SDK** | 同上，整套编排迁移才有 handoff/guardrail | 不划算；思想可平价移植 [31][32] |

---

## 4. 多智能体客服模式与转人工

1. **路由 agent + 专精 agent（triage → specialists）**：OpenAI Agents SDK 的官方叙事就是客服——triage
   agent 通过 handoff 把会话交给 refund/billing/FAQ 专精 agent，handoff 即一个 LLM 可见的工具
   `transfer_to_xxx`，配合 `handoff_description` 与 `input_filter` 控制历史可见性 [31]。Anthropic 的
   routing 模式与之同构："classifies an input and directs it to a specialized followup task" [4]。
2. **Orchestrator-workers**：Anthropic 多智能体研究系统（lead agent + 并行 subagent）证明该模式在
   "heavy parallelization, information that exceeds single context windows, and interfacing with numerous
   complex tools" 时收益巨大（内部研究评估超单 agent 90.2%，token 使用解释 80% 方差），但代价是
   "about 15× more tokens than chats"，且状态错误会复利、需要 checkpoint/retry [15]。
   **对客服单问题场景，这套重型的收益前提大多不成立。**
3. **LlamaIndex 的三种实现**：AgentWorkflow + `can_handoff_to`（agent 间直接移交）、orchestrator agent
   （专精 agent 作为工具，结果总是返回编排者）、自定义 planner；同时"streaming events as it goes so you
   can keep users informed of progress" [29]。
4. **Pydantic AI 的两种等价物**：delegation（子 agent 作为工具，`usage=ctx.usage` 聚合计费）与
   programmatic hand-off（应用代码决定下一个 agent，用 `message_history` 传递上下文，联合输出类型
   `output_type=FlightDetails | Failed` 表达失败信号）[27]。
5. **Human-in-the-loop 转人工**：
   - Pydantic AI：工具标 `requires_approval=True`，run 以 `DeferredToolRequests` 结束、人工批准后带
     `DeferredToolResults` 恢复；`ToolDenied` 可拒绝并告知模型原因 [23]。这比本项目现在
     "escalate_to_human 工具返回一段文本"更完整——转人工后 conversation 生命周期与 agent 循环的衔接、
     客服坐席的实际操作，都可以挂在这条机制上。
   - LlamaIndex：工具内 `ctx.wait_for_event(HumanResponseEvent, ...)` 暂停，外部
     `send_event` 续跑，支持离线场景序列化 Context [40]。
   - OpenAI：handoff 的 `on_handoff` 回调可携带 `EscalationData(reason=...)` 类结构化信息，
     授权检查必须在回调开头做 [31]。
6. **落地建议口径**（引用 Anthropic 原则）：客服 agent 的复杂度起点应是"单 agent + 少数工具"，
   "only increasing complexity when needed" [4]；先有清晰的转人工通道（HITL），再考虑多专精 agent。

---

## 5. 评测

### 5.1 RAGAS（检索 + 生成质量）

- 论文（arXiv 2309.15217，Es et al.）：RAGAS 是"reference-free evaluation of Retrieval Augmented
  Generation"框架，覆盖"the ability of the retrieval system to identify relevant and focused context
  passages"、LLM 对检索内容的使用忠实度与整体生成质量；免除人工标注以实现"faster evaluation cycles of
  RAG architectures" [33]。
- 官方文档指标体系（类名核实自 docs.ragas.io）[34]：
  - **Faithfulness**（`Faithfulness`）：答案对检索上下文的事实一致性——LLM 把答案拆成 claims 逐条核对
    是否被上下文支持；
  - **Answer/Response Relevancy**（`ResponseRelevancy`）：由答案反向生成问题、比对 embedding 相似度；
  - **Context Precision**（`LLMContextPrecisionWithReference` 等）：检索 chunk 排序的精确率，按排名加权；
  - **Context Recall**（`LLMContextRecall`）：参考答案中的 claims 有多少能归因到检索上下文；
  - 噪声敏感度 `NoiseSensitivity`（无关 chunk 是否导致答错）、实体召回 `ContextEntityRecall`；
  - Agent 维度：`ToolCallAccuracy`、`ToolCallF1`、`AgentGoalAccuracy`、`TopicAdherence`——对"带工具的
    客服 agent"同样可用。
- Anthropic 的工程佐证：其多智能体系统的评测实践是"~20 real-query test cases"起步 + 单次 LLM-as-judge
  （0-1 rubric：factual accuracy、citation accuracy、completeness、source quality、tool efficiency）与
  人工判断最吻合 [15]。

### 5.2 检索质量与幻觉（Chroma RIG 等）

- **Chroma《Evaluating Retrieval Quality in Retrieval-Augmented Generation》（RIG 研究笔记，2024-04）**。
  **来源状态声明**：原 URL `research.trychroma.com/evaluating-retrieval-quality` 现已 301 到
  `trychroma.com/research/evaluating-retrieval-quality` 并返回 404；官网 research 栏目现仅保留
  2024-05 之后的 5 篇（Context-1、Context Rot、Generative Benchmarking、Evaluating Chunking、
  Embedding Adapters）[37]，且本环境无法访问 Wayback 存档。因此以下为该笔记**广为引用的核心结论**，
  未能逐字核对原文，具体数字引用时请以存档为准：
  1. 现有的检索质量估计方法（让 LLM 自评检索质量、用 query-文档 embedding 相似度估计）都不可靠——
     LLM 无法可靠自评"检索结果好不好"；
  2. 提出 **RIG（Retrieval-Informed Generation）协议**：强制 LLM 仅基于检索内容作答，并按结构化格式
     输出每条陈述的 supported/unsupported（含引用），把"检索不足"从隐性幻觉变成显式信号；
  3. 幻觉率与检索质量强相关：检索全为相关内容时幻觉率趋近于 0；混入无关 chunk（distractor）会显著
     推高幻觉率（广被转述的量级：无关上下文下传统 RAG 幻觉率可接近 ~98%，RIG 协议下降到个位数百分比）；
  4. 工程含义：**"度量并保住检索质量"比"增加生成侧花样"更能降低幻觉**，且 RIG 让每次回答都自带
     可校验的引用面。
- **可核验的替代一手来源**（同一问题的学术论文）：
  - Béchard & Ayodele, *Reducing hallucination in structured outputs via RAG*（arXiv 2404.08189）：
    实验显示带 RAG（含引用约束）可"significantly reduces hallucinations in the output"，幻觉率从
    68% 降到 10%（搜索摘要级核实）[38]；
  - Salemi & Zamani, *Evaluating Retrieval Quality in Retrieval-Augmented Generation*（arXiv
    2404.13781，注意与 Chroma 笔记同名）：提出 eRAG——把每个文档单独喂给 LLM、用下游任务表现作为该
    文档的相关性标签，与端到端 RAG 表现的 Kendall τ 相关性提升 0.168-0.494，且最多省 50× 显存 [36]；
  - IRCoT 亦报告交错检索"reduces model hallucination" [11]。

### 5.3 对评测体系的建议口径

上线任何 Agentic RAG 改造前后，用同一套 RAGAS 小集（faithfulness / answer relevancy / context
precision / context recall）[33][34] + 20 条真实客服问题回归 [15]，避免"加了重排/改写但不知道是变好
还是变坏"。

---

## 6. 对本项目的适用性映射（一句话级）

> 现状：单 Agent（Pydantic AI）+ `retrieve_knowledge` 单次稠密检索（Chroma, embedding-3 1024 维）+
> 3 个业务工具 + 直接生成；JDDC QA 对短文本语料；SSE 会话；glm-4-flash。

**性价比高（建议做）**：

1. **混合检索 RRF**：`retrieve_knowledge` 内部换成 `QueryFusionRetriever`（现有 Chroma 稠密 +
   BM25/keyword 稀疏，`retriever_weights` 偏向稀疏），解决 QA 对短文本 + 型号/专有名词的召回 [8][1]。
2. **查询改写**：多轮指代消解（"它什么时候到"→ 完整问题），直接用 QueryFusionRetriever 的 query
   generation（`num_queries` 控制），glm-4-flash 即可胜任 [8][1]。
3. **重排 + 阈值过滤**：`SimilarityPostprocessor`（分数阈值剔除无关块）+ 本地
   `SentenceTransformerRerank`（或 bge-reranker 类）对 top-k 重排；这是综述定义的标准 post-retrieval
   优化 [18][1]。
4. **CRAG-lite 自检重试**：不训练任何模型——检索分数阈值/轻量评估器判"incorrect"→ 重写 query 重试
   一次（LangGraph 教程同款 grade → rewrite 循环，但放在 Pydantic AI 工具或 `ModelRetry` 里实现）→
   仍失败则转人工；符合 CRAG"plug-and-play"定位 [17][6]。
5. **RIG-lite 引用约束**：提示词强制"仅依据检索内容作答 + 标注引用 + 说不清楚就说不清楚"，配合
   `output_validator` 抽查 faithfulness，降低幻觉并让回答可校验 [35][24]。
6. **转人工升级为真 HITL**：`escalate_to_human` 换成 `requires_approval=True` 的 deferred tool，
   人工批准/拒绝走 `DeferredToolRequests`/`DeferredToolResults` 生命周期 [23]。
7. **RAGAS 小评测集**：20-50 条 QA 对 + faithfulness / context precision / recall / answer relevancy，
   作为每次检索改造的回归基线 [33][34]。

**过度设计（不建议做）**：

1. **多跳/迭代检索（IRCoT/SubQuestion 全家桶）**：客服 QA 单跳为主，SubQuestionQueryEngine 的多源
   合成收益覆盖不了其 token 成本 [10][11]。
2. **Self-RAG 训练路线**：需要训练 7B/13B 反思模型，与"调 glm-4-flash API"路线根本冲突 [16]。
3. **Adaptive-RAG 的训练分类器**：按复杂度路由三档管线；单知识库场景用提示词/规则近似即可，不值得
   训练 [14]。
4. **RouterQueryEngine 等框架级路由**：本项目 4 个工具，Pydantic AI 的工具选择本身就是路由 [13][12]。
5. **Auto-merging 层级索引 / Sentence-window**：QA 对本身是问答粒度短文本，无"长文档层级"可合并；
   官方自测收益也仅"roughly the same" [20]。仅当未来引入长文档知识源再考虑 sentence-window [19]。
6. **多专精 agent 编排**：4 工具 + 单知识库，单 agent + Anthropic"最简方案"原则足够；15× token 的
   orchestrator-workers 只有超上下文/超多工具时才回本 [4][15][27]。
7. **Durable Execution（Temporal 等）**：MVP 阶段 SQLite + uvicorn 足够，限制反而多（仅 async、2MB
   payload）[25]。

---

## 7. "Pi agent" 考证（用户提出的路线评估）

**最可能指代**：badlogic（Mario Zechner）的 **pi（pi-mono）**——2025 年下半年走红的极简编码
agent，2026-01 经 Armin Ronacher 专文介绍后热度更高 [42][43][44]。次要可能：中文语境里把
**Pydantic AI** 简称 "pi"（本项目的现有编排栈）。

**pi 是什么**：TypeScript 单仓四包 [42]：

- `pi-ai`：统一多 provider LLM API（含任意 OpenAI 兼容端点、流式、TypeBox 工具、成本跟踪），作者称已用于 7 个生产项目；
- `pi-agent-core`：agent loop（工具执行/校验/事件流）+ Agent 类（状态、消息队列、transport 抽象）；
- `pi-tui`：终端 UI 框架；`pi-coding-agent`：CLI 本体（会话/AGENTS.md/斜杠命令/无头 JSON-RPC 模式）。

设计哲学：默认仅 read/write/edit/bash 4 个工具、系统提示+工具 <1000 token、无权限提示、
**明确拒绝 MCP** [42]。

**适配度评估（结论：不建议为本项目换栈）**：

1. CLI 本体是**编码** agent，不是客服对话 harness；可复用的是 pi-ai / pi-agent-core 两个库；
2. 引入即意味着把后端从 Python/FastAPI/Pydantic AI/LlamaIndex/Chroma 整体重写为 Node，或额外维护
   TS sidecar——增量成本远超收益（对照第 3.5 节的框架迁移成本结论）；
3. pi 的"极简 agent loop、少工具、上下文工程"理念，Python 侧的等价物就是项目现有的 Pydantic AI
   （Anthropic"最简方案"原则同款 [4]）。
4. 若"pi"本意是 Pydantic AI：配套方案 `agentic-rag-plan.md` 正是"Pydantic AI agent + RAG 检索"的深化。

---

## 8. 智谱平台能力现状（2026-09，一手核实）

- **Rerank API 存在且可用**：`POST https://open.bigmodel.cn/api/paas/v4/rerank`；请求体
  `model`（schema 枚举 `rerank`）、`query`（≤4096 字）、`documents`（≤128 条、每条 ≤4096 字）、
  `top_n`、`return_documents`；响应 `results[].index / relevance_score` [45]。
  计费页另列 GLM-rerank、GLM-rerank-pro（0.8 元/千 token）与知识库检索内置重排开关
  （`rerank_status`）[48]——可用模型名以控制台实测为准，API schema 仅列 `rerank`。
- **模型矩阵**：旗舰 GLM-5.x（1M 上下文）；GLM-4.6（200K，官方标注"擅长高级编码、复杂推理与
  **工具调用**"）；**免费档**：GLM-4.7-Flash（200K/128K）、GLM-4.5-Flash（128K/96K）、
  GLM-4-Flash-250414（128K/16K）[46]。本项目现用 glm-4-flash 存在直接免费升级路径。
- **Embedding-3**（8K 上下文）与 **Rerank**（4K 上下文）模型均在列 [46]。
- **平台工具**：Web Search API（返回大模型友好的标题/URL/摘要），可作 CRAG"检索失败分支"的
  中文联网兜底 [47]。

---

## 9. 检索组件与中文评测基准补充

- **混合检索**：LlamaIndex `QueryFusionRetriever` + `BM25Retriever`（包
  `llama-index-retrievers-bm25`）做稀疏×稠密融合，官方指南覆盖 simple fusion / reciprocal rank
  fusion / relative score & dist-based fusion 三篇 [49][50][51][52]。中文注意：BM25 默认按英文
  词切分，需自定义 jieba 分词函数（0.14 版本的参数名实现时核对）。
- **重排**：`FlagEmbeddingReranker`（BAAI bge-reranker 系列 cross-encoder；`bge-reranker-v2-m3`
  多语言）可本地 CPU 运行 [53][54][55]，与智谱 rerank API 二选一或 A/B 对照。
- **中文评测基准**：
  - C-MTEB（随 C-Pack 发布，35 数据集 6 类任务；检索子集含 **EcomRetrieval** 电商 s2p、
    **DuRetrieval**）——选型 embedding/rerank 模型时参考 [56]；
  - DuReader（百度中文 MRC/问答，DuRetrieval 即源于此）[57]；
  - JDDC（本项目语料出处：100 万+多轮电商客服对话，LREC 2020；官方 baseline 为 seq2seq）[58][59]。
    日常回归用 RAGAS 小集（见 5.3）即可，不必引入大基准。

---

## 参考文献

[1] Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey*（Naive/Advanced/Modular 三段式分类）:
https://arxiv.org/abs/2312.10997 （全文 https://arxiv.org/html/2312.10997v5 ）
[2] LlamaIndex 官方文档 *Introduction to RAG*（五阶段、检索器清单）:
https://developers.llamaindex.ai/python/framework/understanding/rag/
[3] Singh et al., *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG*:
https://arxiv.org/abs/2501.09136
[4] Anthropic, *Building effective agents*（workflow vs agent、六种模式、最简方案原则，2024-12-19）:
https://www.anthropic.com/engineering/building-effective-agents
[5] LlamaIndex 官方博客, *Agentic RAG with LlamaIndex*:
https://www.llamaindex.ai/blog/agentic-rag-with-llamaindex-2721b8a49ff6
[6] LangChain 官方教程, *Build a custom RAG agent with LangGraph*（grade/rewrite 自纠错循环）:
https://docs.langchain.com/oss/python/langgraph/agentic-rag
[7] Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*（HyDE）:
https://arxiv.org/abs/2212.10496
[8] LlamaIndex QueryFusionRetriever 源码（RRF k=60、num_queries、FUSION_MODES、retriever_weights）:
https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/fusion_retriever.py
[9] Press et al., *Measuring and Narrowing the Compositionality Gap in Language Models*（Self-Ask）:
https://arxiv.org/abs/2210.03350
[10] LlamaIndex 文档, *Sub-question query engine*:
https://developers.llamaindex.ai/python/examples/query_engine/sub_question_query_engine/
[11] Trivedi et al., *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive
Multi-Step Questions*（IRCoT）: https://arxiv.org/abs/2212.10509
[12] LlamaIndex 文档, *Routers*（LLM/Pydantic Selector、RouterQueryEngine、RouterRetriever）:
https://developers.llamaindex.ai/python/framework/module_guides/querying/router/
[13] Pydantic AI 文档, *Tools*（@agent.tool / tool_plain / Tool / toolsets）:
https://pydantic.dev/docs/ai/tools-toolsets/tools/
[14] Jeong et al., *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through
Question Complexity*（NAACL 2024）: https://arxiv.org/abs/2403.14403
[15] Anthropic, *How we built our multi-agent research system*（orchestrator-workers、token 数据、评测方法）:
https://www.anthropic.com/engineering/built-multi-agent-research-system
[16] Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*:
https://arxiv.org/abs/2310.11511
[17] Yan et al., *Corrective Retrieval Augmented Generation*（CRAG）:
https://arxiv.org/abs/2401.15884
[18] LlamaIndex 文档, *Node Postprocessors*（rerank/过滤/重排器全家桶）:
https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/
[19] LlamaIndex 文档/示例, *Metadata Replacement + Node Sentence Window* 与 *SentenceWindowNodeParser*:
https://developers.llamaindex.ai/python/examples/node_postprocessor/metadatareplacementdemo/
https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/sentence_window/
[20] LlamaIndex 示例, *Auto-merging retrieval*（HierarchicalNodeParser + AutoMergingRetriever）:
https://developers.llamaindex.ai/python/examples/retrievers/auto_merging_retriever/
[21] LlamaIndex 文档, *Agent Memory*（Memory 类、三种 memory block）:
https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
[22] Pydantic AI 文档, *Messages and chat history*（message_history、ModelMessagesTypeAdapter、
run_id/conversation_id、ProcessHistory、sanitize_messages）:
https://pydantic.dev/docs/ai/core-concepts/message-history/
[23] Pydantic AI 文档, *Deferred Tools*（人工审批 / 外部执行、DeferredToolRequests/Results）:
https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/
[24] Pydantic AI 文档, *Output*（output_type、ToolOutput/NativeOutput/PromptedOutput、output_validator、
流式 API）: https://pydantic.dev/docs/ai/core-concepts/output/
[25] Pydantic AI 文档, *Durable Execution (Temporal)*（TemporalDurability、PydanticAIWorkflow、限制清单）:
https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/
[26] Pydantic AI 文档, *Graphs*（pydantic-graph 状态机）:
https://pydantic.dev/docs/ai/graph/graph/
[27] Pydantic AI 文档, *Multi-agent applications*（五级复杂度、delegation、programmatic hand-off）:
https://pydantic.dev/docs/ai/guides/multi-agent-applications/
[28] LlamaIndex 文档, *Agents*（FunctionAgent/AgentWorkflow、ChatMemoryBuffer 默认记忆）:
https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/
[29] LlamaIndex 文档, *Building a multi-agent system*（AgentWorkflow、can_handoff_to、orchestrator 模式）:
https://developers.llamaindex.ai/python/framework/understanding/agent/multi_agent/
[30] LangGraph 官方仓库 redirects.json（旧 crag/self_rag/adaptive_rag/customer-support 教程 → 新 agentic-rag）:
https://github.com/langchain-ai/langgraph/blob/main/docs/redirects.json
[31] OpenAI Agents SDK 文档, *Handoffs*:
https://openai.github.io/openai-agents-python/handoffs/
[32] OpenAI Agents SDK 文档, *Guardrails*:
https://openai.github.io/openai-agents-python/guardrails/
[33] Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*:
https://arxiv.org/abs/2309.15217
[34] RAGAS 官方文档, *Metrics*（Faithfulness/ResponseRelevancy/ContextPrecision/ContextRecall/NoiseSensitivity/ToolCallAccuracy 等类名）:
https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
[35] Chroma Research, *Evaluating Retrieval Quality in Retrieval-Augmented Generation*（RIG 笔记，2024-04；
**原文已下线**，Wayback 存档入口:
https://web.archive.org/web/2024/https://research.trychroma.com/evaluating-retrieval-quality ——
本环境不可达，正文相关数字标注为未能重新核对）:
https://research.trychroma.com/evaluating-retrieval-quality
[36] Salemi & Zamani, *Evaluating Retrieval Quality in Retrieval-Augmented Generation*（eRAG，注意与 [35] 同名）:
https://arxiv.org/abs/2404.13781
[37] Chroma Research 当前栏目（现存 5 篇研究笔记清单，佐证 RIG 笔记已下线）:
https://www.trychroma.com/research
[38] Béchard & Ayodele, *Reducing hallucination in structured outputs via Retrieval-Augmented Generation*:
https://arxiv.org/abs/2404.08189
[39] Cormack et al., *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*
（SIGIR 2009，RRF k=60 原始出处，由 [8] 源码注释引用）:
https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
[40] LlamaIndex 文档, *Human in the loop*（InputRequiredEvent / HumanResponseEvent / ctx.wait_for_event）:
https://developers.llamaindex.ai/python/framework/understanding/agent/human_in_the_loop/
[41] Pydantic AI 文档总入口（ai.pydantic.dev 301 → pydantic.dev/docs/ai）:
https://pydantic.dev/docs/ai/overview/
[42] Mario Zechner, *What I learned building an opinionated and minimal coding agent*（pi 设计哲学、
四包架构、4 工具、拒绝 MCP、pi-ai 生产使用）:
https://mariozechner.at/posts/2025-11-30-pi-coding-agent/
[43] badlogic/pi-mono 仓库（pi-ai / pi-agent-core / pi-tui / pi-coding-agent）:
https://github.com/badlogic/pi-mono
[44] Armin Ronacher, *Pi: The Minimal Agent Within OpenClaw*（2026-01-31）:
https://lucumr.pocoo.org/2026/1/31/pi/
[45] 智谱 BigModel 官方文档, *文本重排序 API*（端点 /paas/v4/rerank、参数与响应 schema）:
https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E6%96%87%E6%9C%AC%E9%87%8D%E6%8E%92%E5%BA%8F
[46] 智谱 BigModel 官方文档, *模型概览*（GLM-5.x/4.6/免费 Flash 档、embedding-3、rerank）:
https://docs.bigmodel.cn/cn/guide/start/model-overview
[47] 智谱 BigModel 官方文档, *联网搜索 Web Search API 使用指南*:
https://docs.bigmodel.cn/cn/guide/tools/web-search
[48] 智谱 BigModel 官方文档, *知识库服务计费*（GLM-rerank / GLM-rerank-pro 0.8 元/千 token、
知识库检索 rerank_status）:
https://docs.bigmodel.cn/cn/guide/tools/knowledge/price
[49] LlamaIndex 指南, *Simple Fusion Retriever*（QueryFusionRetriever 组合多路检索器）:
https://developers.llamaindex.ai/python/framework/integrations/retrievers/simple_fusion/
[50] LlamaIndex 指南, *Reciprocal Rerank Fusion Retriever*（RRF 模式）:
https://developers.llamaindex.ai/python/framework/integrations/retrievers/reciprocal_rerank_fusion/
[51] LlamaIndex 指南, *BM25 Retriever*（llama-index-retrievers-bm25）:
https://developers.llamaindex.ai/python/framework/integrations/retrievers/bm25_retriever/
[52] LlamaIndex API 参考, *QueryFusionRetriever*（FusionMode / num_queries / retriever_weights）:
https://developers.llamaindex.ai/python/framework-api-reference/retrievers/query_fusion/
[53] LlamaIndex API 参考, *FlagEmbeddingReranker*:
https://developers.llamaindex.ai/python/framework-api-reference/postprocessor/flag_embedding_reranker/
[54] BAAI/bge-reranker-large 模型卡（cross-encoder 原理）:
https://huggingface.co/BAAI/bge-reranker-large
[55] FlagOpen/FlagEmbedding 仓库（BGE 系列 + C-MTEB 评测代码）:
https://github.com/FlagOpen/FlagEmbedding
[56] Xiao et al., *C-Pack: Packaged Resources To Advance General Chinese Embedding*（C-MTEB）:
https://arxiv.org/abs/2309.07597
[57] baidu/DuReader 仓库（中文 MRC/问答数据集与 baseline）:
https://github.com/baidu/DuReader
[58] Yang et al., *The JDDC Corpus: A Large-Scale Multi-Turn Chinese Dialogue Dataset for E-commerce
Customer Service*（LREC 2020）: https://arxiv.org/abs/1911.09969
[59] SimonJYang/JDDC-Baseline-Seq2Seq（JDDC 官方 baseline）:
https://github.com/SimonJYang/JDDC-Baseline-Seq2Seq
