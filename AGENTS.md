# AGENTS.md

## 项目概述 (Project Purpose)

**客服 Agent(智能客服机器人)**:RAG 知识库问答 + 多轮对话(上下文连续)+
工具调用(查订单/查物流/转人工)+ Web 会话管理界面。当前为可运行的 MVP。

## 技术栈 (2026-09-01 已拍板,勿随意更换)

- **Agent 编排**:Pydantic AI 2.x(`pydantic-ai-slim[openai]`)
- **LLM / Embedding**:智谱 GLM(`glm-4-flash` 免费 / `embedding-3` 1024 维),
  OpenAI 兼容端点 `https://open.bigmodel.cn/api/paas/v4/`
- **RAG**:LlamaIndex 0.14(切分/索引/检索)+ Chroma 1.x(本地持久化向量库)
  + BM25 混合检索(rank-bm25/jieba)+ 交叉编码器重排(默认智谱 rerank API;
  设计路线/调优方法/结果台账见 docs/rag-hybrid-design.md,2026-09-03 落地)
- **后端**:FastAPI + SQLite(aiosqlite),SSE 流式对话;uv 管理依赖(Python 3.13)
- **前端**:Vue 3 + Vite + Element Plus + markdown-it/highlight.js
- 决策过程与备选方案见下方「调研结论」;详细启动步骤见 README.md。

## 环境 (Environment)

- 平台:Windows(win32),shell 为 Git Bash。
- 工作区路径包含中文(`客服agent`),Bash 命令引用路径务必加引号。
- **chromadb 1.5.9 的 Windows bug**:持久化目录用含中文的**绝对路径**时,
  写入正常但重新打开报 `Error loading hnsw index`。`app/config.py` 已用
  「相对路径 + `os.chdir(BACKEND_DIR)`」规避,**勿把 CHROMA_PATH 改回绝对路径**。
- Git Bash 下 curl 内联中文会因 GBK 编码导致 400(服务端本身正常);
  测试含中文的接口时用 `printf` 写 UTF-8 临时文件 + `--data-binary @file`。

## 常用命令 (Commands)

```bash
# 后端(在 backend/ 下)
uv sync                                    # 安装依赖
uv run uvicorn app.main:app --port 8000 --reload   # 启动后端
uv run python scripts/prepare_dataset.py   # 下载 JDDC 语料 → 抽取 QA 对 → data/raw/kb.jsonl
uv run python scripts/ingest.py            # 建向量索引(--fake 无 Key 干跑 / --reset 重建)
uv run python scripts/eval_rag.py          # 检索评测:50 题固定种子,Recall/Precision/MRR(--json 落盘)
uv run python scripts/tune_rag.py          # 检索超参网格搜索(缓存双路候选+重排分数,组合秒级)

# 前端(在 frontend/ 下)
npm install
npm run dev        # http://localhost:5173(Vite 代理 /api → 127.0.0.1:8000)
npm run build      # 生产构建验证

# 无 Key 干跑:.env 置 EMBED_FAKE=1 + ingest --fake(检索链路可测,对话仍需 Key)
```

无自动化测试框架;冒烟验证方式:uvicorn 启动 + `/api/health` + 会话 CRUD;
Agent 工具链可用 pydantic_ai 的 TestModel 离线验证(`agent.override(model=TestModel())`,
注意 2.x 的 override 是**同步**上下文管理器)。

## 约定 (Conventions)

- 目录:`backend/app`(应用)、`backend/scripts`(数据脚本)、`frontend/src`。
- 密钥只进 `backend/.env`(已 gitignore);模板是 `backend/.env.example`。
- 生成物不入库:`data/raw/`(下载语料、kb.jsonl)、`data/chroma/`(向量库)、`data/app.db`。
  `data/mock/orders.json` 是演示订单,入库。
- 会话上下文连续的实现:每轮请求从 SQLite 加载该会话全部消息,映射为
  Pydantic AI 的 `message_history`(上限 60 条,见 `HISTORY_MAX_MESSAGES`)。
- API 风格:`/api` 前缀;SSE 事件 `delta` / `citations` / `done` / `error`。
- 依赖版本:pydantic-ai 与 llama-index 迭代快,升级后必须重跑冒烟
  (2.x 的 `OpenAIChatModel` 必须显式传模型名;`BaseEmbedding` 在
  `llama_index.core.embeddings` 下)。
- 检索默认走混合链路 `app/rag/hybrid.py`(BM25×向量加权融合 α=0.5 → 智谱交叉编码器
  重排候选 20 → 返回 5 块);`RETRIEVAL_MODE=dense` 回退旧行为。调参改配置后跑
  `eval_rag.py` 确认 Recall@5 ≥ 0.95。
- 查询拆分(分点检索)默认开启:`retrieve_knowledge` 内先经 `decompose_agent`
  (Pydantic AI 结构化输出)拆检索点,逐点召回后「原句锚定 top-5 + 拆分点新块追加」
  合并;`QUERY_DECOMPOSITION=0` 关闭。合并策略做过 4 版对比,勿改回全局重排序
  (会把单意图 gold 挤出 top-5,详见设计文档 §7.2)。
- 单例锁必须用 `threading.RLock`:`get_hybrid_retriever` 持锁构建时会嵌套调
  `get_corpus` 再加锁,`threading.Lock` 会同线程自死锁(表现为请求永久挂起,勿改回)。
- `backend/eval/results/` 是评测生成物(gitignore);指标口径与调参协议见设计文档 §4。

## 调研结论 (2026-09 网络调研 → 已落地)

目标:构建客服 agent(知识库 RAG 问答 + 多轮对话 + 工具调用 + 渠道接入)。

- **路线 A(低代码一体化)**:自托管 Dify / FastGPT / MaxKB / RAGFlow,适合快速上线。
- **路线 B(代码自建,本项目采用)**:LangGraph 或 Pydantic AI(编排)+ LlamaIndex
  (检索)+ Qdrant/pgvector/Chroma(向量库)+ FastAPI。**最终选型:Pydantic AI +
  LlamaIndex + Chroma + FastAPI + Vue 3**(用户 2026-09-01 拍板)。
- 向量库共识:轻量自托管选 Qdrant/Chroma;已有 PostgreSQL 选 pgvector;超大规模选 Milvus。
- 知识库数据:JDDC 京东客服语料(官方基线仓库脱敏子集,GitHub 直连),仅限学习/研究。

## 注意事项

- 在创建任何新文件或目录前,先 `ls` 确认当前工作区状态,避免覆盖。
- JDDC 语料仅限学习/研究用途,商用需另行获得授权。
- 换 embedding 模型或维度后必须 `ingest.py --reset` 重建索引。
- git 仓库已 init 但尚无提交;首次提交由用户决定。
