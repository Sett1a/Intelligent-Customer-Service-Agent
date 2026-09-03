"""全局配置:环境变量 + 路径。"""

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "glm-4-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
# 本地伪向量干跑开关
EMBED_FAKE = os.getenv("EMBED_FAKE", "0").lower() in ("1", "true", "yes")

DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MOCK_DIR = DATA_DIR / "mock"
DB_PATH = DATA_DIR / "app.db"

# chromadb 1.5.9 在 Windows 上无法从含非 ASCII(中文)的绝对路径重新加载 hnsw 段
# (写入正常、重开即报 "Error loading hnsw index"),因此 Chroma 一律使用相对路径,
# 并把工作目录钉在 backend/,保证该相对路径与启动位置无关。
CHROMA_PATH = Path("data") / "chroma"
os.chdir(BACKEND_DIR)

COLLECTION_NAME = "customer_service_kb"  # Chroma 1.x 要求名称 >= 3 字符
# 传给模型的会话历史上限(条数,一条=一轮中的一个角色)
HISTORY_MAX_MESSAGES = 60

# ---------- 检索(RAG)超参数 ----------
# 检索模式:dense=纯向量(旧行为) | hybrid=BM25×向量 双路召回+融合+交叉编码器重排
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
# dense 模式 / 旧路径:向量召回条数
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "4"))

# 混合召回:两路各自的候选深度与权重(权重同时作用于 RRF 与加权融合)
DENSE_TOPK = int(os.getenv("DENSE_TOPK", "50"))
BM25_TOPK = int(os.getenv("BM25_TOPK", "50"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "1.0"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "1.0"))
# 融合方式:weighted=分数 min-max 归一后加权(调优最优) | rrf=倒数排名融合(名次不敏感)
FUSION = os.getenv("FUSION", "weighted")
RRF_K = int(os.getenv("RRF_K", "60"))  # RRF 平滑常数(标准值 60,仅 rrf 模式使用)
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))  # weighted 模式下稠密路权重

# 重排:zhipu=智谱 rerank API(交叉编码器,推荐) | local=本地 cross-encoder
# (需 uv sync --extra rerank-local,首次运行下载 ONNX 模型) | off=不重排
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "zhipu")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank")  # local 模式下填 cross-encoder 模型名
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))  # 送入重排的融合候选数
HYBRID_FINAL_K = int(os.getenv("HYBRID_FINAL_K", "5"))  # 重排后返回给 LLM 的块数

# 查询拆分(分点检索):用 LLM 子 agent 把问题拆成多个检索点,逐点召回后跨查询融合。
# 拆分失败自动退回整句检索;单意图问题由模型输出空列表,行为等同整句检索。
QUERY_DECOMPOSITION = os.getenv("QUERY_DECOMPOSITION", "1").lower() in ("1", "true", "yes")
DECOMP_MAX_QUERIES = int(os.getenv("DECOMP_MAX_QUERIES", "3"))  # 最多拆出的检索点数
