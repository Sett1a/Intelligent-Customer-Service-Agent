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
SIMILARITY_TOP_K = 4
# 传给模型的会话历史上限(条数,一条=一轮中的一个角色)
HISTORY_MAX_MESSAGES = 60
