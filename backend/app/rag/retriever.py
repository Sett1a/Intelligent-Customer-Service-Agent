"""检索器:Chroma 持久化索引 → LlamaIndex retriever。"""

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import CHROMA_PATH, COLLECTION_NAME, SIMILARITY_TOP_K
from app.rag.embeddings import build_embed_model


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(COLLECTION_NAME)


def collection_count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0


def get_retriever(top_k: int | None = None):
    """返回 LlamaIndex 异步检索器;索引未构建时抛 RuntimeError。"""
    if collection_count() == 0:
        raise RuntimeError(
            "知识库索引为空:请先运行 uv run python scripts/prepare_dataset.py "
            "&& uv run python scripts/ingest.py"
        )
    store = ChromaVectorStore(chroma_collection=get_collection())
    index = VectorStoreIndex.from_vector_store(store, embed_model=build_embed_model())
    return index.as_retriever(similarity_top_k=top_k or SIMILARITY_TOP_K)
