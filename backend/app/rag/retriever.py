"""检索器入口:hybrid 模式走混合检索(默认),dense 模式保持纯向量旧行为。"""

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import CHROMA_PATH, COLLECTION_NAME, RETRIEVAL_MODE, SIMILARITY_TOP_K
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
    """返回检索器:hybrid(默认)或 dense;索引未构建时抛 RuntimeError。

    两者都暴露 LlamaIndex 的 retrieve/aretrieve(list[NodeWithScore]),
    agent.py 的 retrieve_knowledge 工具对实现无感知。
    """
    if collection_count() == 0:
        raise RuntimeError(
            "知识库索引为空:请先运行 uv run python scripts/prepare_dataset.py "
            "&& uv run python scripts/ingest.py"
        )
    if RETRIEVAL_MODE == "hybrid":
        from app.rag.hybrid import get_hybrid_retriever

        return get_hybrid_retriever()
    store = ChromaVectorStore(chroma_collection=get_collection())
    index = VectorStoreIndex.from_vector_store(store, embed_model=build_embed_model())
    return index.as_retriever(similarity_top_k=top_k or SIMILARITY_TOP_K)
