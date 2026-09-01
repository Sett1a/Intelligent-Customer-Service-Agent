"""构建/重建 Chroma 知识库索引。

用法:
  uv run python scripts/ingest.py            # 真实 embedding-3(需 ZHIPU_API_KEY)
  uv run python scripts/ingest.py --fake     # 干跑:本地伪向量,无需 Key(仅自测链路)
  uv run python scripts/ingest.py --reset    # 重建前清空 collection
  uv run python scripts/ingest.py --limit 200    # 仅索引前 N 条(轻量测试)
"""

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))  # 直接运行脚本时让 `app` 包可导入

KB_PATH = BACKEND_DIR / "data" / "raw" / "kb.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="使用本地伪向量(无需 API Key)")
    parser.add_argument("--reset", action="store_true", help="重建前清空 collection")
    args = parser.parse_args()

    import chromadb
    from llama_index.core import (
        Document,
        Settings,
        StorageContext,
        VectorStoreIndex,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from app.config import CHROMA_PATH, COLLECTION_NAME
    from app.rag.embeddings import build_embed_model

    if not KB_PATH.exists():
        print(f"知识库源文件不存在:{KB_PATH}")
        print("请先运行:uv run python scripts/prepare_dataset.py")
        return 1

    docs = []
    with KB_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            docs.append(
                Document(
                    text=f"问题:{item['question']}\n回答:{item['answer']}",
                    metadata={
                        "source": item.get("source", "JDDC"),
                        "question": item["question"],
                        "qid": item.get("id"),
                    },
                )
            )
    print(f"[load] {len(docs)} 条知识文档")

    embed_model = build_embed_model(fake=args.fake)
    Settings.embed_model = embed_model

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if args.reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[reset] 已清空旧索引")
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION_NAME)

    store = ChromaVectorStore(chroma_collection=col)
    storage_context = StorageContext.from_defaults(vector_store=store)
    VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=[SentenceSplitter(chunk_size=400, chunk_overlap=50)],
        show_progress=True,
    )
    print(f"[done] 索引完成,collection '{COLLECTION_NAME}' 共 {col.count()} 个向量块")
    client.close()  # 优雅关闭:等待后台压缩落盘,否则 Windows 下重开索引会损坏
    return 0


if __name__ == "__main__":
    sys.exit(main())
