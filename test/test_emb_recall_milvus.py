import os
import shutil
import sys

sys.path.append(".")
sys.path.append("tinyrag/searcher/emb_recall")
sys.path.append("tinyrag/embedding")
sys.path.append("tinyrag/parser")

from emb_retriever import EmbRetriever
from hf_emb import HFSTEmbedding
from txt_parser import TXTParser


def test_emb_recall_milvus():
    base_dir = "test_milvus"
    batch_size = 200
    max_chunks = 3000
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    file_path = "data/醉枕江山.txt"
    emb_model = HFSTEmbedding(path="models/bge-small-zh-v1.5")
    txt_parser = TXTParser(
        file_path=file_path,
        model=emb_model,
        chunk_size=250,
        chunk_overlap=40,
    )
    parsed_items = txt_parser.parse()

    assert parsed_items is not None
    assert len(parsed_items) > 0

    retriever = EmbRetriever(index_dim=512, base_dir=base_dir)
    rows = []
    for idx, item in enumerate(parsed_items):
        rows.append(
            {
                "id": idx,
                "vector": item["embedding"],
                "text": item["content"],
            }
        )
        if len(rows) >= batch_size:
            print('batch'+str(idx))
            retriever.insert_batch(rows)
            retriever.client.flush(collection_name=retriever.collection_name)
            rows = []
    if rows:
        retriever.insert_batch(rows)
        retriever.client.flush(collection_name=retriever.collection_name)
    retriever.save()

    loaded = EmbRetriever(index_dim=512, base_dir=base_dir)
    loaded.load()
    query = "太平公主"
    query_emb = emb_model.get_embedding(query)
    result = loaded.search(query_emb, top_n=30)

    assert len(result) > 0
    print(f"loaded_chunks: {len(rows)} / total_chunks: {len(parsed_items)}")
    for idx, text, score in result:
        print(f"id={idx} score={score}")
        print(text)
        print()


if __name__ == "__main__":
    test_emb_recall_milvus()
