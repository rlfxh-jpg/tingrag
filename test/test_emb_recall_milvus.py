import os
import shutil
import sys

sys.path.append(".")
sys.path.append("tinyrag/searcher/emb_recall")

from emb_retriever import EmbRetriever


def test_emb_recall_milvus():
    base_dir = "tmp_test_milvus"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    retriever = EmbRetriever(index_dim=4, base_dir=base_dir)
    retriever.insert([1.0, 0.0, 0.0, 0.0], "alpha")
    retriever.insert([0.0, 1.0, 0.0, 0.0], "beta")
    retriever.insert([0.0, 0.0, 1.0, 0.0], "gamma")
    retriever.save()

    loaded = EmbRetriever(index_dim=4, base_dir=base_dir)
    loaded.load()
    result = loaded.search([1.0, 0.0, 0.0, 0.0], top_n=2)

    assert len(result) == 2
    assert result[0][1] == "alpha"
    print(result)


if __name__ == "__main__":
    test_emb_recall_milvus()
