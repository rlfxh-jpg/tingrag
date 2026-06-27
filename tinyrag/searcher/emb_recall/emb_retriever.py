import os
from typing import List

from pymilvus import MilvusClient


class EmbRetriever:
    def __init__(self, index_dim: int, base_dir="data/db/faiss_idx") -> None:
        self.index_dim = index_dim
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.db_path = os.path.join(self.base_dir, "milvus_lite.db")
        self.collection_name = f"index_{self.index_dim}"
        self.client = MilvusClient(self.db_path)
        self._next_id = 0

    def insert(self, emb: list, doc: str):
        self._ensure_collection()
        self.client.insert(
            collection_name=self.collection_name,
            data=[
                {
                    "id": self._next_id,
                    "vector": emb,
                    "text": doc,
                }
            ],
        )
        self._next_id += 1

    def save(self, index_name=""):
        if index_name:
            self.collection_name = index_name
        self._ensure_collection()

    def load(self, index_name=""):
        if index_name:
            self.collection_name = index_name
        self._ensure_collection()
        self._next_id = self._infer_next_id()

    def search(self, embs: list, top_n=5):
        self._ensure_collection()
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[embs],
            limit=top_n,
            output_fields=["text"],
        )
        recall_list = []
        for item in search_res[0]:
            entity = item.get("entity", {})
            recall_list.append((item["id"], entity.get("text"), item["distance"]))
        return recall_list

    def _ensure_collection(self):
        if not self.client.has_collection(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.index_dim,
            )

    def _infer_next_id(self) -> int:
        query_res = self.client.query(
            collection_name=self.collection_name,
            filter="id >= 0",
            output_fields=["id"],
            limit=16384,
        )
        if not query_res:
            return 0
        return max(item["id"] for item in query_res) + 1
