from typing import List, Optional

from pymilvus import MilvusClient


class EmbRetriever:
    def __init__(
        self,
        index_dim: int,
        base_dir="data/db/faiss_idx",
        milvus_uri: str = "http://127.0.0.1:19530",
        milvus_token: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.index_dim = index_dim
        self.base_dir = base_dir
        self.collection_name = collection_name or f"index_{self.index_dim}"
        self.client = MilvusClient(uri=milvus_uri, token=milvus_token)
        self._next_id = 0

    def insert(self, emb: list, doc: str):
        self._ensure_collection()
        self._ensure_loaded()
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

    def insert_batch(self, rows: List[dict]):
        if not rows:
            return
        self._ensure_collection()
        self._ensure_loaded()
        self.client.insert(
            collection_name=self.collection_name,
            data=rows,
        )
        self._next_id = max(row["id"] for row in rows) + 1

    def save(self, index_name=""):
        if index_name:
            self.collection_name = index_name
        self._ensure_collection()
        self._ensure_loaded()
        self.client.flush(collection_name=self.collection_name)

    def load(self, index_name=""):
        if index_name:
            self.collection_name = index_name
        self._ensure_collection()
        self._ensure_loaded()
        self._next_id = self._infer_next_id()

    def search(self, embs: list, top_n=5):
        self._ensure_collection()
        self._ensure_loaded()
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

    def _ensure_loaded(self):
        load_state = self.client.get_load_state(collection_name=self.collection_name)
        state = str(load_state.get("state", "")).lower()
        if "loaded" not in state:
            self.client.load_collection(collection_name=self.collection_name)

    def _infer_next_id(self) -> int:
        self._ensure_loaded()
        query_res = self.client.query(
            collection_name=self.collection_name,
            filter="id >= 0",
            output_fields=["id"],
            limit=16384,
        )
        if not query_res:
            return 0
        return max(item["id"] for item in query_res) + 1
