# rag/store.py — Backend abstraction: ChromaDB or Qdrant

# Backend abstraction: ChromaDB (default) or Qdrant.
# IMPORTANT: use `import config`, never `from config import ...`

from __future__ import annotations
import hashlib
from pathlib import Path

import config

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


# ── Base class ─────────────────────────────────────────────────────────────────

# Abstract base class. Defines the unified interface.
class RAGStore:
    """Abstract base class. Defines the unified interface."""

    def add_chunks(self, chunks: list[dict]) -> None:
        raise NotImplementedError

    def query(self, embedding: list[float], top_k: int, min_score: float) -> list[dict]:
        raise NotImplementedError

    @classmethod
    def list_stores(cls) -> list[str]:
        raise NotImplementedError

    def exists(self) -> bool:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def delete_store(self) -> None:
        raise NotImplementedError


# ── ChromaDB implementation ────────────────────────────────────────────────────

# ChromaDB backend. Persists locally to disk.
class ChromaRAGStore(RAGStore):
    """ChromaDB backend. Persists locally to disk."""

    def __init__(self, store_name: str):
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb nicht installiert. Bitte: pip install chromadb")
        self.store_name = store_name
        self.store_path = config.RAG_STORES_DIR / store_name / "chroma"
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.store_path))
        self._collection = self._client.get_or_create_collection(
            name=f"medskript_{store_name}",
            metadata={"hnsw:space": "cosine"},
        )

    # Adds chunks to store (upsert — idempotent).
    def add_chunks(self, chunks: list[dict]) -> None:
        """Adds chunks to store (upsert — idempotent)."""
        if not chunks:
            return
        ids, embeddings, documents, metadatas = [], [], [], []
        for c in chunks:
            # IDs: remove special characters
            safe_id = str(c["id"]).replace("/", "_").replace("\\", "_")
            # Metadata: None → "" / 0
            meta = {k: (v if v is not None else ("" if isinstance(v, str) or v is None else 0))
                    for k, v in c.get("metadata", {}).items()}
            # Ensure no None values in meta
            meta = {k: (v if v is not None else "") for k, v in meta.items()}
            ids.append(safe_id)
            embeddings.append(c["embedding"])
            documents.append(c["text"])
            metadatas.append(meta)

        # Batch-weise einfügen (100er Batches)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            self._collection.upsert(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

    def query(self, embedding: list[float], top_k: int, min_score: float) -> list[dict]:
        """Semantische Suche. Gibt sortierte Ergebnis-Dicts zurück.
        Oversampling: Anfrage top_k*3 damit min_score ein echter Cutoff ist, nicht nur ein
        Filter auf einem bereits zu kleinen Ergebnis-Pool.
        """
        try:
            # Mehr anfordern als top_k damit der score-Filter Spielraum hat
            n_request = max(top_k * 3, 10)
            try:
                col_size = self._collection.count()
                n_request = min(n_request, col_size) if col_size > 0 else n_request
            except Exception:
                pass
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=max(n_request, 1),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"❌ ChromaDB query Fehler: {e}")
            return []

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Distance → Score: 0=identisch, 2=maximal → score = 1 - dist/2
            score = max(0.0, 1.0 - dist / 2.0)
            if score < min_score:
                continue
            output.append({
                "id": "",
                "text": doc,
                "score": round(score, 4),
                "source": meta.get("source", ""),
                "page": meta.get("page", 0),
                "type": meta.get("type", "text"),
                "image_path": meta.get("image_path") or None,
                "metadata": meta,
            })
        output.sort(key=lambda x: x["score"], reverse=True)
        # Nach Score-Cutoff auf top_k begrenzen
        return output[:top_k]

    @classmethod
    def list_stores(cls) -> list[str]:
        stores_dir = config.RAG_STORES_DIR
        if not stores_dir.exists():
            return []
        return [d.name for d in stores_dir.iterdir() if d.is_dir() and (d / "chroma").exists()]

    def exists(self) -> bool:
        try:
            return self._collection.count() > 0
        except Exception:
            return False

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def delete_store(self) -> None:
        try:
            self._client.delete_collection(f"medskript_{self.store_name}")
        except Exception:
            pass
        import shutil
        if self.store_path.exists():
            shutil.rmtree(self.store_path, ignore_errors=True)


# ── Qdrant Implementation ─────────────────────────────────────────────────────

class QdrantRAGStore(RAGStore):
    """Qdrant-Backend. Verbindet zu Docker-Container."""

    def __init__(self, store_name: str):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client nicht installiert. Bitte: pip install qdrant-client")
        self.store_name = store_name
        self.collection_name = f"medskript_{store_name}"
        self._client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    def _str_to_int_id(self, s: str) -> int:
        return int(hashlib.md5(s.encode()).hexdigest()[:16], 16)

    def _ensure_collection(self, vector_size: int) -> None:
        try:
            self._client.get_collection(self.collection_name)
        except Exception:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        vector_size = len(chunks[0]["embedding"])
        self._ensure_collection(vector_size)
        points = []
        for c in chunks:
            int_id = self._str_to_int_id(str(c["id"]))
            payload = {**c.get("metadata", {}), "text": c["text"]}
            points.append(PointStruct(
                id=int_id,
                vector=c["embedding"],
                payload=payload,
            ))
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self._client.upsert(collection_name=self.collection_name, points=points[i:i + batch_size])

    def query(self, embedding: list[float], top_k: int, min_score: float) -> list[dict]:
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=top_k,
                score_threshold=min_score,
                with_payload=True,
            )
        except Exception as e:
            print(f"❌ Qdrant query Fehler: {e}")
            return []
        output = []
        for r in results:
            p = r.payload or {}
            output.append({
                "id": str(r.id),
                "text": p.get("text", ""),
                "score": round(r.score, 4),
                "source": p.get("source", ""),
                "page": p.get("page", 0),
                "type": p.get("type", "text"),
                "image_path": p.get("image_path") or None,
                "metadata": p,
            })
        return output

    @classmethod
    def list_stores(cls) -> list[str]:
        if not QDRANT_AVAILABLE:
            return []
        try:
            client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
            cols = client.get_collections().collections
            return [c.name.replace("medskript_", "") for c in cols if c.name.startswith("medskript_")]
        except Exception:
            return []

    def exists(self) -> bool:
        try:
            return self._client.get_collection(self.collection_name).points_count > 0
        except Exception:
            return False

    def count(self) -> int:
        try:
            return self._client.get_collection(self.collection_name).points_count
        except Exception:
            return 0

    def delete_store(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass


# ── Factory ───────────────────────────────────────────────────────────────────

_STORE_CACHE: dict[str, RAGStore] = {}

def create_store(store_name: str, backend: str = None) -> RAGStore:
    """
    Factory-Funktion mit Caching. Erstellt passenden Store je nach Backend
    und verhindert Mehrfach-Instanziierung (Memory-Leak-Prävention).
    """
    backend = backend or config.RAG_BACKEND
    cache_key = f"{backend}_{store_name}"
    
    global _STORE_CACHE
    if cache_key not in _STORE_CACHE:
        match backend:
            case "qdrant":
                _STORE_CACHE[cache_key] = QdrantRAGStore(store_name)
            case _:
                _STORE_CACHE[cache_key] = ChromaRAGStore(store_name)
                
    return _STORE_CACHE[cache_key]


def list_available_stores(backend: str = None) -> list[str]:
    """Listet alle vorhandenen Store-Namen auf."""
    backend = backend or config.RAG_BACKEND
    match backend:
        case "qdrant":
            return QdrantRAGStore.list_stores()
        case _:
            return ChromaRAGStore.list_stores()
