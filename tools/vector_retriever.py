"""
ChromaDB vector retrieval tool.

Provides the ``VectorRetriever`` class which stores and queries ``SourceItem``
documents using sentence-transformer embeddings.  No external API key is
required — embeddings are computed locally via sentence-transformers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from schemas.report_schema import SourceItem

logger = logging.getLogger(__name__)

_CHROMA_PERSIST_DIR = ".chroma"


class VectorRetriever:
    """Thin async wrapper around a ChromaDB collection backed by local embeddings.

    Uses ``sentence-transformers`` (model ``all-MiniLM-L6-v2`` by default) to
    embed documents before storing them, and to embed queries at retrieval time.
    ChromaDB is persisted to ``.chroma/`` in the working directory so state
    survives restarts.

    Args:
        collection_name: Name of the ChromaDB collection to use or create.
        embedding_model:  Sentence-transformers model identifier.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._collection_name = collection_name
        self._model = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(
            path=_CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorRetriever initialised: collection=%r, model=%r",
            collection_name,
            embedding_model,
        )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def add_documents(self, docs: list[SourceItem]) -> None:
        """Embed and store a list of SourceItem documents in the collection.

        Duplicate URLs are silently skipped (ChromaDB upsert semantics).

        Args:
            docs: Source items to store.
        """
        if not docs:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._add_documents_sync, docs)

    async def search(self, query: str, top_k: int = 5) -> list[SourceItem]:
        """Retrieve the most semantically similar SourceItems for a query.

        Args:
            query: Plain-text search query.
            top_k: Number of results to return (default 5).

        Returns:
            List of SourceItem objects ordered by descending similarity.
            Returns an empty list if the collection is empty or on error.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sync, query, top_k)

    async def clear_collection(self) -> None:
        """Delete and recreate the collection, removing all stored documents."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._clear_collection_sync)

    # ------------------------------------------------------------------
    # Internal synchronous helpers (run inside executor)
    # ------------------------------------------------------------------

    def _add_documents_sync(self, docs: list[SourceItem]) -> None:
        """Synchronous implementation of add_documents."""
        texts = [f"{doc.title}\n{doc.snippet}" for doc in docs]
        embeddings: list[list[float]] = self._model.encode(texts).tolist()
        self._collection.upsert(
            ids=[doc.url for doc in docs],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"url": doc.url, "title": doc.title, "snippet": doc.snippet}
                for doc in docs
            ],
        )
        logger.info(
            "Added %d documents to collection %r", len(docs), self._collection_name
        )

    def _search_sync(self, query: str, top_k: int) -> list[SourceItem]:
        """Synchronous implementation of search."""
        try:
            count = self._collection.count()
            if count == 0:
                logger.warning("Collection %r is empty — returning no results.", self._collection_name)
                return []

            effective_k = min(top_k, count)
            query_embedding: list[float] = self._model.encode([query])[0].tolist()
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=effective_k,
                include=["metadatas"],
            )
            items: list[SourceItem] = []
            for meta in (results.get("metadatas") or [[]])[0]:
                items.append(
                    SourceItem(
                        url=meta.get("url", ""),
                        title=meta.get("title", ""),
                        snippet=meta.get("snippet", ""),
                    )
                )
            logger.info(
                "VectorRetriever returned %d results for query: %r", len(items), query
            )
            return items
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "VectorRetriever search failed for query %r: %s", query, exc
            )
            return []

    def _clear_collection_sync(self) -> None:
        """Synchronous implementation of clear_collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared collection %r", self._collection_name)
