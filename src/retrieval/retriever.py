import time
from dataclasses import dataclass

import numpy as np
import faiss

from src.embeddings.embedding_model import generate_embeddings
from src.vectorstore.faiss_manager import load_index, load_metadata
from src.utils.config import Settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    document: str
    page: int
    chunk_id: str


class Retriever:
    """Retrieve relevant chunks from a FAISS index."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.index = load_index(settings.index_path, settings.hnsw_ef_search)
        self.metadata = load_metadata(settings.metadata_path)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a query with similarity filtering."""
        top_k = top_k or self.settings.top_k
        threshold = threshold if threshold is not None else self.settings.similarity_threshold

        start = time.perf_counter()
        query_embedding = generate_embeddings(
            [query], self.settings.embedding_model, batch_size=1, normalize=True
        )

        distances, indices = self.index.search(query_embedding, top_k)
        elapsed = time.perf_counter() - start
        logger.info("Retrieval completed in %.3fs", elapsed)

        results: list[RetrievedChunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            score = 1.0 / (1.0 + dist)
            if score < threshold:
                continue

            meta = self.metadata[idx]
            results.append(RetrievedChunk(
                text=meta["text"],
                score=score,
                document=meta["document"],
                page=meta["page"],
                chunk_id=meta["id"],
            ))

        logger.info("Retrieved %d chunks above threshold %.2f", len(results), threshold)
        return results

    def retrieve_mmr(
        self,
        query: str,
        top_k: int | None = None,
        candidates: int = 20,
        lambda_mult: float = 0.5,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve with Maximal Marginal Relevance for diversity."""
        top_k = top_k or self.settings.top_k
        threshold = threshold if threshold is not None else self.settings.similarity_threshold

        query_embedding = generate_embeddings(
            [query], self.settings.embedding_model, batch_size=1, normalize=True
        )

        distances, indices = self.index.search(query_embedding, candidates)

        valid = [(d, i) for d, i in zip(distances[0], indices[0]) if i != -1]
        if not valid:
            return []

        candidate_dists, candidate_indices = zip(*valid)
        candidate_embeddings = np.array([
            self._reconstruct(idx) for idx in candidate_indices
        ], dtype=np.float32)

        query_sims = 1.0 / (1.0 + np.array(candidate_dists))

        selected: list[int] = []
        remaining = list(range(len(candidate_indices)))

        for _ in range(min(top_k, len(remaining))):
            if not remaining:
                break

            best_score = -float("inf")
            best_idx = -1

            for r in remaining:
                relevance = query_sims[r]
                if selected:
                    sel_embs = candidate_embeddings[selected]
                    sims = candidate_embeddings[r] @ sel_embs.T
                    redundancy = float(np.max(sims))
                else:
                    redundancy = 0.0

                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = r

            selected.append(best_idx)
            remaining.remove(best_idx)

        results: list[RetrievedChunk] = []
        for s in selected:
            score = query_sims[s]
            if score < threshold:
                continue
            idx = candidate_indices[s]
            meta = self.metadata[idx]
            results.append(RetrievedChunk(
                text=meta["text"],
                score=float(score),
                document=meta["document"],
                page=meta["page"],
                chunk_id=meta["id"],
            ))

        return results

    def _reconstruct(self, idx: int) -> np.ndarray:
        """Reconstruct a vector from the index."""
        return self.index.reconstruct(int(idx))
