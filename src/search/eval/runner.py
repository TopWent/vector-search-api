from __future__ import annotations

from dataclasses import dataclass

from ..embedder import Embedder
from ..index import VectorIndex
from .ranking import mean_reciprocal_rank, ndcg_at_k, recall_at_k, reciprocal_rank


@dataclass(frozen=True)
class EvalResult:
    n_queries: int
    n_corpus: int
    k: int
    mrr: float
    ndcg: float
    recall: float


def evaluate(
    embedder: Embedder,
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> EvalResult:
    """Encode corpus and queries, run retrieval, return MRR/nDCG/Recall at k.

    qrels maps query_id to {doc_id: relevance grade}.
    """
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[i] for i in corpus_ids]

    corpus_vecs = embedder.encode(corpus_texts)
    index = VectorIndex(dim=embedder.dim)
    index.add(corpus_ids, corpus_vecs)

    qids = list(queries.keys())
    query_texts = [queries[q] for q in qids]
    query_vecs = embedder.encode(query_texts)

    rr_values: list[float] = []
    ndcg_values: list[float] = []
    recall_values: list[float] = []

    search_k = max(k, 100)

    for qid, qvec in zip(qids, query_vecs, strict=True):
        relevance = qrels.get(qid, {})
        relevant = {d for d, r in relevance.items() if r > 0}
        if not relevant:
            continue

        results = index.search(qvec, k=search_k)
        retrieved = [doc_id for doc_id, _ in results]

        rr_values.append(reciprocal_rank(retrieved, relevant))
        ndcg_values.append(ndcg_at_k(retrieved, relevance, k))
        recall_values.append(recall_at_k(retrieved, relevant, k))

    return EvalResult(
        n_queries=len(rr_values),
        n_corpus=len(corpus),
        k=k,
        mrr=mean_reciprocal_rank(rr_values),
        ndcg=sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0,
        recall=sum(recall_values) / len(recall_values) if recall_values else 0.0,
    )
