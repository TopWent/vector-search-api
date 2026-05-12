from search.embedder import HashEmbedder
from search.eval.runner import evaluate


def _toy_corpus():
    return {
        "d1": "machine learning",
        "d2": "deep learning",
        "d3": "cooking pasta",
        "d4": "italian cuisine",
        "d5": "vector databases",
        "d6": "kubernetes",
    }


def test_evaluate_returns_metric_struct():
    corpus = _toy_corpus()
    queries = {"q1": "machine learning"}
    qrels = {"q1": {"d1": 1}}

    result = evaluate(HashEmbedder(dim=32), corpus, queries, qrels, k=5)

    assert result.n_queries == 1
    assert result.n_corpus == 6
    assert result.k == 5
    assert 0.0 <= result.mrr <= 1.0
    assert 0.0 <= result.ndcg <= 1.0
    assert 0.0 <= result.recall <= 1.0


def test_evaluate_skips_queries_without_relevant_docs():
    corpus = _toy_corpus()
    queries = {"q1": "machine learning", "q2": "nothing relevant"}
    qrels = {"q1": {"d1": 1}, "q2": {}}

    result = evaluate(HashEmbedder(dim=32), corpus, queries, qrels, k=5)
    assert result.n_queries == 1


def test_evaluate_empty_qrels_yields_zero_metrics():
    corpus = _toy_corpus()
    queries = {"q1": "x"}
    qrels = {"q1": {}}

    result = evaluate(HashEmbedder(dim=32), corpus, queries, qrels, k=5)
    assert result.n_queries == 0
    assert result.mrr == 0.0
    assert result.ndcg == 0.0
    assert result.recall == 0.0
