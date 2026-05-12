import math

from search.eval.ranking import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank_first():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_second():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5


def test_reciprocal_rank_third():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1.0 / 3.0


def test_reciprocal_rank_none():
    assert reciprocal_rank(["a", "b", "c"], {"x"}) == 0.0


def test_reciprocal_rank_empty_retrieved():
    assert reciprocal_rank([], {"a"}) == 0.0


def test_recall_full_set_retrieved():
    assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5


def test_recall_zero_when_no_relevant():
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_recall_zero_when_k_is_zero():
    assert recall_at_k(["a", "b"], {"a"}, k=0) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 2.0 / 3.0


def test_precision_zero_when_no_overlap():
    assert precision_at_k(["a", "b"], {"x", "y"}, k=2) == 0.0


def test_dcg_perfect_first_position():
    assert dcg_at_k(["a"], {"a": 1}, k=1) == 1.0


def test_dcg_second_position_discounted():
    score = dcg_at_k(["x", "a"], {"a": 1}, k=2)
    assert math.isclose(score, 1.0 / math.log2(3))


def test_ndcg_perfect_ranking_returns_one():
    relevance = {"a": 3, "b": 2, "c": 1}
    assert math.isclose(ndcg_at_k(["a", "b", "c"], relevance, k=3), 1.0)


def test_ndcg_reverse_ranking_is_less_than_one():
    relevance = {"a": 3, "b": 2, "c": 1}
    score = ndcg_at_k(["c", "b", "a"], relevance, k=3)
    assert 0.0 < score < 1.0


def test_ndcg_zero_when_nothing_relevant_retrieved():
    relevance = {"a": 1, "b": 1}
    assert ndcg_at_k(["x", "y", "z"], relevance, k=3) == 0.0


def test_ndcg_zero_when_no_grades():
    assert ndcg_at_k(["a", "b"], {}, k=2) == 0.0


def test_ndcg_handles_binary_relevance():
    relevance = {"a": 1, "b": 1, "c": 1}
    full = ndcg_at_k(["a", "b", "c"], relevance, k=3)
    assert math.isclose(full, 1.0)


def test_ndcg_one_relevant_at_top_then_irrelevant():
    relevance = {"a": 1}
    score = ndcg_at_k(["a", "x", "y"], relevance, k=3)
    assert math.isclose(score, 1.0)


def test_mean_reciprocal_rank_average():
    assert mean_reciprocal_rank([1.0, 0.5, 0.0]) == 0.5


def test_mean_reciprocal_rank_empty():
    assert mean_reciprocal_rank([]) == 0.0
