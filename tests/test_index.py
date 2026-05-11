import numpy as np
import pytest

from search.index import VectorIndex


def _unit(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-12)


def test_empty_search_returns_empty_list():
    idx = VectorIndex(dim=4)
    result = idx.search(_unit([1, 0, 0, 0]), k=5)
    assert result == []


def test_add_then_search_returns_nearest():
    idx = VectorIndex(dim=4)
    vectors = np.stack([
        _unit([1, 0, 0, 0]),
        _unit([0, 1, 0, 0]),
        _unit([0, 0, 1, 0]),
    ])
    idx.add(["a", "b", "c"], vectors)

    result = idx.search(_unit([1, 0, 0, 0]), k=2)
    assert result[0][0] == "a"
    assert result[0][1] == pytest.approx(1.0, abs=1e-5)


def test_search_clamps_k_to_size():
    idx = VectorIndex(dim=4)
    idx.add(["a"], np.stack([_unit([1, 0, 0, 0])]))
    assert len(idx.search(_unit([1, 0, 0, 0]), k=100)) == 1


def test_add_validates_shape():
    idx = VectorIndex(dim=4)
    with pytest.raises(ValueError):
        idx.add(["a"], np.zeros((1, 3), dtype=np.float32))


def test_add_validates_dtype():
    idx = VectorIndex(dim=4)
    with pytest.raises(TypeError):
        idx.add(["a"], np.zeros((1, 4), dtype=np.float64))


def test_search_validates_dtype_and_shape():
    idx = VectorIndex(dim=4)
    idx.add(["a"], np.stack([_unit([1, 0, 0, 0])]))

    with pytest.raises(TypeError):
        idx.search(np.zeros(4, dtype=np.float64), k=1)

    with pytest.raises(ValueError):
        idx.search(np.zeros(3, dtype=np.float32), k=1)


def test_size_grows_with_adds():
    idx = VectorIndex(dim=4)
    assert idx.size == 0
    idx.add(["a", "b"], np.stack([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])]))
    assert idx.size == 2


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "idx.bin"
    idx = VectorIndex(dim=4)
    idx.add(["a", "b"], np.stack([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])]))
    idx.save(path)

    restored = VectorIndex.load(path, dim=4)
    assert restored.size == 2
    result = restored.search(_unit([0, 1, 0, 0]), k=1)
    assert result[0][0] == "b"


def test_load_missing_path_returns_empty_index(tmp_path):
    restored = VectorIndex.load(tmp_path / "missing.bin", dim=4)
    assert restored.size == 0
