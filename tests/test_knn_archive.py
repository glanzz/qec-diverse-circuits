import numpy as np
import pytest
from qec_diverse_circuits.knn_archive import KNNCircuitArchive
from qec_diverse_circuits.constants import COMPOSITE_FEATURES_SIZE


def test_padding_vectors_match_declared_dimension():
    archive = KNNCircuitArchive()
    _, _, _, _, embeddings = archive.search_n_nearest(
        [np.zeros((COMPOSITE_FEATURES_SIZE,)).tolist()]
    )
    for emb in embeddings:
        assert len(emb) == COMPOSITE_FEATURES_SIZE


def test_search_on_empty_archive_returns_all_padding_as_far():
    archive = KNNCircuitArchive()
    distance_sum, farpoints, ids, distances, embeddings = archive.search_n_nearest(
        [np.zeros((COMPOSITE_FEATURES_SIZE,)).tolist()]
    )
    assert farpoints == 10
    assert all(d == 2 for d in distances)
    assert all(i == -1 for i in ids)


def test_insert_then_search_finds_nearest_neighbor():
    archive = KNNCircuitArchive()
    close_vector = np.zeros((COMPOSITE_FEATURES_SIZE,))
    close_vector[0] = 1.0
    far_vector = np.ones((COMPOSITE_FEATURES_SIZE,)) * 5.0

    archive.insert_vector([1, 2], [close_vector.tolist(), far_vector.tolist()])

    query = np.zeros((COMPOSITE_FEATURES_SIZE,))
    _, _, ids, distances, _ = archive.search_n_nearest([query.tolist()])

    assert ids[0] == 1
    assert distances[0] == pytest.approx(1.0, abs=1e-6)


def test_insert_vector_rejects_non_finite_vector():
    archive = KNNCircuitArchive()
    bad_vector = [float("inf")] + [0.0] * (COMPOSITE_FEATURES_SIZE - 1)
    with pytest.raises(ValueError):
        archive.insert_vector([1], [bad_vector])


def test_insert_vector_rejects_all_zero_vector():
    archive = KNNCircuitArchive()
    zero_vector = [0.0] * COMPOSITE_FEATURES_SIZE
    with pytest.raises(ValueError):
        archive.insert_vector([1], [zero_vector])


def test_delete_removes_entry():
    archive = KNNCircuitArchive()
    vector = np.ones((COMPOSITE_FEATURES_SIZE,))
    archive.insert_vector([1], [vector.tolist()])
    assert archive._vectors.shape[0] == 1

    archive.delete(1)
    assert archive._vectors.shape[0] == 0
