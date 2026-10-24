import numpy as np

from qec_diverse_circuits.constants import COMPOSITE_FEATURES_SIZE


class KNNCircuitArchive:
    """In-memory k-nearest-neighbor novelty archive, a drop-in replacement
    for the Milvus-backed archive this environment originally used. Keeps
    every inserted vector in a plain numpy array and does brute-force L2
    search -- appropriate at this archive's scale (a few hundred to a few
    thousand entries per training run) and removes the external-service
    dependency (a standalone Gym package shouldn't require users to stand up
    a vector database just to try it).

    Matches MilvusCircuitStorage's public interface exactly
    (search_n_nearest, insert_vector) so DiverseCircuitGenerationEnv's
    storage-access code is unchanged."""

    def __init__(self, persistent=False, dim=COMPOSITE_FEATURES_SIZE):
        self.persistent = persistent
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float64)
        self._ids = []

    def search_n_nearest(self, query_vector, threshold=10):
        query = np.asarray(query_vector[0], dtype=np.float64)

        ids = [-1] * threshold
        distances = [-1] * threshold
        embeddings = []

        n_stored = self._vectors.shape[0]
        if n_stored > 0:
            deltas = self._vectors - query
            squared_l2 = np.einsum("ij,ij->i", deltas, deltas)
            order = np.argsort(squared_l2)[:threshold]
            for i, idx in enumerate(order):
                distances[i] = float(squared_l2[idx]) ** 0.5
                ids[i] = self._ids[idx]
                embeddings.append(self._vectors[idx].tolist())

        filled = min(n_stored, threshold)
        while filled < threshold:
            distances[filled] = 2
            embeddings.append([-1.0] * self.dim)
            filled += 1

        distance_sum = sum(distances)
        farpoints = sum(1 for d in distances if d > 0.01)

        return distance_sum, farpoints, ids, distances, embeddings

    def insert_vector(self, ids, vectors):
        for v in vectors:
            arr = np.asarray(v, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise ValueError("Refusing to insert vector with non-finite values into novelty archive")
            if not np.any(arr):
                raise ValueError("Refusing to insert all-zero (degenerate) vector into novelty archive")

        new_rows = np.asarray(vectors, dtype=np.float64).reshape(-1, self.dim)
        self._vectors = np.vstack([self._vectors, new_rows])
        self._ids.extend(ids)

    def delete(self, key):
        if key not in self._ids:
            return
        idx = self._ids.index(key)
        self._vectors = np.delete(self._vectors, idx, axis=0)
        self._ids.pop(idx)

    def close(self):
        pass
