import itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra


_PERMS_3 = list(itertools.permutations([0, 1, 2]))  # 6 permutations


def build_graph_from_edges(edge_index: np.ndarray, edge_weight_mm: np.ndarray, n_vertices: int) -> csr_matrix:
    """
    Build undirected graph from edge_index and edge_weight_mm.
    This uses TRUE mesh adjacency edges from the cortical surface.

    Args:
        edge_index: [E, 2] array of edge endpoints
        edge_weight_mm: [E] array of edge weights in mm
        n_vertices: number of vertices

    Returns:
        CSR matrix representing the graph with weights in mm
    """
    u = edge_index[:, 0]
    v = edge_index[:, 1]
    w = edge_weight_mm

    # Make symmetric (undirected graph)
    row = np.concatenate([u, v])
    col = np.concatenate([v, u])
    data = np.concatenate([w, w])

    G = csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))
    return G


def build_knn_geodesic(vertices_m: np.ndarray, k: int = 10) -> csr_matrix:
    """
    Build undirected kNN graph on vertices (in meters), edge weights in mm.
    This approximates geodesic distances on the cortical surface when true adjacency is not available.
    """
    V = np.asarray(vertices_m, dtype=np.float64)
    tree = cKDTree(V)
    dists, nbrs = tree.query(V, k=k + 1)  # include self
    dists = dists[:, 1:]
    nbrs = nbrs[:, 1:]

    n = V.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = nbrs.reshape(-1)
    w_mm = (dists.reshape(-1) * 1000.0).astype(np.float64)

    # make symmetric
    rows2 = np.concatenate([rows, cols])
    cols2 = np.concatenate([cols, rows])
    w2 = np.concatenate([w_mm, w_mm])

    return coo_matrix((w2, (rows2, cols2)), shape=(n, n)).tocsr()


def project_coords_to_vertices(coords: np.ndarray, vertices_m: np.ndarray) -> np.ndarray:
    """
    coords: [K,3] in meters
    returns vertex indices [K]
    """
    tree = cKDTree(vertices_m)
    _, idx = tree.query(np.asarray(coords, dtype=np.float64), k=1)
    return idx.astype(int)


def euclid_mm(vertices_m: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(vertices_m[a] - vertices_m[b]) * 1000.0)


def optimal_assignment(C_true, C_pred, dist_fn, K=3):
    """
    Returns:
      best_mean_dist, best_perm
    """
    best = float("inf")
    best_perm = None
    for perm in _PERMS_3 if K == 3 else itertools.permutations(range(K)):
        d = 0.0
        for i in range(K):
            d += dist_fn(C_true[i], C_pred[perm[i]])
        d /= K
        if d < best:
            best = d
            best_perm = perm
    return best, best_perm


def compute_metrics_one(
    C_true: np.ndarray,
    C_pred: np.ndarray,
    vertices_m: np.ndarray,
    knn_graph: csr_matrix,
    R_mm,
    A0=None,
    thresholds_mm=(5, 10, 15, 20, 30),
):
    """
    Compute permutation-invariant metrics for one sample.
    Uses TRUE mesh-geodesic distances if graph was built from mesh edges.

    - MAE_euc_mm: mean optimal assignment Euclidean error (mm)
    - MAE_geo_mm: mean optimal assignment geodesic error (mm) - TRUE mesh-geodesic if available
    - Err_norm_geo_over_R: mean(d_geo / R_i)
    - Err_excess_geo_mm: mean(max(0, d_geo - R_i))
    - Hit@r for geo and euc
    - err_strong/err_weak (proxy if A0 uniform)

    Note: 'geo' here is shortest path on the graph (TRUE mesh-geodesic if built from edges).
    """
    C_true = np.asarray(C_true, dtype=int)
    C_pred = np.asarray(C_pred, dtype=int)
    K = len(C_true)

    # Compute geodesic distance matrix D [K, K] using dijkstra
    # D[i, j] = geodesic distance from C_true[i] to C_pred[j] in mm
    D_geo = dijkstra(knn_graph, directed=False, indices=C_true)[:, C_pred]  # [K, K] in mm

    # Handle inf/NaN values (disconnected components)
    D_geo = np.where(np.isfinite(D_geo), D_geo, np.inf)

    # Compute Euclidean distance matrix
    def dist_euc(a, b):
        return euclid_mm(vertices_m, a, b)

    D_euc = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            D_euc[i, j] = dist_euc(C_true[i], C_pred[j])

    # Find optimal assignment using geodesic distances
    # For each permutation, compute mean distance
    best_mean_geo = float("inf")
    best_perm_geo = None
    for perm in _PERMS_3 if K == 3 else itertools.permutations(range(K)):
        mean_dist = float(np.mean([D_geo[i, perm[i]] for i in range(K)]))
        if mean_dist < best_mean_geo:
            best_mean_geo = mean_dist
            best_perm_geo = perm

    # Fallback to Euclidean if geodesic assignment failed
    if not np.isfinite(best_mean_geo) or best_perm_geo is None:
        best_mean_euc = float("inf")
        best_perm_euc = None
        for perm in _PERMS_3 if K == 3 else itertools.permutations(range(K)):
            mean_dist = float(np.mean([D_euc[i, perm[i]] for i in range(K)]))
            if mean_dist < best_mean_euc:
                best_mean_euc = mean_dist
                best_perm_euc = perm
        if best_perm_euc is not None:
            best_perm_geo = best_perm_euc
        else:
            # Last resort: use identity permutation
            best_perm_geo = list(range(K))

    # per-source distances (with chosen perm)
    d_geo = np.array([D_geo[i, best_perm_geo[i]] for i in range(K)], dtype=np.float64)
    d_euc = np.array([D_euc[i, best_perm_geo[i]] for i in range(K)], dtype=np.float64)

    mae_geo = float(np.mean(d_geo))
    mae_euc = float(np.mean(d_euc))

    # Radius handling
    R = np.asarray(R_mm, dtype=np.float64).reshape(-1)
    if R.size == 1:
        R = np.full((K,), float(R[0]))
    if R.size != K:
        R = np.full((K,), float(np.mean(R)))
    err_norm = float(np.mean(d_geo / (R + 1e-12)))
    err_excess = float(np.mean(np.maximum(0.0, d_geo - R)))

    # Hit rates
    hits = {}
    for thr in thresholds_mm:
        thr = float(thr)
        hits[f"hit_geo@{int(thr)}mm"] = float(np.mean(d_geo <= thr))
        hits[f"hit_euc@{int(thr)}mm"] = float(np.mean(d_euc <= thr))

    # strong/weak definition:
    # If A0 varies: strong = max A0, weak = min A0
    # Else: use localization difficulty proxy: strong=best localized, weak=worst localized.
    if A0 is not None:
        a = np.asarray(A0, dtype=np.float64).reshape(-1)
        if a.size == 1:
            a = np.full((K,), float(a[0]))
        if a.size != K:
            a = np.full((K,), float(np.mean(a)))
        strong_i = int(np.argmax(a))
        weak_i = int(np.argmin(a))
    else:
        strong_i = int(np.argmin(d_geo))
        weak_i = int(np.argmax(d_geo))

    out = {
        "mae_geo_mm": mae_geo,
        "mae_euc_mm": mae_euc,
        "err_norm_geo_over_R": err_norm,
        "err_excess_geo_mm": err_excess,
        "err_strong_geo_mm": float(d_geo[strong_i]),
        "err_weak_geo_mm": float(d_geo[weak_i]),
    }
    out.update(hits)
    return out


def aggregate_metrics(metrics_list):
    """
    Aggregate metrics from a list of metric dictionaries.
    Filters out None values and empty lists.
    """
    if not metrics_list:
        raise ValueError("metrics_list is empty")

    # Filter out None values
    valid_metrics = [m for m in metrics_list if m is not None]

    if not valid_metrics:
        raise ValueError("No valid metrics in metrics_list (all are None)")

    keys = sorted(valid_metrics[0].keys())

    # Ensure all metrics have the same keys
    for m in valid_metrics:
        if set(m.keys()) != set(keys):
            missing = set(keys) - set(m.keys())
            extra = set(m.keys()) - set(keys)
            raise ValueError(f"Metric keys mismatch. Missing: {missing}, Extra: {extra}")

    return {k: float(np.mean([m[k] for m in valid_metrics])) for k in keys}
