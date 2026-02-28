import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster


def get_gmm_frames(data, n_components=4, max_iter=30):
    """
    Runs GMM with EM, recording frames at each iteration.
    Each frame: {means, covariances, weights, labels, log_likelihood, desc}
    Covariance ellipses are decomposed into angle + semi-axes for JS rendering.
    """
    X = np.array(data)
    n = len(X)
    frames = []

    for it in range(1, max_iter + 1):
        gmm = GaussianMixture(n_components=n_components, max_iter=it, n_init=1,
                              random_state=42, warm_start=False)
        gmm.fit(X)
        labels = gmm.predict(X).tolist()
        ll = float(gmm.score(X) * n)

        # Decompose each covariance into ellipse params
        ellipses = []
        for i in range(n_components):
            cov = gmm.covariances_[i]
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            # 2 * sqrt(eigenvalue) covers ~95% of distribution
            width = 2 * 2 * np.sqrt(eigenvalues[0])
            height = 2 * 2 * np.sqrt(eigenvalues[1])
            ellipses.append({
                'cx': float(gmm.means_[i][0]),
                'cy': float(gmm.means_[i][1]),
                'rx': float(width / 2),
                'ry': float(height / 2),
                'angle': float(angle),
                'weight': float(gmm.weights_[i])
            })

        converged = gmm.converged_
        frames.append({
            'iteration': it,
            'means': gmm.means_.tolist(),
            'ellipses': ellipses,
            'labels': labels,
            'log_likelihood': round(ll, 2),
            'converged': bool(converged),
            'desc': f"EM Iteration {it}: Log-Likelihood = {ll:.1f}" + (" — Converged!" if converged else "")
        })

        if converged:
            break

    return frames


def run_hierarchical(data, n_clusters=4, linkage_method='ward'):
    """
    Runs Agglomerative Clustering and returns:
    - labels
    - dendrogram_data: linkage matrix in a JS-friendly format
    - merge_steps: list of {step, merged, height, size} for animation
    """
    X = np.array(data)
    n = len(X)

    # Compute full linkage matrix
    if linkage_method == 'ward':
        Z = linkage(X, method='ward')
    else:
        Z = linkage(X, method=linkage_method)

    # Get labels for the target n_clusters
    labels = fcluster(Z, t=n_clusters, criterion='maxclust') - 1  # 0-indexed

    # Build merge steps for animation
    merge_steps = []
    for i, row in enumerate(Z):
        merge_steps.append({
            'step': i,
            'left': int(row[0]),
            'right': int(row[1]),
            'distance': round(float(row[2]), 4),
            'size': int(row[3])
        })

    # Build dendrogram-like tree for JS rendering
    # Each node: {id, x (index position), y (height/distance), left, right, is_leaf}
    nodes = []
    # Leaf nodes
    for i in range(n):
        nodes.append({'id': i, 'is_leaf': True, 'x': i, 'y': 0})

    # Internal nodes
    for i, row in enumerate(Z):
        left_id = int(row[0])
        right_id = int(row[1])
        new_id = n + i
        left_node = nodes[left_id]
        right_node = nodes[right_id]
        nodes.append({
            'id': new_id,
            'is_leaf': False,
            'x': (left_node['x'] + right_node['x']) / 2.0,
            'y': round(float(row[2]), 4),
            'left': left_id,
            'right': right_id
        })

    return {
        'labels': labels.tolist(),
        'n_clusters': int(n_clusters),
        'nodes': nodes,
        'merge_steps': merge_steps,
        'max_distance': round(float(Z[-1, 2]), 4) if len(Z) > 0 else 0
    }


def compute_metrics(data, labels):
    """
    Computes Silhouette Score and Davies-Bouldin Index for given labels.
    Returns dict with scores, or None values if not computable.
    """
    X = np.array(data)
    labels_arr = np.array(labels)
    # Filter out noise (-1) for metric computation
    mask = labels_arr >= 0
    X_valid = X[mask]
    labels_valid = labels_arr[mask]
    n_clusters = len(set(labels_valid.tolist()))

    result = {'silhouette': None, 'davies_bouldin': None}

    if n_clusters < 2 or len(X_valid) < n_clusters + 1:
        return result

    try:
        result['silhouette'] = round(float(silhouette_score(X_valid, labels_valid)), 4)
    except Exception:
        pass
    try:
        result['davies_bouldin'] = round(float(davies_bouldin_score(X_valid, labels_valid)), 4)
    except Exception:
        pass

    return result


def compute_silhouette_plot(data, labels):
    """
    Computes per-sample silhouette values for a silhouette plot.
    Returns: {samples: [{index, cluster, value}, ...], avg: float, n_clusters: int}
    """
    from sklearn.metrics import silhouette_samples as _silhouette_samples
    X = np.array(data)
    labels_arr = np.array(labels)
    mask = labels_arr >= 0
    X_valid = X[mask]
    labels_valid = labels_arr[mask]
    n_clusters = len(set(labels_valid.tolist()))

    if n_clusters < 2 or len(X_valid) < n_clusters + 1:
        return {'samples': [], 'avg': 0, 'n_clusters': 0}

    sample_values = _silhouette_samples(X_valid, labels_valid)
    avg = float(np.mean(sample_values))

    # Build per-sample data sorted by cluster then value
    samples = []
    for cluster_id in sorted(set(labels_valid.tolist())):
        cluster_mask = labels_valid == cluster_id
        cluster_vals = sample_values[cluster_mask]
        sorted_indices = np.argsort(cluster_vals)
        for idx in sorted_indices:
            samples.append({
                'cluster': int(cluster_id),
                'value': round(float(cluster_vals[idx]), 4)
            })

    return {
        'samples': samples,
        'avg': round(avg, 4),
        'n_clusters': n_clusters
    }


def compute_cluster_profile(data, labels):
    """
    Computes Mean, Median, Std for each feature per cluster.
    Returns: {clusters: [{id, count, features: [{name, mean, median, std}]}]}
    """
    X = np.array(data)
    labels_arr = np.array(labels)
    mask = labels_arr >= 0
    X_valid = X[mask]
    labels_valid = labels_arr[mask]

    feature_names = ['X', 'Y'] + ([f'F{i}' for i in range(2, X.shape[1])] if X.shape[1] > 2 else [])
    clusters = []

    for cid in sorted(set(labels_valid.tolist())):
        cluster_points = X_valid[labels_valid == cid]
        features = []
        for fi in range(X.shape[1]):
            col = cluster_points[:, fi]
            features.append({
                'name': feature_names[fi] if fi < len(feature_names) else f'F{fi}',
                'mean': round(float(np.mean(col)), 4),
                'median': round(float(np.median(col)), 4),
                'std': round(float(np.std(col)), 4)
            })
        clusters.append({
            'id': int(cid),
            'count': int(len(cluster_points)),
            'features': features
        })

    return {'clusters': clusters}


def compute_feature_importance(data, labels):
    """
    Computes feature importance using between-cluster variance / total variance.
    Higher ratio = feature contributes more to cluster separation.
    Returns: [{name, importance, between_var, within_var}]
    """
    X = np.array(data)
    labels_arr = np.array(labels)
    mask = labels_arr >= 0
    X_valid = X[mask]
    labels_valid = labels_arr[mask]
    n_clusters = len(set(labels_valid.tolist()))

    if n_clusters < 2:
        return []

    feature_names = ['X', 'Y'] + ([f'F{i}' for i in range(2, X.shape[1])] if X.shape[1] > 2 else [])
    result = []

    for fi in range(X_valid.shape[1]):
        col = X_valid[:, fi]
        total_var = float(np.var(col))
        if total_var == 0:
            result.append({'name': feature_names[fi] if fi < len(feature_names) else f'F{fi}',
                           'importance': 0, 'between_var': 0, 'within_var': 0})
            continue

        grand_mean = np.mean(col)
        between_var = 0
        within_var = 0
        for cid in set(labels_valid.tolist()):
            cluster_col = col[labels_valid == cid]
            n_c = len(cluster_col)
            between_var += n_c * (np.mean(cluster_col) - grand_mean) ** 2
            within_var += np.sum((cluster_col - np.mean(cluster_col)) ** 2)

        between_var /= len(col)
        within_var /= len(col)
        importance = between_var / total_var if total_var > 0 else 0

        result.append({
            'name': feature_names[fi] if fi < len(feature_names) else f'F{fi}',
            'importance': round(float(importance), 4),
            'between_var': round(float(between_var), 4),
            'within_var': round(float(within_var), 4)
        })

    return sorted(result, key=lambda x: x['importance'], reverse=True)


def compute_auto_k(data, max_k=10):
    """
    Runs K-Means for K=2..max_k, computes silhouette score for each.
    Returns: {scores: [{k, silhouette}], best_k: int, best_score: float}
    """
    X = np.array(data)
    max_k = min(max_k, len(X) - 1)
    scores = []

    for k_val in range(2, max_k + 1):
        km = KMeans(n_clusters=k_val, n_init=10, max_iter=100, random_state=42)
        km.fit(X)
        try:
            sil = float(silhouette_score(X, km.labels_))
        except Exception:
            sil = 0
        scores.append({'k': k_val, 'silhouette': round(sil, 4)})

    best = max(scores, key=lambda x: x['silhouette'])
    return {
        'scores': scores,
        'best_k': best['k'],
        'best_score': best['silhouette']
    }


def compute_pca_projection(data, labels=None):
    """
    Projects data to 2D using PCA.
    Returns: {points: [[pc1, pc2], ...], explained_variance: [v1, v2], labels: [...]}
    """
    from sklearn.decomposition import PCA
    X = np.array(data)
    pca = PCA(n_components=min(2, X.shape[1]))
    projected = pca.fit_transform(X)
    return {
        'points': projected.tolist(),
        'explained_variance': [round(float(v), 4) for v in pca.explained_variance_ratio_],
        'labels': labels if labels else [-1] * len(X)
    }


def compute_parallel_coords(data, labels):
    """
    Prepares data for parallel coordinates plot.
    Normalises each feature to [0,1]. Returns: {features: [name,...], lines: [{values: [...], cluster: int},...]}
    """
    X = np.array(data)
    labels_arr = np.array(labels)
    n_features = X.shape[1]
    feature_names = ['X', 'Y'] + [f'F{i}' for i in range(2, n_features)]

    # Min-max normalise per feature
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid division by zero
    X_norm = (X - mins) / ranges

    lines = []
    for i in range(len(X)):
        lines.append({
            'values': [round(float(v), 4) for v in X_norm[i]],
            'cluster': int(labels_arr[i]) if i < len(labels_arr) else -1
        })
    return {'features': feature_names[:n_features], 'lines': lines}


def compute_box_plot_data(data, labels):
    """
    Computes box plot statistics (min, q1, median, q3, max) per feature per cluster.
    Returns: {features: [name,...], clusters: [{id, feature_stats: [{min,q1,median,q3,max},...]},...]}
    """
    X = np.array(data)
    labels_arr = np.array(labels)
    mask = labels_arr >= 0
    X_valid = X[mask]
    labels_valid = labels_arr[mask]
    n_features = X.shape[1]
    feature_names = ['X', 'Y'] + [f'F{i}' for i in range(2, n_features)]

    clusters = []
    for cid in sorted(set(labels_valid.tolist())):
        cluster_pts = X_valid[labels_valid == cid]
        stats = []
        for fi in range(n_features):
            col = cluster_pts[:, fi]
            q1, med, q3 = float(np.percentile(col, 25)), float(np.median(col)), float(np.percentile(col, 75))
            stats.append({
                'min': round(float(col.min()), 4),
                'q1': round(q1, 4),
                'median': round(med, 4),
                'q3': round(q3, 4),
                'max': round(float(col.max()), 4)
            })
        clusters.append({'id': int(cid), 'count': int(len(cluster_pts)), 'feature_stats': stats})

    return {'features': feature_names[:n_features], 'clusters': clusters}


def compute_elbow_data(data, max_k=10):
    """
    Runs K-Means for K=1..max_k and returns list of {k, inertia} for elbow plot.
    """
    X = np.array(data)
    max_k = min(max_k, len(X))
    results = []
    for k_val in range(1, max_k + 1):
        km = KMeans(n_clusters=k_val, n_init=10, max_iter=100, random_state=42)
        km.fit(X)
        results.append({'k': k_val, 'inertia': round(float(km.inertia_), 2)})
    return results


def compute_k_distance(data, k_neighbors=5):
    """
    Computes k-th nearest neighbor distance for each point, sorted ascending.
    Used for DBSCAN eps estimation (k-distance plot).
    """
    X = np.array(data)
    k_neighbors = min(k_neighbors, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = distances[:, -1]
    k_distances_sorted = np.sort(k_distances).tolist()
    return [round(d, 4) for d in k_distances_sorted]


def run_dbscan(data, eps=0.5, min_samples=5):
    """
    Runs DBSCAN clustering on the given 2D data.
    Returns: labels, n_clusters, n_noise
    """
    X = np.array(data)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    labels = db.labels_.tolist()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = labels.count(-1)
    return labels, n_clusters, n_noise


def get_dbscan_frames(data, eps=0.5, min_samples=5):
    """
    Manually runs DBSCAN and records animation frames for each step.
    Each frame: {current_idx, neighbors, labels, n_clusters, n_noise, desc}
    """
    from sklearn.neighbors import BallTree

    X = np.array(data)
    n = len(X)
    labels = [-1] * n           # -1 = unvisited/noise
    visited = [False] * n
    cluster_id = 0
    frames = []

    tree = BallTree(X)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        # Find neighbors within eps
        neighbor_indices = tree.query_radius(X[i:i + 1], r=eps)[0].tolist()

        if len(neighbor_indices) < min_samples:
            # Mark as noise (stays -1)
            frames.append({
                'current_idx': i,
                'neighbors': neighbor_indices,
                'labels': labels[:],
                'n_clusters': cluster_id,
                'n_noise': labels.count(-1),
                'desc': f"Point {i}: only {len(neighbor_indices)} neighbors (need {min_samples}). Marked as Noise."
            })
        else:
            # Start a new cluster
            labels[i] = cluster_id
            frames.append({
                'current_idx': i,
                'neighbors': neighbor_indices,
                'labels': labels[:],
                'n_clusters': cluster_id + 1,
                'n_noise': labels.count(-1),
                'desc': f"Point {i}: {len(neighbor_indices)} neighbors found. New cluster {cluster_id} started!"
            })

            # Expand cluster
            seed_set = list(neighbor_indices)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = tree.query_radius(X[q:q + 1], r=eps)[0].tolist()
                    if len(q_neighbors) >= min_samples:
                        # Add new neighbors to seed set
                        for nn in q_neighbors:
                            if nn not in seed_set:
                                seed_set.append(nn)
                if labels[q] == -1:
                    labels[q] = cluster_id

                j += 1

            # Frame after expanding the cluster
            frames.append({
                'current_idx': i,
                'neighbors': seed_set,
                'labels': labels[:],
                'n_clusters': cluster_id + 1,
                'n_noise': labels.count(-1),
                'desc': f"Cluster {cluster_id} fully expanded: {len(seed_set)} points."
            })

            cluster_id += 1

    # Final frame
    frames.append({
        'current_idx': -1,
        'neighbors': [],
        'labels': labels[:],
        'n_clusters': cluster_id,
        'n_noise': labels.count(-1),
        'desc': f"DBSCAN complete! Found {cluster_id} cluster(s) with {labels.count(-1)} noise point(s)."
    })

    return frames

def generate_dataset(n_samples=300, centers=4, type='blobs'):
    """
    Generates a 3D dataset. Returns: list of [x, y, z] points.
    All algorithms work on x,y; z is available for 3D visualization.
    """
    if type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=3,
                          cluster_std=0.60, random_state=0)
    elif type == 'moons':
        X2, y = make_moons(n_samples=n_samples, noise=0.05, random_state=0)
        X2 = X2 * 3
        z = np.sin(X2[:, 0]) * 0.8 + np.random.randn(n_samples) * 0.15
        X = np.column_stack([X2, z])
    elif type == 'circles':
        X2, y = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=0)
        X2 = X2 * 4
        z = np.sqrt(X2[:, 0]**2 + X2[:, 1]**2) * 0.5 + np.random.randn(n_samples) * 0.1
        X = np.column_stack([X2, z])
    elif type == 'aniso':
        X3, y = make_blobs(n_samples=n_samples, centers=centers, n_features=3,
                           cluster_std=0.60, random_state=170)
        transformation = [[0.6, -0.6, 0.2], [-0.4, 0.8, 0.3], [0.1, -0.2, 0.9]]
        X = np.dot(X3, transformation)
    else:
        X = np.random.rand(n_samples, 3) * 10 - 5

    return X.tolist()

def _detect_outliers(X, labels, centroids_arr, threshold=2.0, metric='euclidean'):
    """
    Detects outliers: points whose distance to their assigned centroid
    exceeds the cluster mean distance + threshold * std deviation.
    Returns a list of outlier point indices.
    """
    from scipy.spatial.distance import cdist
    outliers = []
    k = len(centroids_arr)
    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        cluster_points = X[mask]
        center = centroids_arr[cluster_id:cluster_id + 1]
        distances = cdist(cluster_points, center, metric=metric).flatten()
        mean_d = np.mean(distances)
        std_d = np.std(distances)
        cutoff = mean_d + threshold * std_d
        # Map back to global indices
        global_indices = np.where(mask)[0]
        for idx, d in zip(global_indices, distances):
            if d > cutoff:
                outliers.append(int(idx))
    return outliers


def run_kmeans_step_detailed(data, k, centroids=None, phase='init', distance_metric='euclidean'):
    """
    Runs K-means in detailed steps: 'assign' -> 'update' loop.
    Returns: centroids, labels, phase, inertia, converged, description, outliers
    """
    from scipy.spatial.distance import cdist

    # Map user-friendly names to scipy metric names
    metric_map = {'manhattan': 'cityblock', 'euclidean': 'euclidean', 'cosine': 'cosine'}
    distance_metric = metric_map.get(distance_metric, distance_metric)

    X = np.array(data)

    # 1. Initialization
    if not centroids:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=1, random_state=None)
        kmeans.fit(X)
        new_centroids = kmeans.cluster_centers_
        labels = [-1] * len(X)
        return (new_centroids.tolist(), labels, 'assign', 0, False,
                "Initialized K centroids using K-Means++ algorithm.", [])

    current_centroids = np.array(centroids)

    def _assign(pts, cents):
        dist_matrix = cdist(pts, cents, metric=distance_metric)
        return np.argmin(dist_matrix, axis=1)

    def _inertia(pts, lbs, cents):
        total = 0
        for i, point in enumerate(pts):
            center = cents[lbs[i]]
            total += np.sum((point - center) ** 2)
        return total

    # 2. Assignment Phase
    if phase == 'assign':
        labels = _assign(X, current_centroids)
        inertia = _inertia(X, labels, current_centroids)
        outliers = _detect_outliers(X, labels, current_centroids, metric=distance_metric)
        return (current_centroids.tolist(), labels.tolist(), 'update', inertia, False,
                f"Assigned each point to its closest centroid ({distance_metric}).", outliers)

    # 3. Update Phase
    elif phase == 'update':
        labels = _assign(X, current_centroids)
        new_centroids = np.array([X[labels == i].mean(0) if np.sum(labels == i) > 0
                                  else current_centroids[i] for i in range(k)])
        converged = np.allclose(current_centroids, new_centroids, rtol=1e-4)

        inertia = _inertia(X, labels, new_centroids)

        outliers = _detect_outliers(X, labels, new_centroids, metric=distance_metric)
        next_phase = 'assign' if not converged else 'converged'
        desc = "Updated centroids to the mean of their clusters." if not converged else "Centroids stabilized. Converged!"
        return (new_centroids.tolist(), labels.tolist(), next_phase, inertia, converged, desc, outliers)

    return (centroids, [], 'init', 0, False, "Ready to start.", [])
