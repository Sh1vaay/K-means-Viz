from flask import Flask, render_template, request, jsonify
from kmeans_logic import (generate_dataset, run_kmeans_step_detailed, run_dbscan,
                          get_dbscan_frames, compute_metrics, compute_elbow_data,
                          compute_k_distance, get_gmm_frames, run_hierarchical,
                          compute_silhouette_plot, compute_cluster_profile,
                          compute_feature_importance, compute_auto_k,
                          compute_pca_projection, compute_parallel_coords,
                          compute_box_plot_data)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    dataset_type = data.get('type', 'blobs')
    k = int(data.get('k', 4))
    points = generate_dataset(centers=k, type=dataset_type)
    return jsonify({'points': points})

@app.route('/step', methods=['POST'])
def step():
    req = request.json
    points = req.get('points')
    k = int(req.get('k', 3))
    centroids = req.get('centroids')
    phase = req.get('phase', 'init')
    distance_metric = req.get('distance_metric', 'euclidean')
    new_centroids, labels, next_phase, inertia, converged, desc, outliers = run_kmeans_step_detailed(
        points, k, centroids, phase, distance_metric=distance_metric)
    return jsonify({
        'centroids': new_centroids, 'labels': labels, 'phase': next_phase,
        'inertia': inertia, 'converged': converged, 'description': desc,
        'outliers': outliers
    })

@app.route('/dbscan', methods=['POST'])
def dbscan():
    req = request.json
    points = req.get('points')
    eps = float(req.get('eps', 0.5))
    min_samples = int(req.get('min_samples', 5))
    labels, n_clusters, n_noise = run_dbscan(points, eps, min_samples)
    return jsonify({'labels': labels, 'n_clusters': n_clusters, 'n_noise': n_noise})

@app.route('/dbscan_frames', methods=['POST'])
def dbscan_frames():
    req = request.json
    points = req.get('points')
    eps = float(req.get('eps', 0.5))
    min_samples = int(req.get('min_samples', 5))
    frames = get_dbscan_frames(points, eps, min_samples)
    return jsonify({'frames': frames})

@app.route('/gmm_frames', methods=['POST'])
def gmm_frames():
    req = request.json
    points = req.get('points')
    n_components = int(req.get('n_components', 4))
    frames = get_gmm_frames(points, n_components)
    return jsonify({'frames': frames})

@app.route('/hierarchical', methods=['POST'])
def hierarchical():
    req = request.json
    points = req.get('points')
    n_clusters = int(req.get('n_clusters', 4))
    method = req.get('linkage_method', 'ward')
    result = run_hierarchical(points, n_clusters, method)
    return jsonify(result)

@app.route('/metrics', methods=['POST'])
def metrics():
    req = request.json
    result = compute_metrics(req.get('points'), req.get('labels'))
    return jsonify(result)

@app.route('/elbow', methods=['POST'])
def elbow():
    req = request.json
    data = compute_elbow_data(req.get('points'), int(req.get('max_k', 10)))
    return jsonify({'data': data})

@app.route('/k_distance', methods=['POST'])
def k_distance():
    req = request.json
    distances = compute_k_distance(req.get('points'), int(req.get('k_neighbors', 5)))
    return jsonify({'distances': distances})

@app.route('/silhouette_plot', methods=['POST'])
def silhouette_plot():
    req = request.json
    result = compute_silhouette_plot(req.get('points'), req.get('labels'))
    return jsonify(result)

@app.route('/cluster_profile', methods=['POST'])
def cluster_profile():
    req = request.json
    result = compute_cluster_profile(req.get('points'), req.get('labels'))
    return jsonify(result)

@app.route('/feature_importance', methods=['POST'])
def feature_importance():
    req = request.json
    result = compute_feature_importance(req.get('points'), req.get('labels'))
    return jsonify(result)

@app.route('/auto_k', methods=['POST'])
def auto_k():
    req = request.json
    result = compute_auto_k(req.get('points'), int(req.get('max_k', 10)))
    return jsonify(result)

@app.route('/pca_projection', methods=['POST'])
def pca_projection():
    req = request.json
    result = compute_pca_projection(req.get('points'), req.get('labels'))
    return jsonify(result)

@app.route('/parallel_coords', methods=['POST'])
def parallel_coords():
    req = request.json
    result = compute_parallel_coords(req.get('points'), req.get('labels'))
    return jsonify(result)

@app.route('/box_plot_data', methods=['POST'])
def box_plot_data():
    req = request.json
    result = compute_box_plot_data(req.get('points'), req.get('labels'))
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
