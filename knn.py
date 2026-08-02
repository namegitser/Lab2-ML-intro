import numpy as np

def euclidean_distance(points, query):
    # Ensure query is flattened properly if passed as a 2D row vector
    if query.ndim > 1 and query.shape[0] == 1:
        query = query.ravel()
    diffs = points - query
    return np.sqrt(np.sum(diffs ** 2, axis=1))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def knn_predict(query, X_train, y_train, k=3):
    distances = euclidean_distance(X_train, query)
    nearest_idx = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_idx]
    values, counts = np.unique(nearest_labels, return_counts=True)
    return values[np.argmax(counts)]

def predict_grid(xx, yy, X_train, y_train, k=3):
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Clean and robust: computes predictions point-by-point safely
    predictions = [knn_predict(point, X_train, y_train, k) for point in grid_points]
    
    return np.array(predictions).reshape(xx.shape)
