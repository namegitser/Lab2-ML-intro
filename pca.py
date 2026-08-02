import numpy as np

def pca_via_svd(data, n_components):
    # Center the data
    mean = np.mean(data, axis=0)
    centered = data - mean

    # Perform SVD
    U, S, Vt = np.linalg.svd(centered)

    # Principal components
    components = Vt[:n_components]

    # Project data
    projected = centered @ components.T

    return projected, components, mean