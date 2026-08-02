# 📘 Machine Learning Primitives: Complete Reference Manual & Lab Documentation

This comprehensive technical reference guide bridges mathematical theory with concrete code execution for three foundational machine learning algorithms implemented from scratch using NumPy: **K-Nearest Neighbors (KNN)**, **Gradient Descent (1D & 2D)**, and **Principal Component Analysis (PCA)**.

---

## 🗺️ System Architecture Overview

The pipeline processes high-dimensional data matrices and optimization parameters through local ML primitive scripts, outputting high-resolution diagnostic visual assets to trace algorithm execution:

## 🧠 Core Fundamentals Needed to Understand the Code

Before exploring the algorithms, it is essential to understand three foundational computational mechanics that drive the implementation:

### 1. NumPy Broadcasting Rules
When executing element-wise operations like `diffs = points - query`, the operand dimensions often mismatch. For example, `points` has a shape of `(60, 2)` (60 training rows, 2 features), while a single `query` point has a shape of `(2,)`. 

Instead of deploying slow, explicit Python loops, NumPy automatically **broadcasts** the smaller array. It virtually duplicates the `(2,)` row vector down sixty times to create a matching `(60, 2)` array structure, executing the subtraction across all elements
instantly at compiled C-speeds.

### 2. Multi-Dimensional Grid Generation
To build continuous decision boundaries, `np.meshgrid` interpolates 1D axis boundaries into corresponding 2D coordinate matrices (`xx` and `yy`). 
* `xx` contains the X-coordinates for every matrix junction across the canvas layout.
* `yy` tracks the matching vertical Y-coordinates.

Collapsing these matrices using `.ravel()` and joining them via column stacking (`np.c_[xx.ravel(), yy.ravel()]`) maps the entire continuous surface into an explicit two-column lookup table `[[x1, y1], [x2, y2], ...]` readable by vector distance modules.

### 3. Pointer References vs. Deep Memory Copies
In Python, appending an array object to a list (`history.append(point)`) binds a reference pointer to that specific variable index rather than taking a snapshot of the current state. If `point` is modified in a subsequent loop iteration, all logged positions update to the new values retrospectively. 

To preserve chronological history states, the code implements **`point.copy()`**, which allocates a unique slice of physical memory to freeze the values at that specific moment.

---

## 📍 Module 1: K-Nearest Neighbors (KNN)

### 1.1 The Mathematics
KNN is an instance-based classifier that assigns labels to new queries by checking the spatial proximity of local, memorized vector parameters.

#### A. Euclidean Distance ($L_2$ Norm)
Measures the straight-line spatial gap between an established training coordinate $P$ and an incoming query parameter $q$ inside an $n$-dimensional plane:
$$D_{\text{Euclidean}}(P, q) = \Vert P - q \Vert_2 = \sqrt{\sum_{i=1}^{n} (P_i - q_i)^2}$$

#### B. Cosine Similarity
Evaluates directional alignment while ignoring vector magnitudes, yielding an orientation measure derived from the inner angle $\theta$:
$$\text{Sim}_{\text{Cosine}}(a, b) = \cos(\theta) = \frac{a \cdot b}{\Vert a \Vert_2 \Vert b \Vert_2} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \cdot \sqrt{\sum_{i=1}^{n} a_i^2}}$$

#### C. Plurality Voting
Once the indices of the $k$ nearest neighbors are isolated, the final class assignment $\hat{y}$ is selected via a majority vote across an indicator matching matrix $\mathbb{I}$:
$$\hat{y} = \arg\max_{c} \sum_{i \in N_k(q)} \mathbb{I}(y_i = c)$$

### 1.2 Programmatic Implementation (`knn.py`)
```python
import numpy as np

def euclidean_distance(points, query):
    if query.ndim > 1 and query.size == query.shape[-1]:
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
    predictions = [knn_predict(point, X_train, y_train, k) for point in grid_points]
    return np.array(predictions).reshape(xx.shape)
```

---

## 📉 Module 2: Gradient Descent (1D & 2D)

### 2.1 The Mathematics
Gradient Descent is a first-order optimization method designed to minimize a cost surface $f(w)$ by stepping iteratively in the direction of steepest local descent.

#### A. Taylor Expansion Optimization Update Rule
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla f(\mathbf{w}^{(t)})$$
Where $\eta$ represents the scalar **learning rate** hyperparameter.

#### B. 1D Loss Surface Target
The 1D script optimizes a standard parabola bowl shifted away from the origin:
$$f(x) = (x - 3)^2 \implies \frac{df(x)}{dx} = 2(x - 3)$$
The minimum rests at $x = 3$, where the derivative evaluates to a flat slope ($0$).

#### C. 2D Anisotropic Elliptical Surface Target
The 2D function incorporates scaling variance between distinct axes, creating a multi-input partial derivative vector:
$$f_2(x, y) = x^2 + 5y^2 \implies \nabla f_2(x, y) = \begin{bmatrix} \frac{\partial f_2}{\partial x} \\ \frac{\partial f_2}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x \\ 10y \end{bmatrix}$$
Because the $Y$-dimension is scaled $5\times$ steeper than the $X$-axis, updates face sharp directional differences across the topography.

### 2.2 Programmatic Implementation (`gradient_descent.py`)
```python
import numpy as np

def f(x):
    return (x - 3) ** 2

def grad_f(x):
    return 2 * (x - 3)

def gradient_descent_1d(start, lr, steps):
    history = [start]
    x = start
    for _ in range(steps):
        x = x - lr * grad_f(x)
        history.append(x)
    return x, history

def f2(point):
    x, y = point
    return x**2 + 5*y**2

def grad_f2(point):
    x, y = point
    return np.array([2*x, 10*y])

def gradient_descent_2d(start, lr, steps):
    point = np.array(start, dtype=float)
    history = [point.copy()]
    for _ in range(steps):
        point = point - lr * grad_f2(point)
        history.append(point.copy())
    return point, np.array(history)
```

#### Learning Rate ($\eta$) Optimization Dynamics:
1. **Under-correction ($\eta = 0.1$)**: The steps are conservative and steady, smoothly approaching the minimum along the slope.
2. **Over-correction Oscillation ($\eta = 0.6$)**: The steps overshoot the center axis and land on the opposite wall of the bowl, creating an oscillating path that dampens over time.
3. **Explosive Divergence ($\eta = 1.1$)**: The update vector overshoots the target so aggressively that it climbs higher up the opposite wall with each step, causing the tracking logs to explode toward infinity.

---

## 📐 Module 3: Principal Component Analysis (PCA)

### 3.1 The Mathematics
PCA maximizes variance capture across a dataset by rotating coordinate frameworks into decoupled, orthogonal principal components.

#### A. Covariance Center Matrix Scaling
To isolate directional variance from overall translation shifts, column mean matrices $\mu$ are calculated and subtracted from sample rows:
$$X_{\text{centered}} = X - \mu$$

#### B. Singular Value Decomposition (SVD)
The centered data matrix is factored without computing an expensive covariance matrix product ($X^T X$):
$$X_{\text{centered}} = U \Sigma V^T$$
* $U$: Orthogonal matrix mapping row space dependencies.
* $\Sigma$: Diagonal matrix containing singular values $\sigma_i$ that track variance magnitude in descending order.
* $V^T$: Orthogonal matrix whose rows represent the right-singular vectors, serving as the **Principal Components (eigenvectors)**.

#### C. Projection Map and Coordinate Subspace Reconstruction
Compressing to a lower $k$-dimensional workspace extracts the first $k$ rows of $V^T$ to construct the projection matrix $W$:
$$\text{Projected Coordinates} = X_{\text{centered}} W^T$$
Mapping these compressed metrics back onto the primary component axis for visual confirmation uses the inverse transform:
$$\text{Reconstructed Coordinates} = (\text{Projected Coordinates}) W$$

### 3.2 Programmatic Implementation (`pca.py`)
```python
import numpy as np

def pca_via_svd(data, n_components):
    mean = np.mean(data, axis=0)
    centered = data - mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    projected = centered @ components.T

    return projected, components, mean
```

---

## 📊 Environment Configuration & Diagnostic Tracking

### File Automation and Directory Structuring
To automate the assignment workflow, configure a dedicated `/plots` workspace and update the chart saving behavior:

```python
import os
import matplotlib.pyplot as plt

# Creates the tracking directories automatically
os.makedirs('plots', exist_ok=True)
```

To save files cleanly for submission, include `plt.savefig()` directly before calling `plt.show()` in your notebook cells:
```python
# Appendix logic applied across workspace cells
plt.savefig('plots/knn_continuous_boundary.png', bbox_inches='tight', dpi=300)
plt.show()
```
