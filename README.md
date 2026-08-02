# Lab 2: Machine Learning Primitives

A pure Python and NumPy implementation of three foundational machine learning modules: K-Nearest Neighbors (KNN), Gradient Descent (1D & 2D), and Principal Component Analysis (PCA). This project implements the underlying algorithms from scratch without relying on high-level ML frameworks like Scikit-Learn.

## 🚀 Modules Implemented

### 1. K-Nearest Neighbors (KNN)
* Implemented core vector calculations including **Euclidean Distance** and **Cosine Similarity**.
* Features an optimized grid classification method (`predict_grid`) to evaluate coordinate ranges over a meshgrid.
* Outputs include single-point classification visualizations and smooth, continuous decision boundaries generated with `plt.contourf`.

### 2. Gradient Descent
* **1D Optimization**: Traces optimization paths for the loss surface \(f(x) = (x - 3)^2\). Evaluates convergence patterns across conservative, oscillating, and diverging learning rates (\(lr \in \{0.1, 0.6, 1.1\}\)).
* **2D Optimization**: Solves the elliptical loss function \(f_2(x, y) = x^2 + 5y^2\). Plots the precise trajectory step-by-step over structural background contour levels.

### 3. Principal Component Analysis (PCA)
* Deconstructs a high-dimensional, linear 2D trend vector matrix using Singular Value Decomposition (**SVD**).
* Extracts dataset mean alignment values and orthogonal component matrices.
* Overlays principal component variation vectors (PC1 and PC2) as directional arrows directly from the dataset's center mass.
* Maps a complete geometric reconstruction detailing how 2D data compresses orthogonally onto a 1D Principal Direction subspace line.

---

## 📁 Directory Structure

```text
├── ml_primitives/
│   ├── knn.py                     # Euclidean distance & grid neighborhood voting logic
│   ├── gradient_descent.py        # 1D/2D cost evaluation functions & update loops
│   ├── pca.py                     # SVD-based matrix centering & dimensionality reduction
│   ├── ml_primitives_starter.ipynb # Primary analysis & plotting playground notebook
│   ├── Lab2_Instructions.md       # Original project specification metrics
│   ├── requirements.txt           # Environment library setup specifications
│   ├── .gitignore                 # Blocks caching compilation folders and environments
│   └── README.md                  # Project overview documentation
└── plots/                         # Auto-generated verification charts & output metrics
```

---

## 🔧 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com
   cd Lab2-ML-intro
   ```

2. **Initialize a Local Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**:
   Open VS Code, select the active `.venv` kernel interpreter environment, and execute cells inside `ml_primitives_starter.ipynb`.

---

## 📊 Visual Diagnostics
All plotting routines automatically output high-resolution, tightly cropped asset logs saved inside the `/plots` folder upon execution:
* `knn_query_classification.png` & `knn_continuous_boundary.png`
* `gd_1d_loss_comparison.png` & `gd_2d_trajectory.png`
* `pca_component_arrows.png` & `pca_projection_complete_map.png`
