
# 📘 Module 6: Clustering — Learning Notes

> **Goal**: Group similar data points without labels using unsupervised clustering algorithms, understand when to use each method, and learn how to evaluate cluster quality.

---

## 📌 Table of Contents

1. [Clustering Basics](#1-clustering-basics)
2. [K-Means](#2-k-means)
3. [MiniBatch K-Means](#3-minibatch-k-means)
4. [DBSCAN](#4-dbscan)
5. [Agglomerative Clustering](#5-agglomerative-clustering)
6. [MeanShift](#6-meanshift)
7. [OPTICS](#7-optics)
8. [BIRCH](#8-birch)
9. [Cluster Evaluation](#9-cluster-evaluation)

---

## 1. Clustering Basics

### Definition
**Clustering** is an unsupervised learning task where the goal is to group similar data points together **without any labels**. The algorithm discovers natural groupings in the data.

### Common Use Cases
- Customer segmentation
- Document/topic grouping
- Anomaly detection
- Image compression
- Gene expression analysis

### Types of Clustering

| Type | Description | Example Algorithms |
|------|-------------|-------------------|
| **Partitioning** | Divides data into k non-overlapping groups | K-Means, MiniBatch K-Means |
| **Density-based** | Finds clusters of arbitrary shape based on density | DBSCAN, OPTICS |
| **Hierarchical** | Builds a tree of nested clusters | Agglomerative, BIRCH |
| **Mean-shift** | Shifts points toward density peaks | MeanShift |

### Key Concepts
- **No ground truth**: Unlike classification, there are no correct answers to compare against.
- **Feature scaling matters**: Most clustering algorithms are distance-based.
- **Choosing k**: Many algorithms require specifying the number of clusters upfront.

---

## 2. K-Means

### What is K-Means?
Partitions data into **k clusters** by minimizing the within-cluster sum of squares (inertia). Each point is assigned to the nearest centroid.

### How It Works
1. Randomly initialize `k` centroids
2. Assign each point to the nearest centroid
3. Recalculate centroids as the mean of assigned points
4. Repeat steps 2–3 until convergence

### Code Example

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_scaled)

print("Labels:", kmeans.labels_)
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Inertia:", kmeans.inertia_)

# Predict new points
new_points = np.array([[0, 0], [3, 3]])
new_scaled = scaler.transform(new_points)
print("New point clusters:", kmeans.predict(new_scaled))
```

### Elbow Method (Finding Optimal k)

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, "bo-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()
```

---

## 3. MiniBatch K-Means

### What is MiniBatch K-Means?
A faster variant of K-Means that uses **random mini-batches** of data to update centroids. Much faster on large datasets with nearly identical results.

### Code Example

```python
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate larger dataset
X, y_true = make_blobs(n_samples=10000, centers=5, cluster_std=0.80, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mbk = MiniBatchKMeans(n_clusters=5, batch_size=100, random_state=42, n_init=10)
mbk.fit(X_scaled)

print("Labels shape:", mbk.labels_.shape)
print("Cluster centers:\n", mbk.cluster_centers_)
print("Inertia:", mbk.inertia_)
```

### Speed Comparison

```python
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=50000, centers=10, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# K-Means
start = time.time()
KMeans(n_clusters=10, random_state=42, n_init=10).fit(X_scaled)
print(f"K-Means time: {time.time() - start:.2f}s")

# MiniBatch K-Means
start = time.time()
MiniBatchKMeans(n_clusters=10, random_state=42, n_init=10).fit(X_scaled)
print(f"MiniBatch K-Means time: {time.time() - start:.2f}s")
```

---

## 4. DBSCAN

### What is DBSCAN?
**Density-Based Spatial Clustering of Applications with Noise.** Finds clusters of arbitrary shape and identifies outliers as noise.

### Key Parameters
- `eps`: Maximum distance between two points to be considered neighbors
- `min_samples`: Minimum number of points to form a dense region

### How It Works
1. For each point, find neighbors within `eps`
2. If a point has ≥ `min_samples` neighbors, it's a **core point**
3. Connect core points and their neighbors into clusters
4. Points not reachable from any core point are labeled **noise (-1)**

### Code Example

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Generate non-linear data (K-Means struggles here)
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=5)
db.fit(X_scaled)

print("Labels:", db.labels_)
print("Number of clusters:", len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))
print("Number of noise points:", (db.labels_ == -1).sum())
```

### Finding Optimal eps (k-distance graph)

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# k-distance graph (k = min_samples)
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel("5th Nearest Neighbor Distance")
plt.title("k-Distance Graph for Optimal eps")
plt.grid(True)
plt.show()
```

---

## 5. Agglomerative Clustering

### What is Agglomerative Clustering?
A **bottom-up** hierarchical clustering method. Each point starts as its own cluster, and the two closest clusters are merged iteratively.

### Linkage Methods

| Linkage | Distance Between Clusters | When to Use |
|---------|---------------------------|-------------|
| `ward` | Minimizes within-cluster variance | Compact, spherical clusters |
| `complete` | Maximum distance between any two points | Compact clusters |
| `average` | Average distance between all pairs | General purpose |
| `single` | Minimum distance between any two points | Can detect elongated shapes (chaining) |

### Code Example

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

agg = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = agg.fit_predict(X_scaled)

print("Labels:", labels)
print("Unique clusters:", set(labels))
```

### Dendrogram Visualization

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.80, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Compute linkage matrix
Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=15)
plt.title("Agglomerative Clustering Dendrogram")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.show()
```

---

## 6. MeanShift

### What is MeanShift?
A centroid-based algorithm that shifts each point toward the **mode (densest region)** of the data. Does **not** require specifying the number of clusters.

### How It Works
1. For each point, compute the mean of all points within a bandwidth window
2. Shift the point to that mean
3. Repeat until convergence
4. Points that converge to the same location belong to the same cluster

### Code Example

```python
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Estimate bandwidth automatically
bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=200)
print("Estimated bandwidth:", bandwidth)

ms = MeanShift(bandwidth=bandwidth)
ms.fit(X_scaled)

print("Labels:", ms.labels_)
print("Cluster centers:\n", ms.cluster_centers_)
print("Number of clusters:", len(set(ms.labels_)))
```

### Tips
- Use `estimate_bandwidth()` for automatic bandwidth selection.
- Slower than K-Means on large datasets.
- Can find non-spherical clusters.

---

## 7. OPTICS

### What is OPTICS?
**Ordering Points To Identify the Clustering Structure.** An improvement over DBSCAN that works well with **varying density** clusters. Does not require a fixed `eps`.

### Key Parameters
- `min_samples`: Minimum number of neighbors to form a core point
- `xi`: Determines the minimum steepness on the reachability plot to form a cluster
- `cluster_method`: `"xi"` or `"dbscan"`

### Code Example

```python
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np

# Create data with varying density
X1, _ = make_blobs(n_samples=200, centers=[[0, 0]], cluster_std=0.3, random_state=42)
X2, _ = make_blobs(n_samples=100, centers=[[5, 5]], cluster_std=1.0, random_state=42)
X = np.vstack([X1, X2])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

optics = OPTICS(min_samples=5, xi=0.05)
optics.fit(X_scaled)

print("Labels:", optics.labels_)
print("Number of clusters:", len(set(optics.labels_)) - (1 if -1 in optics.labels_ else 0))
print("Noise points:", (optics.labels_ == -1).sum())
```

### Reachability Plot

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X1, _ = make_blobs(n_samples=200, centers=[[0, 0]], cluster_std=0.3, random_state=42)
X2, _ = make_blobs(n_samples=100, centers=[[5, 5]], cluster_std=1.0, random_state=42)
X = np.vstack([X1, X2])
X_scaled = StandardScaler().fit_transform(X)

optics = OPTICS(min_samples=5, xi=0.05)
optics.fit(X_scaled)

# Plot reachability
reachability = optics.reachability_[optics.ordering_]
labels = optics.labels_[optics.ordering_]

plt.figure(figsize=(12, 4))
colors = ["g.", "r.", "b.", "y.", "c."]
for klass, color in zip(range(-1, 5), colors):
    mask = labels == klass
    plt.plot(np.where(mask)[0], reachability[mask], color, markersize=4)
plt.ylabel("Reachability Distance")
plt.title("OPTICS Reachability Plot")
plt.show()
```

---

## 8. BIRCH

### What is BIRCH?
**Balanced Iterative Reducing and Clustering using Hierarchies.** Designed for **very large datasets**. Builds a compact summary (CF Tree) and clusters on that.

### Key Parameters
- `threshold`: Maximum radius of a subcluster
- `branching_factor`: Maximum number of subclusters per node
- `n_clusters`: Number of clusters to extract (or another clustering algo to apply on the leaves)

### Code Example

```python
from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y_true = make_blobs(n_samples=1000, centers=5, cluster_std=0.70, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

brc = Birch(n_clusters=5, threshold=0.5, branching_factor=50)
brc.fit(X_scaled)

print("Labels:", brc.labels_)
print("Number of clusters:", len(set(brc.labels_)))

# Predict new points
import numpy as np
new_points = np.array([[0, 0], [3, 3]])
new_scaled = StandardScaler().fit_transform(X)  # Use same scaler in real code
print("Predicted clusters:", brc.predict(new_points))
```

### BIRCH + Another Clustering Algo

```python
from sklearn.cluster import Birch, KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=5000, centers=10, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Step 1: BIRCH to compress data into subclusters
brc = Birch(n_clusters=None, threshold=0.5)
brc.fit(X_scaled)

# Step 2: Apply KMeans on the subcluster centroids
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans.fit(brc.subcluster_centers_)

# Step 3: Assign original points
labels = kmeans.predict(brc.transform(X_scaled))
print("Labels shape:", labels.shape)
print("Unique clusters:", set(labels))
```

---

## 9. Cluster Evaluation

### Why Evaluate Clustering?
Since there are no labels, we need **internal metrics** to judge cluster quality. If ground truth labels exist, we can also use **external metrics**.

### Evaluation Metrics Overview

| Metric | Type | Range | Best Value | Measures |
|--------|------|-------|------------|----------|
| **Silhouette Score** | Internal | [-1, 1] | Higher | Cohesion vs separation |
| **Calinski-Harabasz** | Internal | [0, ∞) | Higher | Between/within variance ratio |
| **Davies-Bouldin** | Internal | [0, ∞) | Lower | Avg similarity between clusters |
| **Adjusted Rand Index** | External | [-1, 1] | Higher | Agreement with true labels |
| **Normalized Mutual Info** | External | [0, 1] | Higher | Shared information with true labels |

### Code Example: Internal Metrics (No Labels)

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    
    print(f"k={k}  Silhouette={sil:.4f}  Calinski-Harabasz={ch:.2f}  Davies-Bouldin={db:.4f}")
```

### Code Example: External Metrics (With True Labels)

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

print("Adjusted Rand Index:", adjusted_rand_score(y_true, labels))
print("Normalized Mutual Info:", normalized_mutual_info_score(y_true, labels))
print("Fowlkes-Mallows Score:", fowlkes_mallows_score(y_true, labels))
```

### Silhouette Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

sil_values = silhouette_samples(X_scaled, labels)
avg_sil = silhouette_score(X_scaled, labels)

plt.figure(figsize=(8, 5))
y_lower = 10
for i in range(4):
    cluster_sil = sil_values[labels == i]
    cluster_sil.sort()
    size = cluster_sil.shape[0]
    y_upper = y_lower + size
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * size, str(i))
    y_lower = y_upper + 10

plt.axvline(x=avg_sil, color="red", linestyle="--", label=f"Avg = {avg_sil:.3f}")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.title("Silhouette Plot (k=4)")
plt.legend()
plt.show()
```

---

### Full Pipeline: End-to-End Clustering Workflow

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Generate data
X, y_true = make_blobs(n_samples=500, centers=5, cluster_std=0.80, random_state=42)

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit multiple algorithms
models = {
    "KMeans(k=5)": KMeans(n_clusters=5, random_state=42, n_init=10),
    "KMeans(k=3)": KMeans(n_clusters=3, random_state=42, n_init=10),
    "KMeans(k=7)": KMeans(n_clusters=7, random_state=42, n_init=10),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Agglomerative": AgglomerativeClustering(n_clusters=5, linkage="ward"),
}

rows = []
for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).sum()
    
    if n_clusters >= 2:
        sil = silhouette_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
    else:
        sil, ch, db = None, None, None
    
    rows.append({
        "Model": name,
        "Clusters": n_clusters,
        "Noise": noise,
        "Silhouette": sil,
        "Calinski-Harabasz": ch,
        "Davies-Bouldin": db
    })

results = pd.DataFrame(rows)
print(results.round(4))
```

---

### Quick Reference: Choosing an Algorithm

| Scenario | Best Algorithm | Why |
|----------|---------------|-----|
| Known number of clusters, spherical shape | **K-Means** | Fast, simple, works well |
| Very large dataset | **MiniBatch K-Means** / **BIRCH** | Memory efficient |
| Arbitrary cluster shapes | **DBSCAN** / **OPTICS** | Density-based, shape-agnostic |
| Varying cluster density | **OPTICS** | Adapts to local density |
| Hierarchical structure needed | **Agglomerative** | Dendrogram gives structure |
| Unknown number of clusters | **MeanShift** / **DBSCAN** / **OPTICS** | k not required |
| Outlier detection needed | **DBSCAN** / **OPTICS** | Labels outliers as noise |

---

### ✅ Module 6 Checklist
- [x] Understand clustering as unsupervised learning
- [x] Apply K-Means & use the Elbow Method to choose k
- [x] Use MiniBatch K-Means for large datasets
- [x] Use DBSCAN for arbitrary shapes & outlier detection
- [x] Apply Agglomerative Clustering & plot dendrograms
- [x] Use MeanShift for automatic cluster discovery
- [x] Use OPTICS for varying-density data
- [x] Use BIRCH for scalable hierarchical clustering
- [x] Evaluate clusters with Silhouette, Calinski-Harabasz, Davies-Bouldin
- [x] Compare external metrics (ARI, NMI) when labels exist
- [x] Always scale features before distance-based clustering
