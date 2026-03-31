# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
d = pd.read_csv("CC GENERAL.csv")
d = d.drop("CUST_ID", axis=1)

# %%
d = d.dropna()
skewed_cols = [
    'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
    'CASH_ADVANCE', 'PAYMENTS', 'MINIMUM_PAYMENTS',
    'CREDIT_LIMIT', 'PURCHASES_TRX', 'CASH_ADVANCE_TRX'
]
for col in skewed_cols:
    d[col] = np.log1p(d[col])
for col in d.columns:
    lower = d[col].quantile(0.01)
    upper = d[col].quantile(0.99)
    d[col] = np.clip(d[col], lower, upper)
d1 = (d-d.mean())/(d.std())
d1 = d1.to_numpy(dtype=np.float64)

# %%
plt.figure(figsize=(15,8))
sns.boxplot(data=d1)
plt.xticks(rotation=90)
plt.title("Boxplot after Removing Missing Values + Log Transformation + Winsorization + Standardization")
plt.show()

# %%
def kmeans_objective(X,Centroids,distance_lambda):
    #X = number of sample x feature dimension
    #Centroids = number of clusters x feature dimension
    row,col = X.shape
    n_centroids = Centroids.shape[0]

    #Distances = number of samples x number of clusters/centroids
    Distances = np.zeros((row,n_centroids))

    for i in range(row):
        for j in range(n_centroids):
            Distances[i][j] = distance_lambda(X[i],Centroids[j])
    
    #Assignments = number of samples x  (the index of centroid)
    AssignmentsVector = np.argmin(Distances,axis=1)

    #Distance_from_centroid - number of samples x 1 (Distance from nearest centroid)
    DistancesFromNearestCentroid = np.zeros((row,1))
    for i in range(row):
        DistancesFromNearestCentroid[i] = Distances[i][AssignmentsVector[i]]
    
    total = np.sum(DistancesFromNearestCentroid)

    return total,AssignmentsVector

# %%
def update_centroids(X,Centroids):
    # X = number of samples x (feature dimension + 1 (cluster_index column at the end))
    # Centroids = number of clusters x feature_dimension
    k, n_features = Centroids.shape
    CentroidsNew = np.zeros((k,n_features))

    for i in range(k):
        #ClusterPoints = number of filtered samples x feature dimension
        ClusterPoints = X[X[:, -1] == i, :-1]

        if ClusterPoints.shape[0] > 0:
            CentroidsNew[i] = np.mean(ClusterPoints, axis=0)
        else:
            CentroidsNew[i] = Centroids[i]

    return CentroidsNew

# %%
def event_loop(
    X,
    Centroids,
    distance_lambda,
    convergence_lambda,
    pre_event_lambda=None,
    post_event_lambda=None,
    max_iters=100
):
    #X = number of samples x feature dimension
    #Centroid = number of clusters x feature dimension




    CentroidsPrev = Centroids.copy()

    for iteration in range(max_iters):

        objective_value, Assignments = kmeans_objective(
            X, Centroids, distance_lambda
        )

        if pre_event_lambda is not None:
            pre_event_lambda(X, Centroids, Assignments, objective_value, iteration)

        X_aug = np.hstack((X, Assignments.reshape(-1, 1)))


        CentroidsNew = update_centroids(X_aug, Centroids)


        if post_event_lambda is not None:
            post_event_lambda(X, CentroidsNew, iteration)

        # --- Convergence check ---
        if convergence_lambda(CentroidsPrev, CentroidsNew, iteration):
            Centroids = CentroidsNew
            break

        CentroidsPrev = Centroids
        Centroids = CentroidsNew

    return Centroids, Assignments


# %%
distance_lambda = lambda x, c: np.sum((x - c) ** 2)

def centroid_convergence(distance_lambda,threshold=1e-3):
    def convergence_lambda(prev, new, iteration):
        k = prev.shape[0]
        movement = 0.0 
        for i in range(k):
            movement+=distance_lambda(prev[i],new[i])
        # print(f"Iter {iteration}: centroid movement = {movement}")
        return movement < threshold
    return convergence_lambda


# %%
def history_collector():
    history = {
        "objectives": [],
        "Centroids": [],
        "Assignments":[],
        "iteration": 0
    }
    def update(X, Centroids, Assignments, objective_value, iteration):
        history["objectives"].append(objective_value)
        history["Centroids"].append(Centroids.copy())
        history["Assignments"].append(Assignments.copy())
        history["iteration"] = history["iteration"]+1

    return history,update

# %%
def kmeans_plus_plus_init(X, k, distance_lambda, random_state=None):
    """
    X : (n_samples, n_features)
    k : number of clusters
    distance_lambda : distance function
    random_state : optional int for reproducibility
    """

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    centroids = []

    #Choose first centroid randomly
    first_idx = np.random.choice(n_samples)
    centroids.append(X[first_idx])

    #Choose remaining centroids
    for _ in range(1, k):
        # Compute distance to nearest existing centroid
        distances = np.array([
            min(distance_lambda(x, c) for c in centroids)
            for x in X
        ])

        # Square distances (important!)
        probs = distances ** 2
        probs_sum = probs.sum()

        # Numerical safety
        if probs_sum == 0:
            next_idx = np.random.choice(n_samples)
        else:
            probs /= probs_sum
            next_idx = np.random.choice(n_samples, p=probs)

        centroids.append(X[next_idx])

    return np.array(centroids)


# %%
def run_kmeans_for_k(X, k, distance_lambda):
    # Initialize with k-means++
    Centroids = kmeans_plus_plus_init(X, k, distance_lambda)

    history, update_hook = history_collector()

    CentroidFinal, AssignmentsFinal = event_loop(
        X=X,
        Centroids=Centroids,
        distance_lambda=distance_lambda,
        convergence_lambda=centroid_convergence(distance_lambda),
        pre_event_lambda=update_hook,
        max_iters=50
    )

    final_objective = history["objectives"][-1]

    return final_objective, CentroidFinal, AssignmentsFinal, history

# %%
def compute_pij(X, centroids, beta):

    # Step 1: compute squared distances
    # D[i, j] = ||x_i - theta_j||^2
    X_sq = np.sum(X**2, axis=1, keepdims=True)       # (n, 1)
    C_sq = np.sum(centroids**2, axis=1)              # (k,)
    D = X_sq + C_sq - 2 * X @ centroids.T            # (n, k)

    # Step 2: apply -beta * D
    logits = -beta * D

    # Step 3: numerical stability (row-wise)
    logits = logits - np.max(logits, axis=1, keepdims=True)

    # Step 4: softmax
    exp_logits = np.exp(logits)
    P = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return P

# %%
def compute_pij(X, centroids, beta):

    X_sq = np.sum(X**2, axis=1, keepdims=True)      # (n, 1)
    C_sq = np.sum(centroids**2, axis=1)             # (k,)
    D = X_sq + C_sq - 2 * X @ centroids.T           # (n, k)

    logits = -beta * D

    # numerical stability
    logits = logits - np.max(logits, axis=1, keepdims=True)

    exp_logits = np.exp(logits)
    P = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return P

def compute_entropy(P):
    # avoid log(0)
    eps = 1e-12
    P_safe = np.clip(P, eps, 1.0)
    Hi = -np.sum(P_safe * np.log(P_safe), axis=1)   
    H = np.mean(Hi)

    return Hi, H

# %%
def objective_beta(beta, X, centroids, k):
    # Compute P_ij
    P = compute_pij(d1, centroids, beta)

    # Compute entropy
    Hi, Hk = compute_entropy(P)

    # Normalize entropy
    Hk_norm = Hk / np.log(k)

    return Hk_norm


# %%
def simulated_annealing_beta(X, centroids, k,
                             beta_init=0.1,
                             T_init=1.0,
                             alpha=0.95,
                             n_iter=100):

    beta = beta_init
    best_beta = beta
    best_score = objective_beta(beta, X, centroids, k)

    T = T_init

    for _ in range(n_iter):
    
        beta_new = beta + np.random.normal(0, 0.5)
        beta_new = max(beta_new, 1e-6)  

        score_new = objective_beta(beta_new, X, centroids, k)

        delta = score_new - best_score

        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            beta = beta_new

            if score_new < best_score:
                best_score = score_new
                best_beta = beta

        T *= alpha

    return best_beta, best_score

# %%
k_values = range(2, 30)
objectives = []
results = {}

scores = []
beta = 0.1  # you can tune this
alpha = 0.5



for k in k_values:
    obj, C_final, A_final, hist = run_kmeans_for_k(d1, k, distance_lambda)

    objectives.append(obj)

    results[k] = {
        "centroids": C_final,
        "assignments": A_final,
        "history": hist
    }

    beta_opt, score = simulated_annealing_beta(d1, C_final, k)
    results[k]["best_beta"]=beta_opt
    results[k]["score"]=score
    scores.append(score)


    




# %%
k_list = list(k_values)
beta_list = [results[k]["best_beta"] for k in k_list]

plt.figure()

plt.plot(k_list, beta_list, marker='o', color='purple')

plt.xlabel("k (number of clusters)")
plt.ylabel("Optimal Beta")
plt.title("k vs Optimal Beta (from Simulated Annealing)")
plt.grid(True)

plt.show()

# %%
plt.figure()
plt.plot(k_values, scores,color="purple", marker='o', label='Simulated Annealing Optimized Beta')
plt.xlabel("k")
plt.ylabel("Normalized Entropy")
plt.title("Normalized Entropy vs k (Beta optimized via SA)")
plt.grid(True)
plt.legend()
plt.show()

# %%
f = []

for k in results:
    beta = results[k]["best_beta"]
    score = results[k]["score"]
    f.append((k, beta, score))

f1 = sorted(f, key=lambda x: x[2])
N = 10
b1 = [item[1] for item in f1[:N]]


# %%
curves = []
for beta in b1:
    curve = []
    for k in k_values:
        centroids = results[k]["centroids"]

        P = compute_pij(d1, centroids, beta)
        Hi, Hk = compute_entropy(P)

        e_k = Hk / np.log(k)
        curve.append(e_k)

    curves.append(curve)

median_entropy = np.median(curves, axis=0)



# %%
plt.figure()

# Individual curves (light)
for curve in curves:
    plt.plot(k_values, curve, color='gray', alpha=0.3)

# Mean curve (highlight)
plt.plot(k_values, median_entropy, marker='o', color='blue', label='Median entropy')

plt.xlabel("k")
plt.ylabel("Normalized Entropy")
plt.title(f"Entropy vs k (Top {N} beta values)")
plt.legend()
plt.grid(True)
plt.show()

# %%
optimal_index = np.argmin(median_entropy)
optimal_k_entropy = k_list[optimal_index]
print("Optimal k:", optimal_k_entropy)
print("Minimum entropy:", median_entropy[optimal_index])

# %%
plt.figure(figsize=(8, 5))

plt.plot(list(k_values), objectives, marker="o",color="purple")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Final Objective Value")
plt.grid(True, linestyle=":")

plt.show()


# %%
k_list = np.array(list(k_values))
obj = np.array(objectives)

# Normalize for stability
k_norm = (k_list - k_list.min()) / (k_list.max() - k_list.min())
obj_norm = (obj - obj.min()) / (obj.max() - obj.min())

# Line from first to last
line = np.linspace(obj_norm[0], obj_norm[-1], len(obj_norm))

# Distance from line
distances = np.abs(obj_norm - line)

optimal_index = np.argmax(distances)
optimal_k_elbow = k_list[optimal_index]

print("Optimal k (elbow):", optimal_k_elbow)

# %%
def PCA(X,n_components):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    components = eigenvectors[:, :n_components]
    variance = eigenvalues[:n_components]

    return components,variance

def project(X,components):
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    return np.dot(X_centered,components)

# %%
components, variance = PCA(d1,2)
d2 = project(d1,components)
d2 = np.array(d2)
print("Variance= ",np.real(variance))

# %%
assignments = np.array(results[optimal_k_entropy]["assignments"])
centroids = np.array(results[optimal_k_entropy]["centroids"])
centroids_proj = project(centroids, components)
centroids_proj = np.array(centroids_proj)

# %%
plt.figure()

# Data points
plt.scatter(
    d2[:, 0],
    d2[:, 1],
    c=assignments,
    cmap='tab20',
    s=10,
    alpha=0.7
)

# Centroids (correct coloring)
plt.scatter(
    centroids_proj[:, 0],
    centroids_proj[:, 1],
    c=np.arange(len(centroids_proj)),
    cmap='tab20',
    marker='o',
    # s=120,
    edgecolors='black',
    linewidths=1.5,
    label='Centroids'
)

plt.title(f"Clustering Visualization (k={optimal_k_entropy}, PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# %%
import umap


assignments = np.array(results[optimal_k_entropy]["assignments"])

# UMAP projection
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

X_umap = reducer.fit_transform(d1)

centroids = np.array(results[optimal_k_entropy]["centroids"])
centroids_umap = reducer.transform(centroids)

# Plot
plt.figure()

# Data points
plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=assignments,
    cmap='tab20',
    s=10,
    alpha=0.7
)

# Centroids (colored correctly)
plt.scatter(
    centroids_umap[:, 0],
    centroids_umap[:, 1],
    c=np.arange(len(centroids_umap)),
    cmap='tab20',
    marker='o',
    # s=120,
    edgecolors='black',
    linewidths=1.5,
    label='Centroids'
)

plt.title(f"UMAP Visualization (k={optimal_k_entropy})")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.colorbar(label="Cluster")
plt.legend()
plt.grid(True)
plt.show()

# %%
from sklearn.metrics import silhouette_score, davies_bouldin_score

assignments_elbow = np.array(results[optimal_k_elbow]["assignments"])
sil_kmeans = silhouette_score(d1, assignments_elbow)
dbi_kmeans = davies_bouldin_score(d1, assignments_elbow)

print(f"K-means (Elbow k={optimal_k_elbow})")
print("Silhouette:", sil_kmeans)
print("DB Index:", dbi_kmeans)

# %%


assignments_entropy = np.array(results[optimal_k_entropy]["assignments"])

sil_entropy = silhouette_score(d1, assignments_entropy)
dbi_entropy = davies_bouldin_score(d1, assignments_entropy)

print(f"\nK-means (Entropy k={optimal_k_entropy})")
print("Silhouette:", sil_entropy)
print("DB Index:", dbi_entropy)

# %%
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1.5, min_samples=5)  
labels_db = dbscan.fit_predict(d1)

# %%
mask = labels_db != -1
d1_db = d1[mask]
labels_db = labels_db[mask]

# %%
if len(set(labels_db)) > 1:
    sil_db = silhouette_score(d1_db, labels_db)
    dbi_db = davies_bouldin_score(d1_db, labels_db)

    print("\nDBSCAN")
    print("Silhouette:", sil_db)
    print("DB Index:", dbi_db)
else:
    print("\nDBSCAN: Only one cluster found")

# %%
df = pd.DataFrame({
    "Method": [
        f"K-means (Elbow k={optimal_k_elbow})",
        f"K-means (Entropy k={optimal_k_entropy})",
        "DBSCAN"
    ],
    "Silhouette Score ↑": [sil_kmeans, sil_entropy, sil_db],
    "Davies-Bouldin Index ↓": [dbi_kmeans, dbi_entropy, dbi_db]
})

df = df.round(4)

df.style\
  .format({
      "Silhouette Score ↑": "{:.4f}",
      "Davies-Bouldin Index ↓": "{:.4f}"
  })\
  .set_properties(**{'text-align': 'center'})\
  .set_table_styles([
      {'selector': 'th', 'props': [('text-align', 'center')]}
  ])\
  .highlight_max(subset=["Silhouette Score ↑"], color="green")\
  .highlight_min(subset=["Davies-Bouldin Index ↓"], color="green")


