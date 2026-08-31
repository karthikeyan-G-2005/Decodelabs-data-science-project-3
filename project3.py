"""
================================================================================
 PROJECT 3 : UNSUPERVISED LEARNING — CUSTOMER SEGMENTATION
 Architecture :  SCALE -> COMPRESS (PCA) -> CLUSTER (K-Means) -> TRANSLATE
================================================================================
Goal: Use a distance-based algorithm (K-Means) to discover hidden mathematical
groupings in unlabeled retail order data, prove the optimal number of
clusters (Elbow Method + Silhouette Score), and translate the resulting
clusters back into actionable business personas.

Input  : DOC-20260810-WA0013.csv  (raw order-level transactional data)
Output : - customer_features.csv      (engineered customer-level feature table)
         - cluster_assignments.csv    (each customer + assigned cluster)
         - persona_summary.csv        (translated, human-readable personas)
         - elbow_method.png
         - silhouette_scores.png
         - pca_clusters.png
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
INPUT_CSV = "DOC-20260810-WA0013.csv"

# ------------------------------------------------------------------------
# STEP 0 — LOAD RAW DATA
# ------------------------------------------------------------------------
raw = pd.read_csv(INPUT_CSV, parse_dates=["Date"])
print(f"[LOAD] {raw.shape[0]} orders across {raw['CustomerID'].nunique()} customers")

# ------------------------------------------------------------------------
# STEP 1 — FEATURE ENGINEERING (order-level -> customer-level behavior)
# This is the "unlabeled retail data" the slide deck refers to: dozens of
# raw transactional variables that carry no explicit segment label.
# ------------------------------------------------------------------------
snapshot_date = raw["Date"].max() + pd.Timedelta(days=1)
raw["is_coupon"] = raw["CouponCode"].notna().astype(int)
raw["is_bad_outcome"] = raw["OrderStatus"].isin(["Cancelled", "Returned"]).astype(int)

agg = raw.groupby("CustomerID").agg(
    total_spend=("TotalPrice", "sum"),
    avg_order_value=("TotalPrice", "mean"),
    order_count=("OrderID", "count"),
    avg_quantity=("Quantity", "mean"),
    avg_unit_price=("UnitPrice", "mean"),
    avg_items_in_cart=("ItemsInCart", "mean"),
    coupon_usage_rate=("is_coupon", "mean"),
    bad_outcome_rate=("is_bad_outcome", "mean"),
    product_diversity=("Product", "nunique"),
    last_order_date=("Date", "max"),
).reset_index()

agg["recency_days"] = (snapshot_date - agg["last_order_date"]).dt.days
agg = agg.drop(columns=["last_order_date"])

# One-hot ratio features for categorical behavior (payment method, referral
# source, product mix) -> pushes the feature space past 20 columns, exactly
# like the "D > 20 features per customer" scenario in the deck.
cat_ratios = []
for col in ["PaymentMethod", "ReferralSource", "Product"]:
    dummies = pd.get_dummies(raw[[ "CustomerID", col]], columns=[col], prefix=col)
    ratios = dummies.groupby("CustomerID").mean()
    cat_ratios.append(ratios)

features = agg.set_index("CustomerID").join(cat_ratios[0]).join(cat_ratios[1]).join(cat_ratios[2])
features = features.fillna(0)

feature_cols = features.columns.tolist()
print(f"[FEATURES] Engineered {len(feature_cols)} behavioral columns per customer")

X = features.values
customer_ids = features.index.values

# ------------------------------------------------------------------------
# STEP 2 — SCALE (Standardization: z = (x - mean) / std)
# Prevents high-magnitude features (e.g. total_spend in $) from swallowing
# low-magnitude behavioral features (e.g. coupon_usage_rate in [0,1]).
# ------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------------
# STEP 3 — COMPRESS (PCA to the 95% cumulative explained-variance threshold)
# ------------------------------------------------------------------------
pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components = int(np.argmax(cum_var >= 0.95) + 1)
n_components = max(n_components, 2)  # keep at least 2 for 2D visualization
print(f"[PCA] {n_components} components retain {cum_var[n_components-1]*100:.1f}% variance "
      f"(compressed from {X_scaled.shape[1]} raw dimensions)")

pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

# Scree / cumulative variance plot (the "95% Rule" slide)
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="#c0392b")
plt.axhline(0.95, color="#d4a017", linestyle="--", label="95% threshold")
plt.axvline(n_components, color="#2c3e50", linestyle=":", label=f"Chosen k = {n_components}")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("The 95% Rule: Separating Signal from Noise")
plt.legend()
plt.tight_layout()
plt.savefig("pca_variance.png", dpi=150)
plt.close()

# ------------------------------------------------------------------------
# STEP 4 — CLUSTER (K-Means) — prove optimal K with two diagnostic gatekeepers
# ------------------------------------------------------------------------
K_RANGE = range(2, 11)
wcss = []
sil_scores = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca)
    wcss.append(km.inertia_)                       # Within-Cluster Sum of Squares
    sil_scores.append(silhouette_score(X_pca, labels))

# Gatekeeper 1: Elbow Method
plt.figure(figsize=(7, 5))
plt.plot(list(K_RANGE), wcss, marker="o", color="#2980b9")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Diagnostic Gatekeeper 1: The Elbow Method")
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=150)
plt.close()

# Gatekeeper 2: Silhouette Score
plt.figure(figsize=(7, 5))
plt.plot(list(K_RANGE), sil_scores, marker="o", color="#8e44ad")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Diagnostic Gatekeeper 2: The Silhouette Score")
plt.tight_layout()
plt.savefig("silhouette_scores.png", dpi=150)
plt.close()

# Pick K = the value with the highest silhouette score (mathematically proven K)
best_k = list(K_RANGE)[int(np.argmax(sil_scores))]
print(f"[K-SELECTION] Optimal K = {best_k} (silhouette = {max(sil_scores):.3f})")

kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)
features["cluster"] = cluster_labels

# 2D visualization of clusters in PCA space (first two components)
plt.figure(figsize=(7, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", s=25, alpha=0.8)
centers_2d = kmeans.cluster_centers_[:, :2]
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c="black", marker="X", s=200, label="Centroids")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"K-Means Clusters in PCA Space (K={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig("pca_clusters.png", dpi=150)
plt.close()

# ------------------------------------------------------------------------
# STEP 5 — TRANSLATE (reverse-engineer centroids back to human-readable
# original units: inverse PCA transform, then inverse StandardScaler)
# ------------------------------------------------------------------------
centroids_pca = kmeans.cluster_centers_
centroids_scaled = pca.inverse_transform(centroids_pca)
centroids_original = scaler.inverse_transform(centroids_scaled)

centroid_df = pd.DataFrame(centroids_original, columns=feature_cols)
centroid_df.index.name = "cluster"

# Build a business-friendly persona table using the core numeric behaviors
persona_cols = ["total_spend", "avg_order_value", "order_count", "avg_quantity",
                 "coupon_usage_rate", "bad_outcome_rate", "recency_days", "product_diversity"]
persona_table = centroid_df[persona_cols].round(2)
persona_table["customer_count"] = features["cluster"].value_counts().sort_index().values

def label_persona(row):
    if row["avg_order_value"] > persona_table["avg_order_value"].median() and row["coupon_usage_rate"] < persona_table["coupon_usage_rate"].median():
        return "High-Value Loyalist"
    if row["coupon_usage_rate"] > persona_table["coupon_usage_rate"].median() and row["avg_order_value"] < persona_table["avg_order_value"].median():
        return "Budget-Conscious Explorer"
    if row["bad_outcome_rate"] > persona_table["bad_outcome_rate"].median():
        return "At-Risk / High-Return Customer"
    return "Steady Mainstream Shopper"

persona_table["persona_label"] = persona_table.apply(label_persona, axis=1)

print("\n[PERSONA MATRIX]")
print(persona_table.to_string())

# ------------------------------------------------------------------------
# STEP 6 — SAVE OUTPUTS
# ------------------------------------------------------------------------
features.reset_index().to_csv("customer_features.csv", index=False)
features.reset_index()[["CustomerID", "cluster"]].to_csv("cluster_assignments.csv", index=False)
persona_table.to_csv("persona_summary.csv")

print("\n[DONE] Outputs written: customer_features.csv, cluster_assignments.csv, "
      "persona_summary.csv, elbow_method.png, silhouette_scores.png, "
      "pca_variance.png, pca_clusters.png")
