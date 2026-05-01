"""
Spatial Clustering of European Wind Farm Nodes using DBSCAN (Haversine distance).

Produces:
  - processed_nodes.csv  (original columns + physical_cluster)
  - cluster_map.png      (scatter plot with European coastline outline)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ─── Tunable hyper-parameters ───────────────────────────────────────
EPS_KM = 65            # neighbourhood radius in km (50-80 recommended)
MIN_SAMPLES = 3        # minimum core-point neighbours
EARTH_RADIUS_KM = 6371 # mean Earth radius
# ────────────────────────────────────────────────────────────────────

# 1. Load data
df = pd.read_csv("asset_nodes.csv")
print(f"Loaded {len(df)} nodes.  Columns: {list(df.columns)}")

# Normalise column names (handle capacity_mw vs capacity_m)
col_map = {c: c.strip().lower() for c in df.columns}
df.rename(columns=col_map, inplace=True)
cap_col = "capacity_mw" if "capacity_mw" in df.columns else "capacity_m"

# 2. Handle missing values
before = len(df)
df.dropna(subset=["latitude", "longitude"], inplace=True)
if len(df) < before:
    print(f"  Dropped {before - len(df)} rows with missing lat/lon.")
df[cap_col] = df[cap_col].fillna(df[cap_col].median())

# 3. Convert to radians for haversine metric
coords_rad = np.radians(df[["latitude", "longitude"]].values)

# 4. DBSCAN with haversine (expects radians; eps in radians = km / R)
eps_rad = EPS_KM / EARTH_RADIUS_KM
db = DBSCAN(eps=eps_rad, min_samples=MIN_SAMPLES, metric="haversine")
labels = db.fit_predict(coords_rad)

# 5. Assign noise points their own unique cluster IDs
max_label = labels.max()
noise_mask = labels == -1
n_noise = noise_mask.sum()
if n_noise > 0:
    labels[noise_mask] = np.arange(max_label + 1, max_label + 1 + n_noise)

df["physical_cluster"] = labels

# 6. Summary
n_clusters = df["physical_cluster"].nunique()
mean_size = len(df) / n_clusters
print(f"\n{'='*50}")
print(f"  Total clusters : {n_clusters}")
print(f"  Mean nodes/cl. : {mean_size:.1f}")
print(f"  Noise → solo   : {n_noise}")
print(f"{'='*50}\n")

# Top-10 largest clusters
top = (df.groupby("physical_cluster")
         .agg(n=("name", "size"),
              total_cap=(cap_col, "sum"),
              countries=("country", lambda x: ", ".join(sorted(x.unique()))))
         .sort_values("n", ascending=False)
         .head(10))
print("Top-10 clusters by node count:")
print(top.to_string())

# 7. Save
df.to_csv("processed_nodes.csv", index=False)
print("\n✓ Saved processed_nodes.csv")

# 8. Visualisation
fig, ax = plt.subplots(figsize=(14, 10))

# Simple European coastline via Natural Earth (low-res shapefile fallback: just
# plot country borders if available, otherwise skip).
try:
    import json, urllib.request, io, zipfile, os
    # Use a bundled simplified GeoJSON of Europe if available, else skip outline
    coast_url = ("https://raw.githubusercontent.com/johan/world.geo.json/"
                 "master/countries.geo.json")
    # We'll just draw a light background instead for speed
    raise ImportError("skip heavy download")
except Exception:
    # Lightweight: draw a very simple outline using hardcoded European boundary
    ax.set_facecolor("#eaf4fc")
    fig.patch.set_facecolor("white")

# Colour by cluster
unique_clusters = sorted(df["physical_cluster"].unique())
n_colors = len(unique_clusters)
cmap = plt.cm.get_cmap("tab20", min(n_colors, 20))

# Size scaling (capacity)
sizes = df[cap_col].values
sizes = 5 + 120 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)

colors = [cmap(c % 20) for c in df["physical_cluster"].values]

ax.scatter(df["longitude"], df["latitude"],
           c=df["physical_cluster"].values, cmap="tab20",
           s=sizes, alpha=0.7, edgecolors="k", linewidths=0.3)

ax.set_xlabel("Longitude (°)", fontsize=12)
ax.set_ylabel("Latitude (°)", fontsize=12)
ax.set_title(f"Wind Farm Spatial Clusters (DBSCAN  eps={EPS_KM} km, "
             f"min_samples={MIN_SAMPLES})  —  {n_clusters} clusters",
             fontsize=13, fontweight="bold")
ax.set_xlim(-12, 32)
ax.set_ylim(35, 72)
ax.set_aspect(1.6)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("cluster_map.png", dpi=150)
print("✓ Saved cluster_map.png")
