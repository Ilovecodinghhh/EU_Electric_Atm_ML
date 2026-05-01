"""
PCA on Finance Data — Explained Variance Analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Load
df = pd.read_csv("data/Finance20192024/raw_finance.csv", index_col=0, parse_dates=True)
print(f"Shape: {df.shape}  Columns: {list(df.columns)}")
print(f"Date range: {df.index.min()} → {df.index.max()}")
print(f"Missing values:\n{df.isnull().sum()}\n")

# 2. Handle missing values
df.dropna(inplace=True)
print(f"After dropna: {df.shape}")

# 3. Compute daily returns (log returns for stationarity)
returns = np.log(df / df.shift(1)).dropna()
print(f"Returns shape: {returns.shape}\n")

# 4. Standardise
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

# 5. PCA (all components)
n_components = returns_scaled.shape[1]
pca = PCA(n_components=n_components)
pca.fit(returns_scaled)

evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)

print("=" * 55)
print("  Explained Variance Ratio (EVR) per Principal Component")
print("=" * 55)
for i, (v, c) in enumerate(zip(evr, cum_evr), 1):
    bar = "█" * int(v * 50)
    print(f"  PC{i}: {v*100:6.2f}%  (cumulative: {c*100:6.2f}%)  {bar}")
print("=" * 55)

print(f"\n  ➤ PC1 explains: {evr[0]*100:.2f}% of total variance")
print(f"  ➤ Top-3 PCs explain: {cum_evr[2]*100:.2f}% of total variance")
if cum_evr[2] >= 0.85:
    print(f"  ✓ Top-3 PCs exceed 85% threshold ({cum_evr[2]*100:.2f}% ≥ 85%)")
else:
    print(f"  ✗ Top-3 PCs do NOT reach 85% ({cum_evr[2]*100:.2f}% < 85%)")
    # Find how many needed
    n_for_85 = np.searchsorted(cum_evr, 0.85) + 1
    print(f"    → Need {n_for_85} PCs to reach 85%")

# 6. Loadings
print("\n  Component Loadings (weights on original assets):")
loadings = pd.DataFrame(pca.components_, columns=df.columns,
                        index=[f"PC{i+1}" for i in range(n_components)])
print(loadings.round(4).to_string())

# 7. Visualisation — scree plot
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(1, n_components + 1)
ax.bar(x, evr * 100, color="steelblue", alpha=0.7, label="Individual EVR")
ax.plot(x, cum_evr * 100, "ro-", label="Cumulative EVR")
ax.axhline(85, ls="--", color="gray", label="85% threshold")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance (%)")
ax.set_title("Finance PCA — Scree Plot")
ax.set_xticks(x)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("finance_pca_scree.png", dpi=150)
print("\n✓ Saved finance_pca_scree.png")
