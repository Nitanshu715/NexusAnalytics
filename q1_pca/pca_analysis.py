import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import os

np.random.seed(42)
output_dir = os.path.dirname(os.path.abspath(__file__))

n = 500
annual_income = np.random.normal(60000, 25000, n).clip(10000, 200000)
spending_score = np.random.normal(50, 25, n).clip(1, 100)
purchase_freq = np.random.poisson(12, n).clip(1, 52).astype(float)
avg_basket = np.random.normal(150, 80, n).clip(10, 1000)
loyalty_years = np.random.exponential(3, n).clip(0, 20)
online_ratio = np.random.beta(2, 3, n)
num_categories = np.random.randint(1, 10, n).astype(float)
returns_pct = np.random.beta(1.5, 8, n)
discount_usage = np.random.beta(2, 5, n)
weekend_ratio = np.random.beta(3, 4, n)

spending_score += 0.00015 * annual_income + np.random.normal(0, 5, n)
avg_basket += 0.0008 * annual_income + np.random.normal(0, 20, n)
purchase_freq += 0.0001 * annual_income + np.random.normal(0, 2, n)
spending_score = spending_score.clip(1, 100)
avg_basket = avg_basket.clip(10, 1000)
purchase_freq = purchase_freq.clip(1, 52)

mask = np.random.rand(n, 10) < 0.04
data = np.column_stack([
    annual_income, spending_score, purchase_freq, avg_basket,
    loyalty_years, online_ratio, num_categories, returns_pct,
    discount_usage, weekend_ratio
]).astype(float)
data[mask] = np.nan

columns = [
    "Annual Income", "Spending Score", "Purchase Frequency",
    "Avg Basket Size", "Loyalty Years", "Online Purchase Ratio",
    "Num Categories", "Returns Percentage", "Discount Usage", "Weekend Ratio"
]
df = pd.DataFrame(data, columns=columns)

print("=== Dataset Overview ===")
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=columns)

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=columns)

print("\nMissing values after imputation:", df_imputed.isnull().sum().sum())
print("Feature means after scaling (should be ~0):", df_scaled.mean().round(4).values)

pca_full = PCA()
pca_full.fit(df_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("PCA Explained Variance Analysis", fontsize=14, fontweight="bold", y=1.02)

ax1 = axes[0]
bars = ax1.bar(range(1, len(explained) + 1), explained * 100, color="#4C72B0", edgecolor="white", linewidth=0.5)
ax1.set_xlabel("Principal Component", fontsize=11)
ax1.set_ylabel("Explained Variance (%)", fontsize=11)
ax1.set_title("Scree Plot", fontsize=12, fontweight="bold")
ax1.set_xticks(range(1, len(explained) + 1))
for bar, val in zip(bars, explained):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2 = axes[1]
ax2.plot(range(1, len(cumulative) + 1), cumulative * 100, "o-", color="#DD8452",
         linewidth=2, markersize=6, label="Cumulative variance")
ax2.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% threshold")
ax2.axhline(95, color="lightcoral", linestyle="--", linewidth=1, label="95% threshold")
optimal = np.argmax(cumulative >= 0.80) + 1
ax2.axvline(optimal, color="#4C72B0", linestyle=":", linewidth=1.5, label=f"PC {optimal} (80%)")
ax2.set_xlabel("Number of Components", fontsize=11)
ax2.set_ylabel("Cumulative Explained Variance (%)", fontsize=11)
ax2.set_title("Cumulative Explained Variance", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_xticks(range(1, len(cumulative) + 1))
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot_scree.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"\nOptimal components for 80% variance: {optimal}")
print(f"Variance explained by first 3 PCs: {cumulative[2]*100:.2f}%")

pca_3 = PCA(n_components=3)
components_3d = pca_3.fit_transform(df_scaled)

income_tertile = pd.qcut(df_imputed["Annual Income"], 3, labels=["Low", "Medium", "High"])
color_map = {"Low": "#4C72B0", "Medium": "#DD8452", "High": "#55A868"}
colors = [color_map[g] for g in income_tertile]

fig, ax = plt.subplots(figsize=(9, 7))
for group, color in color_map.items():
    mask_g = income_tertile == group
    ax.scatter(components_3d[mask_g, 0], components_3d[mask_g, 1],
               c=color, label=f"{group} Income", alpha=0.65, s=30, edgecolors="white", linewidth=0.3)
ax.set_xlabel(f"PC1 ({pca_3.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({pca_3.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
ax.set_title("PCA Projection — PC1 vs PC2\nColored by Annual Income Tertile", fontsize=13, fontweight="bold")
ax.legend(title="Income Group", fontsize=10, title_fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot_pca_2d.png"), dpi=150, bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(11, 8))
ax3d = fig.add_subplot(111, projection="3d")
for group, color in color_map.items():
    mask_g = income_tertile == group
    ax3d.scatter(components_3d[mask_g, 0], components_3d[mask_g, 1], components_3d[mask_g, 2],
                 c=color, label=f"{group} Income", alpha=0.6, s=25)
ax3d.set_xlabel(f"PC1 ({pca_3.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
ax3d.set_ylabel(f"PC2 ({pca_3.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
ax3d.set_zlabel(f"PC3 ({pca_3.explained_variance_ratio_[2]*100:.1f}%)", fontsize=9)
ax3d.set_title("PCA 3D Projection — PC1, PC2, PC3\nColored by Annual Income Tertile", fontsize=12, fontweight="bold")
ax3d.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot_pca_3d.png"), dpi=150, bbox_inches="tight")
plt.close()

pca_2 = PCA(n_components=2)
pca_2.fit(df_scaled)
loadings = pd.DataFrame(pca_2.components_.T, index=columns, columns=["PC1", "PC2"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("PCA Feature Loadings (PC1 and PC2)", fontsize=13, fontweight="bold")

for i, pc in enumerate(["PC1", "PC2"]):
    ax = axes[i]
    sorted_load = loadings[pc].sort_values()
    colors_bar = ["#DD8452" if v > 0 else "#4C72B0" for v in sorted_load]
    ax.barh(sorted_load.index, sorted_load.values, color=colors_bar, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{pc} Loadings\n({pca_2.explained_variance_ratio_[i]*100:.1f}% variance)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Loading Value", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "plot_pca_loadings.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n=== PCA Component Loadings ===")
print(loadings.round(4))
print("\n=== Explained Variance per Component ===")
for i, (ev, cum) in enumerate(zip(explained, cumulative)):
    print(f"PC{i+1}: {ev*100:.2f}%  |  Cumulative: {cum*100:.2f}%")

print("\nAll Q1 plots saved.")
