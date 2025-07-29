"""
This serves as the final analysis and visualization stage of the evaluation
pipeline. It aggregates all the pre-computed metrics from individual JSON summary
files, normalizes them, and fuses them into composite scores for a holistic
comparison of the generative models.

The script performs the following key steps:
1.  Load Summaries: It recursively finds and loads all JSON summary files from
    the specified metrics directory, creating a unified pandas DataFrame.
2.  Normalize Metrics: It applies min-max normalization to all raw metrics
    (e.g., PSNR, KID, accuracy) to bring them to a common [0, 1] scale.
3.  Fuse Scores: It calculates three different types of fused scores:
    - A 'composite_score' based on a weighted average of the normalized metrics.
    - A 'pca_score' derived from the first principal component of the data.
    - A 'human_fused' score from a Random Forest model trained to predict human
      ratings from the normalized metrics.
4.  Generate Plots: It creates a comprehensive set of visualizations, including
    a correlation heatmap, t-SNE embeddings, and bar charts comparing the
    composite scores across different models, products, and data ratios.
5.  Save Final Results: It saves the complete DataFrame, containing all raw,
    normalized, and fused scores, into a single 'unified_scores.csv' file for
    easy access and reporting.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.manifold import TSNE

# config
METRICS_ROOT = "../../outputs/metrics"
PLOTS_ROOT = "../../outputs/plots"
OUTPUT_CSV = os.path.join("../../outputs/reports/unified_metric_scores.csv")

RAW_COLS = ["psnr", "kid", "ssim", "lpips", "accuracy", "roc_auc"]
THRESHOLDS = {
    "psnr": 25.0,  # >=25 dB best
    "kid": 0.05,  # <=0.05 best
    "lpips": 0.5  # <=0.5 best
}
COMP_WEIGHTS = {
    "psnr_n": 0.1,
    "kid_n": 0.1,
    "ssim_n": 0.1,
    "lpips_n": 0.1,
    "accuracy_n": 0.2,
    "roc_auc_n": 0.2,
    "human_score_n": 0.2,
}


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def minmax_normalize(s, invert=False):
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)
    norm = (s - mn) / (mx - mn)
    return 1 - norm if invert else norm


# load json summaries
def load_summaries(metrics_root):
    records = []
    for obj in os.listdir(metrics_root):
        obj_dir = os.path.join(metrics_root, obj)
        if not os.path.isdir(obj_dir): continue
        for fname in os.listdir(obj_dir):
            if not fname.endswith(".json"): continue
            exp = f"{obj}_{fname[:-5]}"
            summ = json.load(open(os.path.join(obj_dir, fname)))["summary"]
            rec = dict(
                experiment=exp,
                object=obj,
                model=exp.split("_")[-1],
                ratio="_".join(exp.split("_")[1:-1]),
                **{m: summ.get(m, np.nan) for m in RAW_COLS},
                human_score=summ.get("human_score", np.nan)
            )
            records.append(rec)
    df = pd.DataFrame.from_records(records).set_index("experiment")
    return df


# normalize and compute fused scores
def normalize_and_fuse(df):
    # normalize each raw metric to [0,1] and invert
    df["psnr_n"] = minmax_normalize(df["psnr"])
    df["kid_n"] = minmax_normalize(df["kid"], invert=True)
    df["ssim_n"] = minmax_normalize(df["ssim"])
    df["lpips_n"] = minmax_normalize(df["lpips"], invert=True)
    df["accuracy_n"] = minmax_normalize(df["accuracy"])
    df["roc_auc_n"] = minmax_normalize(df["roc_auc"])
    df["human_score_n"] = minmax_normalize(df["human_score"])

    norm_cols = list(COMP_WEIGHTS.keys())
    df[norm_cols] = df[norm_cols].fillna(df[norm_cols].mean())

    # composite = weighted sum
    df["composite_score"] = sum(df[c] * w for c, w in COMP_WEIGHTS.items())

    # PCA‐based fusion
    pca = PCA(n_components=1)
    df["pca_score"] = pca.fit_transform(df[norm_cols])[:, 0]

    return df, norm_cols


# human‐anchored regressor
def train_human_fusion(df, norm_cols):
    df_h = df.dropna(subset=["human_score"])
    if df_h.empty:
        print("No human_scores found")
        return df, None

    X = df_h[norm_cols].values
    y = df_h["human_score"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # linear
    lin = LinearRegression().fit(X_tr, y_tr)
    print("Linear R^2:", r2_score(y_te, lin.predict(X_te)))
    print("Linear MAE:", mean_absolute_error(y_te, lin.predict(X_te)))
    print("Linear coefs:", dict(zip(norm_cols, lin.coef_)))

    # random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
    print("RF R^2:", r2_score(y_te, rf.predict(X_te)))
    print("RF MAE:", mean_absolute_error(y_te, rf.predict(X_te)))

    # retrain
    rf.fit(X, y)
    df["human_fused"] = rf.predict(df[norm_cols].values)
    return df, rf


def make_plots(df, norm_cols):
    ensure_dir(PLOTS_ROOT)

    # normalized metrics
    plt.figure(figsize=(12, 6))
    df.sort_values("composite_score")[norm_cols].plot(kind="bar", width=0.8)
    plt.title("Normalized Metrics by Experiment");
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_ROOT}/normalized_by_experiment.png")
    plt.close()

    # composite vs PCA
    plt.figure(figsize=(6, 6))
    plt.scatter(df["composite_score"], df["pca_score"], s=50, alpha=0.7)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Composite");
    plt.ylabel("PCA Score")
    plt.title("Composite vs PCA Fusion");
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_ROOT}/composite_vs_pca.png")
    plt.close()

    # if human_fused exists
    if "human_fused" in df.columns:
        for xcol in ["composite_score", "pca_score"]:
            plt.figure(figsize=(6, 6))
            plt.scatter(df[xcol], df["human_fused"], s=50, alpha=0.7)
            plt.xlabel(xcol);
            plt.ylabel("Human‐Anchored")
            plt.title(f"{xcol} vs Human‐Fused");
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{PLOTS_ROOT}/{xcol}_vs_human.png")
            plt.close()

        # correlation heatmap
        corr = df[norm_cols + ["composite_score", "pca_score", "human_fused"]].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="RdBu_r")
        plt.title("Correlation Matrix");
        plt.tight_layout()
        plt.savefig(f"{PLOTS_ROOT}/correlation_matrix.png")
        plt.close()

    # average normalized metrics by object/model
    for group, by in [("object", df.groupby("object")), ("model", df.groupby("model"))]:
        avg = by[norm_cols].mean()
        plt.figure(figsize=(8, 4))
        avg.plot(kind="bar")
        plt.title(f"Avg Normalized by {group.capitalize()}");
        plt.ylabel("Value")
        plt.xticks(rotation=0)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_ROOT}/avg_norm_by_{group}.png")
        plt.close()

    # bounded metrics
    bnd = pd.DataFrame(index=df.index)
    for m, th in THRESHOLDS.items():
        if m in df:
            if m in ("kid", "lpips"):
                bnd[f"{m}_bnd"] = 1 - np.minimum(df[m] / th, 1.0)
            else:
                bnd[f"{m}_bnd"] = np.minimum(df[m] / th, 1.0)
    for m in ("ssim", "accuracy", "roc_auc"):
        bnd[f"{m}_bnd"] = df[m].clip(0, 1)
    bnd_cols = [c for c in bnd.columns]
    bnd = bnd.groupby(df["object"])[bnd_cols].mean()
    plt.figure(figsize=(8, 4))
    bnd.plot(kind="bar")
    plt.title("Average Bounded Metrics by Object");
    plt.ylabel("Bounded [0,1]")
    plt.xticks(rotation=0);
    plt.tight_layout()
    plt.savefig(f"{PLOTS_ROOT}/bounded_by_object.png")
    plt.close()

    # tSNE embedding of normalized metrics
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    X = df[norm_cols].values
    tsne_coords = tsne.fit_transform(X)
    df["tsne_1"], df["tsne_2"] = tsne_coords[:, 0], tsne_coords[:, 1]

    # tSNE colored by composite score
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="tsne_1", y="tsne_2",
        hue="composite_score",
        size="composite_score",
        palette="viridis",
        sizes=(30, 200),
        data=df,
        legend="brief",
        alpha=0.8
    )
    plt.title("t-SNE of Normalized Metrics (by composite_score)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_ROOT}/tsne_composite_by_composite_score.png", dpi=300)
    plt.close()

    # tSNE colored by model/ object
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="tsne_1", y="tsne_2",
        hue="model",
        style="object",
        data=df,
        s=100,
        alpha=0.8
    )
    for idx, row in df.iterrows():
        plt.text(row["tsne_1"] + 0.5, row["tsne_2"] + 0.5, idx, fontsize=6, alpha=0.7)
    plt.title("t-SNE of Normalized Metrics (by model/object)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_ROOT}/tsne_composite_by_model.png", dpi=300)
    plt.close()

    avg_model = df.groupby("model")["composite_score"].mean()
    plt.figure()
    avg_model.plot(kind="bar", color=["skyblue", "salmon"])
    plt.ylabel("Average Composite Score")
    plt.title("Average Composite Score by Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("plots/avg_score_by_model.png", dpi=300)
    plt.close()

    # avergae by ratio per category
    ratio_order = ["100real", "70r_30s", "50_50", "30r_70s", "100syn"]

    avg_both = (
        df
        .groupby(["ratio", "model"])["composite_score"]
        .mean()
        .unstack()
        .reindex(ratio_order)
    )

    x = np.arange(len(ratio_order))
    width = 0.35

    plt.figure()
    plt.bar([i - width / 2 for i in x], avg_both["GAN"].values, width, label="GAN")
    plt.bar([i + width / 2 for i in x], avg_both["DM"].values, width, label="DM")

    plt.xticks(x, ratio_order)
    plt.xlabel("Synthetic/Real Ratio")
    plt.ylabel("Composite Score")
    plt.title(f"GAN vs Diffusion Model")

    best_gan = avg_both["GAN"].idxmax()
    best_dm = avg_both["DM"].idxmax()
    gan_x = ratio_order.index(best_gan)
    dm_x = ratio_order.index(best_dm)

    plt.annotate("*", (gan_x - width / 2, avg_both.loc[best_gan, "GAN"] + 0.01), ha="center", va="bottom", color="C0",
                 size=20)
    plt.annotate("*", (dm_x + width / 2, avg_both.loc[best_dm, "DM"] + 0.01), ha="center", va="bottom", color="C1",
                 size=20)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/comp_by_ratio.png", dpi=300)
    plt.close()


def main():
    df = load_summaries(METRICS_ROOT)
    df, norm_cols = normalize_and_fuse(df)
    df, fusion_model = train_human_fusion(df, norm_cols)
    ensure_dir(METRICS_ROOT)
    df.to_csv(OUTPUT_CSV)
    print("\nRaw metrics:")
    print(df[["psnr", "kid", "ssim", "lpips", "accuracy", "roc_auc", "human_score"]].round(4))
    print("\nNormalized metrics:")
    print(df[norm_cols].round(4))
    print("\nComposite & PCA scores:")
    print(df[["composite_score", "pca_score", "human_fused"]].round(4))
    print(f"Unified scores written to {OUTPUT_CSV}")
    make_plots(df, norm_cols)
    print("All plots saved under", PLOTS_ROOT)


if __name__ == "__main__":
    main()
