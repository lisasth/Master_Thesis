"""
This serves as the analysis and visualization of the evaluation pipeline.
It aggregates all the pre-computed metrics from individual JSON summary
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
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms
import lpips
from skimage.metrics import structural_similarity as ssim

# config
datasets_root = "../../data/single"
metrics_root = "../../outputs/metrics"
balance_classes = ["IO", "NIO"]
image_size = 256
human_score_csv = "../../data/Human_Score/scores.csv"

# you can add more products here
category_base = {
    "donut": "Donuts",
    "croissant": "Croissants"
}

# load human scores and normalize
human_df = pd.read_csv(human_score_csv)
if "score" in human_df.columns and "human_score" not in human_df.columns:
    human_df.rename(columns={"score": "human_score"}, inplace=True)

# strip file extension and split parts
human_df["name"] = human_df["filename"].str.replace(r"\.[a-zA-Z]+$", "", regex=True)
parts = human_df["name"].str.split("_")

# extract shape (c or d) and method (dm or gan)
human_df["shape"] = parts.str[0]
human_df["method"] = parts.str[1].str.lower()


# extract ratio string from parts
def extract_ratio(p):
    # drop trailing numeric
    if p[-1].isdigit():
        p = p[:-1]
    rat_parts = p[2:-1]
    if not rat_parts:
        rat = p[2]
    else:
        rat = "_".join(rat_parts)
    rat_low = rat.lower()
    if rat_low == "r":
        return "100real"
    if rat_low == "s":
        return "100syn"
    return rat_low


human_df["ratio"] = parts.apply(extract_ratio)

# group by shape, method, ratio to get mean human_score
human_group = human_df.groupby(["shape", "method", "ratio"])["human_score"].mean()

# prepare models and transforms for KID, PSNR, SSIM, LPIPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# inception for KID
inc_model = models.inception_v3(pretrained=True, aux_logits=True)
inc_model.AuxLogits = torch.nn.Identity()
inc_model.fc = torch.nn.Identity()
inc_model.eval().to(device)
kid_tf = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# LPIPS model
lpips_model = lpips.LPIPS(net="alex").to(device)
lpips_tf = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])


# extract features and compute KID/ PSNR/ SSIM and LPIPS
def extract_feats(folder, sample_list=None):
    feats = []
    files = sorted(os.listdir(folder))
    if sample_list is not None:
        files = [f for f in files if f in sample_list]
    for fn in files:
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(folder, fn)).convert("RGB")
            x = kid_tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats.append(inc_model(x).cpu().numpy().reshape(-1))
    return np.stack(feats, 0) if feats else np.zeros((1, 2048))


def compute_kid(fr, fs):
    m, n = fr.shape[0], fs.shape[0]
    d = fr.shape[1]
    K_xx = (fr @ fr.T / d + 1.0) ** 3
    K_yy = (fs @ fs.T / d + 1.0) ** 3
    K_xy = (fr @ fs.T / d + 1.0) ** 3
    sum_xx = (K_xx.sum() - np.trace(K_xx)) / (m * (m - 1))
    sum_yy = (K_yy.sum() - np.trace(K_yy)) / (n * (n - 1))
    sum_xy = 2.0 * K_xy.sum() / (m * n)
    return float(sum_xx + sum_yy - sum_xy)


def compute_psnr(real_dir, syn_dir):
    files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith((".png", ".jpg"))])
    psnrs = []
    for fn in files:
        rpath = os.path.join(real_dir, fn)
        spath = os.path.join(syn_dir, fn)
        if not os.path.exists(spath):
            continue
        imr = np.array(Image.open(rpath).convert("RGB").resize((image_size, image_size))) / 255.0
        ims = np.array(Image.open(spath).convert("RGB").resize((image_size, image_size))) / 255.0
        mse = np.mean((imr - ims) ** 2)
        psnrs.append(10 * np.log10(1.0 / (mse + 1e-12)))
    return float(np.mean(psnrs)) if psnrs else 0.0


def compute_ssim_lpips(real_dir, syn_dir):
    files = sorted([f for f in os.listdir(real_dir) if f.lower().endswith((".png", ".jpg"))])
    ss_vals, lp_vals = [], []
    for fn in files:
        rpath = os.path.join(real_dir, fn)
        spath = os.path.join(syn_dir, fn)
        if not os.path.exists(spath):
            continue
        ir = Image.open(rpath).convert("RGB").resize((image_size, image_size))
        is_ = Image.open(spath).convert("RGB").resize((image_size, image_size))
        arr_r = np.array(ir)
        arr_s = np.array(is_)
        ss_vals.append(ssim(arr_r, arr_s, channel_axis=-1))
        tr = lpips_tf(ir).unsqueeze(0).to(device)
        ts = lpips_tf(is_).unsqueeze(0).to(device)
        with torch.no_grad():
            lp_vals.append(lpips_model(tr, ts).item())
    return float(np.mean(ss_vals)), float(np.mean(lp_vals))


# map category to shape char for human lookup
shape_map = {"croissant": "c", "donut": "d"}

# update JSON summaries: inject metrics amd human_score
for category in ["donut", "croissant"]:
    cat_dir = os.path.join(metrics_root, category)
    if not os.path.isdir(cat_dir):
        continue

    # find dataset subfolders matching "Donuts_GAN", "Donuts_DM"
    ds_folders = [d for d in os.listdir(datasets_root)
                  if d.lower().startswith(category)]
    for fname in os.listdir(cat_dir):
        if not fname.endswith(".json"):
            continue
        exp = fname[:-5]  # e.g. "30r_70s_GAN"
        suffix = exp.split("_")[-1]  # "GAN" or "DM"
        ratio = exp[:-(len(suffix) + 1)]  # "30r_70s" or "100real"
        ds_name = f"{category_base[category]}_{suffix}"

        if ds_name not in ds_folders:
            print(f"skipping {fname!r}: no matching folder")
            continue

        base_dir = os.path.join(datasets_root, ds_name)
        real_IO, real_NIO = os.path.join(base_dir, "real", balance_classes[0]), \
            os.path.join(base_dir, "real", balance_classes[1])
        syn_IO, syn_NIO = os.path.join(base_dir, "syn", balance_classes[0]), \
            os.path.join(base_dir, "syn", balance_classes[1])

        # determine weights from ratio
        if ratio == "100real":
            w_real, w_syn = 1.0, 0.0
        elif ratio == "100syn":
            w_real, w_syn = 0.0, 1.0
        else:
            m = re.match(r"(\d+)r_?(\d+)s", ratio)
            if m:
                w_real = int(m.group(1)) / 100.0
                w_syn = int(m.group(2)) / 100.0
            else:
                w_real, w_syn = 0.5, 0.5

        # KID
        frIO, fsIO = extract_feats(real_IO), extract_feats(syn_IO)
        frNI, fsNI = extract_feats(real_NIO), extract_feats(syn_NIO)
        kid_avg = round((compute_kid(frIO, fsIO) + compute_kid(frNI, fsNI)) / 2, 5)

        # PSNR
        psnr_avg = round((compute_psnr(real_IO, syn_IO) + compute_psnr(real_NIO, syn_NIO)) / 2, 3)

        # SSIM & LPIPS
        sIO, lpIO = compute_ssim_lpips(real_IO, syn_IO)
        sNI, lpNI = compute_ssim_lpips(real_NIO, syn_NIO)
        s_avg = round((sIO + sNI) / 2, 4)
        lp_avg = round((lpIO + lpNI) / 2, 4)

        # human score
        method = suffix.lower()  # GAN or DM
        ratio_low = ratio.lower()  # e.g. 30r_70s
        shape_char = shape_map[category]  # c or d

        if ratio_low == '100real':
            human_sc = human_group.get((shape_char, method, "100real"), np.nan)
        elif ratio_low == '100syn':
            human_sc = human_group.get((shape_char, method, "100syn"), np.nan)
        else:
            real_h = human_group.get((shape_char, method, "100real"), np.nan)
            syn_h = human_group.get((shape_char, method, "100syn"), np.nan)
            human_sc = w_real * real_h + w_syn * syn_h
            if np.isnan(real_h) or np.isnan(syn_h):
                human_sc = np.nan

        jpath = os.path.join(cat_dir, fname)
        data = json.load(open(jpath))
        data.setdefault("summary", {})
        data["summary"].update({
            "kid": kid_avg,
            "psnr": psnr_avg,
            "ssim": s_avg,
            "lpips": lp_avg,
            "human_score": float(human_sc) if not np.isnan(human_sc) else None
        })
        with open(jpath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"updated {category}/{exp}: human_score={human_sc}")
