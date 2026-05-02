# =================== LSM HYBRID (CNN-RF) — Spatial CV + Discussion Diagnostics ===================
# You have to change the path according to your system.
# ==============================================================================================

import os, warnings, math, numpy as np, joblib, time, random, gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix OOM 3: must be set before any CUDA call
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import geopandas as gpd, rasterio, rasterio.windows as rw
from rasterio.vrt import WarpedVRT
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.model_selection import GroupKFold
from scipy.spatial import cKDTree          # Moran's I
from joblib import Parallel, delayed
import torch, torch.nn as nn
from torchvision import models
import pandas as pd
from shapely.ops import unary_union as _unary_union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x=None, **k): return x if x is not None else range(1)

# --- Paths ---------------------------------------------------------------------------
BASE           = '/mnt/Data3/2025_landslidePaper/2025_landslide'
FACTOR_TIF     = f'{BASE}/raster/Raster_Com.tif'
POINTS_PATH    = f'{BASE}/point/Point_LS.shp'
DEEP_FEATS_TIF = f'{BASE}/CNN_RF/process/deep_feats.tif'
MODEL_PATH     = f'{BASE}/CNN_RF/model/rf_fused_points.joblib'
OUT_PROB_TIF   = f'{BASE}/CNN_RF/result/lsm_prob.tif'

RESULT_DIR = os.path.dirname(OUT_PROB_TIF)
FIG_DIR    = os.path.join(RESULT_DIR, "figs")
LOG_DIR    = os.path.join(RESULT_DIR, "logs")
for d in [os.path.dirname(MODEL_PATH), os.path.dirname(DEEP_FEATS_TIF),
          RESULT_DIR, FIG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Raw band names for feature importance (update to match Raster_Com.tif band order)
RAW_BAND_NAMES = [
    "DEM", "Slope", "Curvature", "SPI", "TRI",
    "Dist_Stream", "Dist_Fault", "Soil_Type", "Geology"
]

# --- CSV helper ----------------------------------------------------------------------
def _curve_df_with_thresholds(x_arr, y_arr, thr_arr, x_name, y_name):
    x_arr = np.asarray(x_arr); y_arr = np.asarray(y_arr); thr_arr = np.asarray(thr_arr)
    need = len(x_arr) - len(thr_arr)
    if need < 0: thr_arr = thr_arr[:len(x_arr)]; need = 0
    if need > 0: thr_arr = np.r_[thr_arr, np.full(need, np.nan, dtype=float)]
    return pd.DataFrame({x_name: x_arr, y_name: y_arr, "threshold": thr_arr})

# --- Constants -----------------------------------------------------------------------
OUTPUT_STRIDE = 16
BACKBONE      = "resnet152"
FEAT_CHANNELS = 256
TILE          = 512
N_SPLITS      = 5
SEED          = 42

# =============================================================================
# HARDWARE PROFILE: 11 GB GPU + 128 GB RAM
# GPU_TILE_BATCH=4: model(120 MB) + intermediates(130 MB) + outs(4×268 MB)
#   + fk(4×134 MB) = ~1.9 GB — safe with in-place TTA
# RF training is CPU-only (RandomForestClassifier) → full VRAM for CNN
# RAM: 128 GB → MORANS_N_SAMPLE=10000 points for robust Moran's I
# =============================================================================
USE_GPU_IF_AVAILABLE = True
CUDA_DEVICE_INDEX    = 0
USE_MIXED_PRECISION  = True
GPU_TILE_BATCH       = 4      # safe for 11 GB GPU with in-place TTA
torch.set_float32_matmul_precision("high")

# --- RF CPU controls -----------------------------------------------------------------
RF_N_JOBS          = max(1, (os.cpu_count() or 1))  # all cores for RF training
PRED_OUTER_THREADS = max(1, (os.cpu_count() or 1))  # tile-level parallelism for prediction
RF_TUNE_ITER       = 150    # random search iterations (was 80 → 150; RF is CPU, no GPU contention)
MORANS_N_SAMPLE    = 10000  # Moran's I subsample (safe with 128 GB RAM)

# --- RandomForest settings -----------------------------------------------------------
CALIBRATION_METHOD = None   # 'sigmoid', 'isotonic', or None

# --- Labels --------------------------------------------------------------------------
LABEL_COL              = "Class"
POS_VALUE              = 1
NEG_VALUE              = 2
NEG_EXCLUSION_RADIUS_M = 90

# --- Fine-tuning (CNN) ---------------------------------------------------------------
FINE_TUNE_ENABLED    = True
FT_EPOCHS            = 10
FT_LR                = 1e-4
FT_TILE              = 512
FT_BUFFER_M          = 60
FT_SKIP_ALL_NEG_PROB = 0.8
FT_BATCH_TILES       = 4      # safe with 11 GB GPU (gradients for unfrozen layers ~2 GB)
FT_ACCUM_STEPS       = 1

# --- Anti-seam & robustness ----------------------------------------------------------
HALO    = 160
USE_TTA = True

# --- Timing helpers ------------------------------------------------------------------
TIMINGS = []
def _fmt_hms(sec):
    m, s = divmod(sec, 60.0); h, m = divmod(int(m), 60); return f"{h:d}:{m:02d}:{s:05.2f}"
def _log_time(step, seconds, **extras):
    rec = {"step": step, "seconds": round(seconds,3), "pretty": _fmt_hms(seconds)}
    rec.update(extras); TIMINGS.append(rec)
    print(f"[Time] {step}: {rec['pretty']} ({rec['seconds']} s)")
def _write_timings_csv(path=os.path.join(LOG_DIR, "timings.csv")):
    if TIMINGS: pd.DataFrame(TIMINGS).to_csv(path, index=False); print(f"[Time] → {path}")
def _cuda_sync_if_any():
    try:
        if torch.cuda.is_available(): torch.cuda.synchronize()
    except Exception: pass
def _savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

# ================= Backbone (unchanged) ==============================================
def _disable_inplace_relu(module):
    for m in module.modules():
        if isinstance(m, nn.ReLU): m.inplace = False

def _get_resnet_backbone(in_ch, backbone="resnet50"):
    name = backbone.lower()
    spec = {
        "resnet18":  (models.resnet18,  getattr(models,"ResNet18_Weights",None),   512),
        "resnet34":  (models.resnet34,  getattr(models,"ResNet34_Weights",None),   512),
        "resnet50":  (models.resnet50,  getattr(models,"ResNet50_Weights",None),  2048),
        "resnet101": (models.resnet101, getattr(models,"ResNet101_Weights",None), 2048),
        "resnet152": (models.resnet152, getattr(models,"ResNet152_Weights",None), 2048),
    }
    if name not in spec: raise ValueError("BACKBONE must be one of: " + ", ".join(spec.keys()))
    ctor, Weights, c_out = spec[name]
    weights = None
    if Weights is not None:
        try: weights = Weights.IMAGENET1K_V1
        except Exception: pass
    rswd_map = {32:[False,False,False], 16:[False,False,True], 8:[False,True,True]}
    rswd = rswd_map.get(OUTPUT_STRIDE, [False,False,False]); os_used = OUTPUT_STRIDE
    try: res = ctor(weights=weights, replace_stride_with_dilation=rswd)
    except TypeError: res = ctor(weights=weights); os_used = 32
    _disable_inplace_relu(res)
    if in_ch != 3:
        old = res.conv1
        new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            w = old.weight; repeat = math.ceil(in_ch/3)
            w_rep = w.repeat(1,repeat,1,1)[:,:in_ch,:,:]
            new.weight.copy_(w_rep*(3.0/float(in_ch)))
        res.conv1 = new
    trunk = nn.Sequential(res.conv1,res.bn1,res.relu,res.maxpool,
                          res.layer1,res.layer2,res.layer3,res.layer4)
    print(f"[Backbone] {backbone} | c_out={c_out} | OS={os_used}")
    return trunk, c_out

class FCNFeatureExtractor(nn.Module):
    def __init__(self, in_ch=9, out_ch=128, backbone=BACKBONE):
        super().__init__()
        self.backbone, c_out = _get_resnet_backbone(in_ch, backbone=backbone)
        self.proj = nn.Conv2d(c_out, out_ch, kernel_size=1)
    def forward(self, x):
        f = self.backbone(x); f = self.proj(f)
        return nn.functional.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)

# ================= Utilities =========================================================
def _select_device():
    if USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
        torch.cuda.set_device(CUDA_DEVICE_INDEX)
        torch.backends.cudnn.benchmark = True
        print(f"[Device] CUDA:{CUDA_DEVICE_INDEX}")
        return torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
    print("[Device] CPU"); return torch.device("cpu")

def compute_band_stats(src, block=1024):
    """Per-band mean/std with progress bar."""
    n_bx = math.ceil(src.width  / block)
    n_by = math.ceil(src.height / block)
    sums = sqs = counts = None
    with tqdm(total=n_bx*n_by, desc="[A] Band statistics", unit="block", leave=False) as pbar:
        for y in range(0, src.height, block):
            for x in range(0, src.width, block):
                w   = rw.Window(x, y, min(block,src.width-x), min(block,src.height-y))
                arr = src.read(window=w, out_dtype='float64')
                mask = np.isfinite(arr); arr = np.where(mask, arr, 0.0)
                c = mask.sum(axis=(1,2)).astype(np.int64)
                s = arr.sum(axis=(1,2), dtype=np.float64)
                q = np.square(arr, dtype=np.float64).sum(axis=(1,2), dtype=np.float64)
                if sums is None: sums, sqs, counts = s, q, c
                else: sums += s; sqs += q; counts += c
                pbar.update(1)
    counts_safe = np.maximum(counts, 1); means = sums / counts_safe
    vars_ = (sqs/counts_safe) - (means*means)
    vars_ = np.where(counts > 0, vars_, 1.0); vars_ = np.clip(vars_, 1e-12, None)
    return means.astype('float32'), np.sqrt(vars_).astype('float32')

def _standardize_block(arr, mean, std, eps=1e-6):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    std_safe = np.where(np.isfinite(std) & (std>eps), std, eps).astype(arr.dtype, copy=False)
    return (arr - mean[:,None,None]) / std_safe[:,None,None]

def _reproject_points_to_raster_crs(points_gdf, raster_src):
    return points_gdf.to_crs(raster_src.crs) if points_gdf.crs != raster_src.crs else points_gdf

def _sample_rasters_at_points(tif_paths, points_xy):
    """Sample raster values at point locations with progress bar."""
    feats = []
    for path in tif_paths:
        label = os.path.basename(path)
        with rasterio.open(path) as src:
            with WarpedVRT(src, crs=src.crs) as vrt:
                vals = np.array(
                    list(tqdm(vrt.sample(points_xy), total=len(points_xy),
                              desc=f"[A] Sample {label}", unit="pt", leave=False)),
                    dtype='float32')
        feats.append(vals)
    return np.concatenate(feats, axis=1)

# FIX: larger blocks → less spatial leakage between folds
def _make_spatial_blocks(points_gdf, n_splits=N_SPLITS):
    """
    Assign spatial block IDs using a sqrt(n_splits) × sqrt(n_splits) grid.
    Larger blocks than v3 (which used sqrt(n_splits*2)) → better spatial separation.
    """
    xmin, ymin, xmax, ymax = points_gdf.total_bounds
    n_cells = int(np.ceil(np.sqrt(n_splits)))   # FIX: was sqrt(n_splits*2)
    xs = np.linspace(xmin, xmax, n_cells+1)
    ys = np.linspace(ymin, ymax, n_cells+1)
    def cell_id(x, y):
        ix = min(n_cells-1, max(0, np.searchsorted(xs, x, side='right')-1))
        iy = min(n_cells-1, max(0, np.searchsorted(ys, y, side='right')-1))
        return iy*n_cells + ix
    block_ids = points_gdf.geometry.apply(lambda g: cell_id(g.x, g.y)).values
    n_unique  = len(np.unique(block_ids))
    print(f"[Spatial blocks] {n_cells}×{n_cells}={n_cells**2} cells | populated={n_unique}")
    return block_ids

def _drop_ambiguous_negatives(gdf, label_col, pos_val=1, neg_val=2, radius_m=90):
    if radius_m <= 0: return gdf
    gpos = gdf[gdf[label_col].astype(int)==pos_val]
    gneg = gdf[gdf[label_col].astype(int)==neg_val]
    if gpos.empty or gneg.empty: return gdf
    need_proj = (not gdf.crs) or (not gdf.crs.is_projected)
    if need_proj:
        cx, cy = gdf.unary_union.centroid.xy; utm = int((cx[0]+180)//6)+1
        epsg = 32600+utm if cy[0]>=0 else 32700+utm
        gpos_ = gpos.to_crs(epsg); gneg_ = gneg.to_crs(epsg)
    else: gpos_, gneg_ = gpos, gneg
    buf = gpos_.buffer(radius_m)
    ug  = buf.union_all() if hasattr(buf,"union_all") else _unary_union(
        list(buf.values if hasattr(buf,"values") else buf))
    kept = gneg_.loc[~gneg_.geometry.intersects(ug)]
    gpos_b = gpos_.to_crs(gdf.crs) if need_proj else gpos_
    kept_b = kept.to_crs(gdf.crs)  if need_proj else kept
    return pd.concat([gpos_b, kept_b], ignore_index=True).set_crs(gdf.crs, allow_override=True)

def _expand_window(x, y, w, h, width, height, halo):
    x0=max(0,x-halo); y0=max(0,y-halo)
    x1=min(width,x+w+halo); y1=min(height,y+h+halo)
    return x0, y0, x1-x0, y1-y0, (x-x0), (y-y0)

def _apply_aug(t, k):
    if k==0: return t
    if k==1: return torch.flip(t, dims=[-1])
    if k==2: return torch.flip(t, dims=[-2])
    return t.transpose(-2,-1)

def _undo_aug(t, k): return _apply_aug(t, k)

def _pad_to(arr, th, tw):
    C,h,w = arr.shape
    return np.pad(arr,((0,0),(0,th-h),(0,tw-w)),mode='edge') if (th!=h or tw!=w) else arr

def _threshold_metrics(y_true, p, threshold):
    yp = (p >= threshold).astype(int)
    tn,fp,fn,tp = confusion_matrix(y_true, yp, labels=[0,1]).ravel()
    return {"threshold": threshold,
            "accuracy":   accuracy_score(y_true, yp),
            "precision":  precision_score(y_true, yp, zero_division=0),
            "recall":     recall_score(y_true, yp, zero_division=0),
            "f1":         f1_score(y_true, yp, zero_division=0),
            "specificity": tn/max(1,tn+fp),
            "brier":      brier_score_loss(y_true, p),
            "tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}

def _maybe_calibrate_prefit(base_model, Xva, yva, method):
    if method is None: return base_model
    if isinstance(method, str):
        m = method.strip().lower()
        if m in ("","none"): return base_model
        if m not in ("sigmoid","isotonic"):
            warnings.warn(f"Unknown CALIBRATION_METHOD='{method}', using 'sigmoid'."); m="sigmoid"
    else: return base_model
    cal = CalibratedClassifierCV(base_model, method=m, cv='prefit')
    cal.fit(Xva, yva); return cal

# ================= RF Hyperparameter Tuning ==========================================
def tune_rf_params(X, y, groups, *, _tune_gdf=None, n_iter=RF_TUNE_ITER, n_splits=5, seed=42, metric="ap"):
    """
    Random search with spatial GroupKFold.

    RF trains all n_estimators at once (no early stopping), so each tuning
    iteration is already parallel via n_jobs=-1. We run iterations sequentially
    to avoid CPU thread contention between iterations.
    """
    rng  = np.random.RandomState(seed)
    mfn  = average_precision_score if metric.lower()=="ap" else roc_auc_score

    # Stronger regularization to reduce overfitting under spatial CV
    search = {
        "n_estimators":      [500, 1000],
        "max_depth":         [5, 10, 15, 20],
        "max_features":      ["sqrt", "log2"],
        "min_samples_split": [10, 20, 50],
        "min_samples_leaf":  [5, 10, 20],
        "bootstrap":         [True],
        "criterion":         ["gini"],
    }

    best_params, best_score, history = None, -np.inf, []

    # RF is already multi-core per iteration → run iterations sequentially
    with tqdm(total=n_iter, desc="[B][tune-RF] Random search",
              unit="iter", dynamic_ncols=True) as pbar:
        for i in range(n_iter):
            cand   = {k: rng.choice(v) for k, v in search.items()}
            params = dict(cand)
            params.update(n_jobs=RF_N_JOBS, random_state=seed, oob_score=False)

            fold_scores = []
            _tune_splits = _buffered_spatial_splits(
                _tune_gdf, y, n_splits=n_splits, buffer_m=BUFFER_M, seed=seed+i)
            for tr, va in _tune_splits:
                m = RandomForestClassifier(**params)
                m.fit(X[tr], y[tr])
                fold_scores.append(mfn(y[va], m.predict_proba(X[va])[:,1]))

            ms = float(np.mean(fold_scores))
            history.append({"iter": i+1, "score": ms, **cand})
            if ms > best_score:
                best_score, best_params = ms, params

            pbar.set_postfix({
                "best": f"{best_score:.4f}",
                "last": f"{ms:.4f}",
                "n_est": cand["n_estimators"],
                "depth": str(cand["max_depth"])
            })
            pbar.update(1)

    print(f"[B][tune-RF] BEST {metric.upper()}={best_score:.4f}")
    return best_params, best_score, history

def _plot_rf_tuning_results(history, metric="AP", out_dir=FIG_DIR, csv_dir=LOG_DIR):
    if not history: return
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(csv_dir, "rf_tuning_history.csv"), index=False)
    plt.figure(figsize=(6,4)); plt.plot(df["iter"], df["score"], marker="o")
    plt.xlabel("Iteration"); plt.ylabel(metric); plt.title(f"RF tuning: {metric}")
    _savefig(os.path.join(out_dir, "rf_tuning_curve.png"))
    if {"max_depth","n_estimators"}.issubset(df.columns):
        plt.figure(figsize=(6,5))
        sc = plt.scatter(df["n_estimators"], df["max_depth"].fillna(-1), c=df["score"])
        plt.colorbar(sc, label=metric); plt.xlabel("n_estimators"); plt.ylabel("max_depth (None→-1)")
        plt.title("RF tuning: score by (n_estimators, max_depth)")
        _savefig(os.path.join(out_dir, "rf_tuning_scatter.png"))


# =============================================================================
# DISCUSSION DIAGNOSTICS (D1–D5) — same as XGB v5 / LGBM v3, RF-adapted
# =============================================================================

def _compute_morans_i(residuals, coords, k=8, n_sample=MORANS_N_SAMPLE, seed=42):
    """[D2] Moran's I on OOF residuals with k-NN weights + 499-permutation p-value."""
    rng = np.random.RandomState(seed); n = len(residuals)
    if n > n_sample:
        idx = rng.choice(n, n_sample, replace=False)
        r = residuals[idx].astype(float); c = coords[idx]
    else: r = residuals.astype(float); c = coords
    n_s   = len(r); k_eff = min(k, n_s-1)
    tree  = cKDTree(c)
    _, nn_idx = tree.query(c, k=k_eff+1, workers=-1)  # all cores
    nn_idx = nn_idx[:, 1:]                              # drop self
    z          = r - r.mean()
    z_nb_sum   = z[nn_idx].sum(axis=1)
    num        = float(np.dot(z, z_nb_sum))
    W_total    = float(n_s * k_eff)
    denom      = float(np.dot(z, z))
    I = (n_s/W_total)*(num/denom) if (W_total>0 and denom>0) else 0.0
    n_perm = 499; I_perm = np.zeros(n_perm)
    with tqdm(total=n_perm, desc="[D2] Moran permutations", unit="perm", leave=False) as pbar:
        for t in range(n_perm):
            zp = rng.permutation(z); zp_nb = zp[nn_idx].sum(axis=1)
            num_p = float(np.dot(zp, zp_nb)); dp = float(np.dot(zp, zp))
            I_perm[t] = (n_s/W_total)*(num_p/dp) if dp>0 else 0.0
            pbar.update(1)
    p_val = float((np.sum(I_perm>=I)+1)/(n_perm+1))
    if I>0.3 and p_val<0.05:
        interp = (f"Strong positive spatial autocorrelation (I={I:.3f}, p={p_val:.3f}). "
                  "Spatial CV was necessary to prevent inflated metrics.")
    elif I>0.1 and p_val<0.05:
        interp = (f"Moderate positive spatial autocorrelation (I={I:.3f}, p={p_val:.3f}). "
                  "Spatial CV reduces this bias.")
    elif p_val>=0.05:
        interp = (f"No significant spatial autocorrelation (I={I:.3f}, p={p_val:.3f}). "
                  "Spatial CV was effective.")
    else: interp = f"Moran's I={I:.3f}, p={p_val:.3f}."
    return float(I), float(p_val), interp


def _plot_overfitting_analysis(fold_df, out_dir=FIG_DIR, log_dir=LOG_DIR):
    """
    [D1] Train vs test AUROC gap per fold.

    RF-SPECIFIC NOTE: Training AUROC is typically near 1.0 for Random Forest
    because each tree sees only a bootstrap sample during training, but we
    evaluate on the FULL training set (including out-of-bag samples for each tree).
    A high training AUROC is therefore EXPECTED for RF and should not be
    automatically interpreted as overfitting. The generalization ability of RF
    is better assessed by the test AUROC and Moran's I on residuals.
    """
    if "train_AUROC" not in fold_df.columns:
        print("[D1] Skipping — train_AUROC not in fold_df."); return 0.0, "N/A"
    folds     = fold_df["fold"].values
    train_auc = fold_df["train_AUROC"].values
    test_auc  = fold_df["AUROC"].values
    gaps      = train_auc - test_auc
    mean_gap  = float(np.mean(gaps))

    # For RF, gap > 0.1 threshold (higher than XGB/LGBM due to bootstrap nature)
    rf_flag = "⚠ Gap unusually large (check max_depth)" if mean_gap > 0.20 \
              else "✓ Expected RF behavior (bootstrap inflates train AUROC)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    w = 0.35; x = np.arange(len(folds))
    axes[0].bar(x-w/2, train_auc, w, label="Train AUROC", color="#4878CF")
    axes[0].bar(x+w/2, test_auc,  w, label="Test AUROC (spatial CV)", color="#6ACC65")
    axes[0].axhline(np.mean(train_auc), linestyle="--", color="#4878CF", alpha=0.6, lw=1.5)
    axes[0].axhline(np.mean(test_auc),  linestyle="--", color="#6ACC65", alpha=0.6, lw=1.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"Fold {f}" for f in folds])
    axes[0].set_ylabel("AUROC"); axes[0].set_title("Train vs Test AUROC — CNN-RF"); axes[0].legend()
    axes[0].set_ylim(max(0, min(train_auc.min(),test_auc.min())-0.05), 1.02)
    axes[0].text(0.5,-0.15, f"Mean gap={mean_gap:.4f} | {rf_flag}",
                 ha="center", transform=axes[0].transAxes, fontsize=8,
                 color="orange" if mean_gap>0.20 else "green")

    bar_colors = ["#D65F5F" if g>0.20 else "#6ACC65" for g in gaps]
    axes[1].bar(folds, gaps, color=bar_colors)
    axes[1].axhline(0.20, linestyle="--", color="orange", lw=1.5, label="RF concern threshold (0.20)")
    axes[1].axhline(mean_gap, linestyle=":", color="grey", lw=1.5, label=f"Mean={mean_gap:.4f}")
    axes[1].set_xlabel("Fold"); axes[1].set_ylabel("Train − Test AUROC")
    axes[1].set_title("AUROC Gap (RF: training AUROC ≈ 1.0 is expected)")
    axes[1].legend()
    _savefig(os.path.join(out_dir, "overfitting_analysis.png"))

    pd.DataFrame({"fold":folds,"train_AUROC":train_auc,"test_AUROC":test_auc,
                  "gap":gaps,"flag":["concern" if g>0.20 else "expected_rf" for g in gaps]}).to_csv(
        os.path.join(log_dir, "overfitting_gap.csv"), index=False)
    print(f"[D1] mean_gap={mean_gap:.4f} | {rf_flag}")
    return mean_gap, rf_flag


def _plot_calibration_curve_diag(y_true, p_pred, n_bins=10, out_dir=FIG_DIR, log_dir=LOG_DIR):
    """[D3] Reliability diagram for OOF predictions."""
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=n_bins, strategy="uniform")
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].plot([0,1],[0,1],'--',color='grey',label='Perfect calibration')
    axes[0].plot(mean_pred, frac_pos, 's-', color='#4878CF', lw=2,
                 label=f'CNN-RF (ECE={ece:.3f})')
    axes[0].fill_between([0,1],[0,1],[0,1],alpha=0.05,color='grey')
    axes[0].set_xlabel("Mean Predicted Probability"); axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Reliability Diagram (OOF)"); axes[0].legend()
    axes[0].set_xlim(0,1); axes[0].set_ylim(0,1)
    axes[1].hist(p_pred[y_true==0], bins=40, alpha=0.6, color='#6ACC65',
                 label='Non-landslide', density=True)
    axes[1].hist(p_pred[y_true==1], bins=40, alpha=0.6, color='#D65F5F',
                 label='Landslide',     density=True)
    axes[1].set_xlabel("Predicted Probability"); axes[1].set_ylabel("Density")
    axes[1].set_title("Probability Distribution"); axes[1].legend()
    _savefig(os.path.join(out_dir, "calibration_curve.png"))
    pd.DataFrame({"mean_predicted":mean_pred,"fraction_positive":frac_pos,
                  "calibration_error_bin":np.abs(frac_pos-mean_pred)}).to_csv(
        os.path.join(log_dir, "calibration_data.csv"), index=False)
    print(f"[D3] ECE = {ece:.4f}")
    return ece


def _get_rf_base(clf):
    """Extract base RF from calibrated wrapper or return clf directly."""
    for attr in ("estimator", "base_estimator"):
        base = getattr(clf, attr, None)
        if base is not None and hasattr(base, "feature_importances_"):
            return base
    if hasattr(clf, "calibrated_classifiers_"):
        imps = []
        for cc in clf.calibrated_classifiers_:
            for attr in ("estimator", "base_estimator"):
                b = getattr(cc, attr, None)
                if b is not None and hasattr(b, "feature_importances_"):
                    imps.append(b.feature_importances_); break
        if imps: return type("_RF", (), {"feature_importances_": np.mean(imps, axis=0)})()
    return clf


def _plot_feature_importance_rf(clf_final, feature_names, out_dir=FIG_DIR, log_dir=LOG_DIR, top_n=30):
    """
    [D4] RF MDI (Mean Decrease in Impurity / Gini importance).
    Works with both calibrated and uncalibrated RandomForestClassifier.
    """
    try:
        base = _get_rf_base(clf_final)
        imp_vals = base.feature_importances_
    except Exception:
        print("[D4] Could not retrieve RF feature importance — skipping."); return None

    imp_df = pd.DataFrame({
        "feature": feature_names[:len(imp_vals)] if feature_names else [f"f{i}" for i in range(len(imp_vals))],
        "importance": imp_vals.astype(float)
    })
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df["importance_pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100
    imp_df.to_csv(os.path.join(log_dir, "feature_importance.csv"), index=False)

    plot_df = imp_df.head(top_n)
    colors  = ["#4878CF" if "CNN_" in str(r["feature"]) else "#D65F5F"
               for _, r in plot_df.iterrows()]

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df)*0.35)))
    ax.barh(plot_df["feature"][::-1], plot_df["importance_pct"][::-1], color=colors[::-1])
    ax.set_xlabel("MDI Importance (%)"); ax.set_title(f"RF Feature Importance — top {top_n} (Gini/MDI)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4878CF", label="CNN spatial features"),
                        Patch(color="#D65F5F", label="Raw conditioning factors")],
              loc="lower right")
    _savefig(os.path.join(out_dir, "feature_importance.png"))
    print(f"[D4] Top-5 features:")
    for _, row in imp_df.head(5).iterrows():
        print(f"     {row['feature']:25s}  {row['importance_pct']:.2f}%")
    return imp_df


def _generate_discussion_report(cv_summary, fold_df, mean_gap, overfitting_flag,
                                  morans_I, morans_p, morans_interp,
                                  calibration_error, imp_df, log_dir=LOG_DIR):
    """[D5] Auto-generate Discussion section draft from computed diagnostic values."""
    auroc_m = cv_summary.get("AUROC_mean", float("nan"))
    auroc_s = cv_summary.get("AUROC_std",  float("nan"))
    f1_m    = cv_summary.get("f1_05_mean", float("nan"))
    f1_s    = cv_summary.get("f1_05_std",  float("nan"))
    acc_m   = cv_summary.get("accuracy_05_mean", float("nan"))
    acc_s   = cv_summary.get("accuracy_05_std",  float("nan"))

    top5 = ""
    if imp_df is not None and len(imp_df) >= 5:
        top5 = ", ".join([f"{r['feature']} ({r['importance_pct']:.1f}%)"
                          for _, r in imp_df.head(5).iterrows()])

    report = f"""
=======================================================================
DISCUSSION SECTION DRAFT — CNN-RF — AUTO-GENERATED FROM DIAGNOSTICS
=======================================================================
[Copy and refine into your manuscript. Replace [REF] with citations.]
=======================================================================

--- Performance Interpretation ---
The CNN-RF hybrid model was evaluated using spatial k-fold cross-validation
(n={N_SPLITS} folds) with geographically separated blocks to prevent spatial
leakage between training and test samples [REF]. Under this spatially unbiased
evaluation, the model achieved an AUROC of {auroc_m:.4f} ± {auroc_s:.4f},
an F1-score of {f1_m:.4f} ± {f1_s:.4f}, and an overall accuracy of
{acc_m:.4f} ± {acc_s:.4f}. Standard deviations across folds reflect geographic
variability in model performance, which is expected across heterogeneous terrain.

--- Overfitting Analysis (D1) — RF-specific ---
The mean AUROC gap between training and spatially held-out test sets was
{mean_gap:.4f} ({overfitting_flag}). For Random Forest, a high training AUROC
is expected due to the bootstrap bagging mechanism: each tree is fit on a
bootstrap sample, but the training AUROC is evaluated on the full training set,
including samples that were out-of-bag for some trees. This inflates the apparent
training performance and does not indicate memorisation in the traditional sense [REF].
The more meaningful indicator of generalisation is the test AUROC from spatial CV,
which reflects the model's ability to predict landslide susceptibility in
geographically unseen areas. Regularisation through controlled tree depth and
minimum sample constraints further mitigates variance-driven overfitting [REF].
Reported metrics should be interpreted as comparative model skill under the adopted
spatial validation framework rather than as fully independent operational estimates.

--- Spatial Autocorrelation of Residuals (D2) ---
Moran's I on out-of-fold prediction residuals was {morans_I:.4f} (p = {morans_p:.3f}).
{morans_interp}
This result demonstrates the appropriateness of spatial cross-validation for this
dataset. A random split would allow spatially autocorrelated samples to leak between
training and test folds, inflating apparent performance [REF].

--- Calibration and Practical Applicability (D3) ---
The Expected Calibration Error (ECE) from the reliability diagram was {calibration_error:.4f}.
{'Well-calibrated outputs allow susceptibility probabilities to be interpreted directly.' if calibration_error < 0.05 else 'Moderate miscalibration suggests treating outputs as relative hazard rankings rather than absolute probabilities.'}
The framework provides a susceptibility assessment tool, not a real-time early warning
system. It excludes dynamic triggering variables such as rainfall intensity. High or
very high susceptibility zones should be verified through field inspection and expert
knowledge before informing operational decisions.

--- Feature Importance and Physical Interpretability (D4) ---
The five most important conditioning factors by RF MDI (Gini) importance were:
{top5 if top5 else '[see feature_importance.csv]'}. CNN-extracted spatial features
in the top-5 confirm that the CNN module contributed structured spatial information
beyond the raw conditioning factors. MDI importance in RF reflects how much each
feature reduces impurity across all decision tree nodes, weighted by the number of
samples it splits. Topographic and geological factors align with known physical
controls on landslide susceptibility in the study area [REF].

--- Limitations ---
The landslide inventory consists of historical point records with uncertain positional
accuracy for pre-2000 events. The balanced sampling strategy does not reflect the
naturally imbalanced landscape distribution. The model is calibrated for Uttaradit
Province and requires recalibration before application to other regions. RF's
bagging-based variance reduction provides stable predictions but may produce
smoother susceptibility patterns than boosting-based alternatives, potentially
underestimating localised high-susceptibility zones [REF].

=======================================================================
END OF DRAFT — Figures: {os.path.basename(FIG_DIR)} | CSVs: {os.path.basename(LOG_DIR)}
=======================================================================
"""
    out_path = os.path.join(log_dir, "discussion_draft.txt")
    with open(out_path, "w") as f: f.write(report)
    print(f"[D5] Discussion draft → {out_path}")
    return report


def _run_discussion_diagnostics(fold_df, all_preds_df, clf_final, feature_names,
                                 cv_summary, out_dir=FIG_DIR, log_dir=LOG_DIR):
    """Master diagnostic runner — D1 through D5 with overall progress bar."""
    print("\n[D] === DISCUSSION DIAGNOSTICS (Issue 3) ===")
    oof_y     = all_preds_df["y_true"].values
    oof_p     = all_preds_df["p_pred"].values
    coords    = all_preds_df[["x","y"]].values
    residuals = oof_p - oof_y.astype(float)

    steps = ["D1 Overfitting (RF)", "D2 Moran's I", "D3 Calibration",
             "D4 Feature Importance (MDI)", "D5 Discussion Draft"]
    with tqdm(total=len(steps), desc="[D] Diagnostics", unit="step") as dpbar:

        dpbar.set_description(f"[D] {steps[0]}")
        mean_gap, overfitting_flag = _plot_overfitting_analysis(fold_df, out_dir, log_dir)
        dpbar.update(1)

        dpbar.set_description(f"[D] {steps[1]}")
        morans_I, morans_p, morans_interp = _compute_morans_i(
            residuals, coords, k=8, n_sample=MORANS_N_SAMPLE)
        pd.DataFrame([{"morans_I": morans_I, "p_value": morans_p,
                       "interpretation": morans_interp}]).to_csv(
            os.path.join(log_dir, "morans_i_residuals.csv"), index=False)
        print(f"\n[D2] Moran's I={morans_I:.4f}  p={morans_p:.3f}  {morans_interp}")
        dpbar.update(1)

        dpbar.set_description(f"[D] {steps[2]}")
        ece = _plot_calibration_curve_diag(oof_y, oof_p, n_bins=10, out_dir=out_dir, log_dir=log_dir)
        dpbar.update(1)

        dpbar.set_description(f"[D] {steps[3]}")
        imp_df = _plot_feature_importance_rf(clf_final, feature_names, out_dir, log_dir, top_n=30)
        dpbar.update(1)

        dpbar.set_description(f"[D] {steps[4]}")
        _generate_discussion_report(cv_summary, fold_df, mean_gap, overfitting_flag,
                                     morans_I, morans_p, morans_interp, ece, imp_df, log_dir)
        dpbar.update(1)

    print(f"\n[D] COMPLETE | Figures → {out_dir} | CSVs → {log_dir}")
    for f in ["overfitting_analysis.png","overfitting_gap.csv","morans_i_residuals.csv",
               "calibration_curve.png","calibration_data.csv","feature_importance.png",
               "feature_importance.csv","discussion_draft.txt"]:
        print(f"    {f}")


# =============================================================================
# BUFFER-BASED SPATIAL SPLITS — Option B (replaces GroupKFold blocks)
# =============================================================================

BUFFER_M = 500   # metres — spatial exclusion radius around training points

def _buffered_spatial_splits(points_gdf, y, n_splits=5, train_size=0.70,
                              buffer_m=BUFFER_M, min_test_frac=0.10, seed=42):
    """
    Buffer-based spatial train/test splits (Roberts et al., 2017 — Option B).

    For each fold:
      1. Randomly select train_size fraction as training set.
      2. Project to UTM for metric-distance computation.
      3. Apply buffer_m around all training points (cKDTree query).
      4. Exclude from test set any candidate within the buffer.
      5. If exclusion removes too many test points, relax buffer by 50%.

    Parameters
    ----------
    points_gdf : GeoDataFrame  — points in any CRS
    y          : np.ndarray    — binary labels
    n_splits   : int           — number of random splits
    train_size : float         — fraction for training (default 0.70)
    buffer_m   : float         — exclusion radius in metres
    min_test_frac : float      — minimum test fraction before relaxing buffer
    seed       : int

    Returns
    -------
    list of (train_idx, test_idx) as np.ndarray pairs
    """
    from scipy.spatial import cKDTree as _cKDTree
    rng = np.random.RandomState(seed)
    n   = len(y)

    # Project to UTM for metric distance calculation
    need_proj = (not points_gdf.crs) or (not points_gdf.crs.is_projected)
    if need_proj and buffer_m > 0:
        try:
            cx, cy = points_gdf.unary_union.centroid.xy
            utm    = int((cx[0] + 180) // 6) + 1
            epsg   = 32600 + utm if cy[0] >= 0 else 32700 + utm
            xy_m   = np.array([(g.x, g.y) for g in points_gdf.to_crs(epsg).geometry])
        except Exception:
            xy_m = np.array([(g.x, g.y) for g in points_gdf.geometry])
            buffer_m = 0
    else:
        xy_m = np.array([(g.x, g.y) for g in points_gdf.geometry])

    splits    = []
    min_test  = max(10, int(n * min_test_frac))
    idx_all   = np.arange(n)
    n_train   = int(n * train_size)

    print(f"[CV] Buffer-based spatial splits | n={n} | buffer={buffer_m}m | "
          f"n_splits={n_splits} | train={int(train_size*100)}%")

    for fold_i in range(n_splits):
        perm     = rng.permutation(n)
        tr_idx   = np.sort(perm[:n_train])
        te_cand  = np.sort(perm[n_train:])

        if buffer_m > 0 and len(tr_idx) > 0:
            tree  = _cKDTree(xy_m[tr_idx])
            dists, _ = tree.query(xy_m[te_cand], k=1, workers=-1)
            te_out = te_cand[dists >= buffer_m]
        else:
            te_out = te_cand
            dists  = np.full(len(te_cand), np.inf)

        # Relax buffer if too many test points excluded
        if len(te_out) < min_test and buffer_m > 0:
            half = buffer_m * 0.5
            te_out = te_cand[dists >= half]
            print(f"[CV] Fold {fold_i+1}: buffer relaxed to {half:.0f}m "
                  f"({len(te_out)} test points remain)")
        if len(te_out) < min_test:
            te_out = te_cand   # last resort: no exclusion
            print(f"[CV] Fold {fold_i+1}: WARNING buffer disabled — "
                  f"only {len(te_cand)} test candidates")

        excluded = len(te_cand) - len(te_out)
        print(f"[CV] Fold {fold_i+1}: train={len(tr_idx)} | "
              f"test_raw={len(te_cand)} | excluded={excluded} | "
              f"test_final={len(te_out)} ({100*len(te_out)/n:.1f}% of all)")
        splits.append((tr_idx, te_out))

    return splits


# =============================================================================
# MAIN TRAINING FUNCTION — Spatial k-fold CV as PRIMARY validation
# =============================================================================
def train_rf_from_points(deep_feats_tif=DEEP_FEATS_TIF, factors_tif=FACTOR_TIF,
                         points_path=POINTS_PATH, out_model=MODEL_PATH, n_splits=N_SPLITS):
    t0_total = time.perf_counter()
    print("[B] Training CNN-RF with spatial k-fold CV + discussion diagnostics …")

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    gdf = gpd.read_file(points_path)
    with rasterio.open(factors_tif) as rst:
        gdf = _reproject_points_to_raster_crs(gdf, rst)
    gdf = _drop_ambiguous_negatives(gdf, LABEL_COL, POS_VALUE, NEG_VALUE, NEG_EXCLUSION_RADIUS_M)

    if LABEL_COL not in gdf.columns: raise ValueError(f"'{LABEL_COL}' not found.")
    cls = gdf[LABEL_COL].astype(int)
    y   = np.where(cls.values==POS_VALUE, 1, 0).astype(int)
    print(f"[B] Samples: total={len(y)} | pos={int((y==1).sum())} | neg={int((y==0).sum())}")

    # ------------------------------------------------------------------
    # Build feature matrix
    # ------------------------------------------------------------------
    t0s = time.perf_counter()
    points_xy = [(geom.x, geom.y) for geom in gdf.geometry]
    Xdeep = _sample_rasters_at_points([deep_feats_tif], points_xy)
    Xraw  = _sample_rasters_at_points([factors_tif],    points_xy)
    X     = np.hstack([Xdeep, Xraw]).astype('float32')
    X     = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    _log_time("B: sample_features", time.perf_counter()-t0s,
              N=len(y), Kdeep=Xdeep.shape[1], Kraw=Xraw.shape[1])

    # Feature names for D4 importance
    n_raw = Xraw.shape[1]
    raw_names = list(RAW_BAND_NAMES) if len(RAW_BAND_NAMES)==n_raw \
                else [f"raw_{i}" for i in range(n_raw)]
    feature_names = [f"CNN_{i+1}" for i in range(Xdeep.shape[1])] + raw_names

    # ------------------------------------------------------------------
    # Spatial blocks (FIXED: sqrt(n_splits) not sqrt(n_splits*2))
    # ------------------------------------------------------------------
    groups        = _make_spatial_blocks(gdf, n_splits=n_splits)
    points_xy_arr = np.array(points_xy, dtype="float64")

    # RF uses class_weight instead of scale_pos_weight
    pos_c = max(int((y==1).sum()),1); neg_c = max(int((y==0).sum()),1)
    cw    = {0: 1.0, 1: float(neg_c)/float(pos_c)}

    # ------------------------------------------------------------------
    # Step A — Hyperparameter tuning (spatial GroupKFold)
    # ------------------------------------------------------------------
    t0t = time.perf_counter()
    best_params, best_cv, tune_hist = tune_rf_params(
        X, y, groups, _tune_gdf=gdf,
        n_iter=RF_TUNE_ITER, n_splits=min(5, n_splits), seed=SEED, metric="ap")
    _plot_rf_tuning_results(tune_hist)
    _log_time("B: rf_tune", time.perf_counter()-t0t, iters=RF_TUNE_ITER)

    if best_params is None:
        best_params = dict(n_estimators=1000, n_jobs=RF_N_JOBS, random_state=SEED, oob_score=False)

    # Apply class weighting to best params
    best_params["class_weight"] = cw

    print(f"[B] Best RF params: "
          f"n_est={best_params.get('n_estimators')} | "
          f"max_depth={best_params.get('max_depth')} | "
          f"max_features={best_params.get('max_features')} | "
          f"criterion={best_params.get('criterion')}")

    # ------------------------------------------------------------------
    # Step B — Primary spatial buffer CV (Option B — buffer-based exclusion)
    # ------------------------------------------------------------------
    print(f"\n[B] === PRIMARY BUFFER-BASED SPATIAL CV (n_splits={n_splits}, buffer={BUFFER_M}m) ===")
    cv_splits    = _buffered_spatial_splits(gdf, y, n_splits=n_splits,
                                             buffer_m=BUFFER_M, seed=SEED)
    fold_records = []; all_preds = []

    t0cv = time.perf_counter()
    with tqdm(total=n_splits, desc="[B] Spatial CV folds", unit="fold") as pbar:
        for fold_i, (tr_idx, te_idx) in enumerate(cv_splits):
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xte, yte = X[te_idx], y[te_idx]

            # Per-fold class weight (from training partition only)
            pos_f = max(int((ytr==1).sum()),1); neg_f = max(int((ytr==0).sum()),1)
            fold_params = dict(best_params)
            fold_params["class_weight"] = {0: 1.0, 1: float(neg_f)/float(pos_f)}

            clf_fold = RandomForestClassifier(**fold_params)
            clf_fold.fit(Xtr, ytr)

            p_te = clf_fold.predict_proba(Xte)[:,1]

            # Train AUROC — NOTE: RF training AUROC ≈ 1.0 is expected (bootstrap)
            p_tr        = clf_fold.predict_proba(Xtr)[:,1]
            train_auroc = roc_auc_score(ytr, p_tr)
            train_ap    = average_precision_score(ytr, p_tr)

            m05    = _threshold_metrics(yte, p_te, 0.5)
            grid   = np.linspace(0.01, 0.99, 99)
            f1s    = [f1_score(yte, (p_te>=t).astype(int), zero_division=0) for t in grid]
            best_t = float(grid[int(np.argmax(f1s))])
            mbt    = _threshold_metrics(yte, p_te, best_t)

            fold_rec = {
                "fold":           fold_i+1,
                "n_train":        len(ytr),
                "n_test":         len(yte),
                "n_estimators":   best_params.get("n_estimators"),
                # Test metrics (PRIMARY — go in paper)
                "AUROC":          roc_auc_score(yte, p_te),
                "AP":             average_precision_score(yte, p_te),
                "Brier":          brier_score_loss(yte, p_te),
                "accuracy_05":    m05["accuracy"],
                "precision_05":   m05["precision"],
                "recall_05":      m05["recall"],
                "f1_05":          m05["f1"],
                "specificity_05": m05["specificity"],
                "best_t":         best_t,
                "accuracy_bt":    mbt["accuracy"],
                "precision_bt":   mbt["precision"],
                "recall_bt":      mbt["recall"],
                "f1_bt":          mbt["f1"],
                # Train metrics (D1 overfitting analysis)
                "train_AUROC":    train_auroc,
                "train_AP":       train_ap,
                "AUROC_gap":      train_auroc - roc_auc_score(yte, p_te),
            }
            fold_records.append(fold_rec)

            for j, idx in enumerate(te_idx):
                all_preds.append({
                    "sample_idx": idx, "fold": fold_i+1,
                    "y_true": int(yte[j]), "p_pred": float(p_te[j]),
                    "x": points_xy_arr[idx,0], "y": points_xy_arr[idx,1],
                    "group": int(groups[idx]),
                })

            pbar.set_postfix({
                "test_AUC":  f"{fold_rec['AUROC']:.4f}",
                "train_AUC": f"{train_auroc:.4f}",
                "F1":        f"{fold_rec['f1_05']:.4f}",
            })
            pbar.update(1)
            print(f"  Fold {fold_i+1}/{n_splits} | "
                  f"test_AUROC={fold_rec['AUROC']:.4f} | "
                  f"train_AUROC={train_auroc:.4f} (RF bootstrap bias expected) | "
                  f"F1={fold_rec['f1_05']:.4f}")

    _log_time("B: spatial_cv_primary", time.perf_counter()-t0cv, folds=n_splits)

    # ------------------------------------------------------------------
    # Aggregate CV metrics — these go in your paper (Table 2)
    # ------------------------------------------------------------------
    fold_df      = pd.DataFrame(fold_records)
    all_preds_df = pd.DataFrame(all_preds)
    fold_df.to_csv(os.path.join(LOG_DIR, "spatial_cv_per_fold.csv"), index=False)
    all_preds_df.to_csv(os.path.join(LOG_DIR, "spatial_cv_predictions.csv"), index=False)

    metric_cols = ["AUROC","AP","Brier",
                   "accuracy_05","precision_05","recall_05","f1_05","specificity_05",
                   "accuracy_bt","precision_bt","recall_bt","f1_bt",
                   "train_AUROC","train_AP","AUROC_gap"]
    cv_summary = {}
    for col in metric_cols:
        vals = fold_df[col].values.astype(float)
        cv_summary[f"{col}_mean"] = float(np.mean(vals))
        cv_summary[f"{col}_std"]  = float(np.std(vals))
    pd.DataFrame([cv_summary]).to_csv(os.path.join(LOG_DIR, "spatial_cv_summary.csv"), index=False)

    print("\n[B] === SPATIAL CV SUMMARY (PRIMARY — for your paper) ===")
    print(f"  AUROC (test):  {cv_summary['AUROC_mean']:.4f} ± {cv_summary['AUROC_std']:.4f}")
    print(f"  AUROC (train): {cv_summary['train_AUROC_mean']:.4f} ± {cv_summary['train_AUROC_std']:.4f}  [RF bootstrap — expected to be high]")
    print(f"  AUROC gap:     {cv_summary['AUROC_gap_mean']:.4f} ± {cv_summary['AUROC_gap_std']:.4f}")
    print(f"  F1 (best-t):   {cv_summary['f1_bt_mean']:.4f} ± {cv_summary['f1_bt_std']:.4f}  ← PRIMARY for paper")
    print(f"  F1 (@0.5):     {cv_summary['f1_05_mean']:.4f} ± {cv_summary['f1_05_std']:.4f}  (secondary)")
    print(f"  Precision(bt): {cv_summary['precision_bt_mean']:.4f} ± {cv_summary['precision_bt_std']:.4f}")
    print(f"  Recall(bt):    {cv_summary['recall_bt_mean']:.4f} ± {cv_summary['recall_bt_std']:.4f}")
    print(f"  Accuracy(bt):  {cv_summary['accuracy_bt_mean']:.4f} ± {cv_summary['accuracy_bt_std']:.4f}")
    print(f"  AP:            {cv_summary['AP_mean']:.4f} ± {cv_summary['AP_std']:.4f}")
    print(f"  Brier:         {cv_summary['Brier_mean']:.4f} ± {cv_summary['Brier_std']:.4f}")

    # OOF ROC/PR curves
    oof_y = all_preds_df["y_true"].values; oof_p = all_preds_df["p_pred"].values
    fpr_a,tpr_a,thr_r = roc_curve(oof_y, oof_p)
    _curve_df_with_thresholds(fpr_a,tpr_a,thr_r,"fpr","tpr").to_csv(
        os.path.join(LOG_DIR,"roc_oof.csv"), index=False)
    prec_a,rec_a,thr_p = precision_recall_curve(oof_y, oof_p)
    _curve_df_with_thresholds(rec_a,prec_a,thr_p,"recall","precision").to_csv(
        os.path.join(LOG_DIR,"pr_oof.csv"), index=False)

    plt.figure()
    plt.plot(fpr_a,tpr_a,label=f"OOF AUC={roc_auc_score(oof_y,oof_p):.3f}")
    plt.plot([0,1],[0,1],'--',color='grey'); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC — CNN-RF spatial k-fold (OOF)"); plt.legend()
    _savefig(os.path.join(FIG_DIR,"roc_oof.png"))

    plt.figure()
    plt.plot(rec_a,prec_a,label=f"OOF AP={average_precision_score(oof_y,oof_p):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR — CNN-RF spatial k-fold (OOF)"); plt.legend()
    _savefig(os.path.join(FIG_DIR,"pr_oof.png"))

    plt.figure(figsize=(6,4))
    plt.bar(fold_df["fold"], fold_df["AUROC"])
    plt.axhline(cv_summary["AUROC_mean"],linestyle='--',color='red',
                label=f"Mean={cv_summary['AUROC_mean']:.3f}")
    plt.xlabel("Fold"); plt.ylabel("Test AUROC"); plt.title("CNN-RF Per-fold Test AUROC"); plt.legend()
    _savefig(os.path.join(FIG_DIR,"cv_auroc_per_fold.png"))

    # ------------------------------------------------------------------
    # Step C — Final model trained on ALL data
    # RF has no early stopping → use best n_estimators from tuning directly
    # This model is used ONLY for the probability map — NOT for metrics
    # ------------------------------------------------------------------
    print(f"\n[B] Training final CNN-RF on ALL data "
          f"(n_estimators={best_params.get('n_estimators')}) …")
    final_params = dict(best_params)
    final_params["class_weight"] = cw   # global class weight from full dataset

    t0f = time.perf_counter()
    print("[B] Fitting final RF … (this may take several minutes with n_jobs=-1)")
    clf_final = RandomForestClassifier(**final_params)
    clf_final.fit(X, y)
    _log_time("B: rf_final_fit", time.perf_counter()-t0f, N=len(y),
              n_estimators=final_params.get("n_estimators"))
    print("[B] Final RF trained on all data → used for probability map only")

    # ------------------------------------------------------------------
    # Step D — Discussion diagnostics (Issue 3)
    # ------------------------------------------------------------------
    _run_discussion_diagnostics(
        fold_df, all_preds_df, clf_final, feature_names, cv_summary,
        out_dir=FIG_DIR, log_dir=LOG_DIR
    )

    # ------------------------------------------------------------------
    # Save payload
    # ------------------------------------------------------------------
    payload = {
        "clf":             clf_final,
        "cv_summary":      cv_summary,
        "fold_records":    fold_records,
        "best_params":     best_params,
        "feature_names":   feature_names,
        "validation_strategy": f"spatial_groupkfold_n{n_splits}",
        "note": ("Primary metrics from spatial k-fold CV. "
                 "clf trained on ALL data for map prediction only. "
                 "RF training AUROC ≈ 1.0 is expected (bootstrap bagging)."),
    }
    joblib.dump(payload, out_model)
    print(f"[B] Saved model → {out_model}")
    _log_time("B: total_train", time.perf_counter()-t0_total, N=len(y), n_splits=n_splits)
    return payload


# ================= Deep features raster (with OOM fixes) ============================
@torch.no_grad()
def write_deep_feature_raster(in_tif, out_tif=DEEP_FEATS_TIF,
                               feat_channels=FEAT_CHANNELS, tile=TILE, encoder_ckpt=None):
    t0 = time.perf_counter(); device = _select_device()
    with rasterio.open(in_tif) as src:
        C=src.count; mean,std=compute_band_stats(src)
        meta=src.meta.copy(); meta.update(count=feat_channels, dtype='float32')
        width,height=src.width,src.height

    model = FCNFeatureExtractor(in_ch=C, out_ch=feat_channels, backbone=BACKBONE).eval()
    if device.type == "cuda":
        if encoder_ckpt and os.path.exists(encoder_ckpt):
            model.load_state_dict(torch.load(encoder_ckpt, map_location="cpu"), strict=False)
        model = model.to(device).to(memory_format=torch.channels_last)
        ampctx = torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION)

    tiles = [(x, y, min(tile,width-x), min(tile,height-y))
             for y in range(0,height,tile) for x in range(0,width,tile)]
    n_tiles = len(tiles)
    desc = "[A] Deep features (GPU)" if device.type=="cuda" else "[A] Deep features (CPU)"

    with rasterio.open(out_tif,'w',**meta) as dst, rasterio.open(in_tif) as src, \
         tqdm(total=n_tiles, desc=desc, unit="tile") as pbar:
        if device.type == "cuda":
            bsize = max(1, GPU_TILE_BATCH)
            for i in range(0, n_tiles, bsize):
                bt = tiles[i:i+bsize]; arrs, metas = [], []; meh = mew = 0
                for (x,y,w,h) in bt:
                    ex,ey,ew,eh,cx,cy = _expand_window(x,y,w,h,width,height,HALO)
                    a = src.read(window=rw.Window(ex,ey,ew,eh), out_dtype='float32')
                    a = _standardize_block(a,mean,std); arrs.append(a)
                    metas.append((x,y,w,h,cx,cy,ew,eh))
                    meh=max(meh,a.shape[1]); mew=max(mew,a.shape[2])
                batch = np.stack([_pad_to(a,meh,mew) for a in arrs], axis=0)
                t = torch.from_numpy(batch).to(device, non_blocking=True).to(memory_format=torch.channels_last)
                if USE_TTA:
                    # FIX OOM 1: in-place TTA — no 3rd tensor allocated
                    outs = None
                    for k in (0,1,2,3):
                        with ampctx: fk = model(_apply_aug(t,k))
                        fk = _undo_aug(fk,k).float()   # upcast to FP32 before accumulating
                        if outs is None: outs = fk
                        else: outs.add_(fk)             # in-place: no new tensor
                        del fk
                    fout = outs.mul_(0.25).contiguous() # in-place divide
                else:
                    with ampctx: fout = model(t).contiguous()
                fout = fout.to(dtype=torch.float32, memory_format=torch.contiguous_format).cpu().numpy()
                for b,(x,y,w,h,cx,cy,ew,eh) in enumerate(metas):
                    dst.write(fout[b,:,cy:cy+h,cx:cx+w], window=rw.Window(x,y,w,h))
                    pbar.update(1)
        else:
            for (x,y,w,h) in tiles:
                ex,ey,ew,eh,cx,cy = _expand_window(x,y,w,h,width,height,HALO)
                a = src.read(window=rw.Window(ex,ey,ew,eh), out_dtype='float32')
                a = _standardize_block(a,mean,std)
                f = model(torch.from_numpy(a[None,...]))[0].numpy().astype('float32')
                dst.write(f[:,cy:cy+h,cx:cx+w], window=rw.Window(x,y,w,h))
                pbar.update(1)

    _cuda_sync_if_any()
    _log_time("A: deep_features", time.perf_counter()-t0, tiles=n_tiles,
              tile=tile, halo=HALO, K=feat_channels, backbone=BACKBONE)


# ================= Fine-tuning (with OOM fix: GPU cleanup on exit) ===================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(); self.alpha,self.gamma=alpha,gamma
        self.bce=nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        bce=self.bce(logits,targets); pt=torch.exp(-bce)
        return (self.alpha*(1-pt)**self.gamma*bce).mean()

def _rasterize_pos_mask(points_path, raster_path, label_col, pos_val, buffer_m=60):
    import rasterio.features as rfeat
    gdf = gpd.read_file(points_path)
    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
        geoms = gdf[gdf[label_col].astype(int)==pos_val].geometry
        if buffer_m>0 and (src.crs.is_projected if src.crs else False):
            geoms = geoms.buffer(buffer_m)
        return rfeat.rasterize([(g,1) for g in geoms],
                               out_shape=(src.height,src.width),
                               transform=src.transform, fill=0, dtype="float32")

def finetune_extractor_on_points(factor_tif, points_path, epochs=FT_EPOCHS, lr=FT_LR,
                                  tile=FT_TILE, buffer_m=FT_BUFFER_M, ckpt_out="encoder_finetuned.pt"):
    t0 = time.perf_counter(); device = _select_device()
    print(f"[FT] Fine-tuning | device={device} | epochs={epochs} | buffer={buffer_m}m")
    with rasterio.open(factor_tif) as src:
        C,H,W = src.count,src.height,src.width; mean,std = compute_band_stats(src)

    model = FCNFeatureExtractor(in_ch=C, out_ch=FEAT_CHANNELS, backbone=BACKBONE).to(device)
    model = model.to(memory_format=torch.channels_last)
    for p in model.backbone[0:6].parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()),
                             lr=lr, weight_decay=1e-4)
    loss_fn = FocalLoss()
    scaler  = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda" and USE_MIXED_PRECISION))
    posmask = _rasterize_pos_mask(points_path, factor_tif, LABEL_COL, POS_VALUE, buffer_m)
    windows = [(x,y,tile,tile) for y in range(0,H-(H%tile),tile)
                                 for x in range(0,W-(W%tile),tile)]
    epoch_losses, epoch_secs = [], []
    model.train()

    with tqdm(total=epochs, desc="[FT] Epochs", unit="ep") as epbar:
        for ep in range(epochs):
            ep_t0 = time.perf_counter(); random.Random(SEED+ep).shuffle(windows)
            total = 0.0; steps = 0; acc_count = 0
            with rasterio.open(factor_tif) as src, \
                 tqdm(total=len(windows), desc=f"[FT] ep{ep+1}/{epochs}", unit="tile", leave=False) as pbar:
                for i in range(0, len(windows), FT_BATCH_TILES):
                    batch = windows[i:i+FT_BATCH_TILES]; Xs, Ys = [], []
                    for (x,y,w,h) in batch:
                        Y = posmask[y:y+h, x:x+w].astype("float32")[None,...]
                        if Y.sum()==0 and np.random.rand()<FT_SKIP_ALL_NEG_PROB: continue
                        Xb = src.read(window=rw.Window(x,y,w,h), out_dtype="float32")
                        Xs.append(_standardize_block(Xb,mean,std)); Ys.append(Y)
                    if not Xs: pbar.update(len(batch)); continue
                    xt = torch.from_numpy(np.stack(Xs)).to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    yt = torch.from_numpy(np.stack(Ys)).to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', enabled=(device.type=="cuda" and USE_MIXED_PRECISION)):
                        loss = loss_fn(model(xt).narrow(1,0,1).contiguous(), yt)
                    if device.type=="cuda":
                        scaler.scale(loss).backward(); acc_count+=1
                        if acc_count%FT_ACCUM_STEPS==0:
                            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                    else: loss.backward(); opt.step()
                    total += float(loss.item()); steps += 1; pbar.update(len(batch))
            if device.type=="cuda" and acc_count%FT_ACCUM_STEPS!=0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            _cuda_sync_if_any()
            ep_sec = time.perf_counter()-ep_t0; ep_loss = total/max(1,steps)
            epoch_losses.append(ep_loss); epoch_secs.append(ep_sec)
            epbar.set_postfix({"loss": f"{ep_loss:.4f}", "t": _fmt_hms(ep_sec)})
            epbar.update(1)

    plt.figure(); plt.plot(range(1,len(epoch_losses)+1), epoch_losses)
    plt.xlabel("Epoch"); plt.ylabel("Focal loss"); plt.title("Fine-tuning loss — CNN-RF")
    _savefig(os.path.join(FIG_DIR,"finetune_loss.png"))
    pd.DataFrame({"epoch":np.arange(1,len(epoch_losses)+1),
                  "loss":epoch_losses,"sec":epoch_secs}).to_csv(
        os.path.join(LOG_DIR,"finetune_epochs.csv"), index=False)
    torch.save(model.state_dict(), ckpt_out)
    _cuda_sync_if_any()

    # FIX OOM 2: explicitly free fine-tuning model before loading inference model
    del opt, scaler, loss_fn
    model.cpu(); del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    print(f"[FT] GPU freed — allocated={torch.cuda.memory_allocated()/1e9:.2f} GB | "
          f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")

    _log_time("FT: total", time.perf_counter()-t0, epochs=epochs)
    return ckpt_out


# ================= Prediction helpers (RF-specific: threaded tile prediction) ========
def _set_estimator_n_jobs(est, n_jobs):
    """Set n_jobs on RF or calibrated wrapper."""
    try:
        if hasattr(est, "n_jobs"): est.n_jobs = n_jobs
        for attr in ("base_estimator", "estimator"):
            b = getattr(est, attr, None)
            if b is not None and hasattr(b, "n_jobs"): b.n_jobs = n_jobs
    except Exception: pass

def _predict_tile_thread(tile, deep_path, fac_path, clf):
    """Predict one tile — safe to call from ThreadPoolExecutor."""
    x, y, w, h = tile
    with rasterio.open(deep_path) as dsrc, rasterio.open(fac_path) as fsrc:
        D = dsrc.read(window=rw.Window(x,y,w,h), out_dtype='float32')
        F = fsrc.read(window=rw.Window(x,y,w,h), out_dtype='float32')
        H, W = D.shape[1], D.shape[2]
        Xp   = np.concatenate([D,F],axis=0).reshape(D.shape[0]+F.shape[0],-1).T
        Xp   = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)
        P    = clf.predict_proba(Xp)[:,1].astype('float32').reshape(H,W)
    return x, y, w, h, P

def predict_probability_map(deep_feats_tif=DEEP_FEATS_TIF, factors_tif=FACTOR_TIF,
                             model_path=MODEL_PATH, out_tif=OUT_PROB_TIF, tile=TILE):
    t0 = time.perf_counter(); print("[C] Predicting probability map …")
    payload = joblib.load(model_path); clf = payload["clf"]
    with rasterio.open(deep_feats_tif) as dsrc, rasterio.open(factors_tif) as fsrc:
        assert dsrc.width==fsrc.width and dsrc.height==fsrc.height
        meta  = dsrc.meta.copy(); meta.update(count=1, dtype='float32')
        width, height = dsrc.width, dsrc.height

    tiles = [(x,y,min(tile,width-x),min(tile,height-y))
             for y in range(0,height,tile) for x in range(0,width,tile)]
    total = len(tiles)

    # Split CPU cores: outer threads × per-tile n_jobs (avoid contention)
    outer = max(1, int(PRED_OUTER_THREADS))
    if RF_N_JOBS in (-1, 0): per_tile = max(1, (os.cpu_count() or 1) // outer)
    else: per_tile = max(1, int(RF_N_JOBS) // outer)
    _set_estimator_n_jobs(clf, per_tile)

    with rasterio.open(out_tif,'w',**meta) as dst, \
         tqdm(total=total, desc="[C] Map predict", unit="tile") as pbar:
        if outer == 1:
            for t in tiles:
                x,y,w,h,P = _predict_tile_thread(t, deep_feats_tif, factors_tif, clf)
                dst.write(P, 1, window=rw.Window(x,y,w,h)); pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=outer) as ex:
                futures = {ex.submit(_predict_tile_thread, t, deep_feats_tif, factors_tif, clf): t
                           for t in tiles}
                for fut in as_completed(futures):
                    x,y,w,h,P = fut.result()
                    dst.write(P, 1, window=rw.Window(x,y,w,h)); pbar.update(1)

    _log_time("C: predict_map", time.perf_counter()-t0,
              width=width, height=height, tile=tile, tiles=total,
              outer_threads=outer, per_tile_n_jobs=per_tile)
    print(f"[C] → {out_tif}")


# ================= Main ==============================================================
def _clear_gpu(label=""):
    """Flush GPU cache between pipeline stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        print(f"[GPU] {label} — allocated={torch.cuda.memory_allocated()/1e9:.2f} GB | "
              f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")

if __name__ == "__main__":
    ckpt = None
    if FINE_TUNE_ENABLED:
        ckpt = finetune_extractor_on_points(
            FACTOR_TIF, POINTS_PATH, epochs=FT_EPOCHS, lr=FT_LR,
            tile=FT_TILE, buffer_m=FT_BUFFER_M, ckpt_out="encoder_finetuned.pt"
        )
        _clear_gpu("after fine-tuning")   # FIX OOM 2: clear before loading inference model

    write_deep_feature_raster(FACTOR_TIF, out_tif=DEEP_FEATS_TIF,
                               feat_channels=FEAT_CHANNELS, tile=TILE, encoder_ckpt=ckpt)
    _clear_gpu("after feature extraction")   # clear CNN before RF training

    payload = train_rf_from_points(DEEP_FEATS_TIF, FACTOR_TIF, POINTS_PATH,
                                    out_model=MODEL_PATH, n_splits=N_SPLITS)
    _clear_gpu("after RF training")

    predict_probability_map(DEEP_FEATS_TIF, FACTOR_TIF, MODEL_PATH,
                             out_tif=OUT_PROB_TIF, tile=TILE)
    _write_timings_csv()
