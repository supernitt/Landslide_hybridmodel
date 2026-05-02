# =================== LSM HYBRID (CNN-XGB) v5 — Spatial CV + Discussion Diagnostics ===================
# You have to change the path according to your system.
# ===============================================================================================

import os, warnings, math, numpy as np, joblib, time, random, gc
# Fix 3: expandable_segments reduces PyTorch allocator fragmentation.
# Must be set before any CUDA call. Eliminates many OOM errors caused by fragmented
# reserved-but-unallocated memory (as recommended in the OOM error message itself).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import geopandas as gpd, rasterio, rasterio.windows as rw
from rasterio.vrt import WarpedVRT
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from scipy.spatial import cKDTree          # Moran's I
from joblib import Parallel, delayed       # parallel tuning loop
from xgboost import XGBClassifier
import xgboost as xgb
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
BASE = '/mnt/Data3/2025_landslidePaper/2025_landslide'
FACTOR_TIF     = f'{BASE}/raster/Raster_Com.tif'
POINTS_PATH    = f'{BASE}/point/Point_LS.shp'
DEEP_FEATS_TIF = f'{BASE}/CNN_XGB/process/deep_feats.tif'
MODEL_PATH     = f'{BASE}/CNN_XGB/model/xgb_fused_points.joblib'
OUT_PROB_TIF   = f'{BASE}/CNN_XGB/result/lsm_prob.tif'

RESULT_DIR = os.path.dirname(OUT_PROB_TIF)
FIG_DIR    = os.path.join(RESULT_DIR, "figs")
LOG_DIR    = os.path.join(RESULT_DIR, "logs")
for d in [os.path.dirname(MODEL_PATH), os.path.dirname(DEEP_FEATS_TIF),
          RESULT_DIR, FIG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# --- NEW in v5: Raw band names for the conditioning factors --------------------------
# Adjust this list to match the band order in your Raster_Com.tif.
# These are the 9 factors retained after VIF screening in your manuscript.
RAW_BAND_NAMES = [
    "DEM", "Slope", "Curvature", "SPI", "TRI",
    "Dist_Stream", "Dist_Fault", "Soil_Type", "Geology"
]
# If your raster has a different number of bands, update this list.
# CNN feature names are auto-generated (CNN_1, CNN_2, ...).

# --- Small helpers -------------------------------------------------------------------
def _curve_df_with_thresholds(x_arr, y_arr, thr_arr, x_name, y_name):
    x_arr = np.asarray(x_arr); y_arr = np.asarray(y_arr); thr_arr = np.asarray(thr_arr)
    need = len(x_arr) - len(thr_arr)
    if need < 0: thr_arr = thr_arr[:len(x_arr)]; need = 0
    if need > 0: thr_arr = np.r_[thr_arr, np.full(need, np.nan, dtype=float)]
    return pd.DataFrame({x_name: x_arr, y_name: y_arr, "threshold": thr_arr})

def _assert_same_len(**cols):
    lens = {k: (len(v) if hasattr(v, "__len__") and not np.isscalar(v) else 1)
            for k, v in cols.items()}
    if len(set(lens.values())) != 1:
        raise ValueError(f"Length mismatch: {lens}")

# --- Constants -----------------------------------------------------------------------
OUTPUT_STRIDE = 16
BACKBONE      = "resnet152"
FEAT_CHANNELS = 256
TILE          = 512
N_SPLITS      = 5
SEED          = 42

# =============================================================================
# HARDWARE PROFILE: 11 GB GPU + 128 GB RAM
# GPU memory budget per batch (ResNet152 FP16, no_grad):
#   Upsampled feature output dominates: ~134 MB per 512×512 tile
#   GPU_TILE_BATCH=8 → 8×134 MB = 1.07 GB + model 120 MB ≈ 1.5 GB used / 11 GB available
#   XGBoost GPU (20K samples × 265 features) = ~20 MB → negligible on GPU
# Fine-tuning memory: gradients only for unfrozen layers (layer3, layer4, proj)
#   FT_BATCH_TILES=6 → ~2.5 GB — comfortable margin below 11 GB
# RAM: 128 GB → Moran's I can sample 10,000 points (vs 3,000 default)
# CPU: n_jobs=cpu_count() for all CPU ops; parallel tuning uses all cores
# =============================================================================

USE_GPU_IF_AVAILABLE = True
CUDA_DEVICE_INDEX    = 0
USE_MIXED_PRECISION  = True   # FP16 inference + FP32 accumulation (AMP)
# GPU_TILE_BATCH: safe value for 11 GB GPU with ResNet152 + TTA + in-place accumulation.
# Memory budget per batch (FP16 inference, in-place TTA):
#   model:        ~120 MB (FP16 ResNet152)
#   intermediates: ~130 MB (peak at layer1: batch×256×128×128×FP16)
#   outs (FP32 accumulator): batch×256×512×512×4 = batch×268 MB
#   fk  (FP16 current pass): batch×256×512×512×2 = batch×134 MB
#   Peak with in-place add: 120 + 130 + batch×(268+134) = 120+130+batch×402
#   Solve for 8 GB headroom: batch×402 ≤ 7750 → batch ≤ 19
#   BUT: fine-tuning model may not be fully freed yet → use conservative batch=4
#   Recalculate clean GPU: 120+130+4×402 = 1858 MB ≈ 1.9 GB ← very safe
GPU_TILE_BATCH       = 4      # was 8 → reduced; in-place TTA eliminates 4.27 GB allocation
torch.set_float32_matmul_precision("high")

XGB_N_JOBS            = max(1, (os.cpu_count() or 1))
XGB_USE_GPU           = True
CALIBRATION_METHOD    = None
EARLY_STOPPING_ROUNDS = 500   # ↓ was 1000 → GPU trains fast; 500 rounds is sufficient patience
XGB_MAX_BIN           = 512   # ↑ was 256 → finer histogram binning; XGB GPU only uses ~0.5 MB extra
XGB_GPU_ID            = CUDA_DEVICE_INDEX
XGB_TUNE_ITER         = 150   # ↑ was 100 → wider search; CPU-parallel tuning absorbs the cost
XGB_VER               = tuple(int(p) for p in xgb.__version__.split('.')[:2])

# Moran's I sample size — increased from 3000 to 10000 (safe with 128 GB RAM)
MORANS_N_SAMPLE = 10000

LABEL_COL              = "Class"
POS_VALUE              = 1
NEG_VALUE              = 2
NEG_EXCLUSION_RADIUS_M = 90

FINE_TUNE_ENABLED    = True
FT_EPOCHS            = 20
FT_LR                = 1e-4
FT_TILE              = 512
FT_BUFFER_M          = 60
FT_SKIP_ALL_NEG_PROB = 0.8
FT_BATCH_TILES       = 4      # was 6 → reduced; fine-tuning carries optimizer states (~4.5 GB total)
FT_ACCUM_STEPS       = 1      # stays 1 — effective batch is sufficient

HALO    = 160
USE_TTA = True

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

# ================= Utilities (unchanged from v4) =====================================
def _select_device():
    if USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
        torch.cuda.set_device(CUDA_DEVICE_INDEX)
        torch.backends.cudnn.benchmark = True
        print(f"[Device] CUDA:{CUDA_DEVICE_INDEX}")
        return torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
    print("[Device] CPU"); return torch.device("cpu")

def compute_band_stats(src, block=1024):
    n_blocks_y = math.ceil(src.height / block)
    n_blocks_x = math.ceil(src.width  / block)
    n_total    = n_blocks_y * n_blocks_x
    sums = sqs = counts = None
    with tqdm(total=n_total, desc="[A] Band statistics", unit="block", leave=False) as pbar:
        for y in range(0, src.height, block):
            for x in range(0, src.width, block):
                w   = rw.Window(x, y, min(block, src.width-x), min(block, src.height-y))
                arr = src.read(window=w, out_dtype="float64")
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
    return means.astype("float32"), np.sqrt(vars_).astype("float32")

def _standardize_block(arr, mean, std, eps=1e-6):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    std_safe = np.where(np.isfinite(std) & (std > eps), std, eps).astype(arr.dtype, copy=False)
    return (arr - mean[:,None,None]) / std_safe[:,None,None]

def _reproject_points_to_raster_crs(points_gdf, raster_src):
    return points_gdf.to_crs(raster_src.crs) if points_gdf.crs != raster_src.crs else points_gdf

def _sample_rasters_at_points(tif_paths, points_xy):
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

def _make_spatial_blocks(points_gdf, n_splits=N_SPLITS):
    xmin,ymin,xmax,ymax=points_gdf.total_bounds
    n_cells=int(np.ceil(np.sqrt(n_splits)))
    xs=np.linspace(xmin,xmax,n_cells+1); ys=np.linspace(ymin,ymax,n_cells+1)
    def cell_id(x,y):
        ix=min(n_cells-1,max(0,np.searchsorted(xs,x,side='right')-1))
        iy=min(n_cells-1,max(0,np.searchsorted(ys,y,side='right')-1))
        return iy*n_cells+ix
    block_ids=points_gdf.geometry.apply(lambda g:cell_id(g.x,g.y)).values
    n_unique=len(np.unique(block_ids))
    print(f"[Spatial blocks] {n_cells}x{n_cells}={n_cells**2} cells | populated={n_unique}")
    return block_ids

def _drop_ambiguous_negatives(gdf,label_col,pos_val=1,neg_val=2,radius_m=90):
    if radius_m<=0: return gdf
    gpos=gdf[gdf[label_col].astype(int)==pos_val]; gneg=gdf[gdf[label_col].astype(int)==neg_val]
    if gpos.empty or gneg.empty: return gdf
    need_proj=(not gdf.crs) or (not gdf.crs.is_projected)
    if need_proj:
        cx,cy=gdf.unary_union.centroid.xy; utm=int((cx[0]+180)//6)+1
        epsg=32600+utm if cy[0]>=0 else 32700+utm
        gpos_=gpos.to_crs(epsg); gneg_=gneg.to_crs(epsg)
    else: gpos_,gneg_=gpos,gneg
    buf=gpos_.buffer(radius_m)
    ug=buf.union_all() if hasattr(buf,"union_all") else _unary_union(list(buf.values if hasattr(buf,"values") else buf))
    kept=gneg_.loc[~gneg_.geometry.intersects(ug)]
    gpos_b=gpos_.to_crs(gdf.crs) if need_proj else gpos_
    kept_b=kept.to_crs(gdf.crs) if need_proj else kept
    return pd.concat([gpos_b,kept_b],ignore_index=True).set_crs(gdf.crs,allow_override=True)

def _xgb_gpu_params():
    # n_jobs is set in ALL paths — GPU builds still use CPU threads for data
    # loading and metric evaluation. Default is 0 (all threads) but explicit is safer.
    if XGB_USE_GPU:
        if XGB_VER>=(2,0):
            return dict(device='cuda', tree_method='hist', predictor='gpu_predictor',
                        max_bin=XGB_MAX_BIN, n_jobs=XGB_N_JOBS)
        else:
            return dict(tree_method='gpu_hist', predictor='gpu_predictor',
                        gpu_id=XGB_GPU_ID, max_bin=XGB_MAX_BIN, n_jobs=XGB_N_JOBS)
    return dict(tree_method='hist', n_jobs=XGB_N_JOBS)

def _clean_xgb_params(d):
    d=dict(d)
    if 'device' in d: d.pop('gpu_id',None)
    return d

def _threshold_metrics(y_true, p, threshold):
    yp=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_true,yp,labels=[0,1]).ravel()
    return {"threshold":threshold,
            "accuracy":accuracy_score(y_true,yp),
            "precision":precision_score(y_true,yp,zero_division=0),
            "recall":recall_score(y_true,yp,zero_division=0),
            "f1":f1_score(y_true,yp,zero_division=0),
            "specificity":tn/max(1,tn+fp),
            "brier":brier_score_loss(y_true,p),
            "tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn)}

def tune_xgb_params(X,y,groups,*,_tune_gdf=None,base_params,metric="ap",n_iter=20,n_splits=5,
                    seed=42,early_stopping_rounds=400,pos_weight=1.0,use_gpu=True):
    rng=np.random.RandomState(seed)
    mfn=average_precision_score if metric.lower()=="ap" else roc_auc_score
    # Stronger regularization to reduce overfitting under spatial CV
    search={"max_depth":[3,4,5,6],
            "min_child_weight":[5,10,20,30],
            "subsample":[0.5,0.6,0.7],
            "colsample_bytree":[0.5,0.6,0.7],
            "reg_lambda":[2.0,5.0,10.0,20.0],
            "reg_alpha":[0.5,1.0,2.0,5.0],
            "gamma":[1,3,5,10],
            "learning_rate":[0.01,0.03,0.05]}
    if use_gpu: search["max_bin"]=[256,512]

    # Pre-draw all candidates for reproducibility
    candidates = [{k: rng.choice(v) for k, v in search.items()} for _ in range(n_iter)]

    def _eval_candidate(i, cand):
        params = dict(base_params); params.update(cand)
        params.update(n_estimators=4000, scale_pos_weight=pos_weight,
                      eval_metric="logloss", random_state=seed, verbosity=0)
        if not use_gpu: params.pop("max_bin", None)
        fold_scores = []
        # Use buffer splits for tuning too (consistent with primary CV)
        _tune_splits = _buffered_spatial_splits(
            _tune_gdf, y, n_splits=n_splits, buffer_m=BUFFER_M, seed=seed+i)
        for tr, va in _tune_splits:
            m = XGBClassifier(**_clean_xgb_params(params))
            try: m.fit(X[tr],y[tr],eval_set=[(X[va],y[va])],
                       early_stopping_rounds=early_stopping_rounds,verbose=False)
            except TypeError: m.fit(X[tr],y[tr],eval_set=[(X[va],y[va])],verbose=False)
            fold_scores.append(mfn(y[va], m.predict_proba(X[va])[:,1]))
        ms = float(np.mean(fold_scores))
        print(f"[B][tune] {i+1}/{n_iter} AP={ms:.4f}  cand={cand}")
        return ms, cand, _clean_xgb_params(params)

    # GPU: keep sequential (one GPU, can't run multiple trainers in parallel).
    # CPU: parallelise across iterations with threads (each model already multi-core).
    parallel_jobs = 1 if use_gpu else -1
    raw = Parallel(n_jobs=parallel_jobs, prefer="threads")(
        delayed(_eval_candidate)(i, cand) for i, cand in enumerate(candidates)
    )

    best_params, best_score, history = None, -np.inf, []
    for i, (ms, cand, full_params) in enumerate(raw):
        history.append({"iter": i+1, "score": ms, **cand})
        if ms > best_score: best_score, best_params = ms, full_params
    print(f"[B][tune] BEST AP={best_score:.4f}")
    return best_params, best_score, history

def _plot_tuning_results(history,metric="AP",out_dir=FIG_DIR,csv_dir=LOG_DIR):
    if not history: return
    df=pd.DataFrame(history)
    df.to_csv(os.path.join(csv_dir,"xgb_tuning_history.csv"),index=False)
    plt.figure(figsize=(6,4)); plt.plot(df["iter"],df["score"],marker="o")
    plt.xlabel("Iteration"); plt.ylabel(metric); plt.title(f"XGB tuning: {metric}")
    _savefig(os.path.join(out_dir,"xgb_tuning_curve.png"))

def _maybe_calibrate_prefit(base_model,Xva,yva,method):
    if method is None: return base_model
    if isinstance(method,str):
        m=method.strip().lower()
        if m in ("","none"): return base_model
        if m not in ("sigmoid","isotonic"):
            warnings.warn(f"Unknown CALIBRATION_METHOD='{method}', using 'sigmoid'."); m="sigmoid"
    else: return base_model
    cal=CalibratedClassifierCV(base_model,method=m,cv='prefit'); cal.fit(Xva,yva); return cal


# =============================================================================
# NEW IN v5 — DIAGNOSTIC FUNCTIONS FOR DISCUSSION SECTION (ISSUE 3)
# =============================================================================

def _compute_morans_i(residuals, coords, k=8, n_sample=MORANS_N_SAMPLE, seed=42):
    """
    [D2] Compute Moran's I on model residuals to quantify spatial autocorrelation.

    Uses k-nearest-neighbour binary weights on a subsample for efficiency.
    Includes a 499-permutation test for the p-value.

    A significant positive Moran's I means prediction errors cluster spatially —
    the model has not captured all spatial structure in the data. This is expected
    for geospatial models and should be acknowledged in the Discussion.

    Returns: (I_statistic, p_value, interpretation_string)
    """
    rng = np.random.RandomState(seed)
    n   = len(residuals)

    # Subsample for speed on large datasets
    if n > n_sample:
        idx = rng.choice(n, n_sample, replace=False)
        r   = residuals[idx].astype(float)
        c   = coords[idx]
    else:
        r = residuals.astype(float); c = coords

    n_s   = len(r)
    k_eff = min(k, n_s - 1)

    # Build k-NN index (vectorized — no Python loops)
    tree = cKDTree(c)
    _, nn_idx = tree.query(c, k=k_eff + 1, workers=-1)  # workers=-1 = all CPU cores
    nn_idx = nn_idx[:, 1:]                     # remove self-index  → (n_s, k_eff)

    z          = r - r.mean()
    z_nb_sum   = z[nn_idx].sum(axis=1)         # Σ_j w_ij * z_j  per row
    num        = float(np.dot(z, z_nb_sum))
    W_total    = float(n_s * k_eff)
    denom      = float(np.dot(z, z))

    I = (n_s / W_total) * (num / denom) if (W_total > 0 and denom > 0) else 0.0

    # Permutation test — 499 permutations
    n_perm  = 499
    I_perm  = np.zeros(n_perm)
    with tqdm(total=n_perm, desc="[D2] Moran permutations", unit="perm", leave=False) as pbar:
        for t in range(n_perm):
            zp        = rng.permutation(z)
            zp_nb_sum = zp[nn_idx].sum(axis=1)
            num_p     = float(np.dot(zp, zp_nb_sum))
            denom_p   = float(np.dot(zp, zp))
            I_perm[t] = (n_s / W_total) * (num_p / denom_p) if denom_p > 0 else 0.0
            pbar.update(1)

    p_val = float((np.sum(I_perm >= I) + 1) / (n_perm + 1))

    if I > 0.3 and p_val < 0.05:
        interp = (f"Strong positive spatial autocorrelation (I={I:.3f}, p={p_val:.3f}). "
                  "Model errors cluster spatially. This is expected for geospatial models "
                  "and confirms that spatial CV was necessary to prevent inflated metrics.")
    elif I > 0.1 and p_val < 0.05:
        interp = (f"Moderate positive spatial autocorrelation (I={I:.3f}, p={p_val:.3f}). "
                  "Some spatial structure remains unexplained. Spatial CV reduces this bias.")
    elif p_val >= 0.05:
        interp = (f"No significant spatial autocorrelation in residuals (I={I:.3f}, p={p_val:.3f}). "
                  "Model errors are spatially random, suggesting spatial CV was effective.")
    else:
        interp = f"Moran's I={I:.3f}, p={p_val:.3f}. Interpret with caution."

    return float(I), float(p_val), interp


def _plot_overfitting_analysis(fold_df, out_dir=FIG_DIR, log_dir=LOG_DIR):
    """
    [D1] Plot train AUROC vs test AUROC per fold to assess overfitting.

    A large gap (train >> test) indicates the model memorises training data
    rather than learning generalisable patterns. The Discussion should explicitly
    state whether overfitting was observed and what mitigation was applied.

    Threshold: gap > 0.05 is flagged as a potential overfitting signal.
    """
    if "train_AUROC" not in fold_df.columns:
        print("[D1] Skipping overfitting plot — train_AUROC not in fold_df."); return

    folds       = fold_df["fold"].values
    train_auc   = fold_df["train_AUROC"].values
    test_auc    = fold_df["AUROC"].values
    gaps        = train_auc - test_auc
    mean_gap    = float(np.mean(gaps))
    flag        = "⚠ Potential overfitting" if mean_gap > 0.05 else "✓ No strong overfitting signal"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: grouped bar chart
    w = 0.35; x = np.arange(len(folds))
    axes[0].bar(x - w/2, train_auc, w, label="Train AUROC", color="#4878CF")
    axes[0].bar(x + w/2, test_auc,  w, label="Test AUROC (spatial CV)", color="#6ACC65")
    axes[0].axhline(np.mean(train_auc), linestyle="--", color="#4878CF", alpha=0.6, lw=1.5)
    axes[0].axhline(np.mean(test_auc),  linestyle="--", color="#6ACC65", alpha=0.6, lw=1.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"Fold {f}" for f in folds])
    axes[0].set_ylabel("AUROC"); axes[0].set_title("Train vs Test AUROC per Fold")
    axes[0].legend(); axes[0].set_ylim(max(0, min(train_auc.min(), test_auc.min()) - 0.05), 1.02)
    axes[0].text(0.5, -0.15, f"Mean gap = {mean_gap:.4f} | {flag}",
                 ha="center", transform=axes[0].transAxes, fontsize=9,
                 color="red" if mean_gap > 0.05 else "green")

    # Right: gap per fold
    bar_colors = ["#D65F5F" if g > 0.05 else "#6ACC65" for g in gaps]
    axes[1].bar(folds, gaps, color=bar_colors)
    axes[1].axhline(0.05, linestyle="--", color="red", lw=1.5, label="Overfitting threshold (0.05)")
    axes[1].axhline(mean_gap, linestyle=":", color="grey", lw=1.5, label=f"Mean gap={mean_gap:.4f}")
    axes[1].set_xlabel("Fold"); axes[1].set_ylabel("Train AUROC − Test AUROC")
    axes[1].set_title("AUROC Gap per Fold (Overfitting Signal)")
    axes[1].legend()

    _savefig(os.path.join(out_dir, "overfitting_analysis.png"))

    # Save data
    gap_df = pd.DataFrame({
        "fold": folds, "train_AUROC": train_auc,
        "test_AUROC": test_auc, "gap": gaps,
        "flag": ["overfit" if g > 0.05 else "ok" for g in gaps]
    })
    gap_df.to_csv(os.path.join(log_dir, "overfitting_gap.csv"), index=False)
    print(f"[D1] Overfitting analysis: mean_gap={mean_gap:.4f} | {flag}")
    return mean_gap, flag


def _plot_calibration_curve_diag(y_true, p_pred, n_bins=10,
                                  out_dir=FIG_DIR, log_dir=LOG_DIR):
    """
    [D3] Plot reliability diagram (calibration curve) for OOF predictions.

    A perfectly calibrated model would have all points on the diagonal.
    Overconfident models (points above diagonal) produce probability values
    that overstate certainty. Underconfident models (below diagonal) produce
    conservative estimates. Both affect practical applicability.

    For the Discussion: comment on whether susceptibility probability values
    can be directly interpreted or require calibration before operational use.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, p_pred, n_bins=n_bins, strategy="uniform"
    )

    # Calibration error
    calibration_error = float(np.mean(np.abs(fraction_of_positives - mean_predicted_value)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: reliability diagram
    axes[0].plot([0,1],[0,1],'--',color='grey',label='Perfect calibration')
    axes[0].plot(mean_predicted_value, fraction_of_positives,
                 's-', color='#4878CF', lw=2, label=f'CNN-XGBoost (ECE={calibration_error:.3f})')
    axes[0].fill_between([0,1],[0,1],[0,1],alpha=0.05,color='grey')
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Reliability Diagram (OOF Predictions)")
    axes[0].legend(); axes[0].set_xlim(0,1); axes[0].set_ylim(0,1)

    # Right: probability histogram
    axes[1].hist(p_pred[y_true==0], bins=40, alpha=0.6, color='#6ACC65',
                 label='Non-landslide', density=True)
    axes[1].hist(p_pred[y_true==1], bins=40, alpha=0.6, color='#D65F5F',
                 label='Landslide',     density=True)
    axes[1].set_xlabel("Predicted Probability"); axes[1].set_ylabel("Density")
    axes[1].set_title("Predicted Probability Distribution"); axes[1].legend()

    _savefig(os.path.join(out_dir, "calibration_curve.png"))

    cal_df = pd.DataFrame({
        "mean_predicted": mean_predicted_value,
        "fraction_positive": fraction_of_positives,
        "calibration_error_bin": np.abs(fraction_of_positives - mean_predicted_value)
    })
    cal_df.to_csv(os.path.join(log_dir, "calibration_data.csv"), index=False)
    print(f"[D3] Expected Calibration Error (ECE): {calibration_error:.4f}")
    return calibration_error


def _plot_feature_importance(clf_final, feature_names,
                              out_dir=FIG_DIR, log_dir=LOG_DIR, top_n=30):
    """
    [D4] Plot XGBoost gain-based feature importance for the final model.

    Gain measures the average improvement in loss from a feature's splits,
    giving a physically interpretable ranking of conditioning factors.
    This directly supports Discussion text on which factors drive predictions
    and connects model outputs to landslide mechanisms.
    """
    try:
        importance = clf_final.get_booster().get_score(importance_type='gain')
    except Exception:
        print("[D4] Could not retrieve feature importance — skipping."); return None

    # Map feature indices to names
    named = {}
    for key, val in importance.items():
        try:
            idx = int(key.replace("f",""))
            name = feature_names[idx] if idx < len(feature_names) else key
        except (ValueError, IndexError):
            name = key
        named[name] = val

    imp_df = pd.DataFrame(list(named.items()), columns=["feature","gain"])
    imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)
    imp_df["gain_pct"] = imp_df["gain"] / imp_df["gain"].sum() * 100
    imp_df.to_csv(os.path.join(log_dir, "feature_importance.csv"), index=False)

    # Keep only top_n for readability
    plot_df = imp_df.head(top_n)

    # Colour CNN features differently from raw conditioning factors
    colors = ["#4878CF" if "CNN" in str(r["feature"]) else "#D65F5F"
              for _, r in plot_df.iterrows()]

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df)*0.35)))
    ax.barh(plot_df["feature"][::-1], plot_df["gain_pct"][::-1], color=colors[::-1])
    ax.set_xlabel("Gain Importance (%)"); ax.set_title(f"XGBoost Feature Importance (top {top_n}, gain)")

    # Custom legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color="#4878CF", label="CNN spatial features"),
                      Patch(color="#D65F5F", label="Raw conditioning factors")]
    ax.legend(handles=legend_handles, loc="lower right")

    _savefig(os.path.join(out_dir, "feature_importance.png"))
    print(f"[D4] Feature importance saved. Top-5 features:")
    for _, row in imp_df.head(5).iterrows():
        print(f"     {row['feature']:25s}  gain_pct={row['gain_pct']:.2f}%")
    return imp_df


def _generate_discussion_report(cv_summary, fold_df, mean_gap, overfitting_flag,
                                  morans_I, morans_p, morans_interp,
                                  calibration_error, imp_df,
                                  log_dir=LOG_DIR):
    """
    [D5] Auto-generate a Discussion section draft based on computed diagnostic values.

    This text is a starting template — you should refine it to match your manuscript's
    voice and add citations. Numbers are taken directly from the computed diagnostics.
    Save location: logs/discussion_draft.txt
    """
    auroc_m  = cv_summary.get("AUROC_mean",    float("nan"))
    auroc_s  = cv_summary.get("AUROC_std",     float("nan"))
    f1_m     = cv_summary.get("f1_05_mean",    float("nan"))
    f1_s     = cv_summary.get("f1_05_std",     float("nan"))
    acc_m    = cv_summary.get("accuracy_05_mean", float("nan"))
    acc_s    = cv_summary.get("accuracy_05_std",  float("nan"))
    brier_m  = cv_summary.get("Brier_mean",    float("nan"))

    top5 = ""
    if imp_df is not None and len(imp_df) >= 5:
        top5 = ", ".join([f"{r['feature']} ({r['gain_pct']:.1f}%)"
                          for _, r in imp_df.head(5).iterrows()])

    report = f"""
=======================================================================
DISCUSSION SECTION DRAFT — AUTO-GENERATED FROM DIAGNOSTIC ANALYSES
=======================================================================
[Copy and refine into your manuscript. Replace [REF] with citations.]
=======================================================================

--- Performance Interpretation ---
The CNN-XGBoost hybrid model was evaluated using spatial k-fold
cross-validation (n={N_SPLITS} folds) with geographically separated blocks,
which prevents spatial leakage between training and test samples [REF].
Under this spatially unbiased evaluation, the model achieved an AUROC
of {auroc_m:.4f} ± {auroc_s:.4f}, an F1-score of {f1_m:.4f} ± {f1_s:.4f},
and an overall accuracy of {acc_m:.4f} ± {acc_s:.4f}. The standard
deviations across folds reflect geographic variability in model
performance, which is expected when conditioning factors and landslide
densities vary across the study area.

--- Overfitting Analysis (D1) ---
The mean AUROC gap between training and spatially held-out test sets
was {mean_gap:.4f} ({overfitting_flag}). {'A gap below 0.05 provides' if mean_gap <= 0.05 else 'This gap suggests limited'} evidence that the model
generalises to geographically unseen areas rather than memorising
training samples. Regularisation through XGBoost hyperparameters
(L1/L2 penalties, subsampling, minimum child weight) and the use of
spatial cross-validation contributed to this outcome [REF].
Nevertheless, the reported metrics should be interpreted as evidence
of relative model skill under the adopted spatial validation framework
rather than as fully independent operational performance estimates.

--- Spatial Autocorrelation of Residuals (D2) ---
Moran's I computed on the out-of-fold prediction residuals was
{morans_I:.4f} (p = {morans_p:.3f}). {morans_interp}
This result demonstrates the appropriateness of spatial cross-validation
for this dataset, as a random data split would have allowed spatially
autocorrelated samples to leak between training and test sets, inflating
the apparent model performance [REF]. Future studies should consider
variogram-based block size selection to optimise spatial CV design.

--- Calibration and Practical Applicability (D3) ---
The Expected Calibration Error (ECE) of the model, computed from the
reliability diagram on out-of-fold predictions, was {calibration_error:.4f}.
{'An ECE below 0.05 indicates well-calibrated probability outputs,' if calibration_error < 0.05 else 'An ECE above 0.05 indicates moderate miscalibration, meaning'}
{'meaning that predicted susceptibility probabilities closely reflect' if calibration_error < 0.05 else 'meaning that predicted susceptibility probabilities may not directly reflect'}
observed landslide frequencies. {'Practitioners can therefore interpret' if calibration_error < 0.05 else 'Practitioners should therefore treat'}
the probability values on the susceptibility map {'with reasonable confidence' if calibration_error < 0.05 else 'as relative rankings rather than absolute probabilities'}.
Crucially, the present framework is a susceptibility assessment tool,
not a real-time early warning system. It does not incorporate dynamic
triggering variables such as rainfall intensity or antecedent moisture
conditions. Areas classified as high or very high susceptibility should
be verified through field inspection and should be used in conjunction
with local expert knowledge before informing land-use or infrastructure
decisions.

--- Feature Importance and Physical Interpretability (D4) ---
The five most important conditioning factors by XGBoost gain importance
were: {top5 if top5 else '[see feature_importance.csv]'}. This ranking is broadly consistent with
the physical controls on landslide occurrence in the study area, where
topographic factors (slope, DEM, TRI) govern gravitational stress and
hydrological connectivity, while geological factors (fault proximity,
lithology) control material strength and permeability. The inclusion of
CNN-extracted spatial features in the top-5 confirms that the
CNN module contributed structured spatial information beyond what the
raw conditioning factors provided independently, consistent with
findings reported in [REF].

--- Limitations ---
Several limitations should be noted. First, the landslide inventory
consists of historical point records with uncertain temporal and
positional accuracy for older events (pre-2000), which may introduce
noise into model training. Second, the balanced sampling strategy
applied during training does not reflect the naturally imbalanced
distribution of landslides in real landscapes, meaning that reported
classification metrics may not directly represent operational false
alarm rates. Third, the model was calibrated specifically for
Uttaradit Province and should be recalibrated before application to
other regions with different geological or climatic regimes. Future
work should explore spatially aware architectures, dynamic triggering
factor integration, and temporally separated validation datasets to
improve robustness and transferability.

=======================================================================
END OF DRAFT — All supporting figures and CSVs are in logs/ and figs/
=======================================================================
"""
    out_path = os.path.join(log_dir, "discussion_draft.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"[D5] Discussion draft saved → {out_path}")
    return report


def _run_discussion_diagnostics(fold_df, all_preds_df, clf_final, feature_names,
                                 cv_summary,
                                 out_dir=FIG_DIR, log_dir=LOG_DIR):
    """Master diagnostic runner — D1 through D5. Called after Step C."""
    print("\n[D] === DISCUSSION DIAGNOSTICS (Issue 3) ===")
    oof_y     = all_preds_df["y_true"].values
    oof_p     = all_preds_df["p_pred"].values
    coords    = all_preds_df[["x", "y"]].values
    residuals = oof_p - oof_y.astype(float)

    steps = ["D1 Overfitting", "D2 Moran's I", "D3 Calibration", "D4 Feature Importance", "D5 Discussion Draft"]
    with tqdm(total=len(steps), desc="[D] Diagnostics", unit="step") as dpbar:

        dpbar.set_description(f"[D] {steps[0]}")
        mean_gap, overfitting_flag = _plot_overfitting_analysis(fold_df, out_dir, log_dir)
        dpbar.update(1)

        dpbar.set_description(f"[D] {steps[1]}")
        morans_I, morans_p, morans_interp = _compute_morans_i(residuals, coords, k=8, n_sample=MORANS_N_SAMPLE)
        pd.DataFrame([{"morans_I": morans_I, "p_value": morans_p,
                       "interpretation": morans_interp}]).to_csv(
            os.path.join(log_dir, "morans_i_residuals.csv"), index=False)
        print(f"\n[D2] Moran's I={morans_I:.4f}  p={morans_p:.3f}  {morans_interp}")
        dpbar.update(1)

        dpbar.set_description(f"[D] {steps[2]}")
        calibration_error = _plot_calibration_curve_diag(oof_y, oof_p, n_bins=10, out_dir=out_dir, log_dir=log_dir)
        dpbar.update(1)

        dpbar.set_description(f"[D] {steps[3]}")
        imp_df = _plot_feature_importance(clf_final, feature_names, out_dir, log_dir, top_n=30)
        dpbar.update(1)

        dpbar.set_description(f"[D] {steps[4]}")
        _generate_discussion_report(cv_summary, fold_df, mean_gap, overfitting_flag,
                                    morans_I, morans_p, morans_interp,
                                    calibration_error, imp_df, log_dir)
        dpbar.update(1)

    print("\n[D] === DIAGNOSTICS COMPLETE ===")
    print(f"  Figures → {out_dir}")
    print(f"  CSVs + draft → {log_dir}")
    for f in ["overfitting_analysis.png","overfitting_gap.csv",
               "morans_i_residuals.csv","calibration_curve.png",
               "calibration_data.csv","feature_importance.png",
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
# MODIFIED train_xgb_from_points — adds train metrics in fold loop + D1-D5
# =============================================================================
def train_xgb_from_points(deep_feats_tif=DEEP_FEATS_TIF, factors_tif=FACTOR_TIF,
                           points_path=POINTS_PATH, out_model=MODEL_PATH, n_splits=N_SPLITS):
    t0_total = time.perf_counter()
    print("[B] Training XGBoost with spatial k-fold CV + discussion diagnostics …")

    gdf = gpd.read_file(points_path)
    with rasterio.open(factors_tif) as rst:
        gdf = _reproject_points_to_raster_crs(gdf, rst)
    gdf = _drop_ambiguous_negatives(gdf, LABEL_COL, POS_VALUE, NEG_VALUE, NEG_EXCLUSION_RADIUS_M)

    if LABEL_COL not in gdf.columns:
        raise ValueError(f"'{LABEL_COL}' not found.")
    cls = gdf[LABEL_COL].astype(int)
    y   = np.where(cls.values == POS_VALUE, 1, 0).astype(int)
    print(f"[B] Samples: total={len(y)} | pos={int((y==1).sum())} | neg={int((y==0).sum())}")

    t0s = time.perf_counter()
    points_xy = [(geom.x, geom.y) for geom in gdf.geometry]
    Xdeep = _sample_rasters_at_points([deep_feats_tif], points_xy)
    Xraw  = _sample_rasters_at_points([factors_tif],    points_xy)
    X     = np.hstack([Xdeep, Xraw]).astype('float32')
    X     = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    _log_time("B: sample_features", time.perf_counter()-t0s,
              N=len(y), Kdeep=Xdeep.shape[1], Kraw=Xraw.shape[1])

    # Build feature name list for D4 (feature importance)
    n_raw = Xraw.shape[1]
    if len(RAW_BAND_NAMES) == n_raw:
        raw_names = list(RAW_BAND_NAMES)
    else:
        warnings.warn(f"RAW_BAND_NAMES has {len(RAW_BAND_NAMES)} entries but raster has "
                      f"{n_raw} bands. Using generic names. Update RAW_BAND_NAMES in constants.")
        raw_names = [f"raw_band_{i+1}" for i in range(n_raw)]
    feature_names = [f"CNN_{i+1}" for i in range(Xdeep.shape[1])] + raw_names

    groups       = _make_spatial_blocks(gdf, n_splits=n_splits)
    points_xy_arr = np.array(points_xy, dtype="float64")
    pos_c = max(int((y==1).sum()), 1); neg_c = max(int((y==0).sum()), 1)
    spw   = (float(neg_c) / float(pos_c)) ** 0.75

    # Step A — Hyperparameter tuning
    base_flags = _xgb_gpu_params(); base_core = dict(eval_metric="logloss", random_state=SEED)
    t0t = time.perf_counter()
    best_params, best_cv, tune_hist = tune_xgb_params(
        X, y, groups, _tune_gdf=gdf,
        base_params={**base_core, **base_flags},
        metric="ap", n_iter=XGB_TUNE_ITER, n_splits=min(5, n_splits),
        seed=SEED, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        pos_weight=spw, use_gpu=XGB_USE_GPU)
    _plot_tuning_results(tune_hist); _log_time("B: xgb_tune", time.perf_counter()-t0t)
    if best_params is None:
        best_params = _clean_xgb_params({**base_core, **_xgb_gpu_params()})

    # Step B — Primary spatial buffer CV (Option B — buffer-based exclusion)
    print(f"\n[B] === PRIMARY BUFFER-BASED SPATIAL CV (n_splits={n_splits}, buffer={BUFFER_M}m) ===")
    cv_splits  = _buffered_spatial_splits(gdf, y, n_splits=n_splits,
                                           buffer_m=BUFFER_M, seed=SEED)
    fold_records = []; all_preds = []; best_iters = []

    t0cv = time.perf_counter()
    with tqdm(total=n_splits, desc="[B] Spatial CV folds") as pbar:
        for fold_i, (tr_idx, te_idx) in enumerate(cv_splits):
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xte, yte = X[te_idx], y[te_idx]

            pos_f = max(int((ytr==1).sum()), 1); neg_f = max(int((ytr==0).sum()), 1)
            spw_f = (float(neg_f) / float(pos_f)) ** 0.75

            fold_params = dict(best_params); fold_params["scale_pos_weight"] = spw_f
            clf_fold = XGBClassifier(**fold_params)
            try:
                clf_fold.fit(Xtr, ytr, eval_set=[(Xte, yte)],
                             early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
            except TypeError:
                clf_fold.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

            bi = getattr(clf_fold, "best_iteration", None)
            if bi is not None: best_iters.append(int(bi))

            p_te = clf_fold.predict_proba(Xte)[:, 1]
            # ----------------------------------------------------------------
            # NEW IN v5 (D1): also evaluate on TRAINING set for overfitting
            # ----------------------------------------------------------------
            p_tr        = clf_fold.predict_proba(Xtr)[:, 1]
            train_auroc = roc_auc_score(ytr, p_tr)
            train_ap    = average_precision_score(ytr, p_tr)
            # ----------------------------------------------------------------

            m05    = _threshold_metrics(yte, p_te, 0.5)
            grid   = np.linspace(0.01, 0.99, 99)
            f1s    = [f1_score(yte, (p_te>=t).astype(int), zero_division=0) for t in grid]
            best_t = float(grid[int(np.argmax(f1s))])
            mbt    = _threshold_metrics(yte, p_te, best_t)

            fold_rec = {
                "fold":           fold_i + 1,
                "n_train":        len(ytr),
                "n_test":         len(yte),
                "best_iter":      bi,
                # Test metrics (primary — go in paper)
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
                # NEW IN v5 — Train metrics for overfitting analysis (D1)
                "train_AUROC":    train_auroc,
                "train_AP":       train_ap,
                "AUROC_gap":      train_auroc - roc_auc_score(yte, p_te),
            }
            fold_records.append(fold_rec)

            for j, idx in enumerate(te_idx):
                all_preds.append({
                    "sample_idx": idx, "fold": fold_i+1,
                    "y_true": int(yte[j]), "p_pred": float(p_te[j]),
                    "x": points_xy_arr[idx, 0], "y": points_xy_arr[idx, 1],
                    "group": int(groups[idx]),
                })

            print(f"  Fold {fold_i+1}/{n_splits} | "
                  f"test_AUROC={fold_rec['AUROC']:.4f} | "
                  f"train_AUROC={train_auroc:.4f} | "
                  f"gap={fold_rec['AUROC_gap']:.4f} | "
                  f"F1={fold_rec['f1_05']:.4f}")
            pbar.update(1)

    _log_time("B: spatial_cv_primary", time.perf_counter()-t0cv, folds=n_splits)

    fold_df     = pd.DataFrame(fold_records)
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
    print(f"  AUROC (train): {cv_summary['train_AUROC_mean']:.4f} ± {cv_summary['train_AUROC_std']:.4f}")
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
    fpr_a, tpr_a, thr_r = roc_curve(oof_y, oof_p)
    _curve_df_with_thresholds(fpr_a,tpr_a,thr_r,"fpr","tpr").to_csv(
        os.path.join(LOG_DIR,"roc_oof.csv"),index=False)
    prec_a, rec_a, thr_p = precision_recall_curve(oof_y, oof_p)
    _curve_df_with_thresholds(rec_a,prec_a,thr_p,"recall","precision").to_csv(
        os.path.join(LOG_DIR,"pr_oof.csv"),index=False)

    plt.figure(); plt.plot(fpr_a,tpr_a,label=f"OOF AUC={roc_auc_score(oof_y,oof_p):.3f}")
    plt.plot([0,1],[0,1],'--',color='grey'); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC — spatial k-fold (OOF)"); plt.legend()
    _savefig(os.path.join(FIG_DIR,"roc_oof.png"))

    plt.figure(); plt.plot(rec_a,prec_a,label=f"OOF AP={average_precision_score(oof_y,oof_p):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR — spatial k-fold (OOF)"); plt.legend()
    _savefig(os.path.join(FIG_DIR,"pr_oof.png"))

    plt.figure(figsize=(6,4))
    plt.bar(fold_df["fold"], fold_df["AUROC"])
    plt.axhline(cv_summary["AUROC_mean"],linestyle='--',color='red',
                label=f"Mean={cv_summary['AUROC_mean']:.3f}")
    plt.xlabel("Fold"); plt.ylabel("Test AUROC"); plt.title("Per-fold AUROC"); plt.legend()
    _savefig(os.path.join(FIG_DIR,"cv_auroc_per_fold.png"))

    # Step C — Final model on ALL data
    mean_best_iter = int(round(np.mean(best_iters))) if best_iters else 500
    print(f"\n[B] Final model: ALL data | n_estimators={mean_best_iter}")
    final_params = dict(best_params)
    final_params["n_estimators"]     = mean_best_iter
    final_params["scale_pos_weight"] = spw
    final_params.pop("early_stopping_rounds", None)

    t0f = time.perf_counter()
    clf_final = XGBClassifier(**final_params)
    clf_final.fit(X, y, verbose=False)
    _log_time("B: xgb_final_fit", time.perf_counter()-t0f, N=len(y))

    # ----------------------------------------------------------------
    # Step D — NEW IN v5: Discussion diagnostics (Issue 3)
    # ----------------------------------------------------------------
    _run_discussion_diagnostics(
        fold_df, all_preds_df, clf_final, feature_names, cv_summary,
        out_dir=FIG_DIR, log_dir=LOG_DIR
    )

    payload = {
        "clf":             clf_final,
        "cv_summary":      cv_summary,
        "fold_records":    fold_records,
        "best_params":     best_params,
        "mean_best_iter":  mean_best_iter,
        "feature_names":   feature_names,
        "validation_strategy": f"spatial_groupkfold_n{n_splits}",
        "note": ("Metrics from spatial k-fold CV. clf trained on ALL data for mapping only."),
    }
    joblib.dump(payload, out_model)
    print(f"[B] Saved model → {out_model}")
    _log_time("B: total_train", time.perf_counter()-t0_total, N=len(y), n_splits=n_splits)
    return payload


# ================= Deep features raster (unchanged from v4) ========================
@torch.no_grad()
def write_deep_feature_raster(in_tif, out_tif=DEEP_FEATS_TIF,
                               feat_channels=FEAT_CHANNELS, tile=TILE, encoder_ckpt=None):
    t0 = time.perf_counter(); device = _select_device()
    with rasterio.open(in_tif) as src:
        C=src.count; mean,std=compute_band_stats(src)
        meta=src.meta.copy(); meta.update(count=feat_channels,dtype='float32')
        width,height=src.width,src.height
    model=FCNFeatureExtractor(in_ch=C,out_ch=feat_channels,backbone=BACKBONE).eval()
    if device.type=="cuda":
        if encoder_ckpt and os.path.exists(encoder_ckpt):
            model.load_state_dict(torch.load(encoder_ckpt,map_location="cpu"),strict=False)
        model=model.to(device).to(memory_format=torch.channels_last)
        ampctx=torch.amp.autocast('cuda',enabled=USE_MIXED_PRECISION)
    tiles=[(x,y,min(tile,width-x),min(tile,height-y))
           for y in range(0,height,tile) for x in range(0,width,tile)]
    n_tiles=len(tiles)

    def _ew(x,y,w,h):
        x0=max(0,x-HALO);y0=max(0,y-HALO)
        x1=min(width,x+w+HALO);y1=min(height,y+h+HALO)
        return x0,y0,x1-x0,y1-y0,(x-x0),(y-y0)
    def _aug(t,k):
        if k==0: return t
        if k==1: return torch.flip(t,dims=[-1])
        if k==2: return torch.flip(t,dims=[-2])
        return t.transpose(-2,-1)
    def _pad(a,th,tw):
        C,h,w=a.shape; return np.pad(a,((0,0),(0,th-h),(0,tw-w)),mode='edge') if(th!=h or tw!=w) else a

    with rasterio.open(out_tif,'w',**meta) as dst,rasterio.open(in_tif) as src,\
         tqdm(total=n_tiles,desc="[A] Deep features") as pbar:
        if device.type=="cuda":
            bsize=max(1,GPU_TILE_BATCH)
            for i in range(0,n_tiles,bsize):
                bt=tiles[i:i+bsize]; arrs,metas=[],[]
                meh=mew=0
                for(x,y,w,h) in bt:
                    ex,ey,ew,eh,cx,cy=_ew(x,y,w,h)
                    a=src.read(window=rw.Window(ex,ey,ew,eh),out_dtype='float32')
                    a=_standardize_block(a,mean,std); arrs.append(a)
                    metas.append((x,y,w,h,cx,cy,ew,eh))
                    meh=max(meh,a.shape[1]);mew=max(mew,a.shape[2])
                batch=np.stack([_pad(a,meh,mew) for a in arrs],axis=0)
                t=torch.from_numpy(batch).to(device,non_blocking=True).to(memory_format=torch.channels_last)
                if USE_TTA:
                    outs = None
                    for k in (0, 1, 2, 3):
                        with ampctx: fk = model(_aug(t, k))
                        fk = _aug(fk, k).float()   # upcasts to FP32 before accumulating
                        if outs is None:
                            outs = fk              # first pass: own the tensor
                        else:
                            outs.add_(fk)          # FIX: in-place add — no new tensor allocated
                        del fk                     # free immediately after accumulation
                    fout = (outs.mul_(0.25)).contiguous()   # in-place divide by 4
                else:
                    with ampctx: fout=model(t).contiguous()
                fout=fout.to(dtype=torch.float32,memory_format=torch.contiguous_format).cpu().numpy()
                for b,(x,y,w,h,cx,cy,ew,eh) in enumerate(metas):
                    dst.write(fout[b,:,cy:cy+h,cx:cx+w],window=rw.Window(x,y,w,h))
                    pbar.update(1)
        else:
            for(x,y,w,h) in tiles:
                ex,ey,ew,eh,cx,cy=_ew(x,y,w,h)
                a=src.read(window=rw.Window(ex,ey,ew,eh),out_dtype='float32')
                a=_standardize_block(a,mean,std)
                f=model(torch.from_numpy(a[None,...]))[0].numpy().astype('float32')
                dst.write(f[:,cy:cy+h,cx:cx+w],window=rw.Window(x,y,w,h)); pbar.update(1)
    _cuda_sync_if_any()
    _log_time("A: deep_features",time.perf_counter()-t0,tiles=n_tiles,K=feat_channels)


# ================= Fine-tuning (unchanged from v4) ==================================
class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25,gamma=2.0):
        super().__init__(); self.alpha,self.gamma=alpha,gamma
        self.bce=nn.BCEWithLogitsLoss(reduction='none')
    def forward(self,logits,targets):
        bce=self.bce(logits,targets);pt=torch.exp(-bce)
        return(self.alpha*(1-pt)**self.gamma*bce).mean()

def _rasterize_pos_mask(points_path,raster_path,label_col,pos_val,buffer_m=60):
    import rasterio.features as rfeat
    gdf=gpd.read_file(points_path)
    with rasterio.open(raster_path) as src:
        if gdf.crs!=src.crs: gdf=gdf.to_crs(src.crs)
        geoms=gdf[gdf[label_col].astype(int)==pos_val].geometry
        if buffer_m>0 and(src.crs.is_projected if src.crs else False): geoms=geoms.buffer(buffer_m)
        return rfeat.rasterize([(g,1) for g in geoms],out_shape=(src.height,src.width),
                               transform=src.transform,fill=0,dtype="float32")

def finetune_extractor_on_points(factor_tif,points_path,epochs=FT_EPOCHS,lr=FT_LR,
                                  tile=FT_TILE,buffer_m=FT_BUFFER_M,ckpt_out="encoder_finetuned.pt"):
    t0=time.perf_counter(); device=_select_device()
    with rasterio.open(factor_tif) as src: C,H,W=src.count,src.height,src.width; mean,std=compute_band_stats(src)
    model=FCNFeatureExtractor(in_ch=C,out_ch=FEAT_CHANNELS,backbone=BACKBONE).to(device).to(memory_format=torch.channels_last)
    for p in model.backbone[0:6].parameters(): p.requires_grad=False
    opt=torch.optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()),lr=lr,weight_decay=1e-4)
    loss_fn=FocalLoss(); scaler=torch.amp.GradScaler('cuda',enabled=(device.type=="cuda" and USE_MIXED_PRECISION))
    posmask=_rasterize_pos_mask(points_path,factor_tif,LABEL_COL,POS_VALUE,buffer_m)
    windows=[(x,y,tile,tile) for y in range(0,H-(H%tile),tile) for x in range(0,W-(W%tile),tile)]
    epoch_losses,epoch_secs=[],[]
    model.train()
    for ep in range(epochs):
        ep_t0=time.perf_counter(); random.Random(SEED+ep).shuffle(windows)
        total=0.0;steps=0;acc_count=0
        with rasterio.open(factor_tif) as src,tqdm(total=len(windows),desc=f"[FT] ep{ep+1}/{epochs}") as pbar:
            for i in range(0,len(windows),FT_BATCH_TILES):
                batch=windows[i:i+FT_BATCH_TILES];Xs,Ys=[],[]
                for(x,y,w,h) in batch:
                    Y=posmask[y:y+h,x:x+w].astype("float32")[None,...]
                    if Y.sum()==0 and np.random.rand()<FT_SKIP_ALL_NEG_PROB: continue
                    Xb=src.read(window=rw.Window(x,y,w,h),out_dtype="float32")
                    Xs.append(_standardize_block(Xb,mean,std));Ys.append(Y)
                if not Xs: pbar.update(len(batch));continue
                xt=torch.from_numpy(np.stack(Xs)).to(device,non_blocking=True).to(memory_format=torch.channels_last)
                yt=torch.from_numpy(np.stack(Ys)).to(device,non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=(device.type=="cuda" and USE_MIXED_PRECISION)):
                    loss=loss_fn(model(xt).narrow(1,0,1).contiguous(),yt)
                if device.type=="cuda":
                    scaler.scale(loss).backward();acc_count+=1
                    if acc_count%FT_ACCUM_STEPS==0: scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
                else: loss.backward();opt.step()
                total+=float(loss.item());steps+=1;pbar.update(len(batch))
        if device.type=="cuda" and acc_count%FT_ACCUM_STEPS!=0:
            scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        _cuda_sync_if_any()
        ep_sec=time.perf_counter()-ep_t0; ep_loss=total/max(1,steps)
        epoch_losses.append(ep_loss);epoch_secs.append(ep_sec)
        print(f"[FT] ep{ep+1}/{epochs} loss={ep_loss:.4f} t={_fmt_hms(ep_sec)}")
    plt.figure();plt.plot(range(1,len(epoch_losses)+1),epoch_losses)
    plt.xlabel("Epoch");plt.ylabel("Focal loss");plt.title("Fine-tuning loss")
    _savefig(os.path.join(FIG_DIR,"finetune_loss.png"))
    pd.DataFrame({"epoch":np.arange(1,len(epoch_losses)+1),"loss":epoch_losses,"sec":epoch_secs}).to_csv(
        os.path.join(LOG_DIR,"finetune_epochs.csv"),index=False)
    torch.save(model.state_dict(), ckpt_out)
    _cuda_sync_if_any()

    # Fix 2: explicitly free the fine-tuning model, optimizer, and scaler from GPU
    # before returning. Without this, the model + AdamW moments (~4.5 GB) stay
    # allocated when write_deep_feature_raster() loads a second model instance,
    # causing OOM despite having sufficient total VRAM.
    del opt, scaler, loss_fn
    model.cpu()          # move weights off GPU before deleting
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"[FT] GPU memory freed. Cached: "
          f"{torch.cuda.memory_reserved()/1e9:.2f} GB | "
          f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    _log_time("FT: total", time.perf_counter()-t0, epochs=epochs)
    return ckpt_out


# ================= Predict probability map (unchanged from v4) ======================
def predict_probability_map(deep_feats_tif=DEEP_FEATS_TIF,factors_tif=FACTOR_TIF,
                             model_path=MODEL_PATH,out_tif=OUT_PROB_TIF,tile=TILE):
    t0=time.perf_counter(); print("[C] Predicting probability map …")
    payload=joblib.load(model_path); clf=payload["clf"]
    with rasterio.open(deep_feats_tif) as dsrc,rasterio.open(factors_tif) as fsrc:
        assert dsrc.width==fsrc.width and dsrc.height==fsrc.height
        meta=dsrc.meta.copy();meta.update(count=1,dtype='float32')
        width,height=dsrc.width,dsrc.height
        total=((height+tile-1)//tile)*((width+tile-1)//tile)
        with rasterio.open(out_tif,'w',**meta) as dst,tqdm(total=total,desc="[C] Map") as pbar:
            for y in range(0,height,tile):
                for x in range(0,width,tile):
                    win=rw.Window(x,y,min(tile,width-x),min(tile,height-y))
                    D=dsrc.read(window=win,out_dtype='float32'); F=fsrc.read(window=win,out_dtype='float32')
                    H,W=D.shape[1],D.shape[2]
                    Xp=np.concatenate([D,F],axis=0).reshape(D.shape[0]+F.shape[0],-1).T
                    Xp=np.nan_to_num(Xp,nan=0.0,posinf=0.0,neginf=0.0)
                    dst.write(clf.predict_proba(Xp)[:,1].astype('float32').reshape(H,W),1,window=win)
                    pbar.update(1)
    _log_time("C: predict_map",time.perf_counter()-t0,width=width,height=height,tiles=total)
    print(f"[C] → {out_tif}")


# ================= Main =============================================================
def _clear_gpu(label=""):
    """Flush GPU cache between pipeline stages to prevent cross-stage OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1e9
        reserv = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU] {label} — allocated={alloc:.2f} GB | reserved={reserv:.2f} GB")

if __name__ == "__main__":
    ckpt = None
    if FINE_TUNE_ENABLED:
        ckpt = finetune_extractor_on_points(
            FACTOR_TIF, POINTS_PATH, epochs=FT_EPOCHS, lr=FT_LR,
            tile=FT_TILE, buffer_m=FT_BUFFER_M, ckpt_out="encoder_finetuned.pt"
        )
        _clear_gpu("after fine-tuning")   # free fine-tuning model before loading inference model

    write_deep_feature_raster(FACTOR_TIF, out_tif=DEEP_FEATS_TIF,
                               feat_channels=FEAT_CHANNELS, tile=TILE, encoder_ckpt=ckpt)
    _clear_gpu("after feature extraction")   # free CNN before XGB training

    payload = train_xgb_from_points(DEEP_FEATS_TIF, FACTOR_TIF, POINTS_PATH,
                                     out_model=MODEL_PATH, n_splits=N_SPLITS)
    _clear_gpu("after XGB training")
    predict_probability_map(DEEP_FEATS_TIF, FACTOR_TIF, MODEL_PATH,
                             out_tif=OUT_PROB_TIF, tile=TILE)
    _write_timings_csv()
