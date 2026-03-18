# =================== LSM HYBRID (CNN-XGB) with Progress + Figures + Metrics ===================
# GPU-utilization upgrade: batched tile inference/training + bigger knobs + full CSV logging.

import os, warnings, math, numpy as np, time, random
import joblib
import geopandas as gpd, rasterio, rasterio.windows as rw
from rasterio.vrt import WarpedVRT
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBClassifier
import xgboost as xgb  # for version-aware GPU flags
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

# --- Paths ----------------------------------------------------------------------------
FACTOR_TIF     = '/media/capybara01/ssd/2025_landslide/raster/Raster_Com.tif'
POINTS_PATH    = '/media/capybara01/ssd/2025_landslide/point/Point_LS.shp'
DEEP_FEATS_TIF = '/media/capybara01/ssd/2025_landslide/CNN_XGB/process/deep_feats2.tif'
MODEL_PATH     = '/media/capybara01/ssd/2025_landslide/CNN_XGB/model/xgb_fused_points2.joblib'
OUT_PROB_TIF   = '/media/capybara01/ssd/2025_landslide/CNN_XGB/result/lsm_prob2.tif'

RESULT_DIR = os.path.dirname(OUT_PROB_TIF)
FIG_DIR    = os.path.join(RESULT_DIR, "figs")
LOG_DIR    = os.path.join(RESULT_DIR, "logs")
for d in [os.path.dirname(MODEL_PATH), os.path.dirname(DEEP_FEATS_TIF),
          RESULT_DIR, FIG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- Small helpers to GUARANTEE CSV columns align ----------
def _curve_df_with_thresholds(x_arr, y_arr, thr_arr, x_name, y_name):
    """Pad 'threshold' to match x/y length so DataFrame construction never fails."""
    x_arr = np.asarray(x_arr); y_arr = np.asarray(y_arr); thr_arr = np.asarray(thr_arr)
    need = len(x_arr) - len(thr_arr)
    if need < 0:
        thr_arr = thr_arr[:len(x_arr)]
        need = 0
    if need > 0:
        thr_arr = np.r_[thr_arr, np.full(need, np.nan, dtype=float)]
    return pd.DataFrame({x_name: x_arr, y_name: y_arr, "threshold": thr_arr})

def _assert_same_len(**cols):
    """Debugging guard: raises early if any columns differ in length."""
    lens = {k: (len(v) if hasattr(v, "__len__") and not np.isscalar(v) else 1) for k, v in cols.items()}
    if len(set(lens.values())) != 1:
        raise ValueError(f"Length mismatch in columns: {lens}")

# --- Stride / output resolution for deep features ------------------------------------
OUTPUT_STRIDE = 16   # {32 (default), 16, 8}

# --- Model / Pipeline ----------------------------------------------------------------
BACKBONE = "resnet152"      # resnet18/34/50/101/152
FEAT_CHANNELS = 256
TILE = 512
N_SPLITS = 5
SEED = 42

# --- Device / Performance for CNN ----------------------------------------------------
USE_GPU_IF_AVAILABLE = True
CUDA_DEVICE_INDEX = 0
USE_MIXED_PRECISION = True
GPU_TILE_BATCH = 2
torch.set_float32_matmul_precision("high")

# --- XGBoost -------------------------------------------------------------------------
XGB_N_JOBS = max(1, (os.cpu_count() or 1))
XGB_USE_GPU = True
CALIBRATION_METHOD = None            # 'sigmoid' or 'isotonic' or None
EARLY_STOPPING_ROUNDS = 1000
XGB_MAX_BIN = 256
XGB_GPU_ID = CUDA_DEVICE_INDEX
XGB_TUNE_ITER = 100
XGB_VER = tuple(int(p) for p in xgb.__version__.split('.')[:2])  # (major, minor)

# --- Labels & Hygiene ----------------------------------------------------------------
LABEL_COL = "Class"
POS_VALUE = 1
NEG_VALUE = 2
NEG_EXCLUSION_RADIUS_M = 90

# --- Fine-tuning (CNN) ---------------------------------------------------------------
FINE_TUNE_ENABLED = True
FT_EPOCHS = 20
FT_LR = 1e-4
FT_TILE = 512
FT_BUFFER_M = 60
FT_SKIP_ALL_NEG_PROB = 0.8
FT_BATCH_TILES = 4
FT_ACCUM_STEPS = 1

# --- Anti-seam & robustness ----------------------------------------------------------
HALO = 160
USE_TTA = True

# --- Timing helpers ------------------------------------------------------------------
TIMINGS = []
def _fmt_hms(sec: float) -> str:
    m, s = divmod(sec, 60.0); h, m = divmod(int(m), 60); return f"{h:d}:{m:02d}:{s:05.2f}"
def _log_time(step: str, seconds: float, **extras):
    rec = {"step": step, "seconds": round(seconds, 3), "pretty": _fmt_hms(seconds)}
    rec.update(extras); TIMINGS.append(rec); print(f"[Time] {step}: {rec['pretty']} ({rec['seconds']} s)")
def _write_timings_csv(path=os.path.join(LOG_DIR, "timings.csv")):
    if TIMINGS:
        pd.DataFrame(TIMINGS).to_csv(path, index=False); print(f"[Time] Saved timing report → {path}")
def _cuda_sync_if_any():
    try:
        if torch.cuda.is_available(): torch.cuda.synchronize()
    except Exception: pass

# ================= Backbone & Feature Extractor =====================================
def _disable_inplace_relu(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.ReLU): m.inplace = False

def _get_resnet_backbone(in_ch: int, backbone: str = "resnet50"):
    name = backbone.lower()
    spec = {
        "resnet18":  (models.resnet18,  getattr(models, "ResNet18_Weights",  None),  512),
        "resnet34":  (models.resnet34,  getattr(models, "ResNet34_Weights",  None),  512),
        "resnet50":  (models.resnet50,  getattr(models, "ResNet50_Weights",  None), 2048),
        "resnet101": (models.resnet101, getattr(models, "ResNet101_Weights", None), 2048),
        "resnet152": (models.resnet152, getattr(models, "ResNet152_Weights", None), 2048),
    }
    if name not in spec:
        raise ValueError("BACKBONE must be one of: " + ", ".join(spec.keys()))
    ctor, Weights, c_out = spec[name]

    weights = None
    if Weights is not None:
        try: weights = Weights.IMAGENET1K_V1
        except Exception: weights = None

    rswd_map = {32:[False,False,False], 16:[False,False,True], 8:[False,True,True]}
    if OUTPUT_STRIDE not in rswd_map:
        warnings.warn(f"Unsupported OUTPUT_STRIDE={OUTPUT_STRIDE}; falling back to 32.")
    rswd = rswd_map.get(OUTPUT_STRIDE, [False,False,False])

    os_used = OUTPUT_STRIDE
    try:
        res = ctor(weights=weights, replace_stride_with_dilation=rswd)
    except TypeError:
        warnings.warn("replace_stride_with_dilation unsupported; using default OS=32.")
        res = ctor(weights=weights); os_used = 32

    _disable_inplace_relu(res)

    if in_ch != 3:
        old = res.conv1
        new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            w = old.weight
            repeat = math.ceil(in_ch / 3)
            w_rep = w.repeat(1, repeat, 1, 1)[:, :in_ch, :, :]
            w_new = w_rep * (3.0 / float(in_ch))
            new.weight.copy_(w_new)
        res.conv1 = new

    trunk = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool,
                          res.layer1, res.layer2, res.layer3, res.layer4)
    print(f"[Backbone] {backbone} | c_out={c_out} | OUTPUT_STRIDE={os_used}")
    return trunk, c_out

class FCNFeatureExtractor(nn.Module):
    def __init__(self, in_ch=9, out_ch=128, backbone=BACKBONE):
        super().__init__()
        self.backbone, c_out = _get_resnet_backbone(in_ch, backbone=backbone)
        self.proj = nn.Conv2d(c_out, out_ch, kernel_size=1)
    def forward(self, x):
        f = self.backbone(x)
        f = self.proj(f)
        return nn.functional.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)

# ================= Utilities ========================================================
def _select_device():
    if USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
        torch.cuda.set_device(CUDA_DEVICE_INDEX)
        print(f"[Device] Using CUDA:{CUDA_DEVICE_INDEX}")
        torch.backends.cudnn.benchmark = True
        return torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
    print("[Device] Using CPU"); return torch.device("cpu")

def compute_band_stats(src, block=1024):
    """
    Robust per-band mean/std over a raster:
    - reads as float64
    - ignores non-finite values
    - accumulates sums and squared sums in float64 to avoid overflow
    - guards zero-count bands and tiny/NaN stds
    """
    sums = None          # float64
    sqs  = None          # float64
    counts = None        # int64

    for y in range(0, src.height, block):
        for x in range(0, src.width, block):
            w = rw.Window(x, y, min(block, src.width - x), min(block, src.height - y))
            arr = src.read(window=w, out_dtype="float64")          # promote to float64
            mask = np.isfinite(arr)
            arr = np.where(mask, arr, 0.0)

            c = mask.sum(axis=(1, 2)).astype(np.int64)
            s = arr.sum(axis=(1, 2), dtype=np.float64)
            q = np.square(arr, dtype=np.float64).sum(axis=(1, 2), dtype=np.float64)

            if sums is None:
                sums, sqs, counts = s, q, c
            else:
                sums   += s
                sqs    += q
                counts += c

    counts_safe = np.maximum(counts, 1)
    means = sums / counts_safe
    # Var = E[x^2] - (E[x])^2
    vars_ = (sqs / counts_safe) - (means * means)
    # if a band had 0 valid pixels, use std=1; clamp tiny/negative due to roundoff
    vars_ = np.where(counts > 0, vars_, 1.0)
    vars_ = np.clip(vars_, 1e-12, None)

    means = means.astype("float32")
    stds  = np.sqrt(vars_).astype("float32")
    return means, stds

def _standardize_block(arr, mean, std, eps=1e-6):
    """
    Standardize a (C,H,W) array using per-band mean/std with numerical guards.
    - Replaces NaN/±inf with 0
    - Ensures std is finite and >= eps
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    std_safe = np.where(np.isfinite(std) & (std > eps), std, eps).astype(arr.dtype, copy=False)
    return (arr - mean[:, None, None]) / std_safe[:, None, None]

def _reproject_points_to_raster_crs(points_gdf, raster_src):
    return points_gdf.to_crs(raster_src.crs) if points_gdf.crs != raster_src.crs else points_gdf

def _sample_rasters_at_points(tif_paths, points_xy):
    feats = []
    for path in tif_paths:
        with rasterio.open(path) as src:
            with WarpedVRT(src, crs=src.crs) as vrt:
                vals = np.array([v for v in vrt.sample(points_xy)], dtype='float32')
        feats.append(vals)
    return np.concatenate(feats, axis=1)

def _make_spatial_blocks(points_gdf, n_splits=N_SPLITS):
    xmin, ymin, xmax, ymax = points_gdf.total_bounds
    nx = ny = int(np.ceil(np.sqrt(n_splits*2)))
    xs = np.linspace(xmin, xmax, nx+1); ys = np.linspace(ymin, ymax, ny+1)
    def cell_id(x,y):
        ix = min(nx-1, max(0, np.searchsorted(xs, x, side='right')-1))
        iy = min(ny-1, max(0, np.searchsorted(ys, y, side='right')-1))
        return iy*nx + ix
    return points_gdf.geometry.apply(lambda g: cell_id(g.x, g.y)).values

def _drop_ambiguous_negatives(gdf, label_col, pos_val=1, neg_val=2, radius_m=90):
    if radius_m <= 0: return gdf
    gpos = gdf[gdf[label_col].astype(int) == pos_val]
    gneg = gdf[gdf[label_col].astype(int) == neg_val]
    if gpos.empty or gneg.empty: return gdf
    need_tmp_metric = (not gdf.crs) or (not gdf.crs.is_projected)
    if need_tmp_metric:
        cx, cy = gdf.unary_union.centroid.xy
        utm_zone = int((cx[0] + 180) // 6) + 1
        epsg = 32600 + utm_zone if cy[0] >= 0 else 32700 + utm_zone
        gpos_ = gpos.to_crs(epsg); gneg_ = gneg.to_crs(epsg)
    else:
        gpos_, gneg_ = gpos, gneg
    buf = gpos_.buffer(radius_m)
    union_geom = buf.union_all() if hasattr(buf, "union_all") else _unary_union(list(buf.values if hasattr(buf, "values") else buf))
    keep_mask = ~gneg_.geometry.intersects(union_geom)
    kept_neg_metric = gneg_.loc[keep_mask]
    gpos_back  = gpos_.to_crs(gdf.crs)  if need_tmp_metric else gpos_
    kept_back  = kept_neg_metric.to_crs(gdf.crs) if need_tmp_metric else kept_neg_metric
    out = pd.concat([gpos_back, kept_back], ignore_index=True)
    return out.set_crs(gdf.crs, allow_override=True)

def _savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()

# ---------- GPU flags that work on XGBoost 1.x and 2.x ----------
def _xgb_gpu_params():
    """
    Return version-safe GPU/CPU flags.
    - XGBoost >= 2.0  -> use device='cuda', tree_method='hist' (NO gpu_id)
    - XGBoost <  2.0  -> use tree_method='gpu_hist', gpu_id=..., predictor='gpu_predictor'
    """
    if XGB_USE_GPU:
        if XGB_VER >= (2, 0):
            return dict(device='cuda', tree_method='hist', predictor='gpu_predictor', max_bin=XGB_MAX_BIN)
        else:
            return dict(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=XGB_GPU_ID, max_bin=XGB_MAX_BIN)
    else:
        return dict(tree_method='hist', n_jobs=XGB_N_JOBS)

def _clean_xgb_params(d: dict) -> dict:
    d = dict(d)
    if 'device' in d:  # in 2.x, having both triggers an error
        d.pop('gpu_id', None)
    return d

# ---------- Random search over a compact, strong space ----------
def tune_xgb_params(
    X, y, groups, *,
    base_params: dict,
    metric: str = "ap",         # 'ap' or 'auc'
    n_iter: int = 20,
    n_splits: int = 5,
    seed: int = 42,
    early_stopping_rounds: int = 400,
    pos_weight: float = 1.0,
    use_gpu: bool = True,
):
    rng = np.random.RandomState(seed)
    gkf = GroupKFold(n_splits=n_splits)
    metric = metric.lower()
    metric_name = metric.upper()
    metric_fn = average_precision_score if metric == "ap" else roc_auc_score

    # search space
    search = {
        "max_depth":          [4, 6, 8, 10],
        "min_child_weight":   [1, 3, 5, 10],
        "subsample":          [0.6, 0.8, 1.0],
        "colsample_bytree":   [0.6, 0.8, 1.0],
        "reg_lambda":         [0.5, 1.0, 1.5, 2.5],
        "reg_alpha":          [0.0, 0.1, 0.5],
        "gamma":              [0, 1, 5],
        "learning_rate":      [0.01, 0.03, 0.05, 0.1],
    }
    if use_gpu:
        search["max_bin"] = [256, 512, 1024]

    def _sample_one():
        return {k: rng.choice(v) for k, v in search.items()}

    best_params, best_score = None, -np.inf
    history = []

    for i in range(n_iter):
        cand = _sample_one()
        params = dict(base_params)
        params.update(cand)
        params.update(
            n_estimators=4000,                    # let ES decide best_iter
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=seed,
            verbosity=0
        )
        if not use_gpu and "max_bin" in params:
            params.pop("max_bin", None)

        fold_scores = []
        for tr, va in gkf.split(X, y, groups):
            model = XGBClassifier(**_clean_xgb_params(params))
            try:
                model.fit(X[tr], y[tr],
                          eval_set=[(X[va], y[va])],
                          early_stopping_rounds=early_stopping_rounds,
                          verbose=False)
            except TypeError:  # old xgboost
                model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
            p = model.predict_proba(X[va])[:, 1]
            fold_scores.append(metric_fn(y[va], p))

        mean_score = float(np.mean(fold_scores))
        history.append({"iter": i+1, "score": mean_score, **cand})
        print(f"[B][tune] {i+1}/{n_iter} {metric_name}={mean_score:.4f}  cand={cand}")
        if mean_score > best_score:
            best_score, best_params = mean_score, _clean_xgb_params(params)

    print(f"[B][tune] BEST {metric_name}={best_score:.4f}")
    return best_params, best_score, history

# ---------- Plotting for tuning history ----------
def _plot_tuning_results(history, metric="AP", out_dir=FIG_DIR, csv_dir=LOG_DIR):
    if not history: return
    df = pd.DataFrame(history)
    csv_path = os.path.join(csv_dir, "xgb_tuning_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"[B][tune] Saved tuning history → {csv_path}")

    plt.figure(figsize=(6,4))
    plt.plot(df["iter"], df["score"], marker="o")
    plt.xlabel("Iteration"); plt.ylabel(metric.upper())
    plt.title(f"XGB tuning: {metric.upper()} per iteration")
    _savefig(os.path.join(out_dir, "xgb_tuning_curve.png"))

    if {"learning_rate","max_depth"}.issubset(df.columns):
        plt.figure(figsize=(6,5))
        sc = plt.scatter(df["learning_rate"], df["max_depth"], c=df["score"])
        plt.colorbar(sc, label=metric.upper())
        plt.xlabel("learning_rate"); plt.ylabel("max_depth")
        plt.title("XGB tuning: score by (learning_rate, max_depth)")
        _savefig(os.path.join(out_dir, "xgb_tuning_scatter_lr_depth.png"))

# ================= Batched Deep Features (GPU) + CPU fallback ======================
@torch.no_grad()
def write_deep_feature_raster(in_tif, out_tif=DEEP_FEATS_TIF,
                              feat_channels=FEAT_CHANNELS, tile=TILE, encoder_ckpt=None):
    t0 = time.perf_counter()
    device = _select_device()
    with rasterio.open(in_tif) as src:
        C = src.count
        mean, std = compute_band_stats(src)
        meta = src.meta.copy(); meta.update(count=feat_channels, dtype='float32')
        width, height = src.width, src.height

    model = FCNFeatureExtractor(in_ch=C, out_ch=feat_channels, backbone=BACKBONE).eval()
    if device.type == "cuda":
        if encoder_ckpt and os.path.exists(encoder_ckpt):
            state = torch.load(encoder_ckpt, map_location="cpu")
            model.load_state_dict(state, strict=False)
        model = model.to(device).to(memory_format=torch.channels_last)
        ampctx = torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION)

    tiles = [(x, y, min(tile, width-x), min(tile, height-y))
             for y in range(0, height, tile) for x in range(0, width, tile)]
    n_tiles = len(tiles)

    desc = "[A] Deep features (GPU batched)" if device.type=="cuda" else "[A] Deep features (CPU)"
    with rasterio.open(out_tif, 'w', **meta) as dst, rasterio.open(in_tif) as src, tqdm(total=n_tiles, desc=desc) as pbar:
        if device.type == "cuda":
            bsize = max(1, int(GPU_TILE_BATCH))
            for i in range(0, n_tiles, bsize):
                batch_tiles = tiles[i:i+bsize]
                batch_arrs, metas = [], []
                max_eh = max_ew = 0
                for (x, y, w, h) in batch_tiles:
                    ex, ey, ew, eh, cx, cy = _expand_window(x, y, w, h, width, height, HALO)
                    arr = src.read(window=rw.Window(ex, ey, ew, eh), out_dtype='float32')
                    arr = _standardize_block(arr, mean, std)
                    batch_arrs.append(arr)
                    metas.append((x, y, w, h, cx, cy, ew, eh))
                    max_eh = max(max_eh, arr.shape[1]); max_ew = max(max_ew, arr.shape[2])
                batch_np = np.stack([_pad_to(a, max_eh, max_ew) for a in batch_arrs], axis=0)
                t = torch.from_numpy(batch_np).to(device, non_blocking=True).to(memory_format=torch.channels_last)

                if USE_TTA:
                    outs = None
                    for k in (0,1,2,3):
                        tk = _apply_aug(t, k)
                        with ampctx:
                            fk = model(tk)
                        fk = _undo_aug(fk, k)
                        outs = fk if outs is None else (outs + fk)
                    fout = (outs / 4.0).contiguous()
                else:
                    with ampctx:
                        fout = model(t).contiguous()

                fout = fout.to(dtype=torch.float32, memory_format=torch.contiguous_format).cpu().numpy()
                for b, (x, y, w, h, cx, cy, ew, eh) in enumerate(metas):
                    fc = fout[b, :, cy:cy+h, cx:cx+w]
                    dst.write(fc, window=rw.Window(x, y, w, h))
                    pbar.update(1)
        else:
            for (x, y, w, h) in tiles:
                ex, ey, ew, eh, cx, cy = _expand_window(x, y, w, h, width, height, HALO)
                arr = src.read(window=rw.Window(ex, ey, ew, eh), out_dtype='float32')
                arr = _standardize_block(arr, mean, std)
                t = torch.from_numpy(arr[None,...])
                f = model(t)[0].numpy().astype('float32')
                fc = f[:, cy:cy+h, cx:cx+w]
                dst.write(fc, window=rw.Window(x, y, w, h))
                pbar.update(1)

    _cuda_sync_if_any()
    dt = time.perf_counter() - t0
    print(f"[A] Wrote deep features → {out_tif}")
    _log_time("A: deep_features", dt, tiles=n_tiles, tile=tile, halo=HALO,
              K=feat_channels, backbone=BACKBONE, device=("CUDA" if torch.cuda.is_available() and USE_GPU_IF_AVAILABLE else "CPU"),
              batch=GPU_TILE_BATCH)

# ================= Fine-tuning (batched) ============================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, logits, targets):
        bce = self.bce(logits, targets); pt = torch.exp(-bce)
        return (self.alpha * (1-pt)**self.gamma * bce).mean()

def _rasterize_pos_mask(points_path, raster_path, label_col, pos_val, buffer_m=60):
    import rasterio.features as rfeat
    gdf = gpd.read_file(points_path)
    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs: gdf = gdf.to_crs(src.crs)
        geoms = gdf[gdf[label_col].astype(int)==pos_val].geometry
        if buffer_m > 0 and (src.crs.is_projected if src.crs else False):
            geoms = geoms.buffer(buffer_m)
        mask = rfeat.rasterize([(geom, 1) for geom in geoms],
                               out_shape=(src.height, src.width),
                               transform=src.transform, fill=0, dtype="float32")
    return mask

def finetune_extractor_on_points(factor_tif, points_path, epochs=FT_EPOCHS, lr=FT_LR,
                                 tile=FT_TILE, buffer_m=FT_BUFFER_M, ckpt_out="encoder_finetuned.pt"):
    t0 = time.perf_counter()
    device = _select_device()
    print(f"[FT] Fine-tuning on {device}  epochs={epochs}  buffer={buffer_m}m  tile={tile}  batch={FT_BATCH_TILES}")
    with rasterio.open(factor_tif) as src:
        C, H, W = src.count, src.height, src.width
        mean, std = compute_band_stats(src)
    model = FCNFeatureExtractor(in_ch=C, out_ch=FEAT_CHANNELS, backbone=BACKBONE).to(device)
    model = model.to(memory_format=torch.channels_last)
    for p in model.backbone[0:6].parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    loss_fn = FocalLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=="cuda" and USE_MIXED_PRECISION))
    posmask = _rasterize_pos_mask(points_path, factor_tif, LABEL_COL, POS_VALUE, buffer_m)

    windows = [(x, y, tile, tile) for y in range(0, H - (H % tile), tile)
                              for x in range(0, W - (W % tile), tile)]
    epoch_losses, epoch_secs = [], []
    model.train()
    for ep in range(epochs):
        ep_t0 = time.perf_counter()
        random.Random(SEED + ep).shuffle(windows)
        total = 0.0; steps = 0; acc_count = 0
        with rasterio.open(factor_tif) as src, tqdm(total=len(windows), desc=f"[FT] epoch {ep+1}/{epochs}") as pbar:
            for i in range(0, len(windows), FT_BATCH_TILES):
                batch = windows[i:i+FT_BATCH_TILES]
                Xs, Ys = [], []
                for (x, y, w, h) in batch:
                    Y = posmask[y:y+h, x:x+w].astype("float32")[None,...]
                    if Y.sum()==0 and np.random.rand() < FT_SKIP_ALL_NEG_PROB:
                        continue
                    X = src.read(window=rw.Window(x, y, w, h), out_dtype="float32")
                    X = _standardize_block(X, mean, std)
                    Xs.append(X); Ys.append(Y)
                if not Xs:
                    pbar.update(len(batch)); continue
                xt = torch.from_numpy(np.stack(Xs, axis=0)).to(device, non_blocking=True)
                yt = torch.from_numpy(np.stack(Ys, axis=0)).to(device, non_blocking=True)
                xt = xt.to(memory_format=torch.channels_last)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device.type=="cuda" and USE_MIXED_PRECISION)):
                    out = model(xt)
                    logits = out.narrow(1, 0, 1).contiguous()
                    loss = loss_fn(logits, yt)
                if device.type == "cuda":
                    scaler.scale(loss).backward()
                    acc_count += 1
                    if acc_count % FT_ACCUM_STEPS == 0:
                        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                else:
                    loss.backward(); opt.step()
                total += float(loss.item()); steps += 1
                pbar.update(len(batch))
        if device.type == "cuda" and acc_count % FT_ACCUM_STEPS != 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        _cuda_sync_if_any()
        ep_sec = time.perf_counter() - ep_t0
        ep_loss = total / max(1, steps)
        epoch_losses.append(ep_loss); epoch_secs.append(ep_sec)
        print(f"[FT] epoch {ep+1}/{epochs}  loss={ep_loss:.4f}  time={_fmt_hms(ep_sec)}")

    # Curves + CSV
    plt.figure(); plt.plot(range(1, len(epoch_losses)+1), epoch_losses)
    plt.xlabel("Epoch"); plt.ylabel("Training loss (focal)"); plt.title("Fine-tuning loss")
    _savefig(os.path.join(FIG_DIR, "finetune_loss.png"))
    pd.DataFrame({"epoch": np.arange(1, len(epoch_losses)+1),
                  "loss": epoch_losses, "seconds": epoch_secs}).to_csv(
        os.path.join(LOG_DIR, "finetune_epochs.csv"), index=False
    )

    torch.save(model.state_dict(), ckpt_out)
    _cuda_sync_if_any()
    dt = time.perf_counter() - t0
    print("[FT] saved encoder weights →", ckpt_out)
    _log_time("FT: total", dt, epochs=epochs, avg_epoch_sec=(round(np.mean(epoch_secs),3) if epoch_secs else None),
              device=("CUDA" if device.type=="cuda" else "CPU"), batch=FT_BATCH_TILES, tile=tile)
    return ckpt_out

# ================= Train XGBoost =====================================================
def _maybe_calibrate_prefit(base_model, Xva, yva, method):
    if method is None: return base_model
    if isinstance(method, str):
        m = method.strip().lower()
        if m in ("", "none"): return base_model
        if m not in ("sigmoid", "isotonic"):
            warnings.warn(f"[B] Unknown CALIBRATION_METHOD='{method}', falling back to 'sigmoid'.")
            m = "sigmoid"
    else:
        return base_model
    cal = CalibratedClassifierCV(base_model, method=m, cv='prefit')
    cal.fit(Xva, yva)
    return cal

def train_xgb_from_points(deep_feats_tif=DEEP_FEATS_TIF, factors_tif=FACTOR_TIF,
                          points_path=POINTS_PATH, out_model=MODEL_PATH, n_splits=N_SPLITS):
    t0_total = time.perf_counter()
    print("[B] Training XGBoost…")
    rng = np.random.RandomState(SEED)

    gdf = gpd.read_file(points_path)
    with rasterio.open(factors_tif) as rst:
        gdf = _reproject_points_to_raster_crs(gdf, rst)
    gdf = _drop_ambiguous_negatives(gdf, LABEL_COL, POS_VALUE, NEG_VALUE, NEG_EXCLUSION_RADIUS_M)

    if LABEL_COL not in gdf.columns: raise ValueError(f"'{LABEL_COL}' not found in {points_path}.")
    cls = gdf[LABEL_COL].astype(int)
    if not cls.isin([POS_VALUE, NEG_VALUE]).all():
        bad = gdf.loc[~cls.isin([POS_VALUE, NEG_VALUE]), LABEL_COL].unique()
        raise ValueError(f"Unexpected values in '{LABEL_COL}': {bad}.")
    y = np.where(cls.values == POS_VALUE, 1, 0).astype(int)
    n_pos, n_neg = int((y==1).sum()), int((y==0).sum())
    print(f"[B] Samples: total={len(y)} | pos={n_pos} | neg={n_neg}")

    # --- Features
    t0_sample = time.perf_counter()
    points_xy = [(geom.x, geom.y) for geom in gdf.geometry]
    Xdeep = _sample_rasters_at_points([deep_feats_tif], points_xy)
    Xraw  = _sample_rasters_at_points([factors_tif], points_xy)
    X = np.hstack([Xdeep, Xraw]).astype('float32')
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    _log_time("B: sample_features", time.perf_counter()-t0_sample,
              N=len(y), Kdeep=Xdeep.shape[1], Kraw=Xraw.shape[1])

    # --- Split & groups
    groups = _make_spatial_blocks(gdf, n_splits=n_splits)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=SEED)
    tr_idx, va_idx = next(gss.split(X, y, groups))
    Xtr, ytr, Xva, yva = X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]

    # class imbalance weight from train partition
    pos = max(int((ytr==1).sum()), 1); neg = max(int((ytr==0).sum()), 1)
    spw = (float(neg)/float(pos))**0.75

    # ====== Hyperparameter tuning ======
    base_flags = _xgb_gpu_params()
    base_core  = dict(eval_metric="logloss", random_state=SEED)
    t0_tune = time.perf_counter()
    best_params, best_cv, tune_hist = tune_xgb_params(
        X, y, groups,
        base_params={**base_core, **base_flags},
        metric="ap",
        n_iter=XGB_TUNE_ITER,
        n_splits=min(5, n_splits),
        seed=SEED,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        pos_weight=spw,
        use_gpu=XGB_USE_GPU
    )
    _plot_tuning_results(tune_hist, metric="AP", out_dir=FIG_DIR, csv_dir=LOG_DIR)
    _log_time("B: xgb_tune", time.perf_counter()-t0_tune, iters=XGB_TUNE_ITER)

    # ====== Final model: refit on train with ES on valid ======
    if best_params is None:
        best_params = {**base_core, **_xgb_gpu_params()}
        best_params = _clean_xgb_params(best_params)
    print("[B] Best XGB params (subset):", {k: best_params[k] for k in
          ["max_depth","min_child_weight","subsample","colsample_bytree",
           "reg_lambda","reg_alpha","gamma","learning_rate","n_estimators","max_bin"]
          if k in best_params})

    base = XGBClassifier(**best_params)
    best_iter = None
    t0_fit = time.perf_counter()
    try:
        base.fit(
            Xtr, ytr,
            eval_set=[(Xtr, ytr), (Xva, yva)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False
        )
        best_iter = getattr(base, "best_iteration", None)
    except TypeError:
        base.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    _log_time("B: xgb_fit", time.perf_counter()-t0_fit,
              early_stopping=("yes" if best_iter is not None else "no"))

    # --- Logloss curve (if available)
    try:
        evals = base.evals_result()
        if "validation_0" in evals and "logloss" in evals["validation_0"]:
            plt.figure()
            plt.plot(evals["validation_0"]["logloss"], label="train")
            if "validation_1" in evals and "logloss" in evals["validation_1"]:
                plt.plot(evals["validation_1"]["logloss"], label="valid")
            plt.xlabel("Boosting round"); plt.ylabel("Logloss"); plt.legend()
            plt.title("XGB logloss (train vs valid)")
            _savefig(os.path.join(FIG_DIR, "xgb_logloss.png"))
    except Exception:
        pass

    # ====== Optional calibration (keeps base model; avoids leakage)
    t0_cal = time.perf_counter()
    clf = _maybe_calibrate_prefit(base, Xva, yva, CALIBRATION_METHOD)
    _log_time("B: calibrate", time.perf_counter()-t0_cal, method=(CALIBRATION_METHOD or "none"))
    print(f"[B] Fit done. Calibration: {('none' if (CALIBRATION_METHOD in (None, '', 'none')) else str(CALIBRATION_METHOD))}  | best_iter={best_iter}")

    # ====== Metrics & plots on held-out valid ======
    p_va = clf.predict_proba(Xva)[:,1]
    p_tr = clf.predict_proba(Xtr)[:,1] if hasattr(clf, "predict_proba") else base.predict_proba(Xtr)[:,1]

    # Save ROC/PR curve data with padded thresholds (to avoid length mismatch)
    fpr_arr, tpr_arr, thr_roc = roc_curve(yva, p_va)
    roc_df = _curve_df_with_thresholds(fpr_arr, tpr_arr, thr_roc, "fpr", "tpr")
    roc_df.to_csv(os.path.join(LOG_DIR, "roc_valid.csv"), index=False)

    prec_arr, rec_arr, thr_pr = precision_recall_curve(yva, p_va)
    pr_df = _curve_df_with_thresholds(rec_arr, prec_arr, thr_pr, "recall", "precision")
    pr_df.to_csv(os.path.join(LOG_DIR, "pr_valid.csv"), index=False)

    # Plot ROC/PR
    plt.figure(); plt.plot(fpr_arr, tpr_arr); plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (AUC={roc_auc_score(yva,p_va):.3f})"); _savefig(os.path.join(FIG_DIR, "roc_curve.png"))
    plt.figure(); plt.plot(rec_arr, prec_arr)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR curve (AP={average_precision_score(yva,p_va):.3f})")
    _savefig(os.path.join(FIG_DIR, "pr_curve.png"))

    # Threshold sweep
    def _threshold_metrics(y_true, p, threshold):
        y_pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        specificity = tn / max(1, (tn+fp))
        return {"threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "specificity": specificity,
                "brier": brier_score_loss(y_true, p),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(yva, (p_va>=t).astype(int), zero_division=0) for t in grid]
    best_t = float(grid[int(np.argmax(f1s))])

    # Save threshold sweep CSV
    pd.DataFrame({"threshold": grid, "f1": f1s}).to_csv(os.path.join(LOG_DIR, "threshold_f1.csv"), index=False)

    metrics_05 = _threshold_metrics(yva, p_va, 0.5)
    metrics_bt = _threshold_metrics(yva, p_va, best_t)
    plt.figure(figsize=(6,4)); plt.plot(grid, f1s); plt.axvline(0.5, linestyle='--'); plt.axvline(best_t, linestyle=':')
    plt.xlabel("Threshold"); plt.ylabel("F1 score"); plt.title(f"Threshold sweep (best_t={best_t:.2f})")
    _savefig(os.path.join(FIG_DIR, "threshold_f1.png"))

    # Summary CSVs (for plotting/analysis)
    summary = {"valid_AUROC": roc_auc_score(yva, p_va), "valid_AP": average_precision_score(yva, p_va),
               "valid_Brier": brier_score_loss(yva, p_va), "train_AUROC": roc_auc_score(ytr, p_tr),
               "train_AP": average_precision_score(ytr, p_tr), "train_Brier": brier_score_loss(ytr, p_tr),
               "best_iteration": best_iter, "calibration": CALIBRATION_METHOD or "none"}
    pd.DataFrame([summary]).to_csv(os.path.join(LOG_DIR, "summary_metrics.csv"), index=False)
    pd.DataFrame([{"set":"valid", **metrics_05}, {"set":"valid_bestT", **metrics_bt}]
                ).to_csv(os.path.join(LOG_DIR, "threshold_metrics.csv"), index=False)
    print("[B] Validation metrics @0.5:", metrics_05)
    print("[B] Validation metrics @bestT:", metrics_bt)

    # Save per-sample predictions for train/valid (useful for stats plots)
    points_xy_arr = np.asarray(points_xy, dtype="float64")
    groups_arr = np.asarray(groups)
    _assert_same_len(idx=tr_idx, y=ytr, p=p_tr,
                     x=points_xy_arr[tr_idx, 0], yx=points_xy_arr[tr_idx, 1], grp=groups_arr[tr_idx])
    df_tr_pred = pd.DataFrame({
        "idx": tr_idx, "set": "train", "y_true": ytr, "p_pred": p_tr,
        "x": points_xy_arr[tr_idx, 0], "y": points_xy_arr[tr_idx, 1], "group": groups_arr[tr_idx],
    })
    _assert_same_len(idx=va_idx, y=yva, p=p_va,
                     x=points_xy_arr[va_idx, 0], yx=points_xy_arr[va_idx, 1], grp=groups_arr[va_idx])
    df_va_pred = pd.DataFrame({
        "idx": va_idx, "set": "valid", "y_true": yva, "p_pred": p_va,
        "x": points_xy_arr[va_idx, 0], "y": points_xy_arr[va_idx, 1], "group": groups_arr[va_idx],
    })
    pd.concat([df_tr_pred, df_va_pred], ignore_index=True).to_csv(
        os.path.join(LOG_DIR, "predictions_train_valid.csv"), index=False
    )

    # ====== Spatial CV report using the tuned params ======
    t0_cv = time.perf_counter()
    gkf = GroupKFold(n_splits=n_splits)
    aucs, praucs, briers = [], [], []
    with tqdm(total=n_splits, desc="[B] Spatial CV") as pbar:
        for tr, te in gkf.split(X, y, groups):
            base_cv = XGBClassifier(**best_params)
            if CALIBRATION_METHOD is None:
                base_cv.fit(X[tr], y[tr]); p = base_cv.predict_proba(X[te])[:,1]
            else:
                cal_cv = CalibratedClassifierCV(base_cv, method=CALIBRATION_METHOD, cv=3).fit(X[tr], y[tr])
                p = cal_cv.predict_proba(X[te])[:,1]
            aucs.append(roc_auc_score(y[te], p))
            praucs.append(average_precision_score(y[te], p))
            briers.append(brier_score_loss(y[te], p))
            pbar.update(1)
    cv_report = {"CV_AUROC_mean":np.mean(aucs), "CV_AUROC_std":np.std(aucs),
                 "CV_AP_mean":np.mean(praucs), "CV_AP_std":np.std(praucs),
                 "CV_Brier_mean":np.mean(briers), "CV_Brier_std":np.std(briers)}
    pd.DataFrame([cv_report]).to_csv(os.path.join(LOG_DIR, "cv_report.csv"), index=False)
    print("[B] Spatial CV:", cv_report)
    _log_time("B: spatial_cv", time.perf_counter()-t0_cv, folds=n_splits)

    joblib.dump({"clf": clf, "best_iter": best_iter,
                 "valid_metrics": summary, "thresholds": {"best_t": best_t}}, out_model)
    print(f"[B] Saved model → {out_model}")
    _log_time("B: total_train", time.perf_counter()-t0_total,
              N=len(y), pos=int((y==1).sum()), neg=int((y==0).sum()))

# ================= Predict Probability Map ==========================================
def _expand_window(x, y, w, h, width, height, halo):
    x0 = max(0, x - halo); y0 = max(0, y - halo)
    x1 = min(width,  x + w + halo); y1 = min(height, y + h + halo)
    return x0, y0, x1 - x0, y1 - y0, (x - x0), (y - y0)

def _apply_aug(t, k):
    if k == 0:  return t
    if k == 1:  return torch.flip(t, dims=[-1])
    if k == 2:  return torch.flip(t, dims=[-2])
    if k == 3:  return t.transpose(-2, -1)
    return t

def _undo_aug(t, k):
    if k == 0:  return t
    if k == 1:  return torch.flip(t, dims=[-1])
    if k == 2:  return torch.flip(t, dims=[-2])
    if k == 3:  return t.transpose(-2, -1)
    return t

def _pad_to(arr, target_h, target_w):
    C, h, w = arr.shape
    pad_h = target_h - h; pad_w = target_w - w
    if pad_h == 0 and pad_w == 0: return arr
    return np.pad(arr, ((0,0),(0,pad_h),(0,pad_w)), mode='edge')

def predict_probability_map(deep_feats_tif=DEEP_FEATS_TIF, factors_tif=FACTOR_TIF,
                            model_path=MODEL_PATH, out_tif=OUT_PROB_TIF, tile=TILE):
    t0 = time.perf_counter()
    print("[C] Predicting probability map…")
    payload = joblib.load(model_path); clf = payload["clf"]
    with rasterio.open(deep_feats_tif) as dsrc, rasterio.open(factors_tif) as fsrc:
        assert dsrc.width==fsrc.width and dsrc.height==fsrc.height
        assert dsrc.transform==fsrc.transform and dsrc.crs==fsrc.crs
        meta = dsrc.meta.copy(); meta.update(count=1, dtype='float32')
        width, height = dsrc.width, dsrc.height
        total = ((height + tile - 1)//tile) * ((width + tile - 1)//tile)
        with rasterio.open(out_tif, 'w', **meta) as dst, tqdm(total=total, desc="[C] Map predict") as pbar:
            for y in range(0, height, tile):
                for x in range(0, width, tile):
                    window = rw.Window(x, y, min(tile, width-x), min(tile, height-y))
                    D = dsrc.read(window=window, out_dtype='float32')
                    F = fsrc.read(window=window, out_dtype='float32')
                    H, W = D.shape[1], D.shape[2]
                    X = np.concatenate([D, F], axis=0).reshape(D.shape[0]+F.shape[0], -1).T
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                    P = clf.predict_proba(X)[:,1].astype('float32').reshape(H, W)
                    dst.write(P, 1, window=window); pbar.update(1)
    dt = time.perf_counter() - t0
    print(f"[C] Wrote probability map → {out_tif}")
    _log_time("C: predict_map", dt, width=width, height=height, tile=tile, tiles=total)

# ================= Main =============================================================
if __name__ == "__main__":
    ckpt = None
    if FINE_TUNE_ENABLED:
        ckpt = finetune_extractor_on_points(
            FACTOR_TIF, POINTS_PATH,
            epochs=FT_EPOCHS, lr=FT_LR, tile=FT_TILE, buffer_m=FT_BUFFER_M,
            ckpt_out="encoder_finetuned.pt"
        )
    write_deep_feature_raster(
        FACTOR_TIF, out_tif=DEEP_FEATS_TIF,
        feat_channels=FEAT_CHANNELS, tile=TILE, encoder_ckpt=ckpt
    )
    train_xgb_from_points(DEEP_FEATS_TIF, FACTOR_TIF, POINTS_PATH, out_model=MODEL_PATH, n_splits=N_SPLITS)
    predict_probability_map(DEEP_FEATS_TIF, FACTOR_TIF, MODEL_PATH, out_tif=OUT_PROB_TIF, tile=TILE)
    _write_timings_csv()
