import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from utils import *
from imageio.v2 import imread




# === Experiments for Project 3: SNR, Missingness, and True Rank r ===
# By Kia

try:
    from utils import softImpute as _softImpute
except Exception:
    _softImpute = None

try:
    from utils import cross_validate_lambda as _cross_validate_lambda
except Exception:
    _cross_validate_lambda = None


# -------------------------------
# Minimal fallbacks (used only if utils.* not found)
# -------------------------------
def _svd_shrink(X, lam):
    """Soft-threshold singular values by lam."""
    U, s, VT = np.linalg.svd(X, full_matrices=False)
    s_shrunk = np.maximum(s - lam, 0.0)
    return (U * s_shrunk) @ VT

def softImpute_fallback(Y, reg=1.0, num_iterations=200, tolerance=1e-5, verbose=False):
    """
    Super-simple SoftImpute: keep observed entries, fill missing with current low-rank estimate.
    Y contains np.nan at missing entries.
    """
    X_old = np.nan_to_num(Y, nan=0.0)
    mask = ~np.isnan(Y)
    for it in range(num_iterations):
        # fill missing with previous estimate
        Y_filled = np.where(mask, Y, X_old)
        X_new = _svd_shrink(Y_filled, reg)
        # convergence check using Frobenius norm
        denom = np.linalg.norm(X_old, ord='fro') + 1e-12
        rel = np.linalg.norm(X_new - X_old, ord='fro') / denom
        if verbose:
            print(f"[SoftImpute] it={it} rel={rel:.3e}")
        if rel < tolerance:
            break
        X_old = X_new
    X_new = np.clip(X_new, 0.0, 1.0)  # for normalized images/synthetic data
    return X_new

def cross_validate_lambda_fallback(
    Y, lambda_grid, holdout_fraction=0.1, num_iterations=200, tolerance=1e-5, random_state=0
):
    rng = np.random.default_rng(random_state)
    mask_obs = ~np.isnan(Y)
    obs_idx = np.argwhere(mask_obs)
    n_hold = max(1, int(holdout_fraction * len(obs_idx)))
    hold_idx = tuple(obs_idx[rng.choice(len(obs_idx), size=n_hold, replace=False)].T)
    # make a copy that hides holdout too
    Y_cv = Y.copy()
    Y_cv[hold_idx] = np.nan
    rmse = {}
    for lam in lambda_grid:
        X_hat = softImpute(Y_cv, reg=lam, num_iterations=num_iterations, tolerance=tolerance)
        err = np.sqrt(np.mean((X_hat[hold_idx] - Y[hold_idx]) ** 2))
        rmse[float(lam)] = float(err)
    best_lambda = min(rmse, key=rmse.get)
    return float(best_lambda), rmse

# Bind the actual callables we’ll use
softImpute = _softImpute if _softImpute is not None else softImpute_fallback
cross_validate_lambda = (
    _cross_validate_lambda if _cross_validate_lambda is not None else cross_validate_lambda_fallback
)

# -------------------------------
# Helpers for synthetic experiments
# -------------------------------
'''def low_rank_matrix(m, n, r, rng=None, normalize=True):
    """Generate X = U V^T with controllable true rank r."""
    rng = np.random.default_rng(None if rng is None else rng)
    U = rng.standard_normal((m, r))
    V = rng.standard_normal((n, r))
    X = U @ V.T
    if normalize:
        # scale to [0,1] so RMSE is in a nice range
        X = (X - X.min()) / (X.max() - X.min() + 1e-12)
    return X
'''
'''def low_rank_matrix(m, n, r, rng=None, normalize=True):
    """Generate X = U V^T with controllable true rank r."""
    rng = np.random.default_rng(None if rng is None else rng)
    U = rng.standard_normal((m, r))
    V = rng.standard_normal((n, r))
    X = U @ V.T
    if normalize:
        # Normalize so that the largest singular value = 1
        norm = np.linalg.norm(X, 'fro')
        X = X / norm
    return X'''
# REPLACE low_rank_matrix in main.py
def low_rank_matrix(m, n, r, rng=None, normalize=True):
    rng = np.random.default_rng(None if rng is None else rng)
    U = rng.standard_normal((m, r))
    V = rng.standard_normal((n, r))
    X = U @ V.T
    if normalize:
        # Normalize so that the largest singular value = 1 (more reasonable scale)
        s_max = np.linalg.svd(X, compute_uv=False)[0]
        X = X / (s_max + 1e-12)
    return X

def add_gaussian_noise(X, snr_db, rng=None):
    """
    Add Gaussian noise to achieve target SNR in dB: 10*log10(P_signal/P_noise) = snr_db
    Uses Frobenius norm for signal power calculation
    """
    rng = np.random.default_rng(None if rng is None else rng)
    # Use Frobenius norm for signal power calculation (per-element power)
    sig_pow = (np.linalg.norm(X, 'fro') ** 2) / X.size
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_pow), size=X.shape)
    Y = X + noise
    return np.clip(Y, 0.0, 1.0)

def apply_missingness(X, missing_fraction, rng=None):
    """Return Y with np.nan at missing entries and a boolean mask of observed entries."""
    rng = np.random.default_rng(None if rng is None else rng)
    mask = rng.random(X.shape) > missing_fraction
    Y = X.copy().astype(float)
    Y[~mask] = np.nan
    return Y, mask

def rmse_on_mask(truth, pred, mask):
    return float(np.sqrt(np.mean(((truth - pred)[mask]) ** 2)))

def pick_lambda(Y, lambda_grid, **kw):
    # consistent wrapper
    lam, _ = cross_validate_lambda(Y, lambda_grid, **kw)
    return lam

# -------------------------------
# Experiment 1: Vary SNR
# -------------------------------
def experiment_snr(
    m=128, n=128, r=10, snr_list=(5, 10, 15, 20, 30),
    missing_fraction=0.5, lam_grid=None, rng=0
):
    rng = np.random.default_rng(rng)
    X = low_rank_matrix(m, n, r, rng=rng)
    if lam_grid is None:
        lam_grid = np.geomspace(0.05, 5.0, 12)

    results = []
    for snr in snr_list:
        Xn = add_gaussian_noise(X, snr_db=snr, rng=rng)
        Y, obs_mask = apply_missingness(Xn, missing_fraction=missing_fraction, rng=rng)

        lam = pick_lambda(
            Y, lam_grid, holdout_fraction=0.1, num_iterations=200, tolerance=1e-5, random_state=42
        )
        Xhat = softImpute(Y, reg=lam, num_iterations=200, tolerance=1e-5, clip_output=False)

        train_rmse = rmse_on_mask(X, Xhat, obs_mask)              # on observed entries
        test_rmse  = rmse_on_mask(X, Xhat, ~obs_mask)             # on missing entries
        results.append((snr, train_rmse, test_rmse))

    # Plot
    snr_vals, train_vals, test_vals = map(np.array, zip(*results))
    plt.figure(figsize=(6,4))
    plt.plot(snr_vals, train_vals, marker="o", label="Train RMSE (observed)")
    plt.plot(snr_vals, test_vals,  marker="o", label="Test RMSE (missing)")
    plt.gca().invert_xaxis()  # worse SNR (left) to better SNR (right) visually rising to left
    plt.xlabel("SNR (dB) – higher is cleaner")
    plt.ylabel("RMSE")
    plt.title(f"Effect of SNR (true rank r={r}, missing={missing_fraction:.0%})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    return results

# -------------------------------
# Experiment 2: Vary Missingness
# -------------------------------
def experiment_missingness(
    m=128, n=128, r=10, snr_db=20, miss_list=(0.2, 0.4, 0.6, 0.8),
    lam_grid=None, rng=1
):
    rng = np.random.default_rng(rng)
    X = low_rank_matrix(m, n, r, rng=rng)
    Xn = add_gaussian_noise(X, snr_db=snr_db, rng=rng)

    if lam_grid is None:
        lam_grid = np.geomspace(0.05, 5.0, 12)

    results = []
    for mf in miss_list:
        Y, obs_mask = apply_missingness(Xn, missing_fraction=mf, rng=rng)
        lam = pick_lambda(
            Y, lam_grid, holdout_fraction=0.1, num_iterations=200, tolerance=1e-5, random_state=7
        )
        Xhat = softImpute(Y, reg=lam, num_iterations=200, tolerance=1e-5, clip_output=False)
        train_rmse = rmse_on_mask(X, Xhat, obs_mask)
        test_rmse  = rmse_on_mask(X, Xhat, ~obs_mask)
        results.append((mf, train_rmse, test_rmse))

    # Plot
    miss_vals, train_vals, test_vals = map(np.array, zip(*results))
    plt.figure(figsize=(6,4))
    plt.plot(miss_vals, train_vals, marker="o", label="Train RMSE (observed)")
    plt.plot(miss_vals, test_vals,  marker="o", label="Test RMSE (missing)")
    plt.xlabel("Missing fraction")
    plt.ylabel("RMSE")
    plt.title(f"Effect of Missingness (true rank r={r}, SNR={snr_db} dB)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    return results

# -------------------------------
# Experiment 3: Vary True Rank r
# -------------------------------
def experiment_true_rank(
    m=128, n=128, r_list=(2, 5, 10, 20, 30), snr_db=20, missing_fraction=0.5,
    lam_grid=None, rng=2
):
    rng = np.random.default_rng(rng)
    if lam_grid is None:
        lam_grid = np.geomspace(0.05, 5.0, 12)

    results = []
    for r in r_list:
        X = low_rank_matrix(m, n, r, rng=rng)
        Xn = add_gaussian_noise(X, snr_db=snr_db, rng=rng)
        Y, obs_mask = apply_missingness(Xn, missing_fraction=missing_fraction, rng=rng)

        lam = pick_lambda(
            Y, lam_grid, holdout_fraction=0.1, num_iterations=200, tolerance=1e-5, random_state=11
        )
        Xhat = softImpute(Y, reg=lam, num_iterations=200, tolerance=1e-5, clip_output=False)
        train_rmse = rmse_on_mask(X, Xhat, obs_mask)
        test_rmse  = rmse_on_mask(X, Xhat, ~obs_mask)
        results.append((r, train_rmse, test_rmse))

    # Plot
    r_vals, train_vals, test_vals = map(np.array, zip(*results))
    plt.figure(figsize=(6,4))
    plt.plot(r_vals, train_vals, marker="o", label="Train RMSE (observed)")
    plt.plot(r_vals, test_vals,  marker="o", label="Test RMSE (missing)")
    plt.xlabel("True rank r")
    plt.ylabel("RMSE")
    plt.title(f"Effect of True Rank (SNR={snr_db} dB, missing={missing_fraction:.0%})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    return results

# ============================================================
# EXTRA: Synthetic grid + HardImpute + Non-random missingness
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

# ---------- HardImpute (keep top-k singular values) ----------
def hardImpute(Y, rank=10, num_iterations=200, tolerance=1e-5, verbose=False):
    """
    Hard-thresholded SVD imputation.
    Y has np.nan on missing entries. We iterate:
      1) fill missing with current estimate
      2) compute rank-k truncated SVD (no shrinkage)
    """
    X_old = np.nan_to_num(Y, nan=0.0)
    mask = ~np.isnan(Y)
    for it in range(num_iterations):
        Y_filled = np.where(mask, Y, X_old)
        # rank-k truncated SVD
        U, s, VT = np.linalg.svd(Y_filled, full_matrices=False)
        k = min(rank, len(s))
        X_new = (U[:, :k] * s[:k]) @ VT[:k, :]
        rel = np.linalg.norm(X_new - X_old, ord='fro') / (np.linalg.norm(X_old, ord='fro') + 1e-12)
        if verbose:
            print(f"[HardImpute] it={it} rel={rel:.3e}")
        if rel < tolerance:
            break
        X_old = X_new
    return np.clip(X_new, 0.0, 1.0)

def cross_validate_rank(Y, k_grid, holdout_fraction=0.1, num_iterations=200, tolerance=1e-5, random_state=0):
    """Pick k for HardImpute by CV on observed entries (like we do for λ)."""
    rng = np.random.default_rng(random_state)
    mask_obs = ~np.isnan(Y)
    obs_idx = np.argwhere(mask_obs)
    n_hold = max(1, int(holdout_fraction * len(obs_idx)))
    hold_idx = tuple(obs_idx[rng.choice(len(obs_idx), size=n_hold, replace=False)].T)
    Y_cv = Y.copy()
    Y_cv[hold_idx] = np.nan

    rmse = {}
    for k in k_grid:
        X_hat = hardImpute(Y_cv, rank=int(k), num_iterations=num_iterations, tolerance=tolerance)
        rmse[int(k)] = float(np.sqrt(np.mean((X_hat[hold_idx] - Y[hold_idx]) ** 2)))
    best_k = min(rmse, key=rmse.get)
    return int(best_k), rmse

# ---------- Non-random missingness makers ----------
def hide_blocks(X, frac_rows=0.2, frac_cols=0.2, rng=None):
    """Hide whole rows and columns (NMAR-ish). Returns Y with NaNs and observed mask."""
    rng = np.random.default_rng(None if rng is None else rng)
    m, n = X.shape
    n_hide_r = int(np.round(frac_rows * m))
    n_hide_c = int(np.round(frac_cols * n))
    rows = rng.choice(m, size=max(0, n_hide_r), replace=False)
    cols = rng.choice(n, size=max(0, n_hide_c), replace=False)

    Y = X.copy().astype(float)
    mask = np.ones_like(X, dtype=bool)
    if len(rows) > 0:
        Y[rows, :] = np.nan
        mask[rows, :] = False
    if len(cols) > 0:
        Y[:, cols] = np.nan
        mask[:, cols] = False
    return Y, mask, rows, cols

# ---------- Combined synthetic grid (rank / SNR / missingness) ----------
def experiment_grid_soft(m=96, n=96, r_list=(2,5,10,20), snr_list=(10,15,20,30),
                         miss_list=(0.2,0.4,0.6), lam_grid=None, rng=123):
    """
    For each (r, SNR, missing), generate X=UV^T, add noise, mask entries,
    pick λ by CV, run SoftImpute, record TEST RMSE.
    Returns a dict of 2D arrays for each sweep and makes heatmaps + line plots.
    """
    rng = np.random.default_rng(rng)
    if lam_grid is None:
        lam_grid = np.geomspace(0.05, 5.0, 12)

    # Helper to get test RMSE for one synthetic case
    def run_case(r, snr_db, miss):
        X = low_rank_matrix(m, n, r, rng=rng)
        Xn = add_gaussian_noise(X, snr_db=snr_db, rng=rng)
        Y, obs_mask = apply_missingness(Xn, missing_fraction=miss, rng=rng)
        lam = pick_lambda(Y, lam_grid, holdout_fraction=0.1, num_iterations=200, tolerance=1e-5, random_state=7)
        Xhat = softImpute(Y, reg=lam, num_iterations=200, tolerance=1e-5, clip_output=False)
        test_rmse = rmse_on_mask(X, Xhat, ~obs_mask)
        return test_rmse

    # Heatmap 1: RMSE vs (r, SNR) at fixed missing
    miss0 = miss_list[1] if len(miss_list) > 1 else miss_list[0]
    mat_r_snr = np.zeros((len(r_list), len(snr_list)))
    for i, r in enumerate(r_list):
        for j, snr in enumerate(snr_list):
            mat_r_snr[i, j] = run_case(r, snr, miss0)

    plt.figure(figsize=(6,4))
    plt.imshow(mat_r_snr, origin='lower', aspect='auto', cmap='viridis',
               extent=[snr_list[0], snr_list[-1], r_list[0], r_list[-1]])
    plt.colorbar(label="Test RMSE")
    plt.xlabel("SNR (dB)")
    plt.ylabel("True rank r")
    plt.title(f"SoftImpute test RMSE | missing={miss0:.0%}")
    plt.tight_layout()

    # Heatmap 2: RMSE vs (missing, SNR) at fixed r
    r0 = r_list[min(1, len(r_list)-1)]
    mat_miss_snr = np.zeros((len(miss_list), len(snr_list)))
    for i, miss in enumerate(miss_list):
        for j, snr in enumerate(snr_list):
            mat_miss_snr[i, j] = run_case(r0, snr, miss)

    plt.figure(figsize=(6,4))
    plt.imshow(mat_miss_snr, origin='lower', aspect='auto', cmap='magma',
               extent=[snr_list[0], snr_list[-1], miss_list[0], miss_list[-1]])
    plt.colorbar(label="Test RMSE")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Missing fraction")
    plt.title(f"SoftImpute test RMSE | true rank r={r0}")
    plt.tight_layout()

    # Line sweeps (fix two, vary one)
    # RMSE vs r
    snr_fix, miss_fix = snr_list[-1], miss_list[1] if len(miss_list)>1 else miss_list[0]
    rmse_vs_r = [run_case(r, snr_fix, miss_fix) for r in r_list]
    plt.figure(figsize=(6,4))
    plt.plot(r_list, rmse_vs_r, marker='o')
    plt.xlabel("True rank r")
    plt.ylabel("Test RMSE")
    plt.title(f"SoftImpute | SNR={snr_fix} dB, missing={miss_fix:.0%}")
    plt.grid(True); plt.tight_layout()

    # RMSE vs SNR
    r_fix = r_list[1] if len(r_list)>1 else r_list[0]
    rmse_vs_snr = [run_case(r_fix, s, miss_fix) for s in snr_list]
    plt.figure(figsize=(6,4))
    plt.plot(snr_list, rmse_vs_snr, marker='o')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Test RMSE")
    plt.title(f"SoftImpute | r={r_fix}, missing={miss_fix:.0%}")
    plt.grid(True); plt.tight_layout()

    # RMSE vs missing
    snr_fix = snr_list[-1]
    rmse_vs_miss = [run_case(r_fix, snr_fix, mf) for mf in miss_list]
    plt.figure(figsize=(6,4))
    plt.plot(miss_list, rmse_vs_miss, marker='o')
    plt.xlabel("Missing fraction")
    plt.ylabel("Test RMSE")
    plt.title(f"SoftImpute | r={r_fix}, SNR={snr_fix} dB")
    plt.grid(True); plt.tight_layout()

    return dict(
        heat_r_snr=mat_r_snr, heat_miss_snr=mat_miss_snr,
        rmse_vs_r=rmse_vs_r, rmse_vs_snr=rmse_vs_snr, rmse_vs_miss=rmse_vs_miss
    )

# ---------- Soft vs Hard comparison ----------
def compare_soft_vs_hard(r=10, snr_db=20, missing_fraction=0.6, k_grid=None, lam_grid=None, rng=99):
    rng = np.random.default_rng(rng)
    if lam_grid is None:
        lam_grid = np.geomspace(0.05, 5.0, 12)
    if k_grid is None:
        # meaningful k range around r
        k_grid = list(range(max(1, r//2), 2*r+1, max(1, r//4)))

    X = low_rank_matrix(128, 128, r, rng=rng)
    Xn = add_gaussian_noise(X, snr_db=snr_db, rng=rng)

    # MAR mask
    Y_soft, obs_mask = apply_missingness(Xn, missing_fraction=missing_fraction, rng=rng)

    # SoftImpute with CV λ
    lam = pick_lambda(Y_soft, lam_grid, holdout_fraction=0.1, num_iterations=200, tolerance=1e-5, random_state=13)
    X_soft = softImpute(Y_soft, reg=lam, num_iterations=200, tolerance=1e-5, clip_output=False)
    soft_test = rmse_on_mask(X, X_soft, ~obs_mask)

    # HardImpute with CV k
    best_k, _ = cross_validate_rank(Y_soft, k_grid, holdout_fraction=0.1, num_iterations=200, tolerance=1e-5, random_state=13)
    X_hard = hardImpute(Y_soft, rank=best_k, num_iterations=200, tolerance=1e-5)
    hard_test = rmse_on_mask(X, X_hard, ~obs_mask)

    # Plot comparison
    plt.figure(figsize=(5.5,4))
    plt.bar(["Soft (λ CV)", f"Hard (k={best_k})"], [soft_test, hard_test])
    plt.ylabel("Test RMSE")
    plt.title(f"Soft vs Hard | r={r}, SNR={snr_db} dB, missing={missing_fraction:.0%}")
    plt.tight_layout()
    return dict(best_lambda=float(lam), best_k=int(best_k), soft_test=float(soft_test), hard_test=float(hard_test))

# === Visual comparison: SoftImpute vs HardImpute reconstructions ===
# === Visual comparison on the real bird image ===
from imageio.v2 import imread

def visualize_soft_vs_hard_image(img_path="image.jpg", missing_fraction=0.5,
                                 snr_db=20, lam_grid=None, k_grid=None, rng=42):
    """
    Compare SoftImpute vs HardImpute on a real grayscale image with noise + missing pixels.
    """
    rng = np.random.default_rng(rng)
    if lam_grid is None:
        lam_grid = np.geomspace(0.05, 5.0, 12)
    if k_grid is None:
        k_grid = list(range(5, 50, 5))  # typical useful ranks

    # --- Load and normalize image ---
    img = imread(img_path, pilmode='L').astype(float)
    img = img / 255.0
    print("Loaded image:", img.shape)

    # --- Add Gaussian noise (based on SNR) using Frobenius norm ---
    sig_pow = (np.linalg.norm(img, 'fro') ** 2) / img.size
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_pow), size=img.shape)
    img_noisy = np.clip(img + noise, 0.0, 1.0)

    # --- Random missingness mask ---
    mask = rng.random(img.shape) > missing_fraction
    img_missing = img_noisy.copy()
    img_missing[~mask] = np.nan

    # --- SoftImpute reconstruction ---
    lam_best = pick_lambda(img_missing, lam_grid, holdout_fraction=0.1,
                           num_iterations=200, tolerance=1e-5, random_state=7)
    img_soft = softImpute(img_missing, reg=lam_best, num_iterations=200, tolerance=1e-5, clip_output=False)

    # --- HardImpute reconstruction ---
    best_k, _ = cross_validate_rank(img_missing, k_grid, holdout_fraction=0.1,
                                    num_iterations=200, tolerance=1e-5, random_state=7)
    img_hard = hardImpute(img_missing, rank=best_k, num_iterations=200, tolerance=1e-5)

    # --- Compute test RMSE on missing entries ---
    soft_rmse = rmse_on_mask(img, img_soft, ~mask)
    hard_rmse = rmse_on_mask(img, img_hard, ~mask)

    # --- Plot comparison ---
    plt.figure(figsize=(15,5))
    plt.subplot(1,4,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(img_noisy, cmap='gray')
    plt.title("Noisy + Missing")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(img_soft, cmap='gray')
    plt.title(f"SoftImpute (λ={lam_best:.2f})\nTest RMSE={soft_rmse:.4f}")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(img_hard, cmap='gray')
    plt.title(f"HardImpute (k={best_k})\nTest RMSE={hard_rmse:.4f}")
    plt.axis("off")

    plt.suptitle(f"Soft vs Hard Imputation on Bird Image | Missing={missing_fraction:.0%}, SNR={snr_db} dB")
    plt.tight_layout()
    plt.show()

    return dict(
        lam_best=float(lam_best), k_best=int(best_k),
        soft_rmse=float(soft_rmse), hard_rmse=float(hard_rmse)
    )

# ---------- Non-random missingness (rows/cols gone) ----------
def experiment_block_missingness(r=10, snr_db=20, frac_rows=0.2, frac_cols=0.2, lam_grid=None, rng=7):
    rng = np.random.default_rng(rng)
    if lam_grid is None:
        lam_grid = np.geomspace(0.05, 5.0, 12)

    X = low_rank_matrix(128, 128, r, rng=rng)
    Xn = add_gaussian_noise(X, snr_db=snr_db, rng=rng)

    # MAR baseline (same total missingness as blocks, roughly)
    miss_equiv = frac_rows + frac_cols - frac_rows*frac_cols
    Y_mar, obs_mar = apply_missingness(Xn, missing_fraction=miss_equiv, rng=rng)

    # Non-random: whole rows/cols missing
    Y_blk, obs_blk, rows, cols = hide_blocks(Xn, frac_rows=frac_rows, frac_cols=frac_cols, rng=rng)

    # CV λ and reconstruct
    lam_mar = pick_lambda(Y_mar, lam_grid, holdout_fraction=0.1, num_iterations=200, tolerance=1e-5, random_state=31)
    X_mar = softImpute(Y_mar, reg=lam_mar, num_iterations=200, tolerance=1e-5, clip_output=False)
    mar_test = rmse_on_mask(X, X_mar, ~obs_mar)

    lam_blk = pick_lambda(Y_blk, lam_grid, holdout_fraction=0.1, num_iterations=200, tolerance=1e-5, random_state=32)
    X_blk = softImpute(Y_blk, reg=lam_blk, num_iterations=200, tolerance=1e-5, clip_output=False)
    blk_test = rmse_on_mask(X, X_blk, ~obs_blk)

    # Plot
    plt.figure(figsize=(5.8,4))
    plt.bar(["MAR (random)", "Blocks (rows/cols)"], [mar_test, blk_test])
    plt.ylabel("Test RMSE")
    plt.title(f"Non-random missingness hurts | missing≈{miss_equiv:.0%}")
    plt.tight_layout()
    return dict(mar_test=float(mar_test), block_test=float(blk_test), frac_rows=float(frac_rows), frac_cols=float(frac_cols))


# === Visual comparison + RMSE bar chart for Soft vs Hard on the bird image ===
from imageio.v2 import imread

def visualize_soft_vs_hard_image(img_path="image.jpg", missing_fraction=0.5,
                                 snr_db=20, lam_grid=None, k_grid=None, rng=42,
                                 save_figs=False):
    """
    Compare SoftImpute vs HardImpute on a real grayscale image
    and plot both the reconstructions and an RMSE bar chart.
    """
    rng = np.random.default_rng(rng)
    if lam_grid is None:
        lam_grid = np.geomspace(0.05, 5.0, 12)
    if k_grid is None:
        k_grid = list(range(5, 60, 5))  # ranks to try for HardImpute

    # --- Load and normalize the bird image ---
    img = imread(img_path, pilmode='L').astype(float)
    img = img / 255.0
    print("Loaded image:", img.shape)

    # --- Add Gaussian noise (controlled by SNR) using Frobenius norm ---
    sig_pow = (np.linalg.norm(img, 'fro') ** 2) / img.size
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_pow), size=img.shape)
    img_noisy = np.clip(img + noise, 0.0, 1.0)

    # --- Apply random missingness ---
    mask = rng.random(img.shape) > missing_fraction
    img_missing = img_noisy.copy()
    img_missing[~mask] = np.nan

    # --- SoftImpute reconstruction ---
    lam_best = pick_lambda(img_missing, lam_grid,
                           holdout_fraction=0.1,
                           num_iterations=200, tolerance=1e-5,
                           random_state=7)
    img_soft = softImpute(img_missing, reg=lam_best,
                          num_iterations=200, tolerance=1e-5, clip_output=False)

    # --- HardImpute reconstruction ---
    best_k, _ = cross_validate_rank(img_missing, k_grid,
                                    holdout_fraction=0.1,
                                    num_iterations=200, tolerance=1e-5,
                                    random_state=7)
    img_hard = hardImpute(img_missing, rank=best_k,
                          num_iterations=200, tolerance=1e-5)

    # --- Compute test RMSEs on missing pixels only ---
    soft_rmse = rmse_on_mask(img, img_soft, ~mask)
    hard_rmse = rmse_on_mask(img, img_hard, ~mask)

    # === Plot 1: Reconstructed images ===
    plt.figure(figsize=(14,5))
    plt.subplot(1,4,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(np.where(np.isnan(img_missing), 1.0, img_missing), cmap='gray')
    #plt.imshow(img_noisy, cmap='gray')
    plt.title(f"Noisy + {missing_fraction*100:.0f}% Missing")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(img_soft, cmap='gray')
    plt.title(f"SoftImpute\nλ={lam_best:.2f}\nRMSE={soft_rmse:.4f}")
    plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(img_hard, cmap='gray')
    plt.title(f"HardImpute\nk={best_k}\nRMSE={hard_rmse:.4f}")
    plt.axis("off")

    plt.suptitle(f"Soft vs Hard Imputation on Bird Image | SNR={snr_db} dB, Missing={missing_fraction:.0%}")
    plt.tight_layout()
    if save_figs:
        plt.savefig("bird_soft_vs_hard_images.png", dpi=150)
    plt.show()

    # === Plot 2: RMSE comparison bar chart ===
    plt.figure(figsize=(4.5,4))
    plt.bar(["SoftImpute", "HardImpute"], [soft_rmse, hard_rmse],
            color=["#4f8dd3", "#e07a5f"])
    plt.ylabel("Test RMSE on Missing Pixels")
    plt.title(f"Soft vs Hard Imputation\n(SNR={snr_db} dB, Missing={missing_fraction:.0%})")
    for i, v in enumerate([soft_rmse, hard_rmse]):
        plt.text(i, v + 0.0005, f"{v:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    if save_figs:
        plt.savefig("bird_soft_vs_hard_rmse.png", dpi=150)
    plt.show()

    return dict(lam_best=float(lam_best), k_best=int(best_k),
                soft_rmse=float(soft_rmse), hard_rmse=float(hard_rmse))


'''
# === Run the comparison ===

if __name__ == "__main__":
    _ = visualize_soft_vs_hard_image(
        img_path="image.jpg",  # your bird image
        missing_fraction=0.5,  # half pixels missing
        snr_db=20,             # moderate noise
        save_figs=True         # saves both plots to PNG
    )
'''

# ------------- Example calls -------------
if __name__ == "__main__":
    # 1) Synthetic grid with CV-λ everywhere
    _grid_out = experiment_grid_soft(
        m=96, n=96,
        r_list=(2, 5, 10, 20),
        snr_list=(10, 15, 20, 30),
        miss_list=(0.2, 0.4, 0.6),
    )

    # 2a) SoftImpute vs HardImpute
    #_cmp = compare_soft_vs_hard(r=10, snr_db=20, missing_fraction=0.6)

    # 2b) Non-random missingness (whole rows/cols hidden)
    _nmar = experiment_block_missingness(r=10, snr_db=20, frac_rows=0.2, frac_cols=0.2)

    plt.show()
# -------------------------------
# Run all three experiments
# -------------------------------
if __name__ == "__main__":
    # Comment/uncomment as needed. These produce 3 figures.
    _ = experiment_snr(
        m=128, n=128, r=10,
        snr_list=(5, 10, 15, 20, 30),
        missing_fraction=0.5
    )
    _ = experiment_missingness(
        m=128, n=128, r=10,
        snr_db=20, miss_list=(0.2, 0.4, 0.6, 0.8)
    )
    _ = experiment_true_rank(
        m=128, n=128, r_list=(2, 5, 10, 20, 30),
        snr_db=20, missing_fraction=0.5
    )
    plt.show()
