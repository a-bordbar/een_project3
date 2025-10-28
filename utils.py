# ADD near the top of utils.py
def nrmse(true, pred):
    true = np.asarray(true, float); pred = np.asarray(pred, float)
    num = np.linalg.norm(true - pred, 'fro')
    den = np.linalg.norm(true, 'fro') + 1e-12
    return num / den

def psnr(true, pred):
    true = np.asarray(true, float); pred = np.asarray(pred, float)
    mse = np.mean((true - pred) ** 2)
    return 10.0 * np.log10(1.0 / (mse + 1e-12))  # assumes data in [0,1]

def rmse(true, pred):
    """Compute Root Mean Squared Error, ignoring NaNs in both arrays."""
    mask = ~np.isnan(true) & ~np.isnan(pred)
    if not np.any(mask):
        return np.nan
    return np.sqrt(np.mean((true[mask] - pred[mask])**2))


import numpy as np
from numpy.linalg import svd

# CHANGED signature and end of softImpute
def softImpute(Y, reg, num_iterations=200, tolerance=1e-5, clip_output=False):
    """
    SoftImpute (simple NumPy version).
    Y: observed matrix with missing entries encoded as np.nan
    reg: regularization lambda (non-negative)
    """
    Y = np.asarray(Y, dtype=float)
    mask = ~np.isnan(Y)        # True where observed
    P_Y = np.where(mask, Y, 0) # P_Omega(Y)

    X_old = np.zeros_like(Y)   # initial estimate (all zeros)
    eps = 1e-12

    for it in range(1, num_iterations + 1):
        P = P_Y + np.where(mask, 0.0, X_old)  # same shape as Y

        U, s, Vt = svd(P, full_matrices=False)

        # 3) Soft-threshold singular values
        s_thresh = np.maximum(0, s - reg)

        nonzero = s_thresh > 0
        if not np.any(nonzero):
            X_new = np.zeros_like(P)
        else:
            U_r = U[:, nonzero]
            Vt_r = Vt[nonzero, :]
            S_r = np.diag(s_thresh[nonzero])
            X_new = U_r @ S_r @ Vt_r

        num = np.linalg.norm(X_new - X_old, ord='fro')
        den = np.linalg.norm(X_old, ord='fro') + eps  #avoid dividing by zero
        rel = num / den

        # stop if relative change is small
        if rel < tolerance:
            X_old = X_new
            break

        # else update and continue
        X_old = X_new

    # Final imputation: observed entries from Y, missing replaced by X_old
    X_completed = np.nan_to_num(P_Y + np.where(mask, 0.0, X_old), nan=0.0)

    if clip_output:
        # only clip for display, not for metrics/experiments
        X_completed = np.clip(X_completed, 0.0, 1.0)
    return X_completed


def cross_validate_lambda(Y, lambdas, holdout_fraction=0.1, num_iterations=100, tolerance=1e-5, random_state=0):
    """
    Cross-validation to select lambda for SoftImpute.

    Y : np.ndarray with np.nan for missing entries
    lambdas : list or array of candidate lambda values
    holdout_fraction : fraction of observed entries to hold out for validation
    """
    rng = np.random.default_rng(random_state)

    mask_obs = ~np.isnan(Y)
    obs_indices = np.argwhere(mask_obs)
    n_obs = len(obs_indices)

    # Select random 10% of observed entries to hold out
    n_holdout = int(holdout_fraction * n_obs)
    holdout_idx = rng.choice(n_obs, n_holdout, replace=False)

    Y_train = Y.copy()
    # Hide holdout entries
    for i, j in obs_indices[holdout_idx]:
        Y_train[i, j] = np.nan

    # True values of held-out entries
    Y_true_holdout = np.full_like(Y, np.nan)
    for i, j in obs_indices[holdout_idx]:
        Y_true_holdout[i, j] = Y[i, j]

    results = {}

    # Evaluate each lambda
    for lam in lambdas:
        Y_pred = softImpute(Y_train, reg=lam, num_iterations=num_iterations, tolerance=tolerance)
        score = rmse(Y_true_holdout, Y_pred)
        results[lam] = score
        print(f"λ={lam:.4f}, RMSE={score:.4f}")

    # Pick lambda with smallest RMSE
    best_lambda = min(results, key=results.get)
    print(f"\n Best λ = {best_lambda} (RMSE={results[best_lambda]:.4f})")
    return best_lambda, results


# REPLACED add_noise_and_missing 
def add_noise_and_missing(img, missing_fraction=0.3, snr_db=20, random_state=None, clip_for_display=True):
    rng = np.random.default_rng(random_state)

    img = img.astype(float)
    if img.max() > 1.0:
        img = img / 255.0

    # Random missing
    mask = rng.random(img.shape) > missing_fraction
    img_missing = img.copy()
    img_missing[~mask] = np.nan

    # Gaussian noise based on Frobenius norm power per entry (||X||_F^2 / (mn))
    signal_power = (np.linalg.norm(img, 'fro') ** 2) / img.size

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=img.shape)

    img_noisy = img_missing.copy()
    img_noisy[mask] = img_noisy[mask] + noise[mask]

    # IMPORTANT: no clipping for numeric experiments; allow caller to clip only for display
    if clip_for_display:
        img_noisy = np.clip(img_noisy, 0.0, 1.0)

    return img_noisy, mask