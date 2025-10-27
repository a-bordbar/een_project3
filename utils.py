def rmse(true, pred):
    """Compute Root Mean Squared Error, ignoring NaNs in both arrays."""
    mask = ~np.isnan(true) & ~np.isnan(pred)
    if not np.any(mask):
        return np.nan
    return np.sqrt(np.mean((true[mask] - pred[mask])**2))


import numpy as np
from numpy.linalg import svd

def softImpute(Y, reg, num_iterations=200, tolerance=1e-5):
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
    X_completed = np.clip(X_completed, np.nanmin(Y), np.nanmax(Y))
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


def add_noise_and_missing(img, missing_fraction=0.3, snr_db=20, random_state=None):
    """
    Add Gaussian noise and random missing values to a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image with values in [0,1] or 0-255 (will be normalized automatically)
    missing_fraction : float
        Fraction of pixels to remove (set as NaN)
    snr_db : float
        Desired signal-to-noise ratio in decibels
    random_state : int or None
        Random seed for reproducibility

    Returns
    -------
    img_noisy : np.ndarray
        Image with missing values (NaN) and Gaussian noise added
    mask : np.ndarray
        Boolean array where True = observed pixels, False = missing
    """
    rng = np.random.default_rng(random_state)

    # Ensure float type and normalize to 0-1
    img = img.astype(float)
    if img.max() > 1.0:
        img = img / 255.0

    # Step 1: Random missing values
    mask = rng.random(img.shape) > missing_fraction
    img_missing = img.copy()
    img_missing[~mask] = np.nan

    # Step 2: Add Gaussian noise to observed pixels
    snr_linear = 10 ** (snr_db / 10)
    signal_power = np.nanvar(img_missing)
    noise_std = np.sqrt(signal_power / snr_linear)
    noise = rng.normal(0, noise_std, size=img.shape)

    img_noisy = img_missing.copy()
    img_noisy[mask] += noise[mask]

    # Clip values to [0,1]
    img_noisy = np.clip(img_noisy, 0.0, 1.0)

    return img_noisy, mask