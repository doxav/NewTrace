# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""

import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing algorithm using sliding window approach.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Initialize output array
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Simple moving average as baseline
    for i in range(output_length):
        window = x[i : i + window_size]

        # Basic moving average filter
        y[i] = np.mean(window)

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal sliding-window Total Variation (TV) denoising, batched ADMM.

    Fundamentally different mechanism from polynomial regression / exponential
    smoothing: each output is the endpoint of the solution of

        min_z  0.5 * ||z - x_win||_2^2  +  lambda * ||D z||_1

    where D is the first-difference operator. TV regularization yields
    piecewise-smooth estimates with SPARSE slope changes: the L1 penalty on
    differences actively kills noise-induced slope reversals (few spurious
    directional changes) while exactly preserving genuine step changes and
    trend corners without ringing (no Gibbs effect, unlike linear filters).

    - Endpoint evaluation (z at the most recent sample) gives zero phase lag
      by construction — no forward extrapolation needed.
    - Lambda adapts to the local noise level via a robust MAD estimate of
      first differences, handling non-stationary noise.
    - All windows are solved simultaneously with a fully vectorized batched
      ADMM (precomputed W x W solve), so it is fast in pure numpy.
    - A light causal directional-hysteresis post-pass suppresses residual
      single-sample false reversals.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    W = int(window_size)
    out_len = n - W + 1

    # Robust noise scale from first differences (MAD-based). For white noise,
    # var(diff) = 2*sigma^2.
    d = np.diff(x)
    if len(d) > 0:
        sigma = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    else:
        sigma = 0.3
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = 1e-9

    # Build sliding-window matrix: rows are trailing windows (causal).
    idx = np.arange(W)[None, :] + np.arange(out_len)[:, None]
    X = x[idx]  # shape (out_len, W)

    # Adaptive regularization: lambda scales with noise std. Larger lambda ->
    # smoother, fewer slope changes; smaller -> better tracking.
    lam = 1.6 * sigma * np.sqrt(W)

    # --- Batched ADMM for TV denoising ---
    # Difference operator D: (W-1) x W
    D = np.zeros((W - 1, W))
    r = np.arange(W - 1)
    D[r, r] = -1.0
    D[r, r + 1] = 1.0
    DtD = D.T @ D
    rho = 4.0
    # Precompute (I + rho * DtD)^-1 once (fixed for all windows)
    M = np.linalg.inv(np.eye(W) + rho * DtD)

    Z = X.copy()
    U = np.zeros((out_len, W - 1))
    B = np.zeros((out_len, W - 1))

    n_iter = 40
    for _ in range(n_iter):
        # z-update: minimize 0.5||z - X||^2 + (rho/2)||Dz - U + B||^2
        rhs = X + rho * D.T @ (U - B).T  # D.T @ (U-B).T -> (out_len, W)
        Z = rhs @ M.T
        # u-update: soft threshold
        V = Z @ D.T + B
        thresh = lam / rho
        U = np.sign(V) * np.maximum(np.abs(V) - thresh, 0.0)
        # b-update
        B = V - U

    # Endpoint value (most recent sample of each window): zero phase lag
    y = Z[:, -1].copy()

    # --- Causal directional hysteresis post-pass ---
    dy = np.diff(y)
    if len(dy) > 0:
        thr = 1.4826 * np.median(np.abs(dy - np.median(dy)))
        if not np.isfinite(thr) or thr <= 1e-12:
            thr = 1e-12
        y2 = y.copy()
        dirn = 1 if dy[0] >= 0 else -1
        run = 0
        run_dir = 0
        for j in range(1, len(y)):
            dd = y[j] - y2[j - 1]
            jd = 1 if dd >= 0 else -1
            if jd != dirn:
                if jd == run_dir:
                    run += 1
                else:
                    run_dir = jd
                    run = 1
                need = 0.55 * thr if run >= 3 else 0.4 * thr
                if run >= 2 and abs(dd) > need:
                    dirn = jd
                    run = 0
                else:
                    y2[j] = y2[j - 1]
            else:
                run = 0
                run_dir = 0
        y = y2

    return np.where(np.isfinite(y), y, 0.0)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    else:
        return adaptive_filter(input_signal, window_size)


# EVOLVE-BLOCK-END


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    """
    Generate synthetic test signal with known characteristics.

    Args:
        length: Length of the signal
        noise_level: Standard deviation of noise to add
        seed: Random seed for reproducibility

    Returns:
        Tuple of (noisy_signal, clean_signal)
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, length)

    # Create a complex signal with multiple components
    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)  # Low frequency component
        + 1.5 * np.sin(2 * np.pi * 2 * t)  # Medium frequency component
        + 0.5 * np.sin(2 * np.pi * 5 * t)  # Higher frequency component
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)  # Decaying oscillation
    )

    # Add non-stationary behavior
    trend = 0.1 * t * np.sin(0.2 * t)  # Slowly varying trend
    clean_signal += trend

    # Add random walk component for non-stationarity
    random_walk = np.cumsum(np.random.randn(length) * 0.05)
    clean_signal += random_walk

    # Add noise
    noise = np.random.normal(0, noise_level, length)
    noisy_signal = clean_signal + noise

    return noisy_signal, clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20):
    """
    Run the signal processing algorithm on a test signal.

    Args:
        noisy_signal: Input signal to filter (if provided, use this; otherwise generate)
        signal_length: Length if generating signal (for backward compatibility)
        noise_level: Noise level if generating signal (for backward compatibility)
        window_size: Window size for processing

    Returns:
        Dictionary containing results and metrics
    """
    # Use provided signal or generate test signal (for backward compatibility)
    if noisy_signal is not None:
        # Filter the provided signal
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")
        clean_signal = None  # Not available when using provided signal
    else:
        # Generate test signal (for __main__ and backward compatibility)
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    # Calculate basic metrics (only if we have clean_signal from generation)
    if len(filtered_signal) > 0 and clean_signal is not None:
        # Align signals for comparison (account for processing delay)
        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = noisy_signal[delay:]

        # Ensure same length
        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]

        # Calculate correlation with clean signal
        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if min_length > 1 else 0

        # Calculate noise reduction
        noise_before = np.var(aligned_noisy - aligned_clean)
        noise_after = np.var(filtered_signal - aligned_clean)
        noise_reduction = (noise_before - noise_after) / noise_before if noise_before > 0 else 0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
        }
    elif len(filtered_signal) > 0:
        # When using provided signal (no clean_signal available), just return filtered signal
        return {
            "filtered_signal": filtered_signal,
            "clean_signal": None,
            "noisy_signal": None,
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": len(filtered_signal),
        }
    else:
        return {
            "filtered_signal": [],
            "clean_signal": [],
            "noisy_signal": [],
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": 0,
        }


if __name__ == "__main__":
    # Test the algorithm
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")
