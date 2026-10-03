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
    Adaptive-order causal local polynomial regression (endpoint Savitzky-Golay).

    Fundamentally different mechanism from exponential smoothing: for each
    output index, a weighted least-squares polynomial is fit to the trailing
    window and evaluated AT THE WINDOW END, so the estimate has zero phase
    lag by construction (no forward extrapolation needed).

    - Recency-weighted (exponential kernel) LS fit: recent samples dominate,
      giving low lag error while the LS averaging suppresses noise.
    - Adaptive polynomial order: degree 2 is accepted only when it reduces
      the fit residual meaningfully vs degree 1; on near-linear/flat segments
      the degree-1 fit is used, which strongly reduces spurious slope
      reversals (slope-change penalty and false reversals).
    - One Huber IRLS reweighting pass makes the fit robust to outliers/spikes
      (step changes and impulse noise) without ringing.
    - Trailing median-of-3 prefilter kills single-sample spikes causally.

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

    # Causal trailing median-of-3 prefilter: xp[i] = median(x[i-2], x[i-1], x[i])
    xp = x.copy()
    if n >= 3:
        xp[2:] = np.median(np.vstack([x[:-2], x[1:-1], x[2:]]), axis=0)

    W = int(window_size)
    # Recency weights: most recent sample (last in window) has weight 1
    idx = np.arange(W)
    lam = 0.86
    w = lam ** (W - 1 - idx)
    sw = np.sqrt(w)

    # Normalized time basis centered at window end (t = 0 at most recent sample)
    t = (idx - (W - 1)).astype(float)
    t /= max(W - 1, 1)

    out_len = n - W + 1
    y = np.empty(out_len)

    # Precompute design matrices
    A1 = np.vstack([np.ones(W), t]).T
    A2 = np.vstack([np.ones(W), t, t * t]).T
    swA1 = A1 * sw[:, None]
    swA2 = A2 * sw[:, None]
    # Pseudo-inverse via normal equations (small fixed systems)
    N1 = swA1.T @ swA1
    N2 = swA2.T @ swA2
    inv1 = np.linalg.pinv(N1)
    inv2 = np.linalg.pinv(N2)

    for k in range(out_len):
        seg = xp[k : k + W]

        # --- Degree-1 weighted fit ---
        b1 = swA1.T @ (seg * sw)
        c1 = inv1 @ b1
        r1 = seg - A1 @ c1
        s1 = float(w @ (r1 * r1))

        # --- Degree-2 weighted fit ---
        b2 = swA2.T @ (seg * sw)
        c2 = inv2 @ b2
        r2 = seg - A2 @ c2
        s2 = float(w @ (r2 * r2))

        # --- Adaptive order selection ---
        # Accept curvature only if it clearly improves the fit
        if s1 > 1e-15 and (s1 - s2) / s1 > 0.12:
            coef, A = c2, A2
        else:
            coef, A = c1, A1

        # --- One Huber IRLS pass for robustness to outliers/steps ---
        resid = seg - A @ coef
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if scale > 1e-12:
            u = np.abs(resid) / (1.345 * scale)
            wr = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
            wr = w * wr
            swr = np.sqrt(wr)
            Ar = A * swr[:, None]
            br = Ar.T @ (seg * swr)
            coef = np.linalg.pinv(Ar.T @ Ar) @ br

        # Evaluate polynomial at window end (t = 0): value = coef[0]
        val = coef[0]
        if not np.isfinite(val):
            val = y[k - 1] if k > 0 else seg[-1]
        y[k] = val

    # Light causal 3-tap binomial smoothing to further suppress slope reversals;
    # endpoint evaluation already has zero lag so this adds negligible delay.
    if out_len >= 3:
        ys = y.copy()
        ys[2:] = 0.6 * y[2:] + 0.3 * y[1:-1] + 0.1 * y[:-2]
        y = ys

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
