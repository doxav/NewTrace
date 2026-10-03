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
    Robust weighted local-linear regression smoother (least-squares family).

    Fundamentally different mechanism from exponential smoothing: for each
    output index, a weighted linear model is fit to the trailing window by
    iteratively-reweighted least squares (IRLS with Huber weights).

    - Exponential time-decay weights emphasize recent samples -> low lag
      without any forward extrapolation (the estimate is evaluated AT the
      window end, i.e., at the current time, so phase delay ~ 0).
    - Huber reweighting downweights outliers/spikes -> few false reversals,
      robust to non-Gaussian noise.
    - Linear (not constant) local model tracks genuine trends and handles
      step changes by re-fitting each window (no ringing, no drift).
    - A short causal median-of-3 postfilter removes residual single-sample
      slope spikes.

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

    # Exponential decay weights over the window: most recent sample weight 1.
    j = np.arange(W, dtype=float)
    half_life = max(W * 0.45, 2.0)
    w0 = 0.5 ** ((W - 1 - j) / half_life)

    # Time design matrix for weighted linear regression: t in [0, W-1],
    # evaluate at t = W-1 (current time).
    t = j
    sw = w0.sum()
    tw = (w0 * t).sum()
    t2w = (w0 * t * t).sum()
    denom = sw * t2w - tw * tw
    if abs(denom) < 1e-12:
        denom = 1e-12

    out = np.empty(n - W + 1)
    c_huber = 1.8

    for i in range(n - W + 1):
        seg = x[i : i + W]
        w = w0.copy()
        val = seg[-1]
        # Two IRLS passes: initial fit, then Huber-reweighted refit.
        for _ in range(2):
            sw_ = w.sum()
            tw_ = (w * t).sum()
            t2w_ = (w * t * t).sum()
            d_ = sw_ * t2w_ - tw_ * tw_
            if abs(d_) < 1e-12:
                break
            sxw = (w * seg).sum()
            txw = (w * t * seg).sum()
            a = (sxw * t2w_ - txw * tw_) / d_    # intercept at t=0
            b = (sw_ * txw - tw_ * sxw) / d_     # slope
            val = a + b * (W - 1)
            # Huber reweighting based on robust residual scale
            r = seg - (a + b * t)
            mad = 1.4826 * np.median(np.abs(r - np.median(r)))
            scale = max(mad, 1e-9)
            w = w0 * np.minimum(1.0, c_huber * scale / np.maximum(np.abs(r), 1e-9))
        if not np.isfinite(val):
            val = out[i - 1] if i > 0 else 0.0
        out[i] = val

    # Causal median-of-3 postfilter: kills isolated one-sample slope spikes
    # (false reversals) without introducing phase shift.
    y = out.copy()
    if len(y) >= 3:
        y[2:] = np.median(np.vstack([out[:-2], out[1:-1], out[2:]]), axis=0)

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
