# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""

import numpy as np


def _savgol_end_weights(window_size, polyorder=2):
    """Precompute Savitzky-Golay weights that evaluate the least-squares
    polynomial fit at the LAST sample of the window (minimal lag)."""
    t = np.arange(window_size, dtype=float) - (window_size - 1)
    V = np.vander(t, polyorder + 1, increasing=True)  # Vandermonde
    # Weights = last row of (V^T V)^-1 V^T  -> evaluate fit at t = 0 (window end)
    return np.linalg.pinv(V)[0]


def enhanced_filter_with_trend_preservation(x, window_size=22, polyorder=2,
                                            median_k=7, ema_alpha=0.5,
                                            deadband_frac=0.25,
                                            reversal_scale=1.2):
    """Four-stage causal filter with asymmetric slope hysteresis:
    running median -> end-point Savitzky-Golay (order 2, evaluated at
    window end for minimal lag) -> light EMA -> hysteresis where
    sub-deadband counter-moves are flattened and a reversal is confirmed
    only when the counter-move exceeds reversal_scale * deadband.
    Suppresses noise-induced reversals while preserving genuine turning
    points with low lag. O(n), vectorized."""
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Stage 1: causal running median (only past samples used)
    if median_k > 1:
        from numpy.lib.stride_tricks import sliding_window_view
        pad = median_k - 1
        xp = np.concatenate([np.full(pad, x[0]), x])
        x = np.median(sliding_window_view(xp, median_k), axis=1)

    # Stage 2: causal Savitzky-Golay evaluated at window end
    w = _savgol_end_weights(window_size, polyorder)
    windows = sliding_window_view(x, window_size)
    y = windows @ w

    # Stage 3: causal EMA post-smoothing (minimal lag)
    if 0 < ema_alpha < 1:
        out = np.empty_like(y)
        acc = y[0]
        for i in range(len(y)):
            acc = ema_alpha * y[i] + (1 - ema_alpha) * acc
            out[i] = acc
        y = out

    # Stage 4: asymmetric adaptive slope hysteresis
    if len(y) > 2:
        d = np.diff(y)
        db = deadband_frac * (np.std(d) + 1e-12)
        rev_db = reversal_scale * db
        sign = 1.0 if d[0] >= 0 else -1.0
        for i in range(len(d)):
            if sign * d[i] < -rev_db:       # confirmed reversal (strong counter-move)
                sign = 1.0 if d[i] > 0 else -1.0
            elif sign * d[i] < 0:
                d[i] = 0.0                  # suppress sub-threshold counter-move
        y = np.concatenate(([y[0]], y[0] + np.cumsum(d)))
    return y


def adaptive_filter(x, window_size=22):
    """Baseline: same four-stage causal smoother (kept for API compatibility)."""
    return enhanced_filter_with_trend_preservation(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Apply the causal Savitzky-Golay filter to the input signal."""
    return enhanced_filter_with_trend_preservation(input_signal, window_size)


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
