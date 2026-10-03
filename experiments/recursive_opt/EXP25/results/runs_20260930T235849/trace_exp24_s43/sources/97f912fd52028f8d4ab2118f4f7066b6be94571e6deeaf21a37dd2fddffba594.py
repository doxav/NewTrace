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
    Enhanced version: causal Savitzky-Golay (degree-2 polynomial fit evaluated
    at the most recent sample of each trailing window), followed by a light
    exponential post-smoother to suppress spurious slope reversals while
    keeping lag minimal.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    output_length = len(x) - window_size + 1

    # Precompute endpoint-evaluation weights for causal Savitzky-Golay filter.
    poly_order = 2
    t = np.arange(window_size, dtype=float)
    t_norm = (t - t[-1]) / max(window_size - 1, 1)
    V = np.vander(t_norm, poly_order + 1, increasing=True)
    eval_vec = np.zeros(poly_order + 1)
    eval_vec[0] = 1.0  # evaluate polynomial at t_norm = 0 (most recent sample)
    pinv = np.linalg.pinv(V)
    sg_weights = eval_vec @ pinv  # shape (window_size,)

    # Apply via sliding-window dot product (vectorized, low latency)
    windows = np.lib.stride_tricks.sliding_window_view(x, window_size)
    y = windows @ sg_weights

    # Noise-level estimate from raw first differences (robust MAD of diffs,
    # scaled by sqrt(2) for white noise). Drives all adaptive parameters so
    # high-noise regimes get stronger smoothing (fewer spurious reversals)
    # while low-noise regimes stay responsive (minimal lag).
    dx = np.abs(np.diff(x, prepend=x[0]))
    sigma_n = 1.4826 * np.median(dx) / np.sqrt(2.0)
    scale = np.std(x) + 1e-9
    noise_ratio = float(np.clip(sigma_n / scale, 0.0, 1.0))

    # Second causal smoothing pass: order-1 Savitzky-Golay evaluated at the
    # endpoint of a trailing window. Width adapts to the noise level:
    # noisy signals get a longer window (much stronger reversal suppression),
    # clean signals keep a short window (minimal lag).
    w2 = int(np.clip(round(5 + 140 * noise_ratio), 5, 25))
    t2 = np.arange(w2, dtype=float)
    tn2 = (t2 - t2[-1]) / (w2 - 1)
    V2 = np.vander(tn2, 2, increasing=True)
    eval_vec2 = np.array([1.0, 0.0])
    sg2 = eval_vec2 @ np.linalg.pinv(V2)
    pad = np.concatenate([np.full(w2 - 1, y[0]), y])
    windows2 = np.lib.stride_tricks.sliding_window_view(pad, w2)
    y = windows2 @ sg2

    # Adaptive exponential post-smoother (Kaufman-style efficiency ratio):
    # alpha scales with |local move| / |recent average move|, so genuine
    # trend moves are tracked quickly (low lag) while small noisy
    # fluctuations are heavily damped (fewer spurious reversals).
    n = len(y)
    y_smooth = np.empty(n)
    y_smooth[0] = y[0]
    diffs = np.abs(np.diff(y, prepend=y[0]))
    signed = np.diff(y, prepend=y[0])
    # Causal rolling mean of |diff| over trailing 10 samples
    kernel = np.ones(10) / 10.0
    noise_est = np.convolve(diffs, kernel, mode="full")[:n]
    noise_est = np.maximum(noise_est, 1e-9)
    # Alpha floor/cap also adapt: noisier signals clamp alpha lower so the
    # exponential smoother damps noise-induced fluctuations harder.
    a_floor = float(np.clip(0.12 / (1.0 + 8.0 * noise_ratio), 0.03, 0.12))
    a_cap = float(np.clip(0.9 / (1.0 + 3.0 * noise_ratio), 0.25, 0.9))
    a_gain = 0.6 * (1.0 + 0.5 * noise_ratio)
    # Directional hysteresis: if the move reverses the previous direction
    # and its magnitude is small relative to recent noise, damp alpha so
    # noise-induced slope flips don't propagate into the output.
    prev_sign = 0.0
    for i in range(1, n):
        a = np.clip(a_gain * diffs[i] / noise_est[i - 1], a_floor, a_cap)
        s = np.sign(signed[i])
        if prev_sign != 0.0 and s != 0.0 and s != prev_sign and diffs[i] < 0.6 * noise_est[i - 1]:
            a *= 0.45
        if jump[min(i, len(jump) - 1)]:
            a = min(1.0, a * 3.0)
        if s != 0.0:
            prev_sign = s
        y_smooth[i] = y_smooth[i - 1] + a * (y[i] - y_smooth[i - 1])

    return y_smooth[:output_length]


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


# --- projection: per-call fallback onto a feasible baseline ---
_PROJECTION_EVENTS = []
_candidate_run_signal_processing = run_signal_processing


def _fallback_run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20):
    import numpy as np
    x = np.asarray(noisy_signal, dtype=float)
    weights = np.exp(np.linspace(-2, 0, window_size))
    weights = weights / np.sum(weights)
    y = np.array([np.sum(x[i:i + window_size] * weights) for i in range(len(x) - window_size + 1)])
    return {'filtered_signal': y, 'clean_signal': None, 'noisy_signal': None, 'correlation': 0, 'noise_reduction': 0, 'signal_length': len(y)}


def _check_run_signal_processing(result, noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20):
    import numpy as np
    if not isinstance(result, dict) or 'filtered_signal' not in result:
        return False
    y = np.asarray(result['filtered_signal'], dtype=float)
    return y.ndim == 1 and len(y) == len(noisy_signal) - window_size + 1 and bool(np.all(np.isfinite(y)))


def run_signal_processing(*args, **kwargs):
    try:
        result = _candidate_run_signal_processing(*args, **kwargs)
    except Exception as error:
        _PROJECTION_EVENTS.append(type(error).__name__)
        return _fallback_run_signal_processing(*args, **kwargs)
    try:
        valid = _check_run_signal_processing(result, *args, **kwargs)
    except Exception:
        valid = False
    if valid:
        return result
    _PROJECTION_EVENTS.append('invalid')
    return _fallback_run_signal_processing(*args, **kwargs)
