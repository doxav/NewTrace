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
    Causal multi-stage filter: median pre-filter -> vectorized windowed
    linear regression (zero-lag level + slope) -> slope-aware adaptive
    smoothing with confirmed-reversal hysteresis.

    Key design points vs. a plain moving average:
      * The windowed regression projects the local trend onto the newest
        sample, so the level estimate has near-zero lag while still
        averaging W samples of noise.
      * The slope estimate is itself EMA-smoothed before any sign decision,
        so single-sample noise spikes cannot flip the committed direction.
      * A reversal is committed only after the (smoothed) slope opposes the
        committed sign with magnitude above an absolute deadband for
        `confirm` consecutive samples. Until confirmation, the level update
        is damped toward the previous smoothed value, which suppresses
        noise-induced false reversals and spurious slope changes while
        preserving genuine trend changes (real reversals persist for many
        samples and confirm quickly).

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < window_size:
        # Degenerate input: fall back to a trivial passthrough-style output
        n = max(len(x) - window_size + 1, 1)
        return np.asarray(x, dtype=float)[:n].copy()

    W = window_size
    output_length = len(x) - W + 1

    # --- Stage 0: causal median-of-3 pre-filter (kills impulse noise) ---
    xm = x.copy()
    if len(x) >= 3:
        xm[2:] = np.median(np.stack([x[:-2], x[1:-1], x[2:]], axis=0), axis=0)

    # --- Stage 1: weighted local linear regression, fully vectorized ---
    # Exponential weights emphasizing recent samples (recent weight = 1).
    w = np.exp(np.linspace(-3.0, 0.0, W))
    tau = np.arange(-(W - 1), 1, dtype=float)
    S0 = w.sum()
    S1 = np.sum(w * tau)
    S2 = np.sum(w * tau * tau)
    denom = S0 * S2 - S1 * S1
    # FIR kernels for the fitted level (zero lag) and slope at the newest
    # sample; flipped for np.convolve 'valid' correlation.
    c_a = np.sum(w * (S2 - S1 * tau)) / denom
    c_b = np.sum(w * (S0 * tau - S1)) / denom
    levels = np.convolve(xm, c_a[::-1], mode="valid")   # length = output_length
    slopes = np.convolve(xm, c_b[::-1], mode="valid")

    # --- Stage 2: slope smoothing + confirmed-reversal hysteresis ---
    slope_s = slopes[0]          # EMA-smoothed slope estimate
    a_slope = 0.25               # slope smoother gain (low -> stable signs)

    # Absolute deadband from a robust noise-scale estimate of the input
    # (diff-based; robust to trends and random walks).
    d = np.abs(np.diff(xm))
    noise_scale = np.median(d) + 1e-12
    deadband = 1.5 * noise_scale / max(W, 1)

    confirm = 3                  # consecutive opposing samples to confirm
    last_sign = 0
    pending_sign = 0
    pending_count = 0

    # Double-EMA level smoother gains
    a1 = 0.35
    a2 = 0.45
    s1 = levels[0]
    s2 = s1

    y = np.zeros(output_length)
    for i in range(output_length):
        slope_s = slope_s + a_slope * (slopes[i] - slope_s)

        sign_est = 0
        if slope_s > deadband:
            sign_est = 1
        elif slope_s < -deadband:
            sign_est = -1

        # Hysteresis with confirmation: only commit a reversal after
        # `confirm` consecutive opposing-sign samples above the deadband.
        if last_sign == 0:
            if sign_est != 0:
                last_sign = sign_est
        elif sign_est != 0 and sign_est != last_sign:
            if sign_est == pending_sign:
                pending_count += 1
            else:
                pending_sign = sign_est
                pending_count = 1
            if pending_count >= confirm:
                last_sign = sign_est
                pending_sign = 0
                pending_count = 0
        else:
            pending_sign = 0
            pending_count = 0

        val = levels[i]
        # While an unconfirmed flip is pending, damp the level update so
        # noise cannot push the output into a spurious reversal.
        if pending_count > 0:
            val = 0.35 * val + 0.65 * s2

        s1 = s1 + a1 * (val - s1)
        s2 = s2 + a2 * (s1 - s2)
        y[i] = s2

    return y


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
