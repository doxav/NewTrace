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
    One-Euro filter (frequency-adaptive exponential smoothing) with
    slope-deadband hysteresis and multi-pass output reversal suppression.

    Paradigm change vs. the previous Holt/smoother stack:
      * One-Euro adapts the smoothing cutoff to the signal's speed:
        low speed -> very low cutoff (heavy smoothing -> few slope changes
        and few false reversals); high speed -> high cutoff (light
        smoothing -> low lag and good tracking). This resolves the
        smoothing-vs-lag trade-off adaptively instead of with a fixed
        window average.
      * A robust noise deadband gates the output slope direction, so the
        trend sign only flips on genuinely large slopes.
      * Multi-pass output reversal hysteresis flattens residual sub-noise
        sign flips (each pass recomputes the robust diff scale, so
        medium-sized flips that survive early passes are removed later).

    Output length = len(x) - window_size + 1.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    W = window_size
    n = len(x)

    # --- Robust noise-scale estimate from first differences ---
    dx_all = np.diff(x, prepend=x[0])
    mad = np.median(np.abs(dx_all - np.median(dx_all)))
    noise_scale = max(mad, 1e-6)
    deadband = max(1.6 * noise_scale, 1e-6)

    # --- One-Euro filter parameters ---
    fc_min = 0.045   # cutoff at low speed: heavy smoothing (few reversals)
    fc_max = 0.85    # cutoff at high speed: light smoothing (low lag)
    beta_oe = 0.55   # speed -> cutoff slope
    d_alpha = 0.012  # smoothing of the derivative estimate (stable speed)

    def alpha_from_cutoff(fc):
        tau = 1.0 / (2.0 * np.pi * max(fc, 1e-6))
        te = 1.0
        return 1.0 / (1.0 + tau / te)

    # --- Causal One-Euro recursion over the full signal ---
    xhat = x[0]
    dx_hat = 0.0
    raw = np.empty(n - W + 1)
    for i in range(n):
        dx_raw = x[i] - xhat  # innovation = speed estimate
        # Smooth the derivative estimate
        dx_hat = d_alpha * dx_raw + (1.0 - d_alpha) * dx_hat
        # Adaptive cutoff: low speed -> low cutoff, high speed -> high cutoff
        fc = fc_min + beta_oe * abs(dx_hat)
        if fc > fc_max:
            fc = fc_max
        a = alpha_from_cutoff(fc)
        prev_xhat = xhat
        xhat = a * x[i] + (1.0 - a) * xhat
        if i >= W - 1:
            # Lead compensation: extrapolate with the smoothed speed to
            # cancel residual lag (scaled by speed confidence).
            speed_conf = min(1.0, abs(dx_hat) / (2.0 * noise_scale))
            lead = 2.5 * speed_conf
            raw[i - W + 1] = xhat + lead * dx_hat

    # --- Slope-deadband gating: zero out sub-noise output slopes ---
    dy = np.diff(raw, prepend=raw[0])
    small = np.abs(dy) < 0.5 * deadband
    if small.any():
        # Replace gated samples with a causal hold of the last ungated value
        y = raw.copy()
        last_val = raw[0]
        for j in range(len(y)):
            if small[j]:
                y[j] = last_val
            else:
                last_val = y[j]
        raw = y

    # --- Multi-pass output reversal hysteresis ---
    y = raw
    if len(y) > 2:
        for eps_mult in (2.2, 1.6, 1.1, 0.75, 0.5):
            base = np.median(np.abs(np.diff(y)))
            if base <= 0:
                continue
            eps = max(eps_mult * base, 0.5 * eps_mult * noise_scale)
            prev_sign = 0.0
            for j in range(1, len(y)):
                dj = y[j] - y[j - 1]
                if dj == 0.0:
                    continue
                s = np.sign(dj)
                if prev_sign != 0.0 and s != prev_sign and abs(dj) < eps:
                    y[j] = y[j - 1]  # hold: suppress noise-induced flip
                else:
                    prev_sign = s

    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


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
