# EVOLVE-BLOCK-START
"""Zero-lag adaptive filter: median-prefiltered triple-EMA (TEMA) cascade.

A 7-point running median removes impulse noise (the main cause of spurious
slope changes and false reversals) with zero phase delay, then
TEMA = 3*E1 - 3*E2 + E3 cancels first- and second-order EMA lag, a
zero-phase median7 on the TEMA output suppresses residual noise, and a
directional hysteresis stage rejects sub-threshold reversals.
"""

import numpy as np


def _ema(x, alpha):
    """Vectorized exponential moving average via cumulative products."""
    a = np.asarray(x, dtype=float)
    out = np.empty_like(a)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def _median7(x):
    """Zero-phase 7-point running median (edge-clamped) to kill impulse noise.

    The wider 7-point window rejects longer impulse bursts and small noise
    clusters that a 5-point median lets through, directly reducing spurious
    slope changes and false reversals with zero phase delay on monotone runs.
    """
    a = np.asarray(x, dtype=float)
    n = len(a)
    pad = np.concatenate(([a[0]] * 3, a, [a[-1]] * 3))
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(pad[i:i + 7])
    return out


def _median7(x):
    """Zero-phase 7-point running median (edge-clamped).

    Used both as input prefilter (kills impulse noise) and on the TEMA
    output (flattens residual noise-induced slope wiggles) with zero phase
    delay on monotone runs.
    """
    a = np.asarray(x, dtype=float)
    n = len(a)
    pad = np.concatenate(([a[0]] * 3, a, [a[-1]] * 3))
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(pad[i:i + 7])
    return out


def _suppress_micro_reversals(y, frac=0.30):
    """Directional hysteresis: hold previous value on sub-threshold reversals.

    Computes a local step-scale as the running median of |diff| (zero-phase),
    then flattens any direction flip whose magnitude is below `frac` of that
    local scale. Genuine trend reversals (large steps) pass through unchanged,
    while noise-induced wiggles are removed with essentially no phase delay.
    """
    a = np.asarray(y, dtype=float)
    d = np.diff(a, prepend=a[:1])
    scale = _median7(np.abs(d))
    out = a.copy()
    for i in range(2, len(a)):
        d_prev = out[i - 1] - out[i - 2]
        d_now = out[i] - out[i - 1]
        if d_prev * d_now < 0.0 and abs(d_now) < frac * scale[i]:
            out[i] = out[i - 1]  # hold: reject micro-reversal
    return out


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Median-prefiltered zero-lag triple-EMA (TEMA) filter.

    A 7-point running median removes impulse noise — the primary driver of
    spurious slope changes and false reversals — with zero phase delay for
    monotonic segments. The cleaned signal then passes through three cascaded
    EMAs (alpha=2/(W+1)); TEMA = 3*E1 - 3*E2 + E3 cancels first- and
    second-order EMA lag. A zero-phase median7 on the TEMA output flattens
    residual noise-induced slope wiggles, a light final smoothing pass
    (alpha=0.26) suppresses residual noise, and directional hysteresis holds
    the previous value on sub-threshold slope reversals, eliminating
    noise-induced false reversals with minimal phase delay.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(f"Input too short ({len(x)}) for window {window_size}")
    x = _median7(x)
    alpha = 2.0 / (window_size + 1.0)
    e1 = _ema(x, alpha)
    e2 = _ema(e1, alpha)
    e3 = _ema(e2, alpha)
    # Zero-phase median on TEMA output flattens residual noise-induced
    # slope wiggles; TEMA's lag cancellation absorbs the cost.
    tema = _median7(3.0 * e1 - 3.0 * e2 + e3)
    # Final smoothing pass — TEMA's lag cancellation absorbs it
    y_full = _ema(tema, 0.26)
    # Directional hysteresis: suppress sub-threshold slope reversals
    y_full = _suppress_micro_reversals(y_full, frac=0.30)
    return y_full[-(len(x) - window_size + 1):]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Apply the selected filtering algorithm to the input signal."""
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
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
