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
    Causal Holt double-exponential filter with spike-robust prefiltering
    and slope-level trend hysteresis.

    Pipeline:
      1. Causal 3-point running median removes impulse spikes that would
         otherwise induce spurious slope reversals.
      2. Damped-trend Holt recursion smooths level+slope (O(n), low latency).
      3. Hysteresis at the slope-estimate level with a robust (MAD-based)
         deadband: the trend direction only flips when the slope clearly
         exceeds the noise scale. Sub-noise slopes are gated to zero.
      4. Light causal EMA polish removes residual micro-oscillations.

    Output length = len(x) - window_size + 1.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    W = window_size
    n = len(x)

    # --- Step 1: causal 3-point median prefilter (spike removal) ---
    xf = x.copy()
    if n >= 3:
        med3 = np.empty(n)
        med3[0] = x[0]
        med3[1] = np.median(x[0:2])
        med3[2:] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
        xf = med3

    # --- Step 2: damped-trend Holt recursion with adaptive level gain ---
    alpha = 0.13   # baseline level smoothing (lower -> smoother, fewer reversals)
    beta = 0.022   # slope smoothing (low -> stable slope, fewer flips)
    phi = 0.80     # trend damping

    # Robust noise-scale estimate from prefiltered first differences
    dx = np.diff(xf, prepend=xf[0])
    mad = np.median(np.abs(dx - np.median(dx)))
    noise_scale = max(mad, 1e-6)
    deadband = max(1.7 * noise_scale, 1e-6)

    level = xf[0]
    slope = 0.0
    last_sign = 0.0

    raw = np.empty(n - W + 1)
    for i in range(n):
        prev_level = level
        # Adaptive alpha: stay smooth during noise, snap to large innovations
        # (steps / genuine trend breaks) so lag stays low without extra reversals.
        innov = xf[i] - (level + phi * slope)
        a_i = alpha
        if abs(innov) > 3.0 * noise_scale:
            a_i = min(0.55, alpha + 0.35 * (abs(innov) / (3.0 * noise_scale) - 1.0))
        level = a_i * xf[i] + (1.0 - a_i) * (level + phi * slope)
        raw_slope = beta * (level - prev_level) + (1.0 - beta) * phi * slope
        # Hysteresis: only flip trend direction when slope clearly exceeds noise
        if raw_slope * last_sign < 0 and abs(raw_slope) < deadband:
            slope = last_sign * deadband * 0.35  # hold direction, decayed
        else:
            slope = raw_slope
            if abs(slope) > deadband:
                last_sign = np.sign(slope)
            elif abs(slope) < 0.5 * deadband:
                slope = 0.0  # gate sub-noise slope: kills micro-oscillation
        if i >= W - 1:
            # Small forward trend extrapolation compensates residual filter
            # lag on smooth/sinusoidal segments; gated on confident slope so
            # it does not overshoot on steps.
            extra = 0.8 * slope if abs(slope) > deadband else 0.2 * slope
            raw[i - W + 1] = level + phi * slope + extra

    # --- Step 3: light causal EMA polish (suppress residual jitter) ---
    gamma = 0.55
    y = np.empty_like(raw)
    y[0] = raw[0]
    for j in range(1, len(raw)):
        y[j] = gamma * raw[j] + (1.0 - gamma) * y[j - 1]

    # --- Step 4: output-level reversal hysteresis (two passes) ---
    # Small output diffs that flip the sign of the previous diff are
    # noise-induced micro-reversals; flatten them to hold the previous
    # value. Adds no drift, so tracking error is essentially unchanged
    # while slope-change / false-reversal counts drop sharply. A second
    # pass with a tighter band removes flips created by the first pass.
    if len(y) > 2:
        for eps_mult in (1.15, 0.85):
            d = np.diff(y)
            base = np.median(np.abs(d))
            if base <= 0:
                continue
            eps = eps_mult * base
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

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
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
