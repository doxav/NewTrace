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
    Adaptive Kalman filter (local-linear state model) with innovation-driven
    process-noise adaptation, spike-robust prefiltering, and output-level
    reversal hysteresis.

    Paradigm change: instead of averaging over a fixed sliding window
    (which trades lag against smoothing at a fixed point on that curve),
    a Kalman filter with a [level, slope] state adapts its effective gain
    to the signal: small gains during quiet/noisy stretches (few slope
    changes, few false reversals) and large gains when the innovation
    grows (steps, genuine trend breaks tracked with minimal lag).

    Pipeline:
      1. Causal 3-point median prefilter removes impulse spikes.
      2. Constant-velocity Kalman recursion:
             F = [[1,1],[0,1]],  H = [1,0]
         Process noise Q is inflated online when the normalized innovation
         exceeds a threshold, so step changes are followed almost
         immediately while quiet segments stay heavily smoothed.
      3. Causal EMA polish suppresses residual micro-oscillation.
      4. Output-level reversal hysteresis flattens sub-noise sign flips
         (no drift, tracking accuracy preserved).

    Output length = len(x) - window_size + 1.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
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

    # Robust measurement-noise scale from first differences of the
    # prefiltered signal (MAD of diffs ~ sigma/sqrt(2) for white noise).
    dx = np.diff(xf, prepend=xf[0])
    mad = np.median(np.abs(dx - np.median(dx)))
    r_noise = max((mad / 0.6745) ** 2, 1e-6)  # measurement variance R

    # --- Step 2: constant-velocity Kalman filter with adaptive Q ---
    # State s = [level, slope]; transition F = [[1,1],[0,1]].
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q_base = np.array([[0.02, 0.0], [0.0, 0.002]]) * r_noise
    Q_hi = np.array([[4.0, 0.0], [0.0, 1.0]]) * r_noise  # inflated (steps)

    s = np.array([xf[0], 0.0])          # initial state
    P = np.array([[r_noise, 0.0], [0.0, r_noise]])

    raw = np.empty(n - W + 1)
    innov_gate = 3.0 * np.sqrt(r_noise)

    for i in range(n):
        # Predict
        s = F @ s
        P = F @ P @ F.T + Q_base

        # Innovation and adaptive process noise: inflate Q when the
        # innovation is large (step / genuine trend break) so the filter
        # snaps to the new level with minimal lag.
        innov = xf[i] - H @ s
        Q = Q_base
        if abs(innov) > innov_gate:
            frac = min(abs(innov) / innov_gate - 1.0, 3.0)
            Q = Q_base + frac * Q_hi
            P = P + Q

        # Update
        PHt = P @ H
        S = H @ PHt + r_noise
        K = PHt / S
        s = s + K * innov
        P = P - np.outer(K, PHt)

        if i >= W - 1:
            # Small forward projection of the slope cancels residual lag
            # without the overshoot a large lead term would cause.
            raw[i - W + 1] = s[0] + 0.35 * s[1]

    # --- Step 3: causal EMA polish (suppress residual jitter) ---
    gamma = 0.55
    y = np.empty_like(raw)
    y[0] = raw[0]
    for j in range(1, len(raw)):
        y[j] = gamma * raw[j] + (1.0 - gamma) * y[j - 1]

    # --- Step 3b: causal 3-tap median on output (kills single-sample flips) ---
    if len(y) >= 3:
        ym = np.empty_like(y)
        ym[0] = y[0]
        ym[1] = 0.5 * (y[0] + y[1])
        ym[2:] = np.median(np.stack([y[:-2], y[1:-1], y[2:]]), axis=0)
        y = ym

    # --- Step 4: output-level reversal hysteresis ---
    # Sub-noise counter-moves are flattened (hold previous value): removes
    # noise-induced sign flips without drift, so tracking error and lag
    # are essentially unchanged while slope changes / false reversals drop.
    if len(y) > 2:
        for eps_mult in (1.6, 1.1, 0.8):
            d = np.diff(y)
            base = np.median(np.abs(d))
            if base <= 0:
                continue
            eps = max(eps_mult * base, 0.4 * eps_mult * np.sqrt(r_noise))
            prev_sign = 0.0
            for j in range(1, len(y)):
                dj = y[j] - y[j - 1]
                if dj == 0.0:
                    continue
                sj = np.sign(dj)
                if prev_sign != 0.0 and sj != prev_sign and abs(dj) < eps:
                    y[j] = y[j - 1]  # hold: suppress noise-induced flip
                else:
                    prev_sign = sj

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
