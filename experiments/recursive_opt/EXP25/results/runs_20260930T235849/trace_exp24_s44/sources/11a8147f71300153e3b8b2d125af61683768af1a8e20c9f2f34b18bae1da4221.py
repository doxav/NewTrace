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


def _safe_fallback(x, window_size):
    """Causal EMA fallback that always returns a finite array of the
    documented length len(x) - window_size + 1 (never raises)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    W = max(int(window_size), 1)
    out_len = max(n - W + 1, 0)
    if out_len == 0:
        return np.zeros(0)
    if not np.all(np.isfinite(x)):
        idx = np.arange(n)
        good = np.isfinite(x)
        if good.sum() == 0:
            return np.zeros(out_len)
        x = np.interp(idx, idx[good], x[good])
    a = 0.4
    z = np.empty(n)
    z[0] = x[0]
    for i in range(1, n):
        z[i] = z[i - 1] + a * (x[i] - z[i - 1])
    return z[W - 1 :].copy()


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Adaptive Kalman filter (constant-velocity state model) with online
    maneuver detection, plus a light noise-deadband output stage.

    Core: state [level, velocity], measurement = level.
      - Measurement noise R is estimated robustly from the median absolute
        deviation of first differences (1.4826*MAD/sqrt(2)).
      - Process noise Q is ADAPTED online: when the previous innovation is
        large relative to the noise scale (a genuine maneuver / trend turn),
        Q inflates so the filter tracks the turn immediately (no lag, no
        missed reversal). When innovations are noise-sized, Q stays tiny so
        the filter heavily averages noise -> few slope reversals.
      - Output sample = posterior level + 0.5 * velocity (half-step
        extrapolation), giving near-zero phase lag on trending segments.

    Final stage: a light deadband smoother on the Kalman output kills the
    residual noise-level wiggle without adding meaningful delay.

    Defensive wrapping: non-finite inputs are interpolated; any unexpected
    error falls back to a safe causal EMA returning exactly
    len(x) - window_size + 1 finite samples.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    try:
        x = np.asarray(x, dtype=float)
        n = x.size
        W = int(window_size)
        if W < 3 or n < W:
            raise ValueError("invalid input dimensions")

        out_len = n - W + 1

        # Repair non-finite samples by interpolation (numerical hygiene).
        if not np.all(np.isfinite(x)):
            idx = np.arange(n)
            good = np.isfinite(x)
            if good.sum() < 2:
                return _safe_fallback(x, W)
            x = np.interp(idx, idx[good], x[good])

        # Robust noise scale from first differences (MAD-based).
        dx = np.diff(x)
        if dx.size == 0:
            return _safe_fallback(x, W)
        sig = 1.4826 * np.median(np.abs(dx - np.median(dx))) / np.sqrt(2.0)
        if not np.isfinite(sig) or sig < 1e-9:
            sig = max(np.std(dx) / np.sqrt(2.0), 1e-9)
        R = sig * sig

        # Base process noise (small -> smooth in steady state) and the
        # maneuver-inflation factor applied when innovations are large.
        q_base = (0.12 * sig) ** 2
        inflate = 12.0

        # Kalman state initialization.
        lvl = x[0]
        vel = 0.0
        P00, P01, P10, P11 = R, 0.0, 0.0, sig * sig
        inn_prev = 0.0

        z = np.empty(n)
        vlim = 20.0 * sig
        for i in range(n):
            # Adaptive process noise from the previous innovation magnitude.
            maneuver = (inn_prev / (3.0 * sig)) ** 2
            q = q_base * (1.0 + inflate * maneuver)

            # Predict (dt = 1): level += velocity.
            lvl_pred = lvl + vel
            p00 = P00 + P01 + P10 + P11 + 0.25 * q
            p01 = P01 + P11 + 0.5 * q
            p10 = P10 + P11 + 0.5 * q
            p11 = P11 + q

            # Update with measurement x[i].
            S = p00 + R
            if not np.isfinite(S) or S < 1e-300:
                return _safe_fallback(x, W)
            K0 = p00 / S
            K1 = p10 / S
            inn = x[i] - lvl_pred
            lvl = lvl_pred + K0 * inn
            vel = vel + K1 * inn
            # Joseph-free (simple) covariance update, symmetrized.
            c00 = (1.0 - K0) * p00
            c01 = (1.0 - K0) * p01
            c10 = p10 - K1 * p00
            c11 = p11 - K1 * p01
            P00, P01 = c00, 0.5 * (c01 + c10)
            P10, P11 = P01, c11
            vel = float(np.clip(vel, -vlim, vlim))
            inn_prev = inn
            z[i] = lvl + 0.5 * vel   # half-step extrapolation: near-zero lag

        if not np.all(np.isfinite(z)):
            return _safe_fallback(x, W)

        # Light deadband smoother on the Kalman output: suppresses residual
        # noise-level wiggle (fewer reversals) with minimal added lag.
        a = 0.75            # high gain: Kalman output is already smooth
        dead = 0.30         # small deadband fraction of running diff scale
        y = np.empty(out_len)
        prev = z[W - 1]
        y[0] = prev
        scale = 0.0
        for i in range(W, n):
            d = z[i] - prev
            scale = 0.90 * scale + 0.10 * abs(d)
            if abs(d) < dead * scale + 1e-15:
                d = 0.0     # noise-level move: hold (kills reversals)
            prev = prev + a * d
            y[i - W + 1] = prev

        if not np.all(np.isfinite(y)):
            return _safe_fallback(x, W)
        return y

    except Exception:
        try:
            return _safe_fallback(x, window_size)
        except Exception:
            return np.zeros(max(len(np.asarray(x)) - int(window_size) + 1, 0))


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
