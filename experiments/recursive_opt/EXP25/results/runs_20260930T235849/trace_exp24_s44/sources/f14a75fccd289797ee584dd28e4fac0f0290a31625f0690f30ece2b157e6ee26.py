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
    adaptation of both process noise and measurement noise, plus a light
    output deadband to suppress noise-induced slope reversals.

    Model: state = [level, slope], F = [[1,1],[0,1]], H = [1,0].

    Adaptation:
      - Measurement noise R is tracked online as an EMA of squared
        innovations (clipped for robustness), so it tracks non-stationary
        noise levels without a fixed assumption.
      - Process noise Q is scaled by an innovation-consistency ratio
        (innov^2 / predicted variance): when the signal moves more than
        the model expects, Q inflates so the filter tracks genuine trend
        changes quickly (low lag); in quiet stretches Q deflates so the
        filter averages noise heavily (few reversals).

    Output: posterior level + 0.5 * slope (half-step extrapolation),
    giving near-zero phase lag. A mild deadband on output diffs
    (relative to a running robust diff scale) zeroes noise-level moves,
    cutting spurious directional reversals at negligible latency cost.

    Wrapped defensively: non-finite inputs are interpolated and any
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

        # --- Initialization from local statistics ---
        if n > 5:
            d0 = np.abs(np.diff(x[: min(n, 30)]))
            r0 = float(np.median(d0)) ** 2
        else:
            r0 = 1.0
        if not np.isfinite(r0) or r0 <= 0:
            r0 = 1.0

        level = float(x[0])
        slope = 0.0
        # Initial covariance
        Pll = r0 * 10.0
        Pls = 0.0
        Pss = r0 * 1.0
        R = r0

        levels = np.empty(n)
        slopes = np.empty(n)
        levels[0] = level
        slopes[0] = slope

        for t in range(1, n):
            # --- Predict ---
            l_pred = level + slope
            s_pred = slope
            # F P F^T
            Pll_p = Pll + 2.0 * Pls + Pss
            Pls_p = Pls + Pss
            Pss_p = Pss

            innov = x[t] - l_pred

            # --- Adaptive process noise: inflate Q when the innovation
            # exceeds what the predicted uncertainty explains ---
            S = Pll_p + R
            ratio = (innov * innov) / (S + 1e-12)
            m = 1.0 + 4.0 * ratio
            if m > 25.0:
                m = 25.0
            q = 0.02 * R * m
            qs = 0.002 * R * m
            Pll_p += q
            Pss_p += qs

            # --- Adaptive measurement noise (robust EMA of innov^2) ---
            R = 0.90 * R + 0.10 * min(innov * innov, 25.0 * R + 1e-12)
            if R < 1e-12:
                R = 1e-12

            # --- Update ---
            S = Pll_p + R
            K0 = Pll_p / S
            K1 = Pls_p / S
            level = l_pred + K0 * innov
            slope = s_pred + K1 * innov
            # Covariance update (Joseph-lite, scalar form)
            Pll = (1.0 - K0) * Pll_p
            Pls = (1.0 - K0) * Pls_p
            Pss = Pss_p - K1 * Pls_p
            if Pss < 1e-12:
                Pss = 1e-12

            levels[t] = level
            slopes[t] = slope

        # Half-step extrapolation: near-zero lag while retaining the
        # Kalman noise averaging.
        z = levels + 0.5 * slopes
        if not np.all(np.isfinite(z)):
            return _safe_fallback(x, W)

        # --- Light deadband EMA on output diffs ---
        a = 0.55            # gain (low lag)
        dead = 0.55         # deadband fraction of running diff scale
        y = np.empty(out_len)
        prev = z[0]
        y[0] = prev
        scale = 0.0
        for i in range(1, out_len):
            d = z[i] - prev
            scale = 0.90 * scale + 0.10 * abs(d)
            if abs(d) < dead * scale + 1e-15:
                d = 0.0     # noise-level move: hold (kills reversals)
            prev = prev + a * d
            y[i] = prev

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
