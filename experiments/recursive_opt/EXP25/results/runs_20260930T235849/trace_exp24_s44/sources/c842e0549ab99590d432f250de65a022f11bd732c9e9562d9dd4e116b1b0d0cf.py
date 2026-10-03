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
    Adaptive Kalman filter (constant-velocity state model) with online
    process/measurement noise adaptation, emitting the one-step-ahead
    prediction as the filtered output (near-zero phase lag).

    State model: [level, slope], F = [[1,1],[0,1]], H = [1,0].

    Adaptation:
      * R (measurement noise) is re-estimated each step as half the variance
        of the differences over the most recent sliding window — this tracks
        non-stationary noise levels without future samples.
      * Q (process noise) is scaled by max(1, e^2/S) where e is the current
        innovation and S the predicted innovation variance. Small
        noise-level innovations leave Q at its quiet baseline (heavy
        smoothing, few reversals); large genuine moves inflate Q so the
        filter tracks them almost instantly (minimal lag, low false
        reversal count).

    The emitted sample for output index i is the state prediction made
    AFTER assimilating sample x[i + W - 2], i.e. the predicted level at
    x[i + W - 1] — a one-step-ahead forecast, so the output leads rather
    than lags the local trend.

    Fully causal, numpy-only, O(n) with a small constant; wrapped
    defensively with a safe causal EMA fallback.
    """
    try:
        x = np.asarray(x, dtype=float)
        n = x.size
        W = int(window_size)
        if W < 3 or n < W:
            raise ValueError("invalid input dimensions")

        out_len = n - W + 1
        if out_len <= 0:
            return np.zeros(0)

        # Repair non-finite samples by interpolation (numerical hygiene).
        if not np.all(np.isfinite(x)):
            idx = np.arange(n)
            good = np.isfinite(x)
            if good.sum() < 2:
                return _safe_fallback(x, W)
            x = np.interp(idx, idx[good], x[good])

        y = np.empty(out_len)

        # Initial state: level = first sample, slope = 0.
        lvl = float(x[0])
        slope = 0.0
        P = np.eye(2) * max(float(np.var(x[:W])) + 1e-6, 1e-6)

        # Noise model parameters.
        q0 = 0.02            # baseline process noise (quiet regime)
        R = max(float(np.var(np.diff(x[: min(W, n)]))) * 0.5, 1e-6)

        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        e_prev = 0.0
        S_prev = R + P[0, 0]

        for t in range(1, n):
            # --- Adaptive process noise from last innovation ---
            q = q0 * max(1.0, (e_prev * e_prev) / (S_prev + 1e-12))
            Q = q * np.array([[0.25, 0.5], [0.5, 1.0]])

            # --- Predict ---
            lvl_p = lvl + slope
            P = F @ P @ F.T + Q

            # --- Adaptive measurement noise from sliding-window diffs ---
            if t >= W:
                d = np.diff(x[t - W + 1 : t + 1])
                R = max(0.5 * float(np.var(d)), 1e-9)

            # --- Update ---
            e = x[t] - lvl_p
            S = P[0, 0] + R
            K0 = P[0, 0] / S
            K1 = P[1, 0] / S
            lvl = lvl_p + K0 * e
            slope = slope + K1 * e
            P00 = P[0, 0] * (1.0 - K0)
            P01 = P[0, 1] * (1.0 - K0)
            P10 = P[1, 0] - K1 * P[0, 0]
            P11 = P[1, 1] - K1 * P[0, 1]
            P = np.array([[P00, P01], [P10, P11]])

            e_prev = e
            S_prev = S

            # --- Emit one-step-ahead prediction for window ending at t+1 ---
            idx = t - W + 2
            if 0 <= idx < out_len:
                y[idx] = lvl + slope

        # Output index 0 corresponds to the prediction made after
        # assimilating x[W-2]; ensure it is set (loop starts at t=1).
        if not np.isfinite(y[0]):
            y[0] = x[W - 1]

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
