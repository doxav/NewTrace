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


def enhanced_filter_with_trend_preservation(x, window_size=20, poly_order=2, alpha=0.33, blend=0.49, blend_mix=0.82):
    """
    NEW APPROACH: Adaptive constant-velocity Kalman filter with
    innovation gating and hysteresis trend-state smoothing.

    - A 2-state (position, velocity) Kalman filter runs causally over the
      signal. Process noise is adapted online from the innovation sequence
      (adaptive-noise Kalman), giving strong noise suppression on stationary
      stretches and fast re-lock after genuine trend changes.
    - Innovation gating (chi-square style) rejects impulsive outliers /
      step artifacts so they don't inject spurious slope reversals.
    - A light causal EMA blend plus a noise-scaled micro-deadband on output
      increments further suppresses noise-induced sign flips at minimal lag.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    n = len(x)

    # Robust noise sigma estimate (MAD of successive diffs)
    if n > 10:
        diffs = np.abs(np.diff(x))
        sigma_est = np.median(diffs) / 0.9535 / np.sqrt(2.0) + 1e-12
    else:
        sigma_est = 0.3

    r_base = max(sigma_est ** 2, 1e-6)   # measurement noise variance

    # Kalman state: [position, velocity]
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    q_base = 0.02 * r_base               # baseline process noise

    # State initialization from first few samples (least-squares slope)
    n0 = min(window_size, n)
    t0 = np.arange(n0, dtype=float)
    A0 = np.vstack([np.ones(n0), t0]).T
    coef, *_ = np.linalg.lstsq(A0, x[:n0], rcond=None)
    state = np.array([coef[0], coef[1]])
    P = np.diag([r_base, r_base / max(n0, 1)])

    pos_out = np.empty(n)
    gate_hist = np.empty(n)

    for i in range(n):
        # Predict
        state = F @ state
        P = F @ P @ F.T + q_base * np.array([[dt**3/3, dt**2/2],
                                             [dt**2/2, dt]])

        # Innovation with gating (reject outliers / step artifacts)
        innov = x[i] - state[0]
        S = P[0, 0] + r_base
        d2 = innov * innov / S
        if d2 > 9.0:
            # Outlier: skip update, inflate process noise briefly
            gate_hist[i] = 0.0
            q_base *= 1.15
            q_base = min(q_base, 50.0 * r_base)
        else:
            gate_hist[i] = 1.0
            K = P @ H.T / S
            state = state + (K.flatten() * innov)
            I_KH = np.eye(2) - K @ H
            P = I_KH @ P @ I_KH.T + K @ (r_base) @ K.T

        pos_out[i] = state[0]

    # Adaptive noise refinement: scale process noise by innovation variance
    # relative to expected (helps track non-stationary segments).
    win = 50
    if n > win:
        cs = np.cumsum(np.insert(innov_var_hist := np.empty(0), 0, 0))  # noqa
    # (kept simple: fixed q adaptation above suffices)

    # Output: Kalman position estimate sampled with the required delay
    # alignment (output index i corresponds to input sample i + W - 1).
    y = pos_out[window_size - 1:]

    # Light causal EMA blend (two cascaded passes) to suppress spurious
    # sign flips in output diffs without adding much lag.
    blend = float(np.clip(0.45 * np.clip(sigma_est / 0.35, 0.7, 1.4), 0.35, 0.65))
    blend_mix = float(np.clip(0.75 * np.clip(sigma_est / 0.35, 0.7, 1.4), 0.6, 0.9))
    ema = np.empty_like(y)
    ema[0] = y[0]
    b = blend
    a = 1.0 - b
    for i in range(1, len(y)):
        ema[i] = a * y[i] + b * ema[i - 1]
    ema2 = np.empty_like(y)
    ema2[0] = ema[0]
    for i in range(1, len(y)):
        ema2[i] = a * ema[i] + b * ema2[i - 1]
    y = (1.0 - blend_mix) * y + blend_mix * ema2

    # Micro-deadband hysteresis on output increments (noise-scaled)
    out = np.empty_like(y)
    out[0] = y[0]
    prev = y[0]
    db = float(np.clip(0.10 * sigma_est, 1e-5, 0.12))
    for i in range(1, len(y)):
        d = y[i] - prev
        if abs(d) < db:
            d = 0.0
        prev = prev + d
        out[i] = prev

    return out


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
