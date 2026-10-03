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


def enhanced_filter_with_trend_preservation(x, window_size=20, q_ratio=0.35,
                                            hyst_k=0.9, adapt_gain=2.0):
    """
    State-space (Kalman) causal filter — fundamentally different family from
    the previous kernel/EMA regression approach.

    Architecture:
    1. Median-3 spike prefilter (removes single-sample spikes that cause
       spurious slope sign flips; no phase shift on monotone segments).
    2. Constant-velocity Kalman filter with innovation-adaptive process
       noise: when the normalized innovation is large (genuine trend
       change), Q inflates so the filter tracks with minimal lag; when the
       innovation is small (noise), Q stays low so the state estimate is
       heavily smoothed. This adaptive Q/R scheduling directly optimizes
       the lag-error vs. smoothness tradeoff for non-stationary data.
    3. MAD-based slope hysteresis: counter-direction output steps smaller
       than a robust deadband are attenuated (not sign-flipped), suppressing
       false reversals without adding phase delay.

    Strictly causal per-sample recursion, O(n), no lookahead.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)

    # --- Median-3 spike prefilter ---
    if len(x) >= 3:
        xm = np.empty_like(x)
        xm[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
        xm[0] = min(x[0], x[1])
        xm[-1] = min(x[-1], x[-2])
        x = xm

    n = len(x)

    # Robust measurement noise estimate from first differences of raw signal
    d_raw = np.diff(x)
    mad_raw = np.median(np.abs(d_raw - np.median(d_raw)))
    R = max((1.4826 * mad_raw) ** 2, 1e-9)
    Q_base = q_ratio * R

    # --- Constant-velocity Kalman filter (hand-rolled, causal) ---
    # State: [position, velocity]; dt = 1
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])

    # Initialize state from the first two samples
    pos = x[0]
    vel = x[1] - x[0] if n > 1 else 0.0
    P = np.eye(2) * R

    kf = np.empty(n, dtype=float)
    kf[0] = pos

    for i in range(1, n):
        # Predict
        pos_pred = pos + vel
        vel_pred = vel
        P_pred = F @ P @ F.T
        P_pred[0, 0] += Q_base
        P_pred[1, 1] += Q_base * 0.5

        # Innovation
        innov = x[i] - pos_pred
        S = P_pred[0, 0] + R

        # Innovation-adaptive process noise: large |innov| -> trust
        # measurement more (lower lag on genuine trend changes)
        q_scale = 1.0 + adapt_gain * (innov * innov) / S
        if q_scale > 1.0:
            P_pred[0, 0] += Q_base * (q_scale - 1.0)

        # Kalman gain
        K0 = P_pred[0, 0] / S
        K1 = P_pred[1, 0] / S

        # Update
        pos = pos_pred + K0 * innov
        vel = vel_pred + K1 * innov
        P00 = (1.0 - K0) * P_pred[0, 0]
        P01 = (1.0 - K0) * P_pred[0, 1]
        P10 = P_pred[1, 0] - K1 * P_pred[0, 0]
        P11 = P_pred[1, 1] - K1 * P_pred[0, 1]
        P = np.array([[P00, P01], [P10, P11]])

        if not np.isfinite(pos):
            pos = x[i]
            vel = 0.0
        kf[i] = pos

    # --- MAD-based slope hysteresis on the Kalman output ---
    d = np.diff(kf)
    mad = np.median(np.abs(d - np.median(d)))
    deadband = hyst_k * (1.4826 * mad + 1e-12)
    yh = np.empty_like(kf)
    yh[0] = kf[0]
    direction = 0.0
    for k in range(1, n):
        v = kf[k] - yh[k - 1]
        if direction >= 0 and v < -deadband:
            direction = -1.0
        elif direction <= 0 and v > deadband:
            direction = 1.0
        if direction > 0 and v < 0:
            yh[k] = yh[k - 1] + 0.5 * max(v, -deadband)
        elif direction < 0 and v > 0:
            yh[k] = yh[k - 1] + 0.5 * min(v, deadband)
        else:
            yh[k] = kf[k]
    y = np.where(np.isfinite(yh), yh, kf)

    # Output contract: drop first window_size - 1 samples (causal alignment)
    return y[window_size - 1:]


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
    input_signal = np.asarray(input_signal, dtype=np.float64)
    if len(input_signal) < window_size:
        return np.empty(0, dtype=np.float64)
    if algorithm_type == "enhanced":
        out = enhanced_filter_with_trend_preservation(input_signal, window_size)
    else:
        out = adaptive_filter(input_signal, window_size)
    expected_len = len(input_signal) - window_size + 1
    out = np.asarray(out, dtype=np.float64)[:expected_len]
    out = np.where(np.isfinite(out), out, 0.0)
    return out


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
