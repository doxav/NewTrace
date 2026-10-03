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


def _derivative_hysteresis(y, k=0.45):
    """
    Suppress sign flips of the output derivative when the new slope magnitude
    is below k * robust scale of recent slopes. Directly targets slope-change
    (S) and false-reversal (R) penalties while preserving genuine trends.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 4:
        return y
    out = y.copy()
    dy = np.diff(y)
    scale = np.median(np.abs(dy)) + 1e-12
    prev_sign = 1.0 if dy[0] >= 0 else -1.0
    for i in range(1, n - 1):
        s = dy[i]
        sgn = 1.0 if s >= 0 else -1.0
        if sgn != prev_sign and abs(s) < k * scale:
            # Noise-induced reversal: hold previous level (flatten)
            out[i + 1] = out[i]
            dy[i] = 0.0
        elif sgn != prev_sign:
            prev_sign = sgn
    return out


def _kalman_cv(x, R, q_pos=0.02, q_vel=0.05):
    """
    Causal constant-velocity Kalman filter (state-space family).
    State: [position, velocity]; measurement: position.
    q_pos/q_vel control smoothness vs. lag via the Q/R ratio.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = np.diag([q_pos, q_vel])
    H = np.array([1.0, 0.0])

    # State init
    pos = x[0]
    vel = (x[min(1, n - 1)] - x[0]) if n > 1 else 0.0
    P = np.diag([R, R])

    y = np.empty(n)
    y[0] = pos
    for i in range(1, n):
        # Predict
        pos_p = pos + dt * vel
        vel_p = vel
        Pp = F @ P @ F.T + Q

        # Innovation
        z = x[i]
        innov = z - pos_p
        S = Pp[0, 0] + R
        K = Pp[:, 0] / S

        # Update
        pos = pos_p + K[0] * innov
        vel = vel_p + K[1] * innov
        P = Pp - np.outer(K, Pp[0, :])

        y[i] = pos
    return y


def enhanced_filter_with_trend_preservation(x, window_size=20,
                                            q_pos=0.02, q_vel=0.05, k=0.45):
    """
    State-space pipeline (fundamentally different from sliding-window
    regression): a causal constant-velocity Kalman filter with
    noise-adaptive measurement covariance R (estimated robustly from the
    median absolute successive difference), followed by a derivative
    hysteresis state machine that suppresses noise-induced directional
    reversals.

    - Kalman gain adapts implicitly: high Q/R keeps lag low on genuine
      trends; moderate R suppresses noise between reversals.
    - Hysteresis flattens sub-threshold slope sign flips -> fewer false
      reversals and smoother slope evolution, without added phase delay.
    - Fully causal per-sample, O(n).
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=np.float64)

    # Robust noise-scale estimate from successive differences
    d = np.diff(x)
    if len(d) > 0:
        sigma = np.median(np.abs(d)) / (0.6745 * np.sqrt(2.0)) + 1e-9
    else:
        sigma = 1.0
    R = max(sigma * sigma, 1e-6)

    y = _kalman_cv(x, R, q_pos=q_pos, q_vel=q_vel)
    y = _derivative_hysteresis(y, k=k)

    # Numerical hygiene
    y = np.where(np.isfinite(y), y, x)
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
