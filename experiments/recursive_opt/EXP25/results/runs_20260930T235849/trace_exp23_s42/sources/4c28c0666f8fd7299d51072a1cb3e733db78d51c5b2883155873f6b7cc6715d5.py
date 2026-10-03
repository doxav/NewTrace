# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

State-space (Bayesian) approach: a constant-velocity Kalman filter with
innovation-gated adaptive process noise and robust measurement-noise scaling.
This is a fundamentally different mechanism from window smoothing: the filter
maintains an explicit probabilistic state [level, velocity] with a Riccati
recursion that optimally balances lag against noise suppression.
"""

import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Baseline moving average filter.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)
    for i in range(output_length):
        y[i] = np.mean(x[i : i + window_size])
    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Adaptive constant-velocity Kalman filter (state-space / Bayesian family).

    Mechanism (qualitatively different from window smoothing):
    - State vector s = [level, velocity], constant-velocity transition model.
    - Measurement noise R is set from a robust MAD estimate of the signal's
      first-difference scale, adapting automatically to noise levels 0.2-0.6.
    - Process noise Q is innovation-gated: small in steady state (strong
      smoothing, few spurious slope reversals), boosted when a large normalized
      innovation signals a genuine trend/step change (fast re-tracking, low
      lag, no ringing drift on random walks). Hysteresis prevents gain chatter.
    - A mild forward extrapolation by the posterior velocity compensates the
      filter's group delay, minimizing lag error without overshoot.
    - Fully causal: output[i] depends only on x[0..i].

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (used for alignment only)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Robust measurement-noise scale from first differences of the raw signal:
    # for white noise, std(diff) = sqrt(2)*sigma, so recover sigma via MAD.
    d = np.diff(x)
    if len(d) > 0:
        sigma = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    else:
        sigma = 0.3
    sigma = max(sigma, 1e-6)
    R = sigma ** 2

    # State transition (constant velocity, dt = 1)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    # Baseline process noise: small -> smooth, reversal-sparse tracking
    q_base = (0.05 * sigma) ** 2
    q_boost = (0.60 * sigma) ** 2

    Q_base = np.array([[q_base / 3.0, q_base / 2.0],
                       [q_base / 2.0, q_base]])
    Q_boost = np.array([[q_boost / 3.0, q_boost / 2.0],
                        [q_boost / 2.0, q_boost]])

    # Initialize state from first few samples (causal)
    k0 = min(5, n)
    level0 = float(np.mean(x[:k0]))
    if n >= 2:
        vel0 = float((x[min(k0, n - 1)] - x[0]) / max(k0 - 1, 1))
    else:
        vel0 = 0.0

    s = np.array([level0, vel0])
    P = np.diag([R, R * 4.0])

    out = np.empty(n)
    boosted = 0.0
    innov_gate_hi = 3.0 * sigma
    innov_gate_lo = 1.5 * sigma

    for i in range(n):
        # ---- Predict ----
        s = F @ s
        Q = Q_base + (Q_boost - Q_base) * boosted
        P = F @ P @ F.T + Q

        # ---- Innovation-gated adaptive gain with hysteresis ----
        z = x[i]
        innov = z - float(H @ s)
        a_innov = abs(innov)
        if a_innov > innov_gate_hi:
            boosted = 1.0
        elif a_innov < innov_gate_lo:
            boosted *= 0.85

        # Outlier rejection: clip extreme innovations (spike robustness)
        innov_c = np.clip(innov, -4.0 * sigma, 4.0 * sigma)

        # ---- Update ----
        S = float(P[0, 0]) + R
        K = P[:, 0] / S
        s = s + K * innov_c

        A = np.eye(2) - np.outer(K, H)
        P = A @ P @ A.T + np.outer(K, K) * R
        P = 0.5 * (P + P.T)  # symmetrize for numerical stability

        # Mild forward extrapolation to compensate group delay
        out[i] = s[0] + s[1] * 1.5

        if not np.isfinite(out[i]):
            out[i] = out[i - 1] if i > 0 else 0.0

    # Align to window-end convention: y[i] corresponds to x[i + window_size - 1]
    y = out[window_size - 1 :]

    # Light causal median-of-3 on output to remove residual single-sample
    # spikes that could induce false directional reversals.
    m = y.copy()
    if len(y) >= 3:
        m[2:] = np.median(np.vstack([y[:-2], y[1:-1], y[2:]]), axis=0)

    return np.where(np.isfinite(m), m, 0.0)


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
