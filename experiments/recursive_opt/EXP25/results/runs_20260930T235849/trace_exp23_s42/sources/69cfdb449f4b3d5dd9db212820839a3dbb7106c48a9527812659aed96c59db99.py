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
    Constant-velocity Kalman filter (state-space / Bayesian family) with
    innovation-gated measurement noise for non-stationary robustness.

    - State: [level, velocity]; prediction step propagates a locally-linear
      model, giving near-zero lag on genuine trends (optimal MMSE estimate).
    - Innovation-gated R: measurement noise is inflated when the innovation
      is small (heavy smoothing, few slope reversals) and restored when a
      large innovation signals a genuine step/trend change (fast re-tracking,
      no ringing, no drift on random walks).
    - Causal median-of-3 on the output suppresses single-sample spikes that
      would create false directional reversals, with no phase shift.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (used for output alignment)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Robust measurement-noise estimate from first differences (MAD-based).
    d = np.abs(np.diff(x))
    r0 = (1.4826 * np.median(d) / np.sqrt(2.0)) ** 2 if len(d) > 0 else 0.1
    if r0 <= 1e-12:
        r0 = 1e-6

    # Kalman parameters: constant-velocity model.
    # q scales process noise (velocity random walk); larger q -> faster
    # tracking, smaller q -> smoother output with fewer reversals.
    q_level = 0.02 * r0
    q_vel = 0.004 * r0
    gate_hi = 3.0    # innovation z-score above which R is reset (genuine change)
    gate_lo = 1.5    # innovation z-score below which R is inflated (steady noise)
    r_mult_max = 9.0 # max inflation of measurement noise in steady state

    # State and covariance initialization
    state = np.array([x[0], 0.0])
    P = np.diag([r0, r0])

    F = np.array([[1.0, 1.0], [0.0, 1.0]])          # state transition
    Q = np.array([[q_level, 0.0], [0.0, q_vel]])    # process noise
    H = np.array([[1.0, 0.0]])                      # measurement model

    out = np.empty(n)
    r = r0

    for i in range(n):
        # --- Predict ---
        state = F @ state
        P = F @ P @ F.T + Q

        # --- Innovation-gated adaptive measurement noise ---
        z = x[i]
        innov = z - float(H @ state)
        S = float(H @ P @ H.T) + r          # innovation covariance
        z_score = abs(innov) / np.sqrt(max(S, 1e-12))
        if z_score > gate_hi:
            r = r0                          # genuine change: trust measurement
        elif z_score < gate_lo:
            r = min(r * 1.15, r_mult_max * r0)  # steady state: smooth heavily
        else:
            r = max(r * 0.95, r0)           # hysteresis decay back to baseline

        # --- Update (Joseph form for numerical stability) ---
        S = float(H @ P @ H.T) + r
        if S <= 1e-12:
            S = 1e-12
        K = (P @ H.T) / S                   # 2x1 Kalman gain
        state = state + (K.flatten() * innov)
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P @ I_KH.T + K @ (np.array([[r]])) @ K.T

        val = state[0]
        if not np.isfinite(val):
            val = out[i - 1] if i > 0 else z
        out[i] = val

    # Align to window-end convention: y[i] corresponds to x[i + window_size - 1]
    seg = out[window_size - 1 :].copy()

    # Causal median-of-3 on the output: removes single-sample spikes that
    # create spurious slope reversals, without adding phase delay.
    m = len(seg)
    if m >= 3:
        med = np.median(np.vstack([seg[:-2], seg[1:-1], seg[2:]]), axis=0)
        seg[2:] = med

    # Final NaN/inf guard
    seg = np.where(np.isfinite(seg), seg, 0.0)
    return seg


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
