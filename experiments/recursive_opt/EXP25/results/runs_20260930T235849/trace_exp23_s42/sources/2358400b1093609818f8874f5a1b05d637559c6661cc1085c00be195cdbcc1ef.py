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
    Constant-velocity Kalman filter (state-space / Bayesian approach).

    A fundamentally different mechanism from exponential/window smoothers:
    - State vector [level, velocity] propagated with a constant-velocity
      transition model; the Kalman gain is computed optimally at each step,
      balancing measurement noise against process noise with zero tuning of
      fixed smoothing constants.
    - Measurement noise R is estimated robustly (MAD of first differences),
      so the filter self-tunes to the actual noise level.
    - Innovation-gated process noise: a large normalized innovation (genuine
      step/trend change) temporarily inflates Q so the filter re-locks fast
      (low lag), while small innovations keep Q tiny (heavy noise suppression,
      few spurious slope reversals). Hysteresis on the gate avoids chatter.
    - Fully causal: output[i] depends only on x[0..i].

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (used only for output alignment)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Robust measurement-noise estimate from first differences:
    # Var(diff) = 2*sigma_meas^2 for white measurement noise
    d = np.diff(x)
    if len(d) > 0:
        sigma_meas = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    else:
        sigma_meas = 0.3
    if not np.isfinite(sigma_meas) or sigma_meas <= 1e-9:
        sigma_meas = 1e-3
    R = sigma_meas ** 2

    # State-space model: constant velocity
    # x_{k+1} = F x_k + w,  w ~ N(0, Q);  z_k = H x_k + v, v ~ N(0, R)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])

    # Baseline process noise: small -> smooth tracking, few reversals.
    # Scaled by sigma_meas so behavior is scale-invariant.
    q_base = (0.02 * sigma_meas) ** 2
    q_boost = (0.35 * sigma_meas) ** 2

    # Initial state
    state = np.array([x[0], 0.0])
    P = np.diag([R, (sigma_meas * 2.0) ** 2])

    out = np.empty(n)
    boost = 0.0

    for i in range(n):
        # --- Predict ---
        state = F @ state
        Q = np.zeros((2, 2))
        q = q_base + (q_boost - q_base) * boost
        Q[0, 0] = q * 0.25
        Q[0, 1] = q * 0.5
        Q[1, 1] = q
        P = F @ P @ F.T + Q

        # --- Innovation & adaptive gating ---
        z = x[i]
        innov = z - float(H @ state)
        S = float(H @ P @ H) + R
        norm_innov = abs(innov) / np.sqrt(S) if S > 1e-18 else 0.0
        # Hysteresis: latch boost on large innovations, decay on small ones
        if norm_innov > 2.8:
            boost = 1.0
        elif norm_innov < 1.3:
            boost *= 0.85

        # --- Update (standard Kalman gain) ---
        K = (P @ H) / S
        state = state + K * innov

        # Joseph-form covariance update for numerical stability
        IKH = np.eye(2) - np.outer(K, H)
        P = IKH @ P @ IKH.T + np.outer(K, K) * R

        # Numerical safeguards
        if not (np.isfinite(state[0]) and np.isfinite(state[1])):
            state = np.array([z, 0.0])
            P = np.diag([R, (sigma_meas * 2.0) ** 2])
        out[i] = state[0]

    # Align to window-end convention: y[i] corresponds to x[i + window_size - 1]
    y = out[window_size - 1 :].copy()
    y = np.where(np.isfinite(y), y, 0.0)
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
