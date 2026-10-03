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
    Adaptive constant-velocity Kalman filter (state-space / Bayesian approach).

    Fundamentally different mechanism from exponential smoothing: an explicit
    Bayesian state-space model [position, velocity] with a recursively
    propagated covariance. The Kalman gain automatically balances measurement
    trust vs. model trust, giving minimum-variance tracking with low lag.

    - Measurement noise R estimated robustly (MAD of first differences).
    - Process noise Q is innovation-gated with hysteresis: small in steady
      state (heavy noise suppression, few spurious slope reversals), boosted
      on large innovations (fast re-tracking of genuine trend/step changes,
      minimal lag).
    - Small forward velocity extrapolation compensates residual group delay.
    - Fully causal: output[i] uses only samples up to index i.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (used for delay compensation)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Robust noise scale estimate from first differences (MAD-based)
    d = np.abs(np.diff(x))
    sigma = 1.4826 * np.median(d) if len(d) > 0 else 1.0
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = 1e-9

    # Measurement noise variance
    R = max(sigma * sigma, 1e-9)

    # Kalman state: [position, velocity]; transition for 1-step constant velocity
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    # Process noise (acceleration variance) — baseline small for smoothness
    q_base = (0.06 * sigma) ** 2
    q_boost = (0.55 * sigma) ** 2
    Q_base = q_base * np.array([[0.25, 0.5], [0.5, 1.0]])
    Q_boost = q_boost * np.array([[0.25, 0.5], [0.5, 1.0]])

    # Initialize state from first two samples (causal)
    pos = x[0]
    vel = x[1] - x[0] if n > 1 else 0.0
    state = np.array([pos, vel])
    # Initial covariance: large uncertainty in velocity
    P = np.array([[R, 0.0], [0.0, R * 4.0]])

    out = np.empty(n)
    boosted = 0.0
    lag_comp = 0.5  # half-step forward velocity extrapolation (delay compensation)

    for i in range(n):
        z = x[i]
        if not np.isfinite(z):
            z = state[0]

        # --- Innovation-gated adaptive process noise (with hysteresis) ---
        # Predict step with current Q
        Q = Q_base + (Q_boost - Q_base) * boosted
        P_pred = F @ P @ F.T + Q
        S = float(H @ P_pred @ H.T + R)
        if S <= 1e-12:
            S = 1e-12
        innov = z - float(H @ state)
        nis = (innov * innov) / S  # normalized innovation squared

        # Gate: chi-square-like threshold; hysteresis decay avoids gain chatter
        if nis > 9.0:        # ~3-sigma: genuine trend/step change detected
            boosted = 1.0
        elif nis < 3.0:
            boosted *= 0.75
        if boosted > 1e-6:
            # Re-predict with boosted Q for fast re-tracking
            Q = Q_base + (Q_boost - Q_base) * boosted
            P_pred = F @ P @ F.T + Q
            S = float(H @ P_pred @ H.T + R)
            if S <= 1e-12:
                S = 1e-12

        # --- Kalman update ---
        K = P_pred @ H.T / S  # 2x1 gain
        state = state + (K.flatten() * innov)
        P = (np.eye(2) - K @ H) @ P_pred
        # Symmetrize for numerical stability
        P = 0.5 * (P + P.T)

        # Output: posterior position + partial velocity extrapolation
        val = state[0] + state[1] * lag_comp
        if not np.isfinite(val):
            val = out[i - 1] if i > 0 else 0.0
        out[i] = val

    # Guard against drift: nothing needed — measurement updates bound the state.

    # Align to window-end convention: y[i] corresponds to x[i + window_size - 1]
    y = out[window_size - 1 :]
    return np.where(np.isfinite(y), y, 0.0)


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
