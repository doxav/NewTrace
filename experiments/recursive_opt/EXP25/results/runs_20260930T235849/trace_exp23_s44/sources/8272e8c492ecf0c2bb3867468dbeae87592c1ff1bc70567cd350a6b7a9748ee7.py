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


def enhanced_filter_with_trend_preservation(x, window_size=20, q_scale=0.02, gate_k=4.0, hyst_scale=0.12):
    """
    Adaptive constant-velocity Kalman filter with innovation gating and
    output hysteresis (a fundamentally different algorithm class from the
    previous polynomial/EMA smoother).

    - State: [level, velocity]; measurement: current sample.
    - Process noise q_scale adapts to the estimated noise level, balancing
      smoothing (low S, R terms) against tracking lag (L terms).
    - Innovation gating: when the measurement deviates from prediction by
      more than gate_k * innovation-std (e.g. a step change), the measurement
      noise is temporarily inflated then trust restored quickly, so
      discontinuities are tracked without large lag spikes.
    - Velocity is lightly damped to reduce spurious directional reversals.
    - Micro-hysteresis deadband on output increments suppresses
      noise-induced slope sign flips.

    The Kalman state at sample index i corresponds to the evaluator's
    aligned index (i - (window_size - 1) delay convention), so the state
    sequence is emitted directly with the required output length.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (used for output length)
        q_scale: Process noise scale (velocity random-walk intensity)
        gate_k: Innovation gating threshold in units of innovation std
        hyst_scale: Hysteresis deadband as a fraction of estimated noise

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    output_length = len(x) - window_size + 1

    # Robust noise sigma estimate from successive diffs (MAD-based)
    if len(x) > 10:
        diffs = np.abs(np.diff(x))
        sigma_est = np.median(diffs) / 0.9535 / np.sqrt(2.0) + 1e-12
    else:
        sigma_est = 0.3

    r = sigma_est ** 2 + 1e-9          # measurement noise variance
    q = q_scale * r                     # velocity process noise variance

    # Constant-velocity transition and measurement matrices
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[r * 0.01, 0.0], [0.0, q]])
    R = np.array([[r]])

    # Initialize from first window via simple least squares (level, slope)
    w0 = x[:window_size]
    t0 = np.arange(window_size, dtype=float)
    tm = t0.mean()
    slope0 = np.sum((t0 - tm) * (w0 - w0.mean())) / (np.sum((t0 - tm) ** 2) + 1e-12)
    state = np.array([w0[-1], slope0])
    # Initial covariance
    P = np.array([[r, 0.0], [0.0, q * 10.0]])

    # Identity for gain computation
    I2 = np.eye(2)

    # Warm-up: run the filter over the first window to reach steady state,
    # emitting nothing (those samples are consumed by the delay alignment).
    for i in range(window_size):
        state = F @ state
        P = F @ P @ F.T + Q
        innov = x[i] - (H @ state)[0]
        S = (H @ P @ H.T)[0, 0] + r
        K = (P @ H.T) / S
        state = state + (K * innov).ravel()
        P = (I2 - K @ H) @ P

    y = np.empty(output_length)
    y[0] = state[0]

    # Main causal Kalman recursion with innovation gating.
    for i in range(window_size, len(x)):
        # Predict
        state = F @ state
        # Light velocity damping: reduces overshoot and spurious reversals
        state[1] *= 0.995
        P = F @ P @ F.T + Q

        # Innovation and gated measurement update
        innov = x[i] - (H @ state)[0]
        S = (H @ P @ H.T)[0, 0] + r
        # Gate: large innovations (step changes / outliers) temporarily
        # inflate measurement variance so the filter follows them within a
        # few samples instead of rejecting or lagging badly.
        if innov * innov > gate_k * gate_k * S:
            r_eff = r * (innov * innov) / (gate_k * gate_k * S)
            S = (H @ P @ H.T)[0, 0] + r_eff
            K = (P @ H.T) / S
            state = state + (K * innov).ravel()
            P = (I2 - K @ H) @ P
            # After a gated jump, boost velocity covariance so the filter
            # quickly re-locks onto the new regime.
            P[1, 1] += q * 20.0
        else:
            K = (P @ H.T) / S
            state = state + (K * innov).ravel()
            P = (I2 - K @ H) @ P

        y[i - window_size + 1] = state[0]

    # Micro-hysteresis deadband on output increments: kills near-zero
    # diff chatter (spurious slope sign flips) while being invisible to
    # genuine dynamics. Threshold scales with estimated noise.
    out = np.empty_like(y)
    out[0] = y[0]
    prev = y[0]
    db = min(max(hyst_scale * sigma_est, 1e-5), 0.15)
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
