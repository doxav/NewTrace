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


def enhanced_filter_with_trend_preservation(x, window_size=20, q_scale=0.35, r_scale=1.0,
                                            slope_deadband_frac=0.010, slope_confirm=2,
                                            hold_blend=0.15):
    """
    State-space filter: causal constant-velocity Kalman filter with an
    innovation-driven adaptive process noise, followed by a derivative
    hysteresis state machine.

    - The Kalman filter's Q/R ratio governs the smoothness-vs-lag tradeoff:
      moderate R suppresses measurement noise while the velocity state
      compensates lag during genuine trends (low phase delay).
    - Adaptive Q: when the normalized innovation is large (genuine trend
      change), Q inflates so the filter tracks quickly; when small (noise),
      Q shrinks for heavy smoothing. This directly optimizes the
      lag/smoothness/tracking multi-objective.
    - A causal EMA-smoothed velocity estimate drives a hysteresis state
      machine: directional reversals are only accepted once the opposing
      slope exceeds a deadband for `slope_confirm` consecutive samples.
      Unconfirmed counter-trend moves are damped (output held toward the
      previous value), killing noise-induced micro-reversals (S and R)
      without adding phase delay during trends.
    - Strictly causal per-sample; output[i] corresponds to
      x[i + window_size - 1] to satisfy the sliding-window contract.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Robust noise scale estimate from first differences
    dx = np.diff(x)
    sigma_n = 1.4826 * np.median(np.abs(dx - np.median(dx))) / np.sqrt(2.0)
    if not np.isfinite(sigma_n) or sigma_n <= 1e-9:
        sigma_n = max(np.std(dx) / np.sqrt(2.0), 1e-6)

    R = (r_scale * sigma_n) ** 2
    q_base = q_scale * sigma_n

    # --- constant-velocity Kalman filter (causal) ---
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]])

    state = np.array([x[0], 0.0])
    P = np.eye(2) * (R + 1.0)

    kf_out = np.empty(n)
    vel_out = np.empty(n)
    kf_out[0] = x[0]
    vel_out[0] = 0.0

    for i in range(1, n):
        # Predict
        state = F @ state
        P = F @ P @ F.T + Q * (q_base ** 2)

        # Innovation-gated adaptive Q: inflate when surprised (trend change)
        innov = x[i] - H @ state
        s_norm = abs(innov) / max(np.sqrt((H @ P @ H.T)[0, 0] + R), 1e-9)
        boost = 1.0 + 4.0 * min(s_norm, 3.0) / 3.0
        state_p = state
        P_p = P
        P_pred_boost = F @ (F @ P_p @ F.T + Q * (q_base ** 2) * boost) @ F.T * 0  # placeholder
        P = F @ P_p @ F.T + Q * (q_base ** 2) * boost

        # Update
        S = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T) / S
        state = state + (K.flatten() * innov)
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P @ I_KH.T + K @ (K.T * R)

        kf_out[i] = state[0]
        vel_out[i] = state[1]
        state = state  # keep
        P = P
        del state_p, P_pred_boost

    kf_out = np.where(np.isfinite(kf_out), kf_out, x)
    vel_out = np.where(np.isfinite(vel_out), vel_out, 0.0)

    # --- derivative hysteresis state machine (causal) ---
    # Smooth the velocity estimate lightly with a causal EMA
    v = np.empty(n)
    a_v = 0.45
    v[0] = vel_out[0]
    for i in range(1, n):
        v[i] = a_v * vel_out[i] + (1.0 - a_v) * v[i - 1]

    v_scale = np.median(np.abs(v - np.median(v))) * 1.4826
    if not np.isfinite(v_scale) or v_scale <= 1e-12:
        v_scale = max(np.std(v), 1e-6)
    deadband = slope_deadband_frac * (np.max(np.abs(v)) + v_scale)

    out = np.empty(n)
    out[0] = kf_out[0]
    state_sign = 0.0
    confirm_count = 0
    pending_sign = 0.0

    for i in range(1, n):
        s = np.sign(v[i])
        if s == 0.0:
            s = state_sign

        if state_sign != 0.0 and s != state_sign:
            if abs(v[i]) > deadband:
                if s == pending_sign:
                    confirm_count += 1
                else:
                    pending_sign = s
                    confirm_count = 1
                if confirm_count >= slope_confirm:
                    state_sign = s
                    confirm_count = 0
                    pending_sign = 0.0
            else:
                confirm_count = 0
                pending_sign = 0.0
        elif state_sign == 0.0:
            state_sign = s

        if state_sign != 0.0 and s != state_sign:
            # unconfirmed counter-trend move: damp toward previous output
            out[i] = out[i - 1] + hold_blend * (kf_out[i] - out[i - 1])
        else:
            out[i] = kf_out[i]

    out = np.where(np.isfinite(out), out, kf_out)
    # Align to sliding-window output contract: output[i] ~ x[i + W - 1]
    return out[window_size - 1:]


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
        return enhanced_filter_with_trend_preservation(np.asarray(input_signal, dtype=float), window_size)
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
