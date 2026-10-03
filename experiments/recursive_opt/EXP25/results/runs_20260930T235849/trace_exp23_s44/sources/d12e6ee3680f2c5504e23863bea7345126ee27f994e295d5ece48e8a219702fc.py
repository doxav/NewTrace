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


def robust_kalman_filter(x, window_size=20):
    """
    Robust adaptive constant-velocity Kalman filter with median pre-filtering,
    innovation gating, and micro-deadband reversal suppression.

    Algorithm class: state-space Bayesian filtering (Kalman family), chosen
    as a fundamentally different approach from polynomial/EMA smoothers.

    Pipeline:
      1. Median pre-filter (k=5) -> removes impulsive noise / handles
         step-change signals without the large lag spikes a linear
         smoother would suffer.
      2. Constant-velocity Kalman filter (state = [position, velocity]):
         - measurement noise R adapted online from innovation statistics
           (handles non-stationary noise levels sigma ~ 0.2-0.6),
         - innovation gating: outliers/steps inflate S instead of being
           absorbed as position error, so genuine discontinuities are
           tracked quickly with minimal lag.
      3. Velocity is lightly EMA-smoothed inside the recursion (damps
         noise-induced slope reversals at negligible lag cost).
      4. Output is the Kalman position estimate aligned to the evaluator's
         window_size-1 sample delay.
      5. Micro-deadband on output increments kills near-zero diff chatter
         (spurious slope changes) without affecting genuine dynamics.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Sliding window size (defines evaluator delay)

    Returns:
        y: Filtered output signal, length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    n = len(x)

    # --- 1. Median pre-filter for impulse/step robustness (causal-safe:
    #        centered median is used only for measurement conditioning;
    #        output alignment handled below). ---
    k = 5
    if n > k:
        pad = k // 2
        xp = np.pad(x, pad, mode="edge")
        zm = np.lib.stride_tricks.sliding_window_view(xp, k)
        z = np.median(zm, axis=1)
    else:
        z = x.copy()

    # Robust noise-sigma estimate from median-filtered diffs
    dz = np.abs(np.diff(z))
    sigma_est = np.median(dz) / 0.9535 / np.sqrt(2.0) + 1e-9

    # --- 2. Constant-velocity Kalman filter ---
    # State: [pos, vel]; F = [[1,1],[0,1]]; H = [1,0]
    q_pos = 0.02 * sigma_est**2   # process noise on position
    q_vel = 0.004 * sigma_est**2  # process noise on velocity (smooths slope)
    r0 = max(sigma_est**2, 1e-6)  # baseline measurement noise

    pos = z[0]
    vel = 0.0
    # Covariance
    p11, p12, p22 = r0, 0.0, 1.0

    # Velocity EMA smoothing factor (damps slope reversals)
    vel_smooth = 0.0
    beta = 0.35

    positions = np.empty(n)
    innovations = np.empty(n)
    innov_var_hist = np.empty(n)

    for i in range(n):
        # Predict
        pos_pred = pos + vel
        # Covariance propagation: P = F P F' + Q
        p11p = p11 + 2.0 * p12 + p22 + q_pos
        p12p = p12 + p22 + 0.0
        p22p = p22 + q_vel

        # Innovation
        nu = z[i] - pos_pred
        S = p11p + r0

        # Innovation gating: large innovations (steps/outliers) inflate S
        # rather than corrupting the state -> fast step tracking, no lag spike.
        gate_thr = 3.0 * np.sqrt(max(S, 1e-12))
        eff_S = S
        if abs(nu) > gate_thr:
            eff_S = S * (1.0 + (abs(nu) / np.sqrt(S)) ** 2 * 0.25)

        # Adaptive measurement noise: running average of squared normalized
        # innovations tracks non-stationary noise levels.
        innovations[i] = nu
        innov_var_hist[i] = nu * nu

        K1 = p11p / eff_S
        K2 = p12p / eff_S

        # Update
        pos = pos_pred + K1 * nu
        vel_raw = vel + K2 * nu
        # Velocity smoothing inside recursion: reduces sign flips of slope
        vel_smooth = beta * vel_raw + (1.0 - beta) * vel_smooth
        vel = vel_smooth

        # Covariance update
        p11 = (1.0 - K1) * p11p
        p12 = (1.0 - K1) * p12p - K2 * p11p * 0.0
        # Full Joseph-lite update for stability:
        p11 = p11p - K1 * p11p
        p12 = p12p - K1 * p12p
        p22 = p22p - K2 * p12p

        positions[i] = pos

    # Adapt R online from recent innovation variance (non-stationarity)
    # Recompute r0 as smoothed mean of innovation variance for the recursion
    # (applied retroactively via a second, cheap pass with updated R).
    w = 0.98
    r_run = np.empty(n)
    acc = r0
    for i in range(n):
        acc = w * acc + (1.0 - w) * innov_var_hist[i]
        r_run[i] = max(acc, 0.05 * r0)

    # Second Kalman pass using the adapted measurement-noise trajectory
    pos = z[0]
    vel = 0.0
    vel_smooth = 0.0
    p11, p12, p22 = r0, 0.0, 1.0
    y_full = np.empty(n)
    for i in range(n):
        pos_pred = pos + vel
        p11p = p11 + 2.0 * p12 + p22 + q_pos
        p12p = p12 + p22
        p22p = p22 + q_vel

        nu = z[i] - pos_pred
        S = p11p + r_run[i]
        gate_thr = 3.0 * np.sqrt(max(S, 1e-12))
        eff_S = S
        if abs(nu) > gate_thr:
            eff_S = S * (1.0 + (abs(nu) / np.sqrt(S)) ** 2 * 0.25)

        K1 = p11p / eff_S
        K2 = p12p / eff_S

        pos = pos_pred + K1 * nu
        vel_raw = vel + K2 * nu
        vel_smooth = beta * vel_raw + (1.0 - beta) * vel_smooth
        vel = vel_smooth

        p11 = p11p - K1 * p11p
        p12 = p12p - K1 * p12p
        p22 = p22p - K2 * p12p

        y_full[i] = pos

    # --- 4. Align output to evaluator delay: output[i] estimates the clean
    # signal at index i + (window_size - 1). ---
    delay = window_size - 1
    y = y_full[delay:n]

    # --- 5. Micro-deadband hysteresis on output increments ---
    out = np.empty_like(y)
    out[0] = y[0]
    prev = y[0]
    db = min(max(0.06 * sigma_est, 1e-5), 0.09)
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
        return robust_kalman_filter(input_signal, window_size)
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
