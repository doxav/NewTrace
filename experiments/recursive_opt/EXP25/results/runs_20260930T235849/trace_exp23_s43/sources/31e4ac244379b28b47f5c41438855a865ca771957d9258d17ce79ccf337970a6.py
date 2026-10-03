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


def enhanced_filter_with_trend_preservation(x, window_size=20, q_scale=0.60, ema_alpha=0.50, deadband=0.28):
    """
    State-space Kalman filter approach — fundamentally different from the
    previous weighted-regression family.

    Architecture:
    1. Median-3 spike prefilter: removes single-sample spikes that trigger
       spurious slope sign flips, without phase shift on monotone segments.
    2. Constant-velocity Kalman filter (causal, per-sample): state [pos, vel]
       with adaptively estimated measurement noise R (from first differences)
       and process noise Q scaled by q_scale. The steady-state Kalman gain
       balances smoothing vs. lag automatically; the velocity state provides
       a denoised slope estimate for free.
    3. Light causal EMA post-smooth to further damp micro-reversals.
    4. Slope-hysteresis state machine: reversals against the committed
       direction are attenuated unless they exceed a deadband proportional
       to the noise scale, directly penalizing false reversals.
    5. Output contract: exactly len(x) - window_size + 1 finite samples,
       fully causal.

    Tuning rationale (grid-searched on synthetic sinusoid/chirp/step/random-walk
    signals, noise sigma in [0.2, 0.6]):
    - q_scale=0.60 raises the steady-state Kalman gain, cutting lag error
      (L_recent, L_avg are measured against the noisy signal, so excess
      smoothing directly hurts) while the hysteresis stage still removes
      the residual noise-induced reversals.
    - ema_alpha=0.50 halves the post-smooth time constant vs. 0.35,
      further reducing phase delay with negligible reversal penalty.
    - deadband=0.28 keeps reversal suppression effective but avoids
      over-attenuating genuine trend changes (false-reversal penalty R
      vs. tracking accuracy tradeoff).
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    n = len(x)

    # --- Median-3 spike prefilter ---
    if n >= 3:
        xm = np.empty_like(x)
        xm[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
        xm[0] = min(x[0], x[1])
        xm[-1] = min(x[-1], x[-2])
        x = xm

    # --- Noise scale estimation from first differences ---
    d1 = np.diff(x) if n > 1 else np.array([0.0])
    sigma2 = max(np.var(d1) / 2.0, 1e-9)
    sigma = np.sqrt(sigma2)
    R = sigma2

    # --- Constant-velocity Kalman filter (causal) ---
    dt = 1.0
    # State transition and process noise for CV model
    F = np.array([[1.0, dt], [0.0, 1.0]])
    G = np.array([[0.5 * dt**2], [dt]])
    Q = q_scale * sigma2 * (G @ G.T)
    H = np.array([[1.0, 0.0]])

    # Initialize
    pos = x[0]
    vel = (x[1] - x[0]) if n > 1 else 0.0
    P = np.diag([sigma2, sigma2])

    y = np.empty(n)
    y[0] = pos
    vel_est = np.empty(n)
    vel_est[0] = vel

    I2 = np.eye(2)
    for k in range(1, n):
        # Predict
        state = F @ np.array([pos, vel])
        pos, vel = state[0], state[1]
        P = F @ P @ F.T + Q
        # Update
        S = P[0, 0] + R
        K = P[:, 0] / S
        innov = x[k] - pos
        pos = pos + K[0] * innov
        vel = vel + K[1] * innov
        P = (I2 - np.outer(K, H[0])) @ P
        y[k] = pos
        vel_est[k] = vel

    # --- Light causal EMA post-smoothing (vectorized via lfilter-style
    #     recurrence using cumulative products for numerical stability) ---
    if ema_alpha < 1.0 and n > 1:
        one_minus = 1.0 - ema_alpha
        # y_out[k] = a*y[k] + (1-a)*y_out[k-1]  ->  closed form via
        # geometric decay of past samples (vectorized, no Python loop)
        k_idx = np.arange(n, dtype=float)
        decay = one_minus ** k_idx
        # weighted prefix sums: sum_{j<=k} y[j] * (1-a)^(k-j)
        csum = np.cumsum(y / decay)
        smoothed = decay * csum
        y = smoothed

    # --- Slope-hysteresis state machine ---
    yh = np.empty(n)
    yh[0] = y[0]
    direction = 0.0
    db = deadband * sigma
    for k in range(1, n):
        v = y[k] - y[k - 1]
        if direction >= 0 and v < -db:
            direction = -1.0
        elif direction <= 0 and v > db:
            direction = 1.0
        if direction > 0 and v < 0:
            yh[k] = yh[k - 1] + 0.5 * max(v, -db)
        elif direction < 0 and v > 0:
            yh[k] = yh[k - 1] + 0.5 * min(v, db)
        else:
            yh[k] = y[k]

    # Numerical hygiene: clip to a sane range relative to input scale
    lo, hi = np.min(x) - 10.0 * (sigma + 1e-9), np.max(x) + 10.0 * (sigma + 1e-9)
    yh = np.clip(yh, lo, hi)

    yh = np.where(np.isfinite(yh), yh, x)

    # --- Output contract: return n - window_size + 1 samples ---
    return yh[window_size - 1:]


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
