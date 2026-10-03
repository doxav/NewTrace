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
    Adaptive constant-velocity Kalman filter with innovation gating.

    Fundamentally different mechanism from polynomial regression / exponential
    smoothing: a Bayesian state-space filter with state [level, velocity].

    - Constant-velocity model: the velocity state tracks genuine trends
      explicitly; the Kalman gain optimally balances model prediction against
      measurement, giving low lag error while suppressing noise.
    - Innovation-gated adaptive process noise: in steady state, low Q gives
      heavy smoothing (few slope reversals); when the normalized innovation
      exceeds a Mahalanobis gate (genuine step/trend change), Q is boosted so
      the filter re-tracks quickly instead of ringing. Hysteresis on the gate
      prevents gain chatter.
    - Robust measurement-noise estimate from MAD of first differences adapts
      to non-stationary noise levels.
    - Causal median-of-3 prefilter kills single-sample spikes; directional
      hysteresis post-pass suppresses noise-induced false reversals.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Causal trailing median-of-3 prefilter: xp[i] = median(x[i-2], x[i-1], x[i])
    xp = x.copy()
    if n >= 3:
        xp[2:] = np.median(np.vstack([x[:-2], x[1:-1], x[2:]]), axis=0)

    # Robust measurement noise variance from first differences (MAD-based).
    # For white noise, var(diff) = 2*sigma^2.
    d = np.diff(xp)
    if len(d) > 0:
        sigma_d = 1.4826 * np.median(np.abs(d - np.median(d)))
    else:
        sigma_d = 0.3
    if sigma_d <= 1e-9:
        sigma_d = 1e-9
    R = (sigma_d ** 2) / 2.0

    # State: [level, velocity]; transition: level += velocity per sample
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    # Baseline process noise: small velocity diffusion -> smooth, few reversals
    q_base = 0.02 * R
    q_boost = 6.0 * R      # boosted during detected genuine changes
    q = q_base

    # State initialization from first few samples
    if n >= 4:
        v0 = (xp[3] - xp[0]) / 3.0
    else:
        v0 = 0.0
    state = np.array([xp[0], v0])
    P = np.diag([R, R])

    # Precompute steady-state-ish Kalman gain pieces per iteration (2x2, cheap)
    out = np.empty(n)
    boosted = 0.0

    for i in range(n):
        # --- Predict ---
        state = F @ state
        Q = np.array([[q / 3.0, q / 2.0], [q / 2.0, q]])
        P = F @ P @ F.T + Q

        # --- Innovation gate with hysteresis (adaptive process noise) ---
        innov = xp[i] - state[0]
        S = P[0, 0] + R
        nis = (innov * innov) / max(S, 1e-12)
        if nis > 9.0:          # ~3-sigma Mahalanobis gate: genuine change
            boosted = 1.0
        elif nis < 3.5:
            boosted *= 0.75
        q_eff = q_base + (q_boost - q_base) * boosted
        Q = np.array([[q_eff / 3.0, q_eff / 2.0], [q_eff / 2.0, q_eff]])
        # Rebuild prediction covariance with the effective Q
        P = F @ (P - Q) @ F.T + Q

        # --- Update ---
        K = (P @ H.T) / max(P[0, 0] + R, 1e-12)   # 2x1 gain
        state = state + (K * innov).ravel()
        I_KH = np.eye(2) - K @ H
        P = I_KH @ P @ I_KH.T + K @ (R * np.eye(2)) @ K.T  # Joseph form

        val = state[0] + 0.35 * state[1]  # slight forward extrapolation
        if not np.isfinite(val):
            val = out[i - 1] if i > 0 else xp[0]
        out[i] = val

    # Align to window-end convention: y[i] corresponds to x[i + window_size - 1]
    seg = out[window_size - 1 :].copy()

    # Causal median-of-3 on output: removes residual single-sample spikes
    m = seg.copy()
    if len(seg) >= 3:
        m[2:] = np.median(np.vstack([seg[:-2], seg[1:-1], seg[2:]]), axis=0)

    # Light causal 3-tap binomial smoothing (negligible added delay)
    y = m.copy()
    if len(m) >= 3:
        y[2:] = 0.6 * m[2:] + 0.3 * m[1:-1] + 0.1 * m[:-2]

    # Directional hysteresis: reversals must persist 2 samples and exceed a
    # robust threshold; otherwise hold previous level (kills false reversals).
    dy = np.diff(y)
    if len(dy) > 0:
        thr = 1.4826 * np.median(np.abs(dy - np.median(dy)))
        if thr <= 1e-12:
            thr = 1e-12
        y2 = y.copy()
        dirn = 1 if dy[0] >= 0 else -1
        run = 0
        run_dir = 0
        for j in range(1, len(y)):
            dd = y[j] - y2[j - 1]
            jd = 1 if dd >= 0 else -1
            if jd != dirn:
                if jd == run_dir:
                    run += 1
                else:
                    run_dir = jd
                    run = 1
                need = 0.6 * thr if run >= 3 else 0.45 * thr
                if run >= 2 and abs(dd) > need:
                    dirn = jd
                    run = 0
                else:
                    y2[j] = y2[j - 1]
            else:
                run = 0
                run_dir = 0
        y = y2

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
