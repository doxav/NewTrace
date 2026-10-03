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

    - State: [level, slope], transition F = [[1,1],[0,1]], measurement H = [1,0].
    - Measurement noise R is estimated robustly (MAD of first differences).
    - Process noise Q is small in steady state (heavy noise suppression, few
      slope reversals) and inflated when the normalized innovation exceeds a
      gate, giving fast re-tracking of steps/trend changes with minimal lag.
    - The Kalman gain is the Bayesian-optimal blend of prediction and
      measurement, so lag error is inherently low without ad-hoc forward
      extrapolation; a small slope-based lead term fine-tunes phase.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (used for lead compensation)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Robust measurement-noise estimate from first differences:
    # Var(diff) = 2*sigma_meas^2 for white noise, so sigma^2 = MAD(diff)^2 / 2.
    d = np.diff(x)
    mad = 1.4826 * np.median(np.abs(d)) if len(d) > 0 else 0.3
    if not np.isfinite(mad) or mad <= 1e-9:
        mad = 1e-9
    R0 = 0.5 * mad * mad

    # Kalman state and covariance
    state = np.array([x[0], 0.0])          # [level, slope]
    P = np.diag([R0, (mad / window_size) ** 2])

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    I = np.eye(2)

    # Baseline process noise (steady state): small -> smooth, reversal-sparse.
    q_level = 0.02 * mad * mad
    q_slope = 2e-5 * mad * mad
    Q_base = np.array([[q_level, 0.0], [0.0, q_slope]])
    # Boosted process noise for genuine change points (fast re-tracking).
    Q_boost = np.array([[4.0 * q_level, 0.0], [0.0, 400.0 * q_slope]])

    gate_hi = 3.0    # inflate Q above this normalized innovation
    gate_lo = 1.5    # deflate back below this (hysteresis)
    boost = 0.0      # 0..1 smoothed gate state

    lead = 0.5       # slope lead term (samples) for residual phase alignment

    out = np.empty(n)
    for i in range(n):
        # Predict
        state = F @ state
        P = F @ P @ F.T + (Q_boost if boost > 0.5 else Q_base)

        # Innovation and normalized gate with hysteresis
        z = x[i]
        innov = z - float(H @ state)
        S = float(H @ P @ H.T) + R0
        nis = abs(innov) / np.sqrt(max(S, 1e-12))
        if nis > gate_hi:
            boost = 1.0
        elif nis < gate_lo:
            boost *= 0.7

        # If boosting, re-predict with inflated Q for fast step response
        if boost > 0.5:
            P = F @ (F @ P @ F.T) * 0.0 + F @ P @ F.T  # keep covariance
            P = P + Q_boost
            innov = z - float(H @ state)

        # Update
        K = (P @ H.T) / S  # 2x1 gain
        state = state + (K * innov).ravel()
        P = (I - K @ H) @ P
        P = 0.5 * (P + P.T)  # symmetrize

        val = state[0] + state[1] * lead
        if not np.isfinite(val):
            val = out[i - 1] if i > 0 else 0.0
        out[i] = val

    # Align to window-end convention: y[i] corresponds to x[i + window_size - 1]
    seg = out[window_size - 1 :].copy()

    # --- Light causal post-processing ---
    # Causal median-of-3: kills residual single-sample spikes (false reversals)
    # with zero phase shift.
    m = seg.copy()
    if len(seg) >= 3:
        m[2:] = np.median(np.vstack([seg[:-2], seg[1:-1], seg[2:]]), axis=0)

    # Directional hysteresis: reject sub-threshold reversals of the output's
    # first difference (pointwise, zero-lag modification).
    dy = np.diff(m)
    y = np.empty_like(m)
    y[0] = m[0]
    if len(dy) > 0:
        thr = 1.4826 * np.median(np.abs(dy - np.median(dy)))
        if not np.isfinite(thr) or thr <= 1e-12:
            thr = 1e-12
        direction = 0
        for j in range(1, len(m)):
            dj = m[j] - y[j - 1]
            s = 1 if dj > 0 else (-1 if dj < 0 else 0)
            if s != 0 and s != direction and abs(dj) < 0.4 * thr:
                dj = 0.0  # reject noise-induced reversal
            elif s != 0 and s != direction:
                direction = s
            y[j] = y[j - 1] + dj
    else:
        y[:] = m

    # Final NaN/inf guard
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
