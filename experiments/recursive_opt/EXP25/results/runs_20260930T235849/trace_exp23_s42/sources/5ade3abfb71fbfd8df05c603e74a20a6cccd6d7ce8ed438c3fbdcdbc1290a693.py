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
    Bayesian state-space filtering: constant-velocity Kalman filter with
    adaptively inflated process noise.

    Fundamentally different mechanism from exponential smoothing: the filter
    maintains a full Gaussian posterior over [position, velocity] and fuses
    each measurement with the predicted state via the Kalman gain, which is
    the MMSE-optimal recursive estimator for linear-Gaussian models.

    - Robust measurement-noise adaptation: R is scaled by the innovation
      magnitude (Huber-style), so outliers/spikes are down-weighted instead
      of being tracked (fewer false reversals, no ringing on steps).
    - Process-noise inflation on persistent innovations lets the filter
      re-lock onto genuine trend/step changes within 2-3 samples (low lag).
    - Velocity-state output with a small steady-state lag correction; the
      Kalman steady-state gain on a CV model has group delay ~1 sample,
      far below window-based smoothers.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (used for alignment)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Robust measurement noise estimate from first differences (MAD-based)
    d = np.abs(np.diff(x))
    sigma = 1.4826 * np.median(d) / np.sqrt(2.0) if len(d) > 0 else 1.0
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = 1e-9

    # --- Constant-velocity Kalman filter (hand-rolled, scalar-friendly) ---
    # State: [position, velocity]; measurement: position
    R0 = sigma ** 2          # baseline measurement noise variance
    q = (0.05 * sigma) ** 2  # baseline process noise (acceleration variance)
    q_boost = (0.60 * sigma) ** 2  # inflated process noise during re-lock

    # State estimate and covariance
    pos = x[0]
    vel = 0.0
    P11, P12, P22 = R0 * 10.0, 0.0, 1.0

    out = np.empty(n)
    boost = 0.0

    for i in range(n):
        # --- Predict ---
        pos_pred = pos + vel
        F11, F12 = 1.0, 1.0
        Pp11 = P11 + 2.0 * P12 + P22 + q * (1.0 + 120.0 * boost)
        Pp12 = P12 + P22
        Pp22 = P22 + q * (0.5 + 60.0 * boost)

        # --- Innovation with Huber-style robustification ---
        innov = x[i] - pos_pred
        s2 = Pp11 + R0
        nis2 = innov * innov / s2
        # Adaptive measurement noise: large innovations are treated as
        # outliers (R inflated) unless they persist (process noise boosted).
        if nis2 > 9.0:
            R_eff = R0 * (nis2 / 9.0)
            boost = min(1.0, boost + 0.5)
        else:
            R_eff = R0
            boost *= 0.7

        s2 = Pp11 + R_eff
        K1 = Pp11 / s2
        K2 = Pp12 / s2

        # --- Update ---
        pos = pos_pred + K1 * innov
        vel = vel + K2 * innov
        P11 = (1.0 - K1) * Pp11
        P12 = (1.0 - K1) * Pp12
        P22 = Pp22 - K2 * Pp12

        if not np.isfinite(pos):
            pos = out[i - 1] if i > 0 else 0.0
            vel = 0.0
        out[i] = pos

    # Align to window-end convention: y[i] corresponds to x[i + window_size - 1]
    seg = out[window_size - 1 :].copy()

    # Steady-state lag correction: CV Kalman has ~1-sample group delay on
    # trends; correct with the velocity state (causal, no future data).
    vel_seg = np.empty_like(seg)
    vel_seg[0] = 0.0
    vel_seg[1:] = np.diff(out[window_size - 2 : -1]) if n > window_size else 0.0
    y = seg + 0.9 * vel_seg

    # Light causal median-of-3 to kill residual single-sample spikes
    if len(y) >= 3:
        ym = y.copy()
        ym[2:] = np.median(np.vstack([y[:-2], y[1:-1], y[2:]]), axis=0)
        y = ym

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
