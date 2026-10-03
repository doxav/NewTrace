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


def enhanced_filter_with_trend_preservation(x, window_size=20, fc_min=0.9, beta=0.35,
                                            d_cutoff=1.2, deadband_scale=0.45, hold=0.4):
    """
    One-Euro adaptive-cutoff filter (fundamentally different family from the
    previous weighted-regression approach).

    Architecture:
    1. Causal median-3 spike prefilter (kills single-sample spikes that
       trigger spurious slope sign flips; no phase shift on monotone runs).
    2. One-Euro filter (Casiez et al.): a first-order low-pass whose cutoff
       frequency ADAPTS to the magnitude of the (also low-pass filtered)
       derivative. Slow/noisy motion -> very low cutoff (heavy smoothing,
       minimizes spurious reversals S and R); fast motion -> cutoff opens
       (minimal lag, preserves genuine trend dynamics L_recent/L_avg).
       This is the canonical adaptive lag/smoothness tradeoff.
    3. Derivative-hysteresis state machine: once a direction is committed,
       counter-moves smaller than a robust noise-scale deadband are damped
       (partial step toward the filter output), eliminating noise-induced
       false reversals without delaying confirmed trend changes.
    4. Strictly causal per-sample; output[i] corresponds to
       x[i + window_size - 1] to satisfy the sliding-window contract.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # --- Causal median-3 spike prefilter ---
    if n >= 3:
        xm = np.empty_like(x)
        xm[0] = x[0]
        xm[1] = min(x[0], x[1])
        xm[2:] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
        x = xm

    # --- Robust noise scale from first differences ---
    dx = np.diff(x) if n > 1 else np.array([0.0])
    sigma = 1.4826 * np.median(np.abs(dx - np.median(dx))) / np.sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = max(np.std(dx) / np.sqrt(2.0), 1e-6)

    # --- One-Euro filter (causal, per-sample) ---
    def alpha_from_cutoff(fc):
        te = 1.0
        tau = 1.0 / (2.0 * np.pi * fc)
        return 1.0 / (1.0 + tau / te)

    a_d = alpha_from_cutoff(d_cutoff)
    dx_hat_prev = 0.0
    x_hat_prev = x[0]
    y = np.empty(n)
    y[0] = x[0]
    d_hat = np.empty(n)
    d_hat[0] = 0.0

    for k in range(1, n):
        d_raw = x[k] - x_hat_prev
        dx_hat = a_d * d_raw + (1.0 - a_d) * dx_hat_prev
        cutoff = fc_min + beta * abs(dx_hat)
        a = alpha_from_cutoff(cutoff)
        x_hat = a * x[k] + (1.0 - a) * x_hat_prev
        y[k] = x_hat
        d_hat[k] = dx_hat
        x_hat_prev = x_hat
        dx_hat_prev = dx_hat

    y = np.where(np.isfinite(y), y, x)
    d_hat = np.where(np.isfinite(d_hat), d_hat, 0.0)

    # --- Derivative hysteresis state machine ---
    db = deadband_scale * sigma + 1e-12
    yh = np.empty(n)
    yh[0] = y[0]
    direction = 0.0
    for k in range(1, n):
        v = y[k] - y[k - 1]
        if direction >= 0 and v < -db:
            direction = -1.0
        elif direction <= 0 and v > db:
            direction = 1.0
        if direction > 0 and v < 0:
            # counter-trend micro-move: damp, don't reverse
            yh[k] = yh[k - 1] + hold * max(v, -db)
        elif direction < 0 and v > 0:
            yh[k] = yh[k - 1] + hold * min(v, db)
        else:
            yh[k] = y[k]
    yh = np.where(np.isfinite(yh), yh, y)

    # --- Output contract: n - window_size + 1 samples ---
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
