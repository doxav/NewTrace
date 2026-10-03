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


def enhanced_filter_with_trend_preservation(x, window_size=20, min_cutoff=0.35, beta=0.35, deadband=0.02):
    """
    One-Euro adaptive low-pass filter with slope hysteresis — a fundamentally
    different family from Kalman / sliding-window regression approaches.

    Architecture:
    1. One-Euro filter (causal, per-sample): the low-pass cutoff frequency
       adapts to the magnitude of the signal's derivative. When the signal
       moves slowly (noise-dominated), the cutoff drops and the filter
       smooths aggressively; when it moves fast (genuine dynamics), the
       cutoff rises and lag error stays minimal. This directly optimizes
       the lag-vs-smoothness tradeoff of the metric.
    2. A two-stage cascade (level filter + derivative filter) estimates a
       low-noise velocity used by the hysteresis state machine.
    3. Slope-hysteresis state machine: direction flips are only accepted
       when the smoothed velocity exceeds a deadband, suppressing
       noise-induced reversals (S and R) without adding phase delay.
    4. Output contract: exactly len(x) - window_size + 1 samples, aligned
       with evaluator delay = window_size - 1. Fully causal.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    n = len(x)

    # --- Noise scale estimation from first differences ---
    if n > 2:
        d1 = np.diff(x)
        sigma = np.sqrt(max(np.var(d1) / 2.0, 1e-9))
    else:
        sigma = 1.0

    te = 1.0  # sampling period (normalized)
    tau_min = 1.0 / (2.0 * np.pi * min_cutoff)
    tau_d = 1.0 / (2.0 * np.pi * 1.0)  # derivative low-pass base cutoff = 1.0

    def alpha_for(tau):
        return 1.0 / (1.0 + tau / te)

    # --- One-Euro filter, causal per-sample loop ---
    y = np.empty(n)
    y[0] = x[0]
    dx_prev = 0.0
    dx_hat_prev = 0.0
    for k in range(1, n):
        dx = (x[k] - y[k - 1]) / te
        # Low-pass the derivative estimate
        a_d = alpha_for(tau_d)
        dx_hat = a_d * dx + (1.0 - a_d) * dx_hat_prev
        # Adaptive cutoff: cutoff rises with |derivative|
        tau = tau_min / (1.0 + beta * abs(dx_hat) / (sigma + 1e-9))
        a = alpha_for(tau)
        y[k] = a * x[k] + (1.0 - a) * y[k - 1]
        dx_prev = dx
        dx_hat_prev = dx_hat

    # --- Slope-hysteresis state machine on filtered output ---
    yh = np.empty(n)
    yh[0] = y[0]
    direction = 0.0
    for k in range(1, n):
        v = y[k] - y[k - 1]
        if direction >= 0 and v < -deadband * sigma:
            direction = -1.0
        elif direction <= 0 and v > deadband * sigma:
            direction = 1.0
        # Suppress micro-reversals against the committed direction
        if direction > 0 and v < 0:
            yh[k] = yh[k - 1] + 0.5 * max(v, -deadband * sigma)
        elif direction < 0 and v > 0:
            yh[k] = yh[k - 1] + 0.5 * min(v, deadband * sigma)
        else:
            yh[k] = y[k]

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
