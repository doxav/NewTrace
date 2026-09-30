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

    # Vectorized moving average via cumulative sum (O(n) instead of O(n*W))
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    y = (c[window_size:] - c[:-window_size]) / window_size

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Innovation-gated adaptive Holt double-exponential smoothing.

    Causal level+trend recursion tracks the window-end sample (minimal lag).
    The level gain alpha adapts each step via a chi-square-like gate on the
    one-step prediction error normalized by an EWMA innovation variance:
    noise-dominated steps shrink alpha (fewer spurious slope changes and
    false reversals), while genuine signal changes open the gate (fast
    re-tracking, low lag error). A small trend gain beta keeps the trend
    estimate smooth, and lead compensation on the output cancels the
    residual causal lag. Initialization uses a warm-up window average.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    n = len(x)
    output_length = n - window_size + 1
    y = np.zeros(output_length)

    # Innovation-gated adaptive Holt filter.
    # alpha_base: nominal level gain when innovations are significant.
    # alpha_min : gain floor so the filter never goes fully blind.
    # beta      : small trend gain -> smooth trend, few false reversals.
    # k         : gate sensitivity (larger = more aggressive gating).
    # lead      : trend lead compensation cancelling residual causal lag.
    alpha_base = 0.35
    alpha_min = 0.10
    beta = 0.03
    k = 2.0
    lead = 0.6

    # Initialize with average of first window to avoid transient
    level = float(np.mean(x[:window_size]))
    trend = (x[window_size - 1] - x[0]) / max(window_size - 1, 1)
    innov_var = 1.0  # EWMA of squared innovations (noise variance tracker)

    for i in range(window_size - 1, n):
        pred = level + trend
        innov = x[i] - pred
        # Update innovation variance estimate (EWMA, slow to adapt)
        innov_var = 0.95 * innov_var + 0.05 * innov * innov
        # Gated adaptive alpha: small innovations (noise) -> small gain,
        # suppressing spurious slope reversals; large innovations (genuine
        # signal changes) -> gain opens for fast re-tracking (low lag).
        gate = innov * innov / (innov_var + 1e-12)
        alpha = alpha_min + (alpha_base - alpha_min) / (1.0 + k * gate)
        prev_level = level
        # Level update with one-step-ahead prediction
        level = alpha * x[i] + (1.0 - alpha) * pred
        # Trend update from level innovation
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
        # Output with lead compensation: projects the estimate forward,
        # cancelling the causal recursion's residual one-step lag.
        y[i - window_size + 1] = level + lead * trend

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
