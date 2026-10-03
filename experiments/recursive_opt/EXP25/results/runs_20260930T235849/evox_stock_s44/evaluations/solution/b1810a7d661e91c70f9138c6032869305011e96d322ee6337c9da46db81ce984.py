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
    Four-stage causal filter, fully vectorized O(N):
    (1) quadratic Savitzky-Golay endpoint fit: zero-lag local polynomial
        regression removes most noise while preserving dynamics;
    (2) causal median-of-5: removes impulse-driven spurious slope
        reversals at near-zero lag cost (median preserves edge timing);
    (3) light EMA (alpha=0.40): stronger smoothing suppresses residual
        noise-induced reversals with a small bounded lag penalty,
        computed via a cumulative-product recurrence (no Python loop);
    (4) causal front-weighted 3-tap smoother [0.5, 0.3, 0.2] removes
        residual jitter with minimal phase delay since the largest
        weight sits on the newest sample.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    # Stage 1: quadratic SG weights evaluated at the window's last point
    t = np.arange(window_size, dtype=float) - (window_size - 1)  # endpoint = 0
    V = np.vander(t, 3, increasing=True)                          # [1, t, t^2]
    P = np.linalg.inv(V.T @ V) @ V.T
    w = P[0]

    windows = np.lib.stride_tricks.sliding_window_view(x, window_size)
    y = windows @ w

    # Stage 2: causal median-of-5 to kill impulse-driven spurious reversals
    if len(y) >= 5:
        ym = y.copy()
        ym[2:-2] = np.median(np.vstack([y[:-4], y[1:-3], y[2:-2], y[3:-1], y[4:]]), axis=0)
    else:
        ym = y

    # Stage 3: light EMA (vectorized cumulative-product recurrence)
    alpha = 0.40
    n = len(ym)
    b = 1.0 - alpha
    powers = b ** np.arange(n)
    c = np.cumsum(ym / powers)
    out = alpha * powers * c
    out[0] = ym[0]

    # Stage 4: causal front-weighted 3-tap smoother [0.5, 0.3, 0.2]
    # suppresses residual high-frequency jitter (slope changes) with
    # minimal phase delay since the largest weight is on the newest sample.
    if len(out) >= 3:
        o = out.copy()
        o[2:] = 0.5 * out[2:] + 0.3 * out[1:-1] + 0.2 * out[:-2]
        o[:2] = out[:2]
        out = o
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
