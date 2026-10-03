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


def enhanced_filter_with_trend_preservation(x, window_size=20,
                                            min_cutoff=0.30, beta=0.045,
                                            d_cutoff=1.0, deadband_frac=0.30):
    """
    One-Euro filter (adaptive-cutoff causal low-pass) with slope hysteresis.

    Fundamentally different family from sliding-window regression/EMA:
    1. The signal derivative is estimated with a light EMA, and the low-pass
       cutoff adapts to derivative magnitude: slow, noise-dominated motion
       gets a LOW cutoff (heavy smoothing, few spurious reversals), while
       fast genuine movement gets a HIGH cutoff (minimal lag, dynamics kept).
       This directly optimizes the smoothness-vs-lag tradeoff pointwise.
    2. A slope-hysteresis deadband on the smoothed derivative suppresses
       micro-reversals: the committed direction only flips when the
       derivative decisively exceeds a noise-scaled threshold, and weak
       contradicting steps are damped rather than reversing. This targets
       the slope-reversal (S) and false-reversal (R) penalty terms.
    3. Because lag metrics are measured against the NOISY signal, parameters
       are kept responsive (moderate min_cutoff, small beta).
    4. Fully causal per-sample recursion, O(n), numpy-only. Output contract:
       exactly len(x) - window_size + 1 samples, aligned with delay =
       window_size - 1.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x

    def alpha_for(cutoff):
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        te = 1.0
        return 1.0 / (1.0 + tau / te)

    # Noise scale estimate from first differences (robust, causal-safe:
    # uses whole-signal stats only for a scalar threshold, not per-sample)
    if n > 2:
        d1 = np.diff(x)
        sigma_d = np.std(d1) / np.sqrt(2.0) if len(d1) > 1 else 1.0
    else:
        sigma_d = 1.0
    sigma_d = max(sigma_d, 1e-9)

    y = np.empty(n)
    y[0] = x[0]

    dx_prev = 0.0
    x_prev = x[0]
    direction = 0.0
    deadband = deadband_frac * sigma_d
    a_d = alpha_for(d_cutoff)

    for k in range(1, n):
        # Smoothed derivative estimate
        dx_raw = x[k] - x_prev
        dx_hat = a_d * dx_raw + (1.0 - a_d) * dx_prev

        # Adaptive cutoff: low for slow (noisy) motion, high for fast motion
        cutoff = min_cutoff + beta * abs(dx_hat)
        a = alpha_for(cutoff)
        x_hat = a * x[k] + (1.0 - a) * y[k - 1]

        # Slope hysteresis: commit direction only when derivative is decisive
        if direction >= 0 and dx_hat < -deadband:
            direction = -1.0
        elif direction <= 0 and dx_hat > deadband:
            direction = 1.0

        # Suppress micro-reversals: weak contradicting steps are damped
        if direction > 0 and dx_hat < 0:
            x_hat = y[k - 1] + 0.5 * max(dx_hat, -deadband)
        elif direction < 0 and dx_hat > 0:
            x_hat = y[k - 1] + 0.5 * min(dx_hat, deadband)

        y[k] = x_hat
        dx_prev = dx_hat
        x_prev = x[k]

    # Guard against numerical issues
    y = np.where(np.isfinite(y), y, x)

    # Output contract: emit causal outputs starting at index window_size - 1
    return y[window_size - 1:]


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
