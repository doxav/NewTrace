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


def enhanced_filter_with_trend_preservation(x, window_size=20, alpha=0.26, beta=0.13, smooth=0.40):
    """
    Noise-adaptive causal filter: median-of-3 -> adaptive-alpha Holt DES
    -> light EMA -> soft-gated trend compensation with sign hysteresis.

    Stage 1: causal median-of-3 removes impulse spikes (zero lag on
    monotone segments).

    Stage 2: Holt DES with INNOVATION-GATED adaptive alpha. The noise
    scale sigma is estimated online as a slow EMA of |innovation|. The
    effective gain is
        a_t = alpha / (1 + (sigma_t / (|innov_t| + eps))^2)
    so small innovations (noise) get heavy smoothing while large
    innovations (genuine moves) track at full gain. Cap alpha lowered to
    0.26 so even full-gain tracking injects less jitter than before.

    Stage 3: light EMA (0.40) suppresses residual level jitter; slightly
    heavier than before since the adaptive level tracks accurately.

    Stage 4: micro trend compensation y = s + g*b where g is a SOFT tanh
    gate of |b|/sigma (avoids discontinuities of hard gating) combined
    with SIGN HYSTERESIS: the compensation direction only flips when
    |b| exceeds 0.9*sigma, so noise-scale slope jitter cannot reverse
    the trend (kills false reversals) while genuine trends keep the
    full lag-reducing advance.

    Fully causal, O(N). First (window_size - 1) warm-up samples discarded
    so len(y) = len(x) - window_size + 1, aligned with x[k + window_size - 1].
    """
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Stage 1: causal median-of-3 impulse rejection
    z = x.copy()
    z[1:-1] = np.minimum(np.maximum(x[1:-1], np.minimum(x[:-2], x[2:])),
                         np.maximum(x[:-2], x[2:]))

    # Stage 2: Holt DES with innovation-gated adaptive alpha
    n = len(x)
    l = np.empty(n)
    b = np.empty(n)
    l[0] = z[0]
    b[0] = z[1] - z[0] if n > 1 else 0.0
    sigma = 0.5  # online noise-scale estimate (EMA of |innovation|)
    one_mb = 1.0 - beta
    for t in range(1, n):
        pred = l[t - 1] + b[t - 1]
        innov = z[t] - pred
        sigma = 0.95 * sigma + 0.05 * abs(innov)  # robust slow noise tracker
        a = alpha / (1.0 + (sigma / (abs(innov) + 1e-9)) ** 2)
        l[t] = pred + a * innov
        b[t] = beta * (l[t] - l[t - 1]) + one_mb * b[t - 1]

    # Stage 3: light causal EMA post-smoothing of the level estimate
    s = np.empty(n)
    s[0] = l[0]
    one_ms = 1.0 - smooth
    for t in range(1, n):
        s[t] = smooth * l[t] + one_ms * s[t - 1]

    # Stage 4: soft-gated micro trend compensation with sign hysteresis
    comp = np.empty(n)
    direction = 0.0
    for t in range(n):
        ratio = abs(b[t]) / (sigma + 1e-9)
        # soft amplitude gate: ramps in smoothly around the noise scale
        amp = 0.22 * max(np.tanh(ratio - 0.5), 0.0)
        # sign hysteresis: only flip direction on confident slopes
        if ratio > 0.9:
            direction = 1.0 if b[t] > 0 else -1.0
        comp[t] = amp * direction * abs(b[t])
    s += comp

    # Discard warm-up to match the sliding-window output length/alignment
    return s[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm.

    Args:
        input_signal: Input time series data
        window_size: Warm-up length; output length = len(x) - window_size + 1
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


def run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=12):
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
