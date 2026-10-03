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
    Causal Savitzky-Golay endpoint regression filter (polynomial least-squares
    family, fundamentally different from exponential smoothing).

    For each output index the last `window_size` samples are fit with a
    low-order polynomial by ordinary least squares, and the polynomial is
    evaluated AT THE NEWEST SAMPLE (window endpoint). Endpoint evaluation of a
    local LS fit has near-zero group delay on linear trends (exact for order-1)
    while the LS averaging over the window suppresses noise strongly.

    - Adaptive polynomial order: quadratic (order 2) is used when genuine
      local curvature is detectable above the noise floor; otherwise linear
      (order 1) is used, which has lower variance and zero lag on trends.
      The blend is smooth to avoid order-switching artifacts.
    - Median-of-3 prefilter removes spikes that cause false reversals.
    - Light causal 3-tap post-smooth + directional hysteresis suppress
      noise-induced slope reversals with negligible added delay.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    from scipy.signal import savgol_coeffs

    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Robust median-of-3 prefilter (suppresses spikes / false reversals)
    xp = x.copy()
    if n >= 3:
        xp[1:-1] = np.median(np.vstack([x[:-2], x[1:-1], x[2:]]), axis=0)

    # Robust noise scale from first differences (MAD-based)
    d = np.abs(np.diff(xp))
    sigma = 1.4826 * np.median(d) / np.sqrt(2.0) if len(d) > 0 else 1.0
    if sigma <= 1e-12:
        sigma = 1e-12

    # Robust curvature scale from second differences
    d2 = np.abs(np.diff(xp, 2))
    curv = 1.4826 * np.median(d2) / np.sqrt(6.0) if len(d2) > 0 else 0.0
    if not np.isfinite(curv) or curv <= 1e-12:
        curv = 1e-12

    # SG endpoint evaluation coefficients: c[j] weights x[i+j] (newest at j=W-1)
    W = window_size
    c2 = savgol_coeffs(W, 2, pos=W - 1)
    c1 = savgol_coeffs(W, 1, pos=W - 1)

    # Vectorized sliding-window dot products (valid mode -> length n-W+1)
    y2 = np.convolve(xp, c2[::-1], mode="valid")
    y1 = np.convolve(xp, c1[::-1], mode="valid")

    # Adaptive order blend: weight on quadratic grows with detectable curvature
    # relative to noise (curvature-to-noise ratio). Smooth logistic blend.
    snr_curv = curv / (curv + sigma)
    w2 = snr_curv ** 1.5  # favor linear unless curvature clearly exceeds noise
    y = (1.0 - w2) * y1 + w2 * y2

    # Light causal 3-tap smoothing (uses only current + 2 past samples)
    if len(y) >= 3:
        ys = np.empty_like(y)
        ys[0] = y[0]
        ys[1] = 0.7 * y[1] + 0.3 * y[0]
        ys[2:] = 0.5 * y[2:] + 0.3 * y[1:-1] + 0.2 * y[:-2]
        y = ys

    # Directional hysteresis: reject sub-threshold slope reversals (zero-lag
    # pointwise modification of the first difference)
    dy = np.diff(y)
    scale = 1.4826 * np.median(np.abs(dy)) if len(dy) > 0 else 1.0
    if scale <= 1e-12:
        scale = 1e-12
    h = 0.40  # reversal acceptance threshold (fraction of slope scale)

    y2h = np.empty_like(y)
    y2h[0] = y[0]
    direction = 0
    for i in range(1, len(y)):
        d_i = dy[i - 1]
        s = 1 if d_i > 0 else (-1 if d_i < 0 else 0)
        if s != 0 and s != direction:
            if abs(d_i) < h * scale:
                d_i = 0.0  # reject sub-threshold reversal
            else:
                direction = s
        y2h[i] = y2h[i - 1] + d_i

    # Numerical safeguards: replace NaN/inf with last valid value
    bad = ~np.isfinite(y2h)
    if bad.any():
        last_good = y2h[0] if np.isfinite(y2h[0]) else 0.0
        for i in range(len(y2h)):
            if bad[i]:
                y2h[i] = last_good
            else:
                last_good = y2h[i]

    return y2h


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
