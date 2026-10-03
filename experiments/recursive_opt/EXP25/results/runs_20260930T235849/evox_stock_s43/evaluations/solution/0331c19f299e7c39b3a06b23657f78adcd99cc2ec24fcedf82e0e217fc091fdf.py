# EVOLVE-BLOCK-START
"""Multi-scale wavelet denoising with soft thresholding (PyWavelets).
The signal is decomposed with a Daubechies wavelet; detail coefficients
(noise lives mostly in fine scales) are soft-thresholded using a
MAD-based universal threshold, then the signal is reconstructed.
Wavelet shrinkage is inherently zero-phase (coefficients are time
aligned), so lag_error ~ 0, while multi-scale separation preserves
genuine slopes and kills noise-induced reversals. A light centered
Savitzky-Golay pass afterwards removes residual roughness."""
import numpy as np
import pywt
from scipy.signal import savgol_filter


def enhanced_filter_with_trend_preservation(x, window_size=20,
                                            wavelet="db4", level=4,
                                            polyorder=2):
    """Wavelet soft-threshold denoiser + light zero-phase SG polish.

    1. wavedec splits the signal into scale bands.
    2. Detail coefficients are soft-thresholded with
       sigma*sqrt(2*ln(N)), sigma estimated robustly via MAD of the
       finest detail band (noise estimate, unaffected by trend).
    3. waverec reconstructs a zero-lag, smooth, trend-preserving signal.
    4. A short centered SG filter (window clipped to signal length)
       smooths residual jitter; output truncated to the harness's
       delay convention (len - window_size + 1 samples).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_out = n - window_size + 1
    if n_out <= 0:
        return np.array([])

    # Pad to a power-of-two-friendly length for clean decomposition
    coeffs = pywt.wavedec(x, wavelet, level=min(level,
                                               pywt.dwt_max_level(n, wavelet)))

    # Robust noise estimate from finest detail coefficients
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
    if sigma <= 0:
        sigma = np.std(detail) if len(detail) > 0 else 1.0
    thr = sigma * np.sqrt(2.0 * np.log(n))

    # Soft-threshold all detail bands; keep approximation untouched
    coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft")
                            for c in coeffs[1:]]

    y = pywt.waverec(coeffs, wavelet)[:n]

    # Light centered SG polish for extra smoothness
    win = window_size if window_size % 2 == 1 else window_size + 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win > polyorder:
        y = savgol_filter(y, window_length=win, polyorder=polyorder,
                          mode="interp")

    return y[:n_out]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Apply the wavelet-denoising zero-phase filter chain."""
    return enhanced_filter_with_trend_preservation(input_signal, window_size)


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
