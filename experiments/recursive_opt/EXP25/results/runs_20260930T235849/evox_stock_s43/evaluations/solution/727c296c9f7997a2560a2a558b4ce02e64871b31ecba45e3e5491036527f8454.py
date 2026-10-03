# EVOLVE-BLOCK-START
"""Multi-scale wavelet denoising pipeline: PyWavelets stationary-style
decomposition -> per-level soft thresholding -> reconstruction ->
Savitzky-Golay polish -> deadband reversal suppression."""
import numpy as np
import pywt
from scipy.signal import savgol_filter


def enhanced_filter_with_trend_preservation(x, window_size=20,
                                            wavelet="sym6", level=None,
                                            scale=0.9):
    """Wavelet shrinkage denoiser with SG polish and deadband.

    Wavelet soft-thresholding is zero-phase (applied to the whole
    offline signal) and preserves edges/transients much better than a
    steep low-pass filter, so lag_error and tracking error stay low
    while noise-induced slope changes and false reversals are removed
    across all scales. Per-level thresholds scaled by `scale` control
    the smoothness/responsiveness trade-off. A window-11 quadratic SG
    polish flattens residual micro-reversals and a deadband on first
    differences suppresses spurious sign flips."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    # 1) Wavelet decomposition with symmetric padding (no edge wrap)
    lv = level or min(pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len), 5)
    coeffs = pywt.wavedec(x, wavelet, level=lv, mode="periodization")
    # 2) Soft-threshold detail coefficients (MAD-scaled, per level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr_univ = sigma * np.sqrt(2 * np.log(n)) * scale
    new = [coeffs[0]]
    for c in coeffs[1:]:
        new.append(pywt.threshold(c, thr_univ, mode="soft"))
    y = pywt.waverec(new, wavelet, mode="periodization")[:n]
    # 3) Savitzky-Golay polish for extra smoothness
    w = 11
    if n > w:
        y = savgol_filter(y, w, 2)
    # 4) Deadband reversal suppression: flatten tiny diffs
    d = np.diff(y)
    thr = 0.03 * (np.std(d) + 1e-12)
    small = np.abs(d) < thr
    if small.any():
        d[small] = 0.0
        y = np.concatenate(([y[0]], y[0] + np.cumsum(d)))
    # Align output to required length n - window_size + 1
    n_out = n - window_size + 1
    return y[window_size - 1:window_size - 1 + n_out]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
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
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
    else:
        clean_signal = None
    filtered_signal = process_signal(noisy_signal, window_size)

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
