# EVOLVE-BLOCK-START
"""Adaptive state-space denoising: Kalman filter with a locally-linear
(position + velocity) model, followed by the Rauch-Tung-Striebel (RTS)
smoother. The RTS backward pass exploits future samples -> zero phase
lag. The process/measurement noise ratio adaptively controls the
smoothness/responsiveness trade-off, and a difference deadband
suppresses noise-induced slope reversals."""
import numpy as np


def enhanced_filter_with_trend_preservation(x, window_size=20,
                                            q_ratio=0.004, deadband=0.05):
    """Kalman filter + RTS smoother with a 2-state (level, velocity)
    constant-velocity model.

    State transition F = [[1,1],[0,1]] (dt=1), measurement H = [1,0].
    Process noise Q = q_ratio * R scales with measurement noise, so the
    filter adapts to the observed noise level. RTS smoothing gives a
    zero-lag estimate. A deadband on first differences damps tiny sign
    flips, cutting spurious slope changes and false reversals while
    preserving genuine dynamics."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    R = np.var(np.diff(x)) / 2.0 + 1e-9  # estimate measurement noise
    Q = q_ratio * R * np.array([[0.25, 0.5], [0.5, 1.0]])

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    I = np.eye(2)

    # Forward pass
    s = np.array([x[0], 0.0])
    P = np.eye(2) * R
    means = np.zeros((n, 2))
    covs = np.zeros((n, 2, 2))
    means[0], covs[0] = s, P
    for k in range(1, n):
        s = F @ s
        P = F @ P @ F.T + Q
        y = x[k] - H @ s
        S = (H @ P @ H.T)[0, 0] + R
        K = (P @ H.T) / S
        s = s + K.ravel() * y
        P = (I - K @ H) @ P
        means[k], covs[k] = s, P

    # RTS backward pass (zero-phase smoothing)
    for k in range(n - 2, -1, -1):
        C = covs[k] @ F.T @ np.linalg.inv(F @ covs[k] @ F.T + Q)
        means[k] = means[k] + C @ (means[k + 1] - F @ means[k])
        covs[k] = covs[k] + C @ (covs[k + 1] - F @ covs[k] @ F.T) @ C.T

    y = means[:, 0].copy()
    # Deadband reversal suppression on first differences
    dy = np.diff(y, prepend=y[0])
    thresh = deadband * np.std(dy)
    dy[np.abs(dy) < thresh] *= 0.3
    y = np.cumsum(dy) + y[0]
    # Align output to required length n - window_size + 1
    n_out = len(x) - window_size + 1
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
