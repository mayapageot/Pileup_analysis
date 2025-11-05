import numpy as np
from utility.double_beta_spectrum import inverse_cdf_ratio2b

def simulate_frequency_pulses(
    S,
    bslspec,
    detector_sigma,
    w,
    nsim=10000,
    seed=1234,
    signal_scale=0.002,
    dt_max=8e-4):
    """
    Simulate frequency-domain pulse realizations with stochastic phase and noise.

    Each realization represents a possible frequency-domain signal including:
      - A deterministic spectral shape `S`
      - Randomized energy sharing between two sub-events (via `inverse_cdf_ratio2b`)
      - Random emission time differences (Δt)
      - Additive complex Gaussian noise with baseline spectrum `bslspec`

    Parameters
    ----------
    S : ndarray, shape (M,)
        Deterministic signal spectrum in frequency domain.
    bslspec : ndarray, shape (M,)
        Baseline noise power spectrum (frequency-dependent variance).
    detector_sigma : float
        Detector-level energy fluctuation amplitude.
    w : ndarray, shape (M,)
        Frequency vector (radians per second).
    nsim : int, optional
        Number of Monte Carlo realizations (default: 10,000).
    seed : int, optional
        Random number generator seed (default: 1234).
    signal_scale : float, optional
        Base amplitude scaling factor for the deterministic signal (default: 0.002).
    dt_max : float, optional
        Maximum random time delay (seconds) between sub-events (default: 8e-4).

    Returns
    -------
    fpulses : ndarray, shape (nsim, M)
        Simulated complex frequency-domain signals for each realization.
    signal_freq : ndarray, shape (nsim, M)
        Deterministic (noise-free) signal component for each realization.
    noise_freq : ndarray, shape (nsim, M)
        Random noise component in frequency domain.
    generated_r : ndarray, shape (nsim,)
        Random energy-sharing ratios (R = E1 / E2).
    generated_dt : ndarray, shape (nsim,)
        Random inter-event time delays (seconds).
    """
    rng = np.random.default_rng(seed)
    M = S.size

    # --- complex Gaussian noise with spectral shape ---
    complex_gauss = (rng.normal(size=(nsim, M)) + 1j * rng.normal(size=(nsim, M))) * np.sqrt(2)
    noise_freq = complex_gauss * np.sqrt(bslspec[None, :])  # (nsim, M)

    # --- deterministic signal per realization ---
    amplitude = signal_scale * rng.normal(loc=1.0, scale=detector_sigma, size=nsim)[:, None]
    signal_freq = amplitude * S[None, :]  # (nsim, M)

    # --- random energy-sharing ratios and delays ---
    generated_r = inverse_cdf_ratio2b(rng.random(nsim))
    generated_dt = rng.uniform(0, dt_max, nsim)

    # --- phase modulation due to inter-event timing ---
    exp_term = np.exp(-1j * w * generated_dt[:, None])  # (nsim, M)
    phase = (1 - generated_r[:, None]) + generated_r[:, None] * exp_term

    # --- final frequency-domain realizations ---
    fpulses = signal_freq * phase + noise_freq

    return fpulses, signal_freq, noise_freq, generated_r, generated_dt