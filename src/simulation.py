import numpy as np
from utility.double_beta_spectrum import inverse_cdf_ratio2b

from scipy.signal import bessel, bilinear, lfilter

def simulate_frequency_pulses(S, nps, detector_sigma, w, nsim=10000, seed=1234, signal_scale=0.002, dt_max=8e-4):
    """
    Simulate frequency-domain pulse realizations with stochastic phase and noise.

    Each realization represents a possible frequency-domain signal including:
      - A deterministic spectral shape `S`
      - Randomized energy sharing between two sub-events (via `inverse_cdf_ratio2b`)
      - Random emission time differences (Δt)
      - Additive complex Gaussian noise with baseline spectrum `nps`

    Parameters
    ----------
    S : ndarray, shape (M,)
        Deterministic signal spectrum in frequency domain.
    nps : ndarray, shape (M,)
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
    noise_freq = complex_gauss * np.sqrt(nps[None, :])  # (nsim, M)

    # --- deterministic signal per realization ---
    amplitude = rng.normal(loc=signal_scale, scale=detector_sigma, size=nsim)[:, None]
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

def simulate_frequency_pulses_fixed_dt_r(S, nps, w, dt ,r ,nsim=10000, seed=1234, signal_scale=0.002):
    rng = np.random.default_rng(seed)
    M = S.size

    # --- complex Gaussian noise with spectral shape ---
    complex_gauss = (rng.normal(size = (nsim, M)) + 1j * rng.normal(size = (nsim, M))) * np.sqrt(2)
    noise_freq = complex_gauss * np.sqrt(nps[None, :])  # (nsim, M)
    exp_term = np.exp(-1j * w * dt)  # (nsim, M)
    phase = (1 - r) + r* exp_term
    signal_freq = signal_scale * S[None, :]
    fpulses = signal_freq * phase * np.exp(-1j * w * np.random.random(nsim)[:,None]*1e-4) + noise_freq
    return fpulses, signal_freq, noise_freq


def pulse_model(t, A, t0, tau_rise, tau_decay1, tau_decay2, p):
    """Pulse model with 1 rise time and 2 decay times."""
    t_shifted = t - t0
    pulse = np.zeros_like(t_shifted)
    mask = t_shifted >= 0
    rise = 1 - np.exp(-t_shifted[mask]/tau_rise)
    decay = p * np.exp(-t_shifted[mask]/tau_decay1) + (1-p) * np.exp(-t_shifted[mask]/tau_decay2)
    pulse[mask] = A * rise * decay
    return pulse

def bessel_filter_coeffs(order, fc, fs):
    """Design a digital Bessel filter using bilinear transform."""
    # Normalize cutoff for digital design
    wc = 2 * np.pi * fc    # analog rad/s cutoff
    # Analog Bessel filter design
    b_a, a_a = bessel(order, wc, analog=True, norm='phase')
    # Convert to digital using bilinear transform
    b_d, a_d = bilinear(b_a, a_a, fs)
    return b_d, a_d

def apply_bessel_to_pulse(t, pulse, fc=450, order=2, fs=1e4):
    b_d, a_d = bessel_filter_coeffs(order, fc, fs)
    filtered_pulse = lfilter(b_d, a_d, pulse)
    return filtered_pulse

def apply_bessel_to_pulse(t, pulse, fc=450, order=2, fs=1e4):
    b_d, a_d = bessel_filter_coeffs(order, fc, fs)
    filtered_pulse = lfilter(b_d, a_d, pulse)
    return filtered_pulse

def filtered_pulse_model(t, A, t0, tau_rise, tau_decay1, tau_decay2, p, fc, order):
    pulse = pulse_model(t, A, t0, tau_rise, tau_decay1, tau_decay2, p)
    filtered_pulse = apply_bessel_to_pulse(t, pulse, fc=fc, order=order, fs=1e4)
    return filtered_pulse.real

if __name__ == "__main__":
    from torch.utils.data import TensorDataset
    import src.dataset as ds
    import src.analysis as an
    import torch
    n_deriv = 1
    channel = 15
    H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(channel, n_deriv = n_deriv, window_fct = np.hanning)
    f1_opt, f2_opt = np.load(f"../outputs/Pileup_filter_functions/channel_{channel}/functions_eff90_V3.npy")
    fpulses, signal_freq, noise_freq, generated_r, generated_dt_single = simulate_frequency_pulses(S, nps,
                                                                                                       8.59898e-05, w,
                                                                                                       signal_scale = 0.001874,
                                                                                                       nsim = 10000,
                                                                                                       dt_max = 0)
    dataset_simu = TensorDataset(torch.from_numpy(np.fft.ifft(fpulses, axis = 1).real.astype(np.float32)))
    PSD_simu_single, _, _ = an.get_PSD_interpole(dataset_simu, H_unit, f1_opt, f2_opt)
    fpulses, signal_freq, noise_freq, generated_r, generated_dt = simulate_frequency_pulses(S, nps, 8.59898e-05, w,
                                                                                                signal_scale = 0.001874,
                                                                                                nsim = 10000)
    dataset_simu = TensorDataset(torch.from_numpy(np.fft.ifft(fpulses, axis = 1).real.astype(np.float32)))
    PSD_simu, Amp_1_simu, Amp_2_simu = an.get_PSD_interpole(dataset_simu, H_unit, f1_opt, f2_opt)
