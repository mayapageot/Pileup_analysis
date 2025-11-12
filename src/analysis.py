import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from scipy.interpolate import interp1d

norm = Normal(0, 1)


def precompute_constants(S, H_unit, w, t, nps):
    """
    Precomputes constants for further calculations.

    Args:
        S (torch.Tensor): Input signal tensor.
        H_unit (torch.Tensor): Unit response tensor.
        w (torch.Tensor): Frequency tensor.
        t (torch.Tensor): Time tensor.
        nps (torch.Tensor): Baseline spectrum tensor.

    Returns:
        tuple: Phase tensor, weighted signal tensor, and normalized spectrum tensor.
    """
    exp_term = torch.exp(-1j * w[None, :] * t[:, None])  # (len(t), len(w))
    S_H = S * H_unit
    S2_over_nps = (S.abs() ** 2) / nps
    S_H_delayed = S_H[None, :] * exp_term
    return S_H_delayed, S_H, S2_over_nps


def compute_vars(S2_over_nps, f1, f2):
    """
    Computes the standard deviation (sigma) for the given inputs.

    Args:
        S2_over_nps (torch.Tensor): Normalized spectrum tensor.
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.

    Returns:
        tuple: Variances of X1, X2 and covariance of X and Y.
    """
    absS2_over_nps = S2_over_nps

    abs_f1 = f1.abs()
    abs_f2 = f2.abs()
    num1 = torch.sum(abs_f1 ** 2 * absS2_over_nps) / 2
    num2 = torch.sum(abs_f2 ** 2 * absS2_over_nps) / 2
    cross = torch.sum((f1 * torch.conj(f2)) * absS2_over_nps) / 2
    R1 = torch.sum(abs_f1 * absS2_over_nps)
    R2 = torch.sum(abs_f2 * absS2_over_nps)
    var1 = num1 / (R1.abs() ** 2)
    var2 = num2 / (R2.abs() ** 2)
    cov12 = cross / (R1 * torch.conj(R2))
    return var1, var2, cov12


def compute_sigma_ratio(mu1, mu2, var1, var2, cov12):
    varY = var1 / mu2 ** 2 + (mu1 ** 2) * var2 / mu2 ** 4 - 2 * mu1 * cov12.real / mu2 ** 3
    return torch.sqrt(varY).real * 2


def compute_mu_sigma(f1, f2, S_H_delayed, r, S_H, S2_over_nps, signal_amp, return_all=False):
    """
    Computes the mean (mu) and standard deviation (sigma) for the given filters and signal.

    Args:
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.
        S_H_delayed (torch.Tensor): Weighted signal delayed tensor.
        r (torch.Tensor): Ratio tensor.
        S_H (torch.Tensor): Weighted signal tensor.
        S2_over_nps (torch.Tensor): Normalized spectrum tensor.
        signal_amp (float): Signal amplitude.
        return_all (bool, optional): If True, returns additional intermediate values. Defaults to False.

    Returns:
        tuple: 
            - muY (torch.Tensor): Ratio of the maximum amplitudes of the two signals.
            - sigmaY (torch.Tensor): Standard deviation of the ratio.
            - (Optional) Additional intermediate values if `return_all` is True.
    """
    s_H_1 = torch.fft.ifft(S_H * f1).real
    s_H_delayed_1 = torch.fft.ifft(S_H_delayed * f1, dim = -1).real
    s_pileup_1 = (1-r[:, None, None]) * s_H_1 + r[:, None, None] * s_H_delayed_1[None, :, :]
    mu1 = torch.max(s_pileup_1.real, dim = -1).values
    s_H_2 = torch.fft.ifft(S_H * f2).real
    s_H_delayed_2 = torch.fft.ifft(S_H_delayed * f2, dim = -1).real
    s_pileup_2 = (1-r[:, None, None]) * s_H_2 + r[:, None, None] * s_H_delayed_2[None, :, :]
    mu2 = torch.max(s_pileup_2.real, dim = -1).values.real
    muY = mu1 / mu2
    var1, var2, cov12 = compute_vars(S2_over_nps, f1, f2)
    sigmaY = compute_sigma_ratio(mu1, mu2, var1, var2, cov12) / signal_amp
    if return_all:
        return muY, sigmaY, mu1, mu2, var1, var2, cov12
    else:
        return muY, sigmaY


def compute_A(muY, sigmaY, N_sigma=1.28):
    """
    Computes the acceptance (A).

    Args:
        muY (torch.Tensor): Ratio of the maximum amplitudes of the two signals.
        sigmaY (torch.Tensor): Standard deviation of the ratio.
        N_sigma (float, optional): Sigma threshold. Defaults to 1.28.

    Returns:
        torch.Tensor: Computed probability value.
    """
    return 1 - norm.cdf((1-muY-N_sigma*sigmaY[0, 0]) / sigmaY)


def compute_J(f1, f2, S_H_delayed, r, S_H, S2_over_nps, signal_amp, ratio_distribution, N_sigma=1.28):
    """
    Computes the J metric for the given inputs.

    Args:
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.
        S_H_delayed (torch.Tensor): Weighted signal delayed tensor.
        r (torch.Tensor): Ratio tensor.
        S_H (torch.Tensor): Weighted signal tensor.
        S2_over_nps (torch.Tensor): Normalized spectrum tensor.
        signal_amp (float): Signal amplitude.
        ratio_distribution (torch.Tensor): Ratio distribution tensor.
        N_sigma (float, optional): Sigma threshold. Defaults to 1.28.

    Returns:
        torch.Tensor: Computed J metric.
    """
    muY, sigmaY = compute_mu_sigma(f1, f2, S_H_delayed, r, S_H, S2_over_nps, signal_amp)
    A = compute_A(muY, sigmaY, N_sigma=N_sigma)
    return torch.mean(A*ratio_distribution[:, None]).real


def optimize_filters(S, H_unit, w, t, r, nps, signal_amp, ratio_distribution, N_sigma = 1.28, n_trials = 1000):
    """
    Optimizes two filters (f1 and f2) to maximize the J metric using gradient-based optimization.

    Args:
        S (torch.Tensor): Input signal tensor.
        H_unit (torch.Tensor): Unit response tensor.
        w (torch.Tensor): Frequency tensor.
        t (torch.Tensor): Time tensor.
        r (torch.Tensor): Ratio tensor.
        nps (torch.Tensor): Baseline spectrum tensor.
        signal_amp (float): Signal amplitude.
        ratio_distribution (torch.Tensor): Ratio distribution tensor.
        N_sigma (float, optional): Sigma threshold for the J metric. Defaults to 1.28.
        n_trials (int, optional): Number of optimization steps. Defaults to 1000.

    Returns:
        tuple:
            - f1 (numpy.ndarray): Optimized first filter.
            - f2 (numpy.ndarray): Optimized second filter.
            - J_values (list): List of J metric values during optimization.
    """
    # Precompute constants required for optimization
    S_H_delayed, S_H, S2_over_nps = precompute_constants(S, H_unit, w, t, nps)
    # Initialize filter parameters as complex tensors
    n = len(H_unit)
    f1_param = torch.nn.Parameter(torch.rand(n, dtype = torch.cfloat))
    f2_param = torch.nn.Parameter(torch.rand(n, dtype = torch.cfloat))

    # Set up the optimizer
    optimizer = torch.optim.Adam([f1_param, f2_param], lr = 1e-2)
    J_values = []  # To store J metric values during optimization
    f1 = torch.abs(f1_param)
    f2 = torch.abs(f2_param)
    for step in range(n_trials):
        optimizer.zero_grad()  # Reset gradients
        # Enforce positivity of the filters
        f1 = torch.abs(f1_param)
        f2 = torch.abs(f2_param)

        # Normalize filters such that mean(|f_i * H * S|) = 1
        norm1 = torch.mean(torch.abs(f1 * H_unit * S))
        norm2 = torch.mean(torch.abs(f2 * H_unit * S))
        f1 = f1 / norm1
        f2 = f2 / norm2
        # Compute the J metric
        J = compute_J(f1, f2, S_H_delayed, r, S_H, S2_over_nps, signal_amp, ratio_distribution, N_sigma = N_sigma)
        J_values.append(J.item())  # Store the current J value

        # Backpropagation and update filter parameters
        J.backward()
        optimizer.step()

        # Uncomment the following lines to print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: J = {J.item():.6f}")

    # Convert optimized filters to numpy arrays
    f1 = np.abs(f1.detach().numpy())
    f2 = np.abs(f2.detach().numpy())
    return f1, f2, J_values


def get_PSD(dataset, H_unit, f1, f2, batch_size=128):
    """
    Computes the Pulse Shape Discriminator (PSD) for the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): Input dataset.
        H_unit (torch.Tensor): Unit transfer function tensor.
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.
        batch_size (int, optional): Batch size for data loading. Defaults to 128.

    Returns:
        numpy.ndarray: Computed PSD values.
    """
    Amp_1 = []
    Amp_2 = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for pulses in loader:
        pulses = pulses.numpy()
        Amp_1.append(np.max(np.fft.ifft(f1 * H_unit * np.fft.fft(pulses, axis=-1), axis=1).real, axis=1))
        Amp_2.append(np.max(np.fft.ifft(f2 * H_unit * np.fft.fft(pulses, axis=-1), axis=1).real, axis=1))
    Amp_1 = np.concatenate(Amp_1)
    Amp_2 = np.concatenate(Amp_2)
    PSD = Amp_1 / Amp_2
    return PSD


def get_PSD_interpole(dataset, H_unit, f1, f2, batch_size=128, interpolation_range=5, jitter_max=20):
    """
    Computes the Pulse Shape Discriminator (PSD) for the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): Input dataset.
        H_unit (torch.Tensor): Unit transfer function tensor.
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.
        batch_size (int, optional): Batch size for data loading. Defaults to 128.
        interpolation_range (int, optional): Range for interpolation around the maximum. Defaults to 5.
        jitter_max (int, optional): Maximum jitter for peak search. Defaults to 20.

    Returns:
        numpy.ndarray: Computed PSD values.
    """
    Amp_1 = []
    Amp_2 = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for pulses in loader:
        if type(pulses) is list:
            pulses = pulses[0]
        pulses = pulses.numpy()
        n_rows, n_cols = pulses.shape
        fpulse = np.fft.fft(pulses, axis=-1)
        filtered_1 = np.roll(np.fft.ifft(f1 * H_unit * fpulse, axis=1).real, shift=n_cols//2, axis=1)
        max_idx_1 = np.argmax(filtered_1[:, n_cols//2-jitter_max:n_cols//2+jitter_max], axis = 1)+n_cols//2-jitter_max
        filtered_2 = np.roll(np.fft.ifft(f2 * H_unit * fpulse, axis=1).real, shift=n_cols//2, axis=1)
        max_idx_2 = np.argmax(filtered_2[:, n_cols//2-jitter_max:n_cols//2+jitter_max], axis = 1)+n_cols//2-jitter_max
        # interpolate to find max
        rows = np.arange(n_rows)
        col_offsets = np.arange(-interpolation_range, interpolation_range + 1)
        cols_1 = max_idx_1[:, None] + col_offsets
        y1 = filtered_1[rows[:, None], cols_1]
        interp1d_1 = interp1d(col_offsets, y1, kind = 'cubic', axis = 1, bounds_error = False,
                              fill_value = 'extrapolate')
        cols_2 = max_idx_2[:, None] + col_offsets
        y2 = filtered_2[rows[:, None], cols_2]
        interp1d_2 = interp1d(col_offsets, y2, kind = 'cubic', axis = 1, bounds_error = False, 
                              fill_value = 'extrapolate')
        fine_x = np.arange(-interpolation_range, interpolation_range + 1, 0.05)
        Amp_1.append(np.max(interp1d_1(fine_x), axis=1))
        Amp_2.append(np.max(interp1d_2(fine_x), axis=1))
    Amp_1 = np.concatenate(Amp_1)
    Amp_2 = np.concatenate(Amp_2)
    PSD = Amp_1 / Amp_2
    return PSD, Amp_1, Amp_2
