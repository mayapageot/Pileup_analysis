import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

norm = Normal(0, 1)

def precompute_constants(S, H_unit, w, t, bslspec, r=0.5):
    """
    Precomputes constants for further calculations.

    Args:
        S (torch.Tensor): Input signal tensor.
        H_unit (torch.Tensor): Unit response tensor.
        w (torch.Tensor): Frequency tensor.
        t (torch.Tensor): Time tensor.
        bslspec (torch.Tensor): Baseline spectrum tensor.
        r (float, optional): Ratio parameter. Defaults to 0.5.

    Returns:
        tuple: Phase tensor, weighted signal tensor, and normalized spectrum tensor.
    """
    exp_term = torch.exp(-1j * w[None, :] * t[:, None])  # (len(t), len(w))
    phase = (1 - r) + r * exp_term
    W = S * H_unit
    S2_over_bslspec = (S.abs() ** 2) / bslspec
    return phase, W, S2_over_bslspec


def compute_sigmas(S2_over_bslspec, f1, f2, signal_amp):
    """
    Computes the standard deviation (sigma) for the given inputs.

    Args:
        S2_over_bslspec (torch.Tensor): Normalized spectrum tensor.
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.
        signal_amp (float): Signal amplitude.

    Returns:
        torch.Tensor: Computed sigma value.
    """
    absS2_over_bslspec = S2_over_bslspec

    abs_f1 = f1.abs()
    abs_f2 = f2.abs()
    num1 = torch.sum(abs_f1 ** 2 * absS2_over_bslspec) / 2
    num2 = torch.sum(abs_f2 ** 2 * absS2_over_bslspec) / 2
    cross = torch.sum((f1 * torch.conj(f2)) * absS2_over_bslspec) / 2
    R1 = torch.sum(abs_f1 * absS2_over_bslspec)
    R2 = torch.sum(abs_f2 * absS2_over_bslspec)
    varX = num1 / (R1.abs() ** 2)
    varY = num2 / (R2.abs() ** 2)
    covXY = cross / (R1 * torch.conj(R2))

    muX = muY = signal_amp
    varZ = varX / muY ** 2 + (muX ** 2) * varY / muY ** 4 - 2 * muX * covXY.real / muY ** 3
    return torch.sqrt(varZ.real) * 2


def compute_J(f1, f2, phase, S_H, S2_over_bslspec, signal_amp, N_sigma=1.28):
    """
    Computes the J metric for the given inputs.

    Args:
        f1 (torch.Tensor): First filter tensor.
        f2 (torch.Tensor): Second filter tensor.
        phase (torch.Tensor): Phase tensor.
        S_H (torch.Tensor): Weighted signal tensor.
        S2_over_bslspec (torch.Tensor): Normalized spectrum tensor.
        signal_amp (float): Signal amplitude.
        N_sigma (float, optional): Sigma threshold. Defaults to 1.28.

    Returns:
        torch.Tensor: Computed J metric.
    """
    amp1 = torch.mean(S_H * f1 * phase, dim=1).real
    amp2 = torch.mean(S_H * f2 * phase, dim=1).real
    F = amp1 / amp2
    sigma = compute_sigmas(S2_over_bslspec, f1, f2, signal_amp)
    true_mu = (1 - F) * 0.9
    true_sigma = torch.sqrt(sigma ** 2 + true_mu ** 2 * 0.03689)
    true_J = 1 - norm.cdf(true_mu / true_sigma - N_sigma)
    return torch.sum(true_J) / true_mu.shape[0]


def optimize_filters(S, H_unit, w, t, bslspec, signal_amp, N_sigma = 1.28, n_trials = 1000):
    """
    Optimizes two filters (f1 and f2) to maximize the J metric using gradient-based optimization.

    Args:
        S (torch.Tensor): Input signal tensor.
        H_unit (torch.Tensor): Unit response tensor.
        w (torch.Tensor): Frequency tensor.
        t (torch.Tensor): Time tensor.
        bslspec (torch.Tensor): Baseline spectrum tensor.
        signal_amp (float): Signal amplitude.
        N_sigma (float, optional): Sigma threshold for the J metric. Defaults to 1.28.
        n_trials (int, optional): Number of optimization steps. Defaults to 1000.

    Returns:
        tuple:
            - f1 (numpy.ndarray): Optimized first filter.
            - f2 (numpy.ndarray): Optimized second filter.
            - J_values (list): List of J metric values during optimization.
    """
    # Precompute constants required for optimization
    phase, S_H, S2_over_bslspec = precompute_constants(S, H_unit, w, t, bslspec, r = 0.5)
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
        J = compute_J(f1, f2, phase, S_H, S2_over_bslspec, signal_amp, N_sigma = N_sigma)
        J_values.append(J.item())  # Store the current J value

        # Backpropagate and update filter parameters
        J.backward()
        optimizer.step()

        # Uncomment the following lines to print progress every 100 steps
        # if step % 100 == 0:
        #     print(f"Step {step}: J = {J.item():.6f}")

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