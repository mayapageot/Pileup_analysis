import numpy as np
from torch.distributions.normal import Normal
from scipy.interpolate import interp1d
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.stats import gaussian_kde
import utility.functions as fn
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
    return var1.real, var2.real, cov12.real


def compute_sigma_ratio(mu1, mu2, var1, var2, cov12):
    varY = var1 / mu2 ** 2 + (mu1 ** 2) * var2 / mu2 ** 4 - 2 * mu1 * cov12 / mu2 ** 3
    return torch.sqrt(varY) * 2


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
    mu2 = torch.max(s_pileup_2.real, dim = -1).values
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


def optimize_filters(S, H_unit, w, t, r, nps, signal_amp, ratio_distribution, N_sigma = 1.28, n_trials = 1000,
                     f1_init=None,f2_init=None,verbose = True):
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
    f1_init = torch.rand(n//2+1, dtype = torch.float) if f1_init is None else f1_init
    f2_init = torch.rand(n//2+1, dtype = torch.float) if f2_init is None else f2_init
    if len(f1_init) > n//2+1:
        f1_init = f1_init[:n//2+1]
    if len(f2_init) > n//2+1:
        f2_init = f2_init[:n//2+1]
    f1_param = torch.nn.Parameter(f1_init.to(S.device))
    f2_param = torch.nn.Parameter(f2_init.to(S.device))

    # Set up the optimizer
    optimizer = torch.optim.Adam([f1_param, f2_param], lr = 1e-2)
    J_values = []  # To store J metric values during optimization
    f1 = torch.abs(f1_param)
    f2 = torch.abs(f2_param)
    f1 = torch.cat([f1, f1[1:-1].flip(0)])
    f2 = torch.cat([f2, f2[1:-1].flip(0)])
    for step in range(n_trials):
        optimizer.zero_grad()  # Reset gradients
        # Enforce positivity of the filters
        f1 = torch.abs(f1_param)
        f2 = torch.abs(f2_param)
        f1 = torch.cat([f1, f1[1:-1].flip(0)])
        f2 = torch.cat([f2, f2[1:-1].flip(0)])
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
        if step % 100 == 0 and verbose:
            print(f"Step {step}: J = {J.item():.6f}")
    print(f"Step {step}: J = {J.item():.6f}")
    # Convert optimized filters to numpy arrays
    f1 = np.abs(f1.detach().cpu().numpy())
    f2 = np.abs(f2.detach().cpu().numpy())
    return f1, f2, J_values


def get_PSD(dataset, H_unit, f1, f2, batch_size=128, window_fct=np.ones):
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
        window = window_fct(pulses.shape[1])  # Apply Hanning window
        pulses = pulses * window
        Amp_1.append(np.max(np.fft.ifft(f1 * H_unit * np.fft.fft(pulses, axis=-1), axis=1), axis=1))
        Amp_2.append(np.max(np.fft.ifft(f2 * H_unit * np.fft.fft(pulses, axis=-1), axis=1), axis=1))
    Amp_1 = np.concatenate(Amp_1)
    Amp_2 = np.concatenate(Amp_2)
    PSD = Amp_1 / Amp_2
    return PSD


def get_PSD_interpole(dataset, H_unit, f1, f2, batch_size=128, interpolation_range=5, jitter_max=20,
                      filter=None, n_deriv = 0, filter_prederiv=None,window_fct=np.ones):
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
    filter = filter if filter is not None else np.ones_like(H_unit, dtype=np.float32)
    H_1_filtered = f1 * H_unit * filter
    H_2_filtered = f2 * H_unit * filter
    for pulses in loader:
        if type(pulses) is list:
            pulses = pulses[0]
        pulses = pulses.numpy()
        pulses -= np.mean(pulses[:500], axis=-1, keepdims=True)
        if filter_prederiv is not None:
            pulses = np.fft.ifft(filter_prederiv * np.fft.fft(pulses, axis=-1), axis=-1).real
        pulses = np.diff(pulses, n=n_deriv, prepend=np.zeros((pulses.shape[0], n_deriv)), axis=-1)
        window = window_fct(pulses.shape[1])
        pulses = pulses * window
        n_rows, n_cols = pulses.shape
        fpulse = np.fft.fft(pulses, axis=-1)
        filtered_1 = np.roll(np.fft.ifft(H_1_filtered * fpulse, axis=1).real, shift=n_cols//2, axis=1)
        max_idx_1 = np.argmax(filtered_1[:, n_cols//2-jitter_max:n_cols//2+jitter_max], axis=1) + n_cols//2-jitter_max
        filtered_2 = np.roll(np.fft.ifft(H_2_filtered * fpulse, axis=1).real, shift=n_cols//2, axis=1)
        max_idx_2 = np.argmax(filtered_2[:, n_cols//2-jitter_max:n_cols//2+jitter_max], axis=1) + n_cols//2-jitter_max
        # interpolate to find max
        rows = np.arange(n_rows)
        col_offsets = np.arange(-interpolation_range, interpolation_range + 1)
        cols_1 = max_idx_1[:, None] + col_offsets
        y1 = filtered_1[rows[:, None], cols_1]
        interp1d_1 = interp1d(col_offsets, y1, kind='cubic', axis=1, bounds_error=False,
                              fill_value='extrapolate')
        cols_2 = max_idx_2[:, None] + col_offsets
        y2 = filtered_2[rows[:, None], cols_2]
        interp1d_2 = interp1d(col_offsets, y2, kind='cubic', axis=1, bounds_error=False,
                              fill_value='extrapolate')
        fine_x = np.arange(-interpolation_range, interpolation_range + 1, 0.05)
        Amp_1.append(np.max(interp1d_1(fine_x), axis=1))
        Amp_2.append(np.max(interp1d_2(fine_x), axis=1))
    Amp_1 = np.concatenate(Amp_1)
    Amp_2 = np.concatenate(Amp_2)
    PSD = Amp_1 / Amp_2
    return PSD, Amp_1, Amp_2


def cubic_interpolate(y, fine_x):
    """
    y: (batch, S)  where S = 2*range + 1
    fine_x: (Nfine,)
    Returns: (batch, Nfine)
    """
    B, S = y.shape
    R = (S - 1) // 2  # interpolation_range

    # normalize X-coordinates to [-1,1]
    fine_x_norm = (fine_x / R).view(1, -1)  # (1, N)

    # Fake Y-coordinate is always 0 (middle of height=1)
    fine_y_norm = torch.zeros_like(fine_x_norm)

    # Stack into (1, N, 2)
    grid = torch.stack([fine_x_norm, fine_y_norm], dim=-1)  # (1, N, 2)

    # Expand for batch: (B, N, 2)
    grid = grid.expand(B, -1, -1)

    # Final shape must be (B, H_out, W_out, 2)
    # Here H_out=1, W_out=N
    grid = grid.unsqueeze(1)  # (B, 1, N, 2)

    # reshape y into a fake 2D image: (B, C=1, H=1, W=S)
    inp = y.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

    out = F.grid_sample(
        inp,
        grid,
        mode="bicubic",
        padding_mode="border",
        align_corners=True,
    )
    # out is (B, 1, 1, N)
    return out.squeeze(1).squeeze(1)  # (B, N)


def cubic_interp1d_scipy(y, fine_x):
    """
    y:      (B, S) sampled on uniform grid x = 0,1,...,S-1
    fine_x: (N,)   arbitrary float positions in the same coordinate system

    Returns:
        (B, N) cubic interpolation matching SciPy interp1d(kind='cubic')
    """

    B, S = y.shape
    N = fine_x.shape[0]

    # Compute tangents m_i exactly like SciPy (Catmull–Rom style)
    # m_i = (y_{i+1} - y_{i-1}) / 2
    ym1 = torch.roll(y, 1, dims=1)
    yp1 = torch.roll(y, -1, dims=1)
    m = (yp1 - ym1) * 0.5

    # Clamp edge derivatives (SciPy behavior)
    # At boundaries it uses one-sided finite differences
    m[:, 0] = (y[:, 1] - y[:, 0])
    m[:, -1] = (y[:, -1] - y[:, -2])

    # ---- Locate interval indices ----
    # floor_x = left index i
    x = fine_x.clamp(0, S - 1 - 1e-12)     # stay inside valid range
    i = x.floor().long()                   # left point index
    t = (x - i.float()).view(1, -1)        # normalize fractional position (1,N)

    # (B,N) gather values at i and i+1
    i1 = i.clamp(0, S - 1)
    i2 = (i + 1).clamp(0, S - 1)

    y_i  = y.gather(1, i1.unsqueeze(0).expand(B, -1))
    y_ip = y.gather(1, i2.unsqueeze(0).expand(B, -1))

    m_i  = m.gather(1, i1.unsqueeze(0).expand(B, -1))
    m_ip = m.gather(1, i2.unsqueeze(0).expand(B, -1))

    # ---- Cubic Hermite spline basis functions ----
    t2 = t * t
    t3 = t2 * t

    h00 =  2*t3 - 3*t2 + 1
    h10 =      t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 =      t3 -    t2

    # ---- Final spline ----
    out = (h00 * y_i
         + h10 * m_i
         + h01 * y_ip
         + h11 * m_ip)

    return out   # shape (B, N)


def batch_iterator(dataset, batch_size=4096, device="cpu", use_loader=True, shuffle=False, num_workers=4, prefetch_factor=2):
    """
    Returns a generator yielding batches of pulses, either via DataLoader or custom get_batch().
    """
    if use_loader:
        # Standard DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True,
                            num_workers=num_workers,
                            prefetch_factor=prefetch_factor)
        for batch in loader:
            yield batch.to(device, dtype=torch.float32)
    else:
        # Custom get_batch loop (vectorized, fast)
        n_batches = (len(dataset) + batch_size - 1) // batch_size
        for i in range(n_batches):
            indices = torch.arange(i*batch_size, min((i+1)*batch_size, len(dataset)))
            batch = dataset.get_batch(indices)
            yield batch.to(device, dtype=torch.float32)


def get_PSD_interpole_torch(dataset, H_unit, f1, f2,
                            batch_size=128, interpolation_range=5,
                            jitter_max=20, filter=None,
                            n_deriv=0, filter_prederiv=None,
                            window_fct=torch.ones,use_loader=False,device="cpu"):
    """
    Fully GPU-enabled PSD computation
    """

    device = H_unit.device
    loader = batch_iterator(dataset, batch_size=batch_size, device='cpu', use_loader=use_loader)

    win_len = dataset.win_length
    if filter is None:
        filter = torch.ones(win_len, device=device)

    H_1_filtered = (f1 * H_unit * filter).to(dtype=torch.complex64)
    H_2_filtered = (f2 * H_unit * filter).to(dtype=torch.complex64)

    Amp_1, Amp_2 = [], []

    # prepare fine X grid for interpolation
    fine_x = torch.arange(-interpolation_range, interpolation_range + 1, 0.05,
                          device = device, dtype = torch.float32)

    # windowing
    window = window_fct(win_len)
    if not torch.is_tensor(window):
        window = torch.tensor(window, device=device, dtype=torch.float32)

    for pulses in loader:
        if isinstance(pulses, (list, tuple)):
            pulses = pulses[0]
        pulses = pulses.to(device, dtype=torch.float32)

        # subtract baseline (first 500 samples)
        pulses = pulses - pulses[:, :500].mean(dim=1, keepdim=True)

        # optional pre-filter in freq
        if filter_prederiv is not None:
            fp = torch.fft.fft(pulses, dim=-1)
            pulses = torch.fft.ifft(fp * filter_prederiv, dim=-1).real

        # derivative
        if n_deriv > 0:
            pulses = pulses.diff(n=n_deriv, dim=-1, prepend=torch.zeros((pulses.shape[0], n_deriv), device=device))

        pulses = pulses * window

        n_rows, n_cols = pulses.shape

        # FFT of pulses
        fpulse = torch.fft.fft(pulses, dim=-1)

        # Filtered output 1
        filt1 = torch.fft.ifft(H_1_filtered * fpulse, dim=-1).real
        filt1 = torch.roll(filt1, shifts=n_cols // 2, dims=1)

        # Peak search band
        lo = n_cols // 2 - jitter_max
        hi = n_cols // 2 + jitter_max

        sub1 = filt1[:, lo:hi]
        max_idx1 = sub1.argmax(dim=1) + lo

        # Same for filter 2
        filt2 = torch.fft.ifft(H_2_filtered * fpulse, dim=-1).real
        filt2 = torch.roll(filt2, shifts=n_cols // 2, dims=1)
        sub2 = filt2[:, lo:hi]
        max_idx2 = sub2.argmax(dim=1) + lo

        # interpolation samples
        offs = torch.arange(-interpolation_range, interpolation_range + 1, device=device)

        idx1 = max_idx1[:, None] + offs[None, :]
        idx2 = max_idx2[:, None] + offs[None, :]

        y1 = filt1.gather(1, idx1)
        y2 = filt2.gather(1, idx2)

        # cubic interpolation
        interp1 = cubic_interp1d_scipy(y1, fine_x)
        interp2 = cubic_interp1d_scipy(y2, fine_x)

        # extract amplitudes
        Amp_1.append(interp1.max(dim=1).values)
        Amp_2.append(interp2.max(dim=1).values)

    Amp_1 = torch.cat(Amp_1)
    Amp_2 = torch.cat(Amp_2)
    PSD = Amp_1 / Amp_2

    # return CPU numpy if desired
    return PSD, Amp_1, Amp_2


def create_NPS(dataset, filter=None, n_deriv = 0, filter_prederiv=None,rms_thr = 0.0005,batch_size=128, window_fct=np.hanning):
    """
    Computes the Noise Power Spectrum (NPS) for the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): Input dataset.
        filter (np.array or None): Filter array. Default is None
        batch_size (int, optional): Batch size for data loading. Defaults to 128.

    Returns:
        numpy.ndarray: Computed NPS values.
    """
    nps_accum = None
    count = 0
    filter = filter if filter is not None else np.ones(dataset[0].shape[0], dtype=np.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for pulses in loader:
        pulses = pulses.numpy()
        if filter_prederiv is not None:
            pulses = np.fft.ifft(filter_prederiv * np.fft.fft(pulses, axis=-1), axis=-1).real
        pulses = pulses[np.std(pulses,axis=-1)<rms_thr]
        pulses = pulses - np.mean(pulses,axis = -1)[:,None]
        pulses = np.diff(pulses, n=n_deriv, prepend=np.zeros((pulses.shape[0], n_deriv)), axis=-1)
        window = window_fct(pulses.shape[1])  # Apply Hanning window
        pulses = pulses * window
        fpulse = np.fft.fft(pulses, axis=-1)
        nps_batch = np.mean(np.abs(filter * fpulse )**2, axis=0)
        if nps_accum is None:
            nps_accum = nps_batch
        else:
            nps_accum += nps_batch
        count += 1
    nps = nps_accum / count
    return nps

def create_NPS_torch(dataset,
                     filter=None,
                     n_deriv=0,
                     filter_prederiv=None,
                     rms_thr=0.0005,
                     batch_size=128,
                     window_fct=torch.ones,
                     use_loader=True,
                     device="cuda"):
    """
    Fully GPU-accelerated Noise Power Spectrum (NPS) computation.
    Note: The NPS is not normalized with respect to the windowing function.
    """

    # Prepare filter
    win_len = dataset.win_length
    device = torch.device(device)

    if filter is None:
        filter = torch.ones(win_len, device=device, dtype=torch.float32)
    else:
        filter = torch.tensor(filter, device=device, dtype=torch.float32)

    # Pre-derivative filter (freq-domain)
    if filter_prederiv is not None:
        filter_prederiv = torch.tensor(filter_prederiv,
                                       device=device,
                                       dtype=torch.complex64)

    # windowing
    window = window_fct(win_len)
    if not torch.is_tensor(window):
        window = torch.tensor(window, device=device, dtype=torch.float32)

    # Batch iterator (same helper as PSD)
    loader = batch_iterator(dataset,
                            batch_size=batch_size,
                            device="cpu",  # pulses come to GPU later
                            use_loader=use_loader)

    nps_accum = None
    count = 0

    for pulses in loader:
        # Allow dataset returning (pulses, label)
        if isinstance(pulses, (list, tuple)):
            pulses = pulses[0]

        pulses = pulses.to(device, dtype=torch.float32)

        # ======== FILTER BEFORE DIFF (F-domain pre-filter) ========
        if filter_prederiv is not None:
            fp = torch.fft.fft(pulses, dim=-1)
            pulses = torch.fft.ifft(fp * filter_prederiv, dim=-1).real

        # ======== RMS THRESHOLD FILTERING ========
        rms = pulses.std(dim=-1)
        mask = rms < rms_thr
        pulses = pulses[mask]
        if pulses.numel() == 0:
            continue

        # ======== REMOVE BASELINE ========
        pulses = pulses - pulses.mean(dim=-1, keepdim=True)

        # ======== N-th DERIVATIVE ========
        if n_deriv > 0:
            zeros = torch.zeros((pulses.shape[0], n_deriv),
                                device=device, dtype=pulses.dtype)
            padded = torch.cat([zeros, pulses], dim=-1)
            pulses = padded.diff(n=n_deriv, dim=-1)

        # ======== APPLY WINDOW ========
        pulses = pulses * window

        # ======== FFT ========
        fpulse = torch.fft.fft(pulses, dim=-1)

        # ======== POWER SPECTRUM ========
        nps_batch = torch.mean(torch.abs(filter * fpulse)**2, dim=0)

        # Accumulate
        if nps_accum is None:
            nps_accum = nps_batch
        else:
            nps_accum += nps_batch

        count += 1

    # ======== FINAL AVERAGE ========
    nps = nps_accum / count
    return nps.detach().cpu().numpy()



def compute_BI(data_pileup, data_single, acceptance, H_unit, f1_opt, f2_opt, window_fct = np.hanning,n_deriv = 0, full_output = False):
    PSD_pileup, Amp_1_pileup, Amp_2_pileup = get_PSD_interpole(data_pileup, H_unit, f1_opt, f2_opt,
                                                                  window_fct = window_fct, n_deriv = n_deriv)
    PSD_single, Amp_1_single, Amp_2_single = get_PSD_interpole(data_single, H_unit, f1_opt, f2_opt,
                                                                  window_fct = window_fct, n_deriv = n_deriv)
    cut = np.percentile(PSD_single, 100 - acceptance * 100)
    rp = np.mean(PSD_pileup < cut)
    BI = fn.K * (1-rp)
    if full_output:
        return BI, rp, PSD_pileup, PSD_single, Amp_1_pileup, Amp_2_pileup, Amp_1_single, Amp_2_single
    else :
        return BI, rp


def compute_BI_uncertainty(PSD_single, PSD_pileup, acceptance, rp, K=fn.K):
    N_single = len(PSD_single)
    N_pileup = len(PSD_pileup)

    # --- 1. Uncertainty from pileup binomial statistics ---
    sigma_rp_pileup = np.sqrt(rp * (1 - rp) / N_pileup)

    # --- 2. Uncertainty of the percentile cut ---
    cut = np.percentile(PSD_single, 100 - acceptance*100)

    # estimate PDFs for PSDsingle and PSDpileup
    kde_single = gaussian_kde(PSD_single)
    kde_pileup = gaussian_kde(PSD_pileup)

    f_single = kde_single.evaluate(cut)[0]
    f_pileup = kde_pileup.evaluate(cut)[0]

    a = 1 - acceptance
    sigma_cut = np.sqrt(a*(1-a)) / (np.sqrt(N_single) * f_single)

    # derivative of rp wrt cut
    drp_dcut = -f_pileup
    sigma_rp_cut = abs(drp_dcut) * sigma_cut
    # --- total ---

    sigma_rp = np.sqrt(sigma_rp_pileup**2 + sigma_rp_cut**2)
    sigma_BI = K * sigma_rp
    return sigma_rp, sigma_BI

import torch
import math

def _gaussian_kde_torch(x: torch.Tensor, points: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Evaluate Gaussian KDE of `points` at locations x.
    points: shape (N,)
    x: shape () or (M,)
    returns: same shape as x (pdf values)
    Uses Silverman's rule of thumb for bandwidth.
    Pure PyTorch implementation so it stays on the same device/dtype as inputs.
    """
    points = points.to(x.device)
    N = points.numel()
    if N == 0:
        return torch.zeros_like(x)

    # compute bandwidth (Silverman rule)
    std = torch.std(points, unbiased=True)
    # if std is zero (all points equal), set small std
    std = std.clamp_min(eps)
    h = 1.06 * std * (N ** (-1/5))
    h = h.clamp_min(eps)

    # Expand for vectorized squared distances
    # x shape (...), points shape (N,)
    # compute ((x - points)/h)**2 sum over points
    # We want pdf(x) = (1 / (N * h * sqrt(2pi))) * sum exp(-0.5 * ((x - points)/h)**2)
    # Vectorized: (x[...,None] - points[None,...]) -> shape (..., N)
    diff = (x.unsqueeze(-1) - points.unsqueeze(0)) / h  # (..., N)
    exponent = -0.5 * diff * diff
    # Use log-sum-exp style? not necessary for typical sizes
    pdf = torch.exp(exponent).sum(dim=-1) / (N * h * math.sqrt(2 * math.pi))
    return pdf


def compute_BI_torch(data_pileup,
                     data_single,
                     acceptance,
                     H_unit,
                     f1_opt,
                     f2_opt,
                     K=None,
                     window_fct=torch.ones,
                     n_deriv=0,
                     compute_uncertainty=False,
                     full_output=False,
                     **kwargs):
    """
    Torch version of compute_BI. Returns BI and rp (and optionally full output).
    - data_pileup, data_single: datasets acceptable by get_PSD_interpole_torch
    - acceptance: float in (0,1)
    - H_unit, f1_opt, f2_opt: passed to get_PSD_interpole_torch
    - K: prefactor (defaults to fn.K if None)
    - window_fct, n_deriv forwarded
    - kwargs forwarded to get_PSD_interpole_torch (batch_size, etc.)
    """
    if K is None:
        try:
            K = fn.K
        except NameError:
            raise ValueError("K not provided and fn.K not found. Pass K explicitly.")

    # Expect PSDs and amps as torch Tensors
    PSD_pileup, Amp_1_pileup, Amp_2_pileup = get_PSD_interpole_torch(
        data_pileup, H_unit, f1_opt, f2_opt, window_fct=window_fct, n_deriv=n_deriv, **kwargs
    )
    PSD_single, Amp_1_single, Amp_2_single = get_PSD_interpole_torch(
        data_single, H_unit, f1_opt, f2_opt, window_fct=window_fct, n_deriv=n_deriv, **kwargs
    )

    # ensure 1-D tensors
    PSD_pileup = PSD_pileup.flatten()
    PSD_single = PSD_single.flatten()

    # percentile: torch.quantile expects q in [0,1]
    # We want the value at 100 - acceptance*100 percentile => 1 - acceptance
    q = torch.tensor(1.0 - acceptance, device=PSD_single.device, dtype=PSD_single.dtype)
    cut = torch.quantile(PSD_single, q)

    rp = torch.mean((PSD_pileup < cut).to(dtype=PSD_pileup.dtype))  # fraction of pileup below cut
    BI = K * (1.0 - rp)
    if compute_uncertainty:
        sigma_rp, sigma_BI = compute_BI_uncertainty_torch(
            PSD_single, PSD_pileup, acceptance, rp, K=K
        )
        if full_output:
            return BI, rp,sigma_rp, sigma_BI, PSD_pileup, PSD_single, Amp_1_pileup, Amp_2_pileup, Amp_1_single, Amp_2_single
        else:
            return BI, rp, sigma_rp, sigma_BI
    if full_output:
        return BI, rp, PSD_pileup, PSD_single, Amp_1_pileup, Amp_2_pileup, Amp_1_single, Amp_2_single
    else:
        return BI, rp


def compute_BI_uncertainty_torch(PSD_single: torch.Tensor,
                                 PSD_pileup: torch.Tensor,
                                 acceptance: float,
                                 rp: torch.Tensor,
                                 K=None,
                                 eps: float = 1e-12):
    """
    Torch version of compute_BI_uncertainty.

    - PSD_single, PSD_pileup: 1D torch tensors
    - acceptance: float in (0,1)
    - rp: torch scalar (fraction of pileup below cut)
    - K: optional prefactor (defaults to fn.K)
    Returns: (sigma_rp, sigma_BI) as torch scalars (same device/dtype as inputs)
    """
    device = PSD_single.device
    dtype = PSD_single.dtype

    if K is None:
        try:
            K = fn.K
        except NameError:
            raise ValueError("K not provided and fn.K not found. Pass K explicitly.")

    N_single = PSD_single.numel()
    N_pileup = PSD_pileup.numel()

    if N_single == 0 or N_pileup == 0:
        raise ValueError("PSD_single and PSD_pileup must be non-empty tensors")

    # --- 1. Uncertainty from pileup binomial statistics ---
    # rp is a torch scalar
    sigma_rp_pileup = torch.sqrt(rp * (1.0 - rp) / N_pileup)

    # --- 2. Uncertainty of the percentile cut ---
    q = torch.tensor(1.0 - acceptance, device=device, dtype=dtype)
    cut = torch.quantile(PSD_single, q)

    # estimate PDFs for PSD_single and PSD_pileup using torch KDE
    # evaluate kde at `cut` (scalar)
    cut_tensor = cut.reshape(1)  # shape (1,)
    f_single = _gaussian_kde_torch(cut_tensor, PSD_single, eps=eps)[0].clamp_min(eps)
    f_pileup = _gaussian_kde_torch(cut_tensor, PSD_pileup, eps=eps)[0].clamp_min(eps)

    a = 1.0 - acceptance
    sigma_cut = math.sqrt(a * (1.0 - a)) / (math.sqrt(N_single) * f_single)

    # derivative of rp wrt cut (drp/dcut) = -pdf_pileup(cut)  (for continuous pdf)
    drp_dcut = -f_pileup
    sigma_rp_cut = torch.abs(drp_dcut) * sigma_cut

    # total
    sigma_rp = torch.sqrt(sigma_rp_pileup ** 2 + sigma_rp_cut ** 2)
    sigma_BI = K * sigma_rp

    return sigma_rp, sigma_BI


def align_waveforms(pulses, shift):
    """
    Vectorized version: shift pulses along the time axis WITHOUT circular wrap-around.
    pulses: (N, L)
    shift: (N,) tensor of integer shifts (positive = right)
    """
    N, L = pulses.shape
    aligned = torch.zeros_like(pulses)

    # Create index grid
    idx = torch.arange(L, device = pulses.device).unsqueeze(0).repeat(N, 1)  # (N, L)
    # Compute shifted indices
    shifted_idx = idx - shift.unsqueeze(1)  # (N, L)

    # Mask valid indices
    mask = (shifted_idx >= 0) & (shifted_idx < L)
    shifted_idx_clipped = torch.clamp(shifted_idx, 0, L - 1)

    # Fill aligned pulses
    aligned[mask] = pulses[torch.arange(N, device = pulses.device).unsqueeze(1).repeat(1, L)[mask],
    shifted_idx_clipped[mask]]
    return aligned


def build_mean_pulse(dataset, rms_thr, batch_size=2048, device="cpu", use_loader=False):
    """
    Compute the mean normalized pulse waveform from a dataset.

    Steps:
        1. Select pulses whose baseline RMS is below `rms_thr`.
        2. Subtract baseline from each pulse.
        3. Align pulses by the position of their maximum amplitude.
        4. Normalize each pulse so its maximum equals 1.
        5. Accumulate and return the average pulse.

    Args:
        dataset: Dataset providing waveforms and a 'win_length' attribute.
        rms_thr (float): RMS threshold for baseline noise filtering.
        batch_size (int): Number of pulses per batch.
        device (str): Device to perform computation on ("cpu" or "cuda").
        use_loader (bool): If True, uses a PyTorch DataLoader; otherwise uses custom iterator.

    Returns:
        torch.Tensor: Mean normalized pulse of shape (win_length,).
    """
    pulse_start_pos, pulse_end_pos = -100, 400
    win_length = dataset.win_length
    loader = batch_iterator(dataset, batch_size=batch_size, device='cpu', use_loader=use_loader)
    target_index = win_length // 2
    baseline_length = win_length // 2 + pulse_start_pos

    mean_pulse = torch.zeros(win_length, dtype=torch.float32, device=device)
    total_count = 0

    for pulses in loader:
        if isinstance(pulses, (list, tuple)):
            pulses = pulses[0]
        pulses = pulses.to(device, dtype=torch.float32)

        baseline_raw = torch.cat((
            pulses[:, :baseline_length],
            pulses[:, win_length // 2 + pulse_end_pos:]
        ), dim=1)

        sel = torch.std(baseline_raw, dim=-1) < rms_thr
        pulses_sel = pulses[sel]
        if pulses_sel.shape[0] == 0:
            continue

        baseline = pulses_sel[:, :baseline_length].mean(dim=1)
        pulses_clean = pulses_sel - baseline[:, None]

        max_pos = torch.argmax(pulses_clean, dim=1)
        shifts = target_index - max_pos

        pulses_aligned = align_waveforms(pulses_clean, shifts)

        max_val = pulses_aligned.max(dim=1).values
        pulses_norm = pulses_aligned / max_val[:, None]

        mean_pulse += pulses_norm.sum(dim=0)
        total_count += pulses_norm.shape[0]

    mean_pulse /= total_count
    return mean_pulse


def build_mean_pulse_filteralignement(dataset, rms_thr, H, batch_size=2048, device="cpu", use_loader=False):
    """
    Compute a mean pulse using frequency-domain filtering + sub-sample alignment.

    Extra steps vs. `build_mean_pulse`:
        - Apply filter H in the frequency domain.
        - Search for peaks within a jitter window.
        - Use cubic interpolation to estimate sub-sample peak position.
        - Apply fractional-time shifting via Fourier shift theorem.

    Args:
        dataset: Dataset providing waveforms and a 'win_length' attribute.
        rms_thr (float): RMS threshold for baseline noise filtering.
        H (torch.Tensor): Frequency-domain filter of shape (win_length,).
        batch_size (int): Number of pulses per batch.
        device (str): Compute device.
        use_loader (bool): Whether to use DataLoader.

    Returns:
        torch.Tensor: Mean filtered+aligned normalized pulse.
    """
    pulse_start_pos, pulse_end_pos = -100, 400
    win_length = dataset.win_length
    loader = batch_iterator(dataset, batch_size=batch_size, device='cpu', use_loader=use_loader)
    target_index = win_length // 2
    baseline_length = win_length // 2 + pulse_start_pos
    jitter_max = 4

    mean_pulse = torch.zeros(win_length, dtype=torch.float32, device=device)
    total_count = 0

    interpolation_range = 5
    fine_x = torch.arange(-interpolation_range, interpolation_range + 1, 0.05, device=device)

    for pulses in loader:
        if isinstance(pulses, (list, tuple)):
            pulses = pulses[0]
        pulses = pulses.to(device, dtype=torch.float32)

        baseline_raw = torch.cat((
            pulses[:, :baseline_length],
            pulses[:, win_length // 2 + pulse_end_pos:]
        ), dim=1)

        sel = torch.std(baseline_raw, dim=-1) < rms_thr
        pulses_sel = pulses[sel]
        if pulses_sel.shape[0] == 0:
            continue

        n_rows, n_cols = pulses_sel.shape

        baseline = pulses_sel[:, :baseline_length].mean(dim=1)
        pulses_clean = pulses_sel - baseline[:, None]

        fpulse = torch.fft.fft(pulses_clean, dim=-1)

        lo = n_cols // 2 - jitter_max
        hi = n_cols // 2 + jitter_max

        filt = torch.fft.ifft(H * fpulse, dim=-1).real
        filt = torch.roll(filt, shifts=n_cols // 2, dims=1)

        sub = filt[:, lo:hi]
        max_idx = sub.argmax(dim=1) + lo

        offs = torch.arange(-interpolation_range, interpolation_range + 1, device=device)
        idx = max_idx[:, None] + offs[None, :]
        y = filt.gather(1, idx)

        interp2 = cubic_interp1d_scipy(y, fine_x)
        shift = fine_x[interp2.argmax(dim=1)]

        phase = torch.exp(-1j * 2 * np.pi * shift[:, None] / win_length)
        pulses_aligned = torch.fft.ifft(fpulse * phase, dim=-1).real

        max_val = pulses_aligned.max(dim=1).values
        pulses_norm = pulses_aligned / max_val[:, None]

        mean_pulse += pulses_norm.sum(dim=0)
        total_count += pulses_norm.shape[0]

    mean_pulse /= total_count
    return mean_pulse
