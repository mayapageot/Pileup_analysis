import sys
from pathlib import Path
# Add src to Python path dynamically
sys.path.append(str(Path(__file__).resolve().parent.parent))
import time
import numpy as np
import src.analysis as an
import src.dataset as ds
import src.simulation as sim
from utility.double_beta_spectrum import pdf_ratio2b
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from scipy.stats import norm
import os
# Ensure we have GPU available
BASE = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]
channels = [3, 5, 9, 11, 13, 15, 17, 19]
# channels = [11, 13, 15]
# channels = [11]
grid_Ns = [i*10 for i in range(1,20)]

BIs = []
rps = []
Js = []
grid_Ns_all = []
channel_data = []
for channel, StdCut in zip(channels, StdCuts):
    window_size = 4096
    meas_name = "000813_20230628T161508"

    nps_bis = np.load(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}.npy")
    nps_bis *= (8. / 3.)  # Adjust for Hann window effect
    acceptance = 0.9
    n_deriv = 0
    _, _, _, _, pulse_factor, _ = ds.get_channel_specs(channel, n_deriv=n_deriv,window_fct=np.hanning)
    signal_amp = ds.get_amp_Q_val(channel) * pulse_factor
    mean_pulse_filtered = np.load(f"{BASE}/outputs/meanpulses_build/channel{channel}_meanpulse_time_0.npy")
    S, w, H_unit = an.compute_H(mean_pulse_filtered, nps_bis, np.hanning)

    # Convert all arrays to torch tensors on GPU
    N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
    S_torch = torch.tensor(S, dtype = torch.cfloat, device = device)
    H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
    w_torch = torch.tensor(w, dtype = torch.cfloat, device = device)
    nps_torch = torch.tensor(nps_bis, dtype = torch.cfloat, device = device)
    signal_amp_torch = torch.tensor(signal_amp, dtype = torch.float32, device = device)


    for grid_N in grid_Ns:
        t_min, t_max, N_t = 0, 8e-4, grid_N
        r_min, r_max, N_r = 0., .5, grid_N

        ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
        ratio_distribution /= np.mean(ratio_distribution)

        t_torch = torch.linspace(t_min, t_max, N_t, dtype=torch.cfloat, device=device)
        r_torch = torch.linspace(r_min, r_max, N_r, dtype=torch.cfloat, device=device)
        ratio_distribution_torch = torch.tensor(ratio_distribution, dtype=torch.cfloat, device=device)

        f1_opt, f2_opt, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
                                                       signal_amp_torch, ratio_distribution_torch, N_sigma = N_sigma,
                                                       n_trials = 500, verbose = False)

        f1_t = torch.tensor(f1_opt, dtype = torch.float32, device = device)
        f2_t = torch.tensor(f2_opt, dtype = torch.float32, device = device)

        S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)

        f1_t /= torch.mean(f1_t * S_H).real
        f2_t /= torch.mean(f2_t * S_H).real

        S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)
        J = an.compute_J(f1_t, f2_t, S_H_delayed, r_torch, S_H, S2_over_nps, signal_amp_torch, ratio_distribution_torch,
                         N_sigma = N_sigma, full_output = False, use_interp = True)
        Js.append(J.item())
        channel_data.append(channel)
        grid_Ns_all.append(grid_N)


        data = np.column_stack([np.array(channel_data), np.array(grid_Ns_all), np.array(Js)])

        header = "channel,grid_N,J"

        np.savetxt(
            f"{BASE}/outputs/Systematics/BI_varying_grid_size.csv",
            data,
            delimiter=",",
            header=header,
            comments=""
        )
        print("Results saved to outputs/Systematics/BI_varying_grid_size.csv")
