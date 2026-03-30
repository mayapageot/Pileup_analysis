import sys
from pathlib import Path
# Add src to Python path dynamically
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import numpy as np
import src.analysis as an
import src.dataset as ds
import src.simulation as sim
from utility.double_beta_spectrum import pdf_ratio2b
import torch
from scipy.stats import norm
import os
# Ensure we have GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]
channels = [3, 5, 9, 11, 13, 15, 17, 19]
fp_model = lambda *params: sim.filtered_pulse_model(*params, fc = 2500, order = 6)

Js = []
channel_data = []

fit_parameters = np.loadtxt(f"{BASE}/outputs/fits/pulse_fit_params.csv", delimiter = ',', skiprows = 1)

for channel, StdCut in zip(channels, StdCuts):
    window_size = 2048
    meas_name = "000813_20230628T161508"
    meanpulse = np.fromfile(ds.find_file(f"{meas_name}_{channel:03}_???.bin_edmean.bin",
                                         specific_subdir="RUN9_pulse_injected_new/"))
    fitted_pulse_function = meanpulse
    dataset_single_name = "pup_n1-d0_000813_20230628T161508"
    file_path_single, path_pos_single = ds.find_files(meas_name, dataset_single_name, channel,
                                                      specific_subdir = "RUN9_pulse_injected_new/")
    data_single = ds.CachedBinaryDataset_withgenerated(file_path_single, path_pos_single, window_size,
                                                       pulse = fitted_pulse_function,
                                                       n_windows = 12288, win_shift = 0)
    nps_bis = np.fromfile(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}_denoised.npy")
    nps_bis *= (8./3.)  # Adjust for Hann window effect
    acceptance = 0.9
    n_deriv = 0

    _, _, _, _, pulse_factor, _ = ds.get_channel_specs(channel, n_deriv = n_deriv,
                                                       window_fct = np.hanning)  # , nps_name="denoised_noise"
    signal_amp = ds.get_amp_Q_val(channel) * pulse_factor

    # meanpulse_bis = fitted_pulse_function(np.arange(-window_size//2,window_size//2))
    mean_pulse = an.build_mean_pulse(data_single, rms_thr = StdCut, batch_size = 2048, device = device,
                                     use_loader = False).cpu().numpy()
    S, w, H_unit = an.compute_H(mean_pulse, nps_bis, np.hanning)
    H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
    mean_pulse_filtered = an.build_mean_pulse_filteralignement(data_single, rms_thr = StdCut,
                                                               H = H_unit_torch, batch_size = 2048,
                                                               device = device, use_loader = False).cpu().numpy()
    S, w, H_unit = an.compute_H(mean_pulse_filtered, nps_bis, np.hanning)

    # Convert all arrays to torch tensors on GPU
    N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
    S_torch = torch.tensor(S, dtype = torch.cfloat, device = device)
    H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
    w_torch = torch.tensor(w, dtype = torch.cfloat, device = device)
    nps_torch = torch.tensor(nps_bis, dtype = torch.cfloat, device = device)
    signal_amp_torch = torch.tensor(signal_amp, dtype = torch.float32, device = device)

    t_min, t_max, N_t = 0, 8e-4, 50
    r_min, r_max, N_r = 0., .5, 50

    ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
    ratio_distribution /= np.mean(ratio_distribution)

    t_torch = torch.linspace(t_min, t_max, N_t, dtype = torch.cfloat, device = device)
    r_torch = torch.linspace(r_min, r_max, N_r, dtype = torch.cfloat, device = device)
    ratio_distribution_torch = torch.tensor(ratio_distribution, dtype = torch.cfloat, device = device)

    f1_opt, f2_opt, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
                                                   signal_amp_torch, ratio_distribution_torch, N_sigma = N_sigma,
                                                   n_trials = 500, verbose = False)
    directory_path = f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/denoise_test/"
    os.makedirs(directory_path, exist_ok = True)
    np.save(directory_path +
            f"channel{channel}_functions_eff{int(acceptance * 100):d}_denoise_test",
            np.array([f1_opt, f2_opt]))
    np.save(directory_path +
            f"channel{channel}_J_{int(acceptance * 100):d}_denoise_test",
            np.array(J_values))
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
    np.savetxt(f"{BASE}/outputs/Systematics/denoise_test.csv",
               np.array([channel_data, Js]).T,
               delimiter = ',', header = 'channel,J', comments = '')
