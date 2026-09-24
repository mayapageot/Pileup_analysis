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
channels = [3, 5, 9, 11, 13, 15, 17, 19]
channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meas_name = "000813_20230628T161508"
acceptance = 0.9
N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100) #number of sigma corresponding to the desired acceptance
#define the time and ratio grid for the analysis
t_min, t_max, N_t = 0, 8e-4, 100
r_min, r_max, N_r = 0., .5, 100
#get the corresponding ratio distribution
ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
ratio_distribution /= np.mean(ratio_distribution)

window_size = 4096 #size of the window used for the analysis, should be the same as the one used for the NPS computation
n_trials =1000 #number of trials for the optimization of the filters


sampling_rate = 10000
BIs = []
BI_uncertainties = []
scaling_factors = []
Ms = []
channel_data = []
for channel in channels:
    #load the meanpulse
    meanpulse = np.load(f"{BASE}/outputs/meanpulses_build/channel{channel}_meanpulse_mean.npy")
    if len(meanpulse) > window_size:
        meanpulse = meanpulse[np.argmax(meanpulse) - window_size//2 : np.argmax(meanpulse) + window_size//2]
    #load the NPS
    nps = np.load(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}.npy")
    nps *= (8. / 3.)  # Adjust for Hann window effect

    S, w, H_unit = an.compute_H(meanpulse, nps, np.hanning,sampling_rate=sampling_rate)

    S_torch = torch.tensor(S, dtype=torch.cfloat, device=device)
    H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
    w_torch = torch.tensor(w, dtype=torch.cfloat, device=device)
    nps_torch = torch.tensor(nps, dtype=torch.cfloat, device=device)
    t_torch = torch.linspace(t_min, t_max, N_t, dtype=torch.cfloat, device=device)
    r_torch = torch.linspace(r_min, r_max, N_r, dtype=torch.cfloat, device=device)
    ratio_distribution_torch = torch.tensor(ratio_distribution, dtype=torch.cfloat, device=device)
    for scaling_factor in [0.6 ,0.7]:
        signal_amp = ds.get_amp_Q_val(channel)*scaling_factor #expected amplitude of the signal at the Q-value
        signal_amp_torch = torch.tensor(signal_amp, dtype=torch.float32, device=device)

        f1_t, f2_t, M_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
                                                   signal_amp_torch, ratio_distribution_torch, N_sigma=N_sigma,
                                                   activation_fct = torch.abs,
                                                   f1_init = None, f2_init = None,
                                                   n_trials=n_trials, use_interp = True, verbose=False)
        M = float(M_values[-1])
        f1_opt,f2_opt = f1_t.cpu().numpy(), f2_t.cpu().numpy()
        np.save(f"{BASE}/outputs/training_Ms/channel_{channel}/M_{int(acceptance * 100):d}_scaling_{scaling_factor:.1f}", np.array(M_values))
        np.save(f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/functions_eff{int(acceptance * 100):d}",np.array([f1_opt, f2_opt]))

        #load the pulse to be injected
        meanpulse_inj = np.load(f"{BASE}/outputs/meanpulses_build/pos_pulses_rawdata/average_pulse_channel{channel}.npy")

        dataset_single_name = "pup_n1-d0_000813_20230628T161508"
        file_path_single, path_pos_single = ds.find_files(meas_name, dataset_single_name, channel,
                                                          specific_subdir="RUN9_pulse_injected_new/")
        data_single = ds.CachedBinaryDataset_withgenerated(file_path_single, path_pos_single, window_size,
                                                           pulse=meanpulse_inj,scaling_factor=scaling_factor,
                                                           n_windows=15999, win_shift=0)
        #load the pileup pulse dataset
        dataset_pileup_name = "pup_n1-d8_000813_20230628T161508"
        file_path_pileup, path_pos_pileup = ds.find_files(meas_name, dataset_pileup_name, channel,
                                                          specific_subdir="RUN9_pulse_injected_new/")
        data_pileup = ds.CachedBinaryDataset_withgenerated(file_path_pileup, path_pos_pileup, window_size,
                                                           pulse=meanpulse_inj,scaling_factor=scaling_factor,
                                                           n_windows=15999, win_shift=0)

        BI, rp, sigma_rp, sigma_BI = an.compute_BI_torch(
            data_pileup,
            data_single,
            acceptance,
            H_unit_torch,
            f1_t,
            f2_t,
            window_fct = np.hanning,
            n_deriv = 0,
            compute_uncertainty = True,
            batch_size = 2048 * 2,
            use_loader = False
        )
        BIs.append(float(BI))
        BI_uncertainties.append(float(sigma_BI))
        Ms.append(M)
        channel_data.append(channel)
        scaling_factors.append(scaling_factor)

        data = np.column_stack([np.array(channel_data), np.array(scaling_factors), np.array(BIs),
                                np.array(BI_uncertainties), np.array(Ms)])

        header = "channel,scaling_factor,BI,BI_uncertainty,M"

        np.savetxt(
            f"{BASE}/outputs/Systematics/BI_scaling_factor_bis.csv",
            data,
            delimiter = ",",
            header = header,
            comments = ""
        )
        print("Results saved to outputs/Systematics/BI_scaling_factor_bis.csv")
