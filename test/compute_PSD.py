import sys
from pathlib import Path
# Add src to Python path dynamically
BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE))
import time
import numpy as np
import src.analysis as an
import src.dataset as ds
import src.simulation as sim
from utility.double_beta_spectrum import pdf_ratio2b
import torch
from scipy.stats import norm
n_deriv = 0
window_size = 4096
meas_name = "000813_20230628T161508"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
BIs = []
BI_uncertainties = []
channel_list = []
M_list = []
acceptance_list = []
channels = [3, 5, 9, 11, 13, 15, 17, 19]
channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]



for channel, StdCut in zip(channels, StdCuts):
    fitted_pulse_function = np.load(f"{BASE}/outputs/meanpulses_build/pos_pulses_rawdata/average_pulse_channel{channel}.npy")
    # Load datasets and channel specifications
    file_path = ds.find_file(f"000813_20230628T161508_{channel:03}_000.bin")
    dataset_single_name = "pup_n1-d0_000813_20230628T161508"
    file_path_single, path_pos_single = ds.find_files(meas_name, dataset_single_name, channel,
                                                      specific_subdir="RUN9_pulse_injected_new/")
    data_single = ds.CachedBinaryDataset_withgenerated(file_path_single, path_pos_single, window_size,
                                                       pulse=fitted_pulse_function,
                                                       n_windows=15999, win_shift=0)

    dataset_pileup_name = "pup_n1-d8_000813_20230628T161508"
    file_path_pileup, path_pos_pileup = ds.find_files(meas_name, dataset_pileup_name, channel,
                                                      specific_subdir="RUN9_pulse_injected_new/")
    data_pileup = ds.CachedBinaryDataset_withgenerated(file_path_pileup, path_pos_pileup, window_size,
                                                       pulse=fitted_pulse_function,
                                                       n_windows=15999, win_shift=0)
    # data_single, data_pileup = ds.create_data_sets_denoise(channel, win_length=2000, shift=0, pos_prefix="_bis")
    nps_bis = np.load(f"{BASE}/outputs/NPS_study/NPS_channel{channel}_win{window_size}_{meas_name}.npy")
    nps_bis *= (8. / 3.)  # Adjust for Hann window effect

    meanpulse_bis = np.load(f"{BASE}/outputs/meanpulses_build/channel{channel}_meanpulse_mean.npy")
    signal_amp = ds.get_amp_Q_val(channel)
    S, w, H_unit = an.compute_H(meanpulse_bis, nps_bis, np.hanning)
    # Convert all arrays to torch tensors on GPU
    S_torch = torch.tensor(S, dtype=torch.cfloat, device=device)
    H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
    w_torch = torch.tensor(w, dtype=torch.cfloat, device=device)
    nps_torch = torch.tensor(nps_bis, dtype=torch.cfloat, device=device)
    signal_amp_torch = torch.tensor(signal_amp, dtype=torch.float32, device=device)

    t_min, t_max, N_t = 0, 8e-4, 100
    r_min, r_max, N_r = 0., .5, 100
    ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
    ratio_distribution /= np.mean(ratio_distribution)

    t_torch = torch.linspace(t_min, t_max, N_t, dtype=torch.cfloat, device=device)
    r_torch = torch.linspace(r_min, r_max, N_r, dtype=torch.cfloat, device=device)
    ratio_distribution_torch = torch.tensor(ratio_distribution, dtype=torch.cfloat, device=device)
    # Loop over acceptance levels
    acceptances = [0.7, 0.8, 0.84, 0.87, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.99]
    acceptance = 0.9
    f1_opt, f2_opt = np.load(f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/functions_eff{int(acceptance*100):d}.npy")
    f1_t, f2_t = torch.tensor(f1_opt, dtype=torch.cfloat, device=device), torch.tensor(f2_opt, dtype=torch.cfloat, device=device)

    BI, rp, PSD_pileup, PSD_single, Amp_1_pileup, Amp_2_pileup, Amp_1_single, Amp_2_single = an.compute_BI_torch(
        data_pileup,
        data_single,
        acceptance,
        H_unit_torch,
        f1_t,
        f2_t,
        window_fct = np.hanning,
        compute_uncertainty = False,
        batch_size = 2048,
        use_loader = False,
        full_output = True
    )
    print(np.shape(PSD_pileup), np.shape(PSD_single))
    np.save(f"{BASE}/outputs/PSDs/PSD_channel{channel}_win{window_size}_{meas_name}_acc{int(acceptance*100):d}.npy",
            np.array([PSD_pileup.numpy(), PSD_single.numpy()]))

