import sys
from pathlib import Path
# Add src to Python path dynamically
sys.path.append(str(Path(__file__).resolve().parent.parent))
import time
import numpy as np
import src.analysis as an
import src.dataset as ds
from utility.double_beta_spectrum import pdf_ratio2b
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from scipy.stats import norm
# Ensure we have GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

channels = [11, 13, 15]  # channels to process
channels = [3, 5, 9, 11, 13, 15, 17, 19]
channels = [9]
for channel in channels:
    # Load datasets
    # data_single, data_pileup,data_noise, _ = ds.create_data_sets(channel,
    #                                                   {"data_name": ["pup_n1-d8_000813_20230628T161508",
    #                                                                  "pup_n1-d0_000813_20230628T161508",
    #                                                                  "000813_20230628T161508"],
    #                                                    "pos_name": ["pup_n1-d8_000813_20230628T161508",
    #                                                                 "pup_n1-d0_000813_20230628T161508",
    #                                                                 "pup_n1-d0_000813_20230628T161508"],
    #                                                    "specific_subdir": ["","",""],
    #                                                    "full_info": [False, False, False]},
    #                                                   win_length=2000, shift=0, pos_prefix="_bis",len_data=2**13+2**12, cached=True)
    data_single, data_pileup, _ = ds.create_data_sets(channel,win_length=2000, shift=0, pos_prefix="_bis",
                                                                  len_data=2 ** 13 + 2 ** 12, cached=True)
    acceptance = 0.9
    n_deriv=0
    meas_name = "000813_20230628T161508"
    w = np.fft.fftfreq(2000, 1e-4) * 2 * np.pi

    H_unit, S, nps, w, pulse_factor, dmeanpulse = ds.get_channel_specs(channel, n_deriv=n_deriv,
                                                           window_fct=np.hanning)  # , nps_name="denoised_noise"
    signal_amp = ds.get_amp_Q_val(channel) * pulse_factor
    t_min, t_max, N_t = 0, 8e-4, 100
    r_min, r_max, N_r = 0., .5, 25

    ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
    ratio_distribution /= np.mean(ratio_distribution)

    meanpulse = np.fromfile(ds.find_file(f"{meas_name}_{channel:03}_???.bin_edmean.bin",
                                      specific_subdir="RUN9_pulse_injected_new/"))

    f1_opt, f2_opt = np.load(
        f"/feynman/home/dphp/mp274748/Pileup_Analysis/outputs/Pileup_filter_functions/channel_{channel}/functions_eff{int(acceptance * 100):d}_d0.npy"
    )
    f1_opt_inter,f2_opt_inter = interp1d(w,f1_opt,fill_value="extrapolate"),interp1d(w,f2_opt,fill_value="extrapolate")
    nps_inter = interp1d(w,nps,fill_value="extrapolate")

    t_torch = torch.linspace(t_min, t_max, N_t, dtype=torch.cfloat, device=device)
    r_torch = torch.linspace(r_min, r_max, N_r, dtype=torch.cfloat, device=device)
    ratio_distribution_torch = torch.tensor(ratio_distribution, dtype=torch.cfloat, device=device)

    N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
    S_torch = torch.tensor(S, dtype=torch.cfloat, device=device)
    H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
    w_torch = torch.tensor(w, dtype=torch.cfloat, device=device)
    nps_torch = torch.tensor(nps, dtype=torch.cfloat, device=device)
    signal_amp_torch = torch.tensor(signal_amp, dtype=torch.float32, device=device)

    f1_t = torch.tensor(f1_opt, dtype=torch.float32, device=device)
    f2_t = torch.tensor(f2_opt, dtype=torch.float32, device=device)

    S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)
    J = an.compute_J(f1_t, f2_t, S_H_delayed, r_torch, S_H, S2_over_nps, signal_amp_torch, ratio_distribution_torch,
                     N_sigma=N_sigma)
    BI, rp, PSD_pileup, PSD_single, Amp_1_pileup, Amp_2_pileup, Amp_1_single, Amp_2_single = an.compute_BI_torch(
        data_pileup,
        data_single,
        acceptance,
        H_unit_torch,
        f1_t,
        f2_t,
        window_fct=np.hanning,
        n_deriv=0,
        full_output=True,
        batch_size=2048 * 2,
        use_loader=False
    )
    print(float(1 - rp), J)
    # for window_size in [4096, 3584, 3072, 2560, 2048, 1536, 1024, 512]:
    #     t0 = time.time()
    #     data_single.set_win_length(window_size)
    #     data_pileup.set_win_length(window_size)
    #     w = np.fft.fftfreq(window_size,1e-4)*2*np.pi
    #     nps_bis = nps_inter(w)
    #     meanpulse_bis = meanpulse[np.argmax(meanpulse) - window_size // 2: np.argmax(meanpulse) + window_size // 2]
    #     S,w,H_unit = ds.compute_H(meanpulse_bis, nps_bis, np.hanning)
    #     # Convert all arrays to torch tensors on GPU
    #     N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
    #     S_torch = torch.tensor(S, dtype=torch.cfloat, device=device)
    #     H_unit_torch = torch.tensor(H_unit, dtype=torch.cfloat, device=device)
    #     w_torch = torch.tensor(w, dtype=torch.cfloat, device=device)
    #     nps_torch = torch.tensor(nps_bis, dtype=torch.cfloat, device=device)
    #     signal_amp_torch = torch.tensor(signal_amp, dtype=torch.float32, device=device)
    #
    #     # f1_opt, f2_opt, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
    #     #                                                signal_amp_torch, ratio_distribution_torch, N_sigma=N_sigma,
    #     #                                                n_trials=500, verbose=True)
    #     # print(f"Channel {channel}, window size {window_size}, time taken: {time.time() - t0:.2f} seconds, J: {J_values[-1]:.4e}")
    #     # Call your analysis function on GPU
    #
    #     f1_t = torch.tensor(f1_opt_inter(w), dtype=torch.float32, device=device)
    #     f2_t = torch.tensor(f2_opt_inter(w), dtype=torch.float32, device=device)
    #
    #     S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)
    #     J = an.compute_J(f1_t, f2_t, S_H_delayed, r_torch, S_H, S2_over_nps, signal_amp_torch, ratio_distribution_torch, N_sigma=1.28)
    #
    #     BI, rp, PSD_pileup, PSD_single, Amp_1_pileup, Amp_2_pileup, Amp_1_single, Amp_2_single = an.compute_BI_torch(
    #         data_pileup,
    #         data_single,
    #         acceptance,
    #         H_unit_torch,
    #         f1_t,
    #         f2_t,
    #         window_fct=np.hanning,
    #         n_deriv=0,
    #         full_output=True,
    #         batch_size = 2048*2,
    #         use_loader = False
    #     )
    #     print(float(1-rp),J)