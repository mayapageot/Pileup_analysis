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
from scipy.optimize import root_scalar
from scipy.stats import norm
import os
# Ensure we have GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
J_values_all = []
meas_values = []
channel_list = []
directory = f"/feynman/work/projets/cupid/data/Ulisse/RUN121/"
rms_amp_data = np.loadtxt(directory+"rms_amp_RUN121.csv",delimiter=",",dtype=str)[1:]

for data in rms_amp_data:
    file_name = data[0]
    meas = int(data[4])
    channel = int(data[3])
    amp = float(data[2])
    try:
        mean_pulse_filtered = np.load(directory+f"MP_RUN121_{file_name.split('.bin')[0]}.npy")
        nps = np.load(directory+f"NPS_RUN121_{file_name.split('.bin')[0]}.npy")
    except:
        continue
    S, w, H_unit = an.compute_H(mean_pulse_filtered, nps, np.hanning)
    signal_amp = amp*0.1
    if signal_amp==0:
        continue
    acceptance = 0.9
    # Convert all arrays to torch tensors on GPU
    N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
    S_torch = torch.tensor(S, dtype = torch.cfloat, device = device)
    H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
    w_torch = torch.tensor(w, dtype = torch.cfloat, device = device)
    nps_torch = torch.tensor(nps, dtype = torch.cfloat, device = device)
    signal_amp_torch = torch.tensor(signal_amp, dtype = torch.float32, device = device)

    t_min, t_max, N_t = 0, 8e-4, 100
    r_min, r_max, N_r = 0., .5, 100

    ratio_distribution = pdf_ratio2b(np.linspace(r_min, r_max, N_r))
    ratio_distribution /= np.mean(ratio_distribution)

    t_torch = torch.linspace(t_min, t_max, N_t, dtype = torch.cfloat, device = device)
    r_torch = torch.linspace(r_min, r_max, N_r, dtype = torch.cfloat, device = device)
    ratio_distribution_torch = torch.tensor(ratio_distribution, dtype = torch.cfloat, device = device)
    try:
        f1_opt, f2_opt, J_values = an.optimize_filters(S_torch, H_unit_torch, w_torch, t_torch, r_torch, nps_torch,
                                                       signal_amp_torch, ratio_distribution_torch, N_sigma = N_sigma,
                                                       n_trials = 500, verbose = False)
    except:
        continue
    # directory_path = f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/WP_test/"
    # os.makedirs(directory_path, exist_ok = True)
    # np.save(directory_path +
    #         f"channel{channel}_functions_eff{int(acceptance * 100):d}_WP_test_{meas}",
    #         np.array([f1_opt, f2_opt]))
    # np.save(directory_path +
    #         f"channel{channel}_J_{int(acceptance * 100):d}_WP_test_{meas}",
    #         np.array(J_values))
    f1_t = torch.tensor(f1_opt, dtype = torch.float32, device = device)
    f2_t = torch.tensor(f2_opt, dtype = torch.float32, device = device)

    S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)

    f1_t /= torch.mean(f1_t * S_H).real
    f2_t /= torch.mean(f2_t * S_H).real

    S_H_delayed, S_H, S2_over_nps = an.precompute_constants(S_torch, H_unit_torch, w_torch, t_torch, nps_torch)
    J = an.compute_J(f1_t, f2_t, S_H_delayed, r_torch, S_H, S2_over_nps, signal_amp_torch, ratio_distribution_torch,
                     N_sigma = N_sigma, full_output = False, use_interp = True)
    J_values_all.append(J.cpu().numpy())
    meas_values.append(meas)
    channel_list.append(channel)
    data =np.column_stack([np.array(channel_list),np.array(meas_values), np.array(J_values_all)])
    print(f"Measurement {meas}: J = {J:.6e}")
    header = "channel,measurement,J_value"
    np.savetxt(
        f"{BASE}/outputs/Systematics/BI_WP_RUN121_Ulysse.csv",
        data,
        comments="",
        delimiter = ",",
        header = header
    )
print("Done.")