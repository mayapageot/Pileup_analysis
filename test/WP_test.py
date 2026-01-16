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
channel=3
LED_amp = [0.028401334, 0.032891195, 0.035842195, 0.038979053, 0.03883822, 0.031177562, 0.021730503, 0.016601065, 0.0141078625]
measurements = [779,    780,    781,    782,    783,    784,    785,    786,    787]
ref_meas = 780
ref_sensitivity = 1.4
ref_amplitude = sim.ROI_amp_from_sensitivity(ref_sensitivity)
WP_gain = np.array(LED_amp)/LED_amp[measurements.index(ref_meas)]
Gains_80V = 13.6
J_values_all = []
meas_values = []
NTL_noise_factor = 1e-7
for meas in measurements:
    mean_pulse_filtered = np.fromfile(f"{BASE}/outputs/meanpulses_build/meanpulse_measurement_{meas}_channel{channel}.bin")
    nps = (np.fromfile(f"{BASE}/outputs/NPS_study/NPS_measurement_{meas}_channel{channel}.bin") +
           NTL_noise_factor*np.abs(np.fft.fft(mean_pulse_filtered))**2)
    S, w, H_unit = an.compute_H(mean_pulse_filtered, nps, np.hanning)
    signal_amp = ref_amplitude * WP_gain[measurements.index(meas)] * Gains_80V
    acceptance = 0.9
    # Convert all arrays to torch tensors on GPU
    N_sigma = norm.ppf(1 - (1 - acceptance) * 100 / 100)
    S_torch = torch.tensor(S, dtype = torch.cfloat, device = device)
    H_unit_torch = torch.tensor(H_unit, dtype = torch.cfloat, device = device)
    w_torch = torch.tensor(w, dtype = torch.cfloat, device = device)
    nps_torch = torch.tensor(nps, dtype = torch.cfloat, device = device)
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
    directory_path = f"{BASE}/outputs/Pileup_filter_functions/channel_{channel}/WP_test/"
    os.makedirs(directory_path, exist_ok = True)
    np.save(directory_path +
            f"channel{channel}_functions_eff{int(acceptance * 100):d}_WP_test_{meas}",
            np.array([f1_opt, f2_opt]))
    np.save(directory_path +
            f"channel{channel}_J_{int(acceptance * 100):d}_WP_test_{meas}",
            np.array(J_values))
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
    data =np.column_stack([np.array(meas_values), np.array(J_values_all)])
    print(f"Measurement {meas}: J = {J:.6e}")
    header = "measurement,J_value"
    np.savetxt(
        f"{BASE}/outputs/Systematics/BI_WP.csv",
        data,
        delimiter = ",",
        header = header
    )
print("Done.")