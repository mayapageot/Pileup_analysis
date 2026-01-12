import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from matplotlib.ticker import LogLocator
import pandas as pd
def plot_psd_vs_dt(PSD_single, PSD_pileup, dt_pileup, cut=0.9, bins=40,PSD_min=-0.01,PSD_max =0.02):
    """
    Plot the Pulse Shape Discriminator (PSD) as a function of time difference (dt).

    This function generates a plot showing the relationship between PSD and dt for
    pile-up simulated pulses, along with a histogram of PSD values for single and
    pile-up pulses.

    Args:
        PSD_single (numpy.ndarray): PSD values for single simulated pulses.
        PSD_pileup (numpy.ndarray): PSD values for pile-up simulated pulses.
        dt_pileup (numpy.ndarray): Time differences for pile-up simulated pulses.
        cut (float, optional): Acceptance cut value for PSD. Defaults to 0.9.
        bins (int, optional): Number of bins for the histogram. Defaults to 40.

    Returns:
        tuple: A tuple containing the figure, main axis, and inset axis objects.
    """
    dt_pileup_sel = dt_pileup[np.abs(PSD_pileup - 1) < 0.05]
    PSD_double_sel = PSD_pileup[np.abs(PSD_pileup - 1) < 0.05]
    bin_means, _, _ = binned_statistic(
        dt_pileup_sel * 1e-1, 1-PSD_double_sel, np.mean, bins
    )
    bin_high, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, 1-PSD_double_sel, lambda x: np.percentile(x, 84.1), bins
    )
    bin_low, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, 1-PSD_double_sel, lambda x:np.percentile(x,15.9), bins
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hlines(1-cut, 0, 0.8, colors="#fc342a", label="90% acceptance cut")
    ax.errorbar(
        bin_edges[:-1],
        bin_means,
        yerr = [bin_means - bin_low, bin_high - bin_means],
        fmt = "o",
        label = "Pile-up Simulated Pulses",
        color = "#2d508c",
    )
    ax.set_xlim(-0.01, 0.8)
    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("1-PSD")
    ax2 = ax.inset_axes([1.0, 0, 1, 1], sharey=ax)
    hist,_,_ = ax2.hist(
        1-PSD_single,
        bins=100,
        range=(PSD_min,PSD_max),
        alpha=0.9,
        label="Single Simulated Pulses",
        color="#ff9200",
        orientation="horizontal",
    )
    ax2.hist(
        1-PSD_pileup,
        bins=100,
        range=(PSD_min,PSD_max),
        alpha=0.9,
        label="Pile-up Simulated Pulses",
        color="#2d508c",
        orientation="horizontal",
    )

    ax2.hlines(1-cut, 0, hist.max(), colors="#fc342a", label="90% acceptance cut")
    ax2.tick_params(left=False, labelleft=False)
    # ax2.set_xticks([50, 100, 150, 200, 250, 300])
    ax2.set_xlabel("Counts")
    ax.set_xticks(np.arange(0, 0.8, 0.1))
    ax2.legend()
    ax.legend(loc="lower right")

    return fig, ax, ax2

def plot_biais_vs_dt(PSD_single, PSD_pileup, dt_pileup, cut=0.9, bins=40):
    """
    Plot the bias (1 - PSD) as a function of time difference (dt).

    This function generates a plot showing the relationship between the bias
    (1 - PSD) and dt for pile-up simulated pulses, along with a histogram of
    bias values for single and pile-up pulses.

    Args:
        PSD_single (numpy.ndarray): PSD values for single simulated pulses.
        PSD_pileup (numpy.ndarray): PSD values for pile-up simulated pulses.
        dt_pileup (numpy.ndarray): Time differences for pile-up simulated pulses.
        cut (float, optional): Acceptance cut value for the bias. Defaults to 0.9.
        bins (int, optional): Number of bins for the histogram. Defaults to 40.

    Returns:
        tuple: A tuple containing the figure, main axis, and inset axis objects.
    """
    dt_pileup_sel = dt_pileup[np.abs(PSD_pileup - 1) < 0.05]
    PSD_double_sel = PSD_pileup[np.abs(PSD_pileup - 1) < 0.05]
    Biais = 1 - PSD_double_sel

    bin_means, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, Biais, np.mean, bins
    )
    bin_high, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, Biais, lambda x: np.percentile(x, 84.1), bins
    )
    bin_low, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, Biais, lambda x: np.percentile(x, 15.9), bins
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hlines(cut, 0, 0.8, colors="#fc342a", label="90% acceptance cut")
    ax.errorbar(
        bin_edges[:-1],
        bin_means,
        yerr=[bin_means - bin_low, bin_high - bin_means],
        fmt="o",
        label="Pile-up Simulated Pulses",
        color="#2d508c",
    )
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("Pileup Bias (1 - PSD)")

    ax2 = ax.inset_axes([1.0, 0, 1, 1], sharey=ax)
    hist,_,_ = ax2.hist(
        1 - PSD_single,
        bins=100,
        range=(-0.008, 0.04),
        density=True,
        alpha=0.7,
        label="Single Simulated Pulses",
        color="#ff9200",
        orientation="horizontal",
    )
    ax2.hist(
        1 - PSD_pileup,
        bins=100,
        range=(-0.008, 0.04),
        density=True,
        alpha=0.7,
        label="Pile-up Simulated Pulses",
        color="#2d508c",
        orientation="horizontal",
    )
    ax2.hlines(cut, 0, hist.max(), colors="#fc342a", label="90% acceptance cut")
    ax2.tick_params(left=False, labelleft=False)
    ax2.set_xlabel("Counts")
    ax.set_xticks(np.arange(0, 0.8, 0.1))
    ax2.legend()
    ax.legend(loc="lower right")

    return fig, ax, ax2
def plot_compted_mu_sigma():
    """
    Plot a 3D scatter plot of computed values (mu, sigma) for a given channel.

    This function visualizes the computed mu and sigma values for a specific
    channel as a 3D scatter plot, using the acceptance level and precomputed
    data.
    """
    import torch
    from scipy.stats import norm
    from utility.double_beta_spectrum import pdf_ratio2b

    channel = 3
    acceptance = 0.9
    muY, sigmaY = np.load(f"../outputs/mu_sigma_channel_{channel}_eff{int(acceptance * 100):d}.npy")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    t_torch = torch.linspace(-8e-4, 8e-4, 200, dtype=torch.cfloat)
    r_torch = torch.linspace(0, 1, 200, dtype=torch.cfloat)

    A = (1 - norm.cdf((1 - muY - sigmaY[0, 0]) / sigmaY)) * pdf_ratio2b(t_torch.numpy()[:, None])
    r_plot, t_plot = np.meshgrid(r_torch[::5].numpy(), t_torch[::5].numpy())

    ax.scatter(r_plot.reshape(-1), t_plot.reshape(-1), A[::5, ::5].reshape(-1))
    plt.show()


def plot_3d():
    """
    Plot a 3D heatmap of the mean PSD values as a function of time difference (dt)
    and energy-sharing ratio (r) for pile-up simulated pulses.

    This function computes a 2D binned statistic (mean) for the PSD values and
    visualizes the result as a 3D heatmap.
    """
    import src.dataset as ds
    from scipy.stats import binned_statistic_2d

    channel = 11
    data_single, data_pileup, dt_pileup, r_pileup, E_pileups = ds.create_data_sets(
        channel, win_length=2000, shift=-2, pos_prefix="_bis", differenciate=True
    )

    PSD_pileup = np.load(f"../outputs/PSDs/PSD_pileup_c5_a90.npy")
    dt_pileup_sel = dt_pileup[np.abs(PSD_pileup - 1) < 0.02]
    r_pileup_sel = r_pileup[np.abs(PSD_pileup - 1) < 0.02]
    PSD_double_sel = PSD_pileup[np.abs(PSD_pileup - 1) < 0.02]

    dt_bins = np.linspace(min(dt_pileup_sel), max(dt_pileup_sel), 30)
    r_bins = np.linspace(min(r_pileup_sel), max(r_pileup_sel), 2)

    # Compute 2D binned statistic — for example, the mean
    statistic, dt_edges, r_edges, binnumber = binned_statistic_2d(
        dt_pileup_sel,
        r_pileup_sel,
        PSD_double_sel,
        statistic='mean',  # or 'median', 'count', 'sum', 'std'
        bins=[dt_bins, r_bins]
    )

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(dt_edges, r_edges, statistic.T, shading='auto')
    plt.xlabel('dt_pileup_sel')
    plt.ylabel('r_pileup_sel')
    plt.title('Mean PSD_double_sel per bin')
    plt.colorbar(label='Mean PSD_double_sel')
    plt.show()

def J_vs_rp():
    import pandas as pd
    out_data = pd.read_csv('../outputs/BI_results_interp_v2.csv')
    plt.scatter(out_data['J'], 1 - out_data['rp'], c = out_data['channel_ID'], cmap = 'tab10')
    plt.plot([0.08, 0.4], [0.08, 0.4], 'k--')
    plt.xlabel('J')
    plt.ylabel('1 - $r_p$')
    plt.show()


def plot_Trial_vs_J():
    fig, ax = plt.subplots()
    for channel in [3, 5, 9, 11, 13, 15, 17, 19]:
        J_values = np.load(f"../outputs/training_Js/channel_{channel}/J_90_V2.npy")
        ax.plot(J_values, c = "k", alpha = 0.3)
    ax.set_xlabel('Trials', fontsize = 16)
    ax.set_ylabel('J', fontsize = 16)
    ax.set_title('Trials vs J', fontsize = 18)

    # Make tick labels bigger
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    # Make the legend bigger
    ax.legend(fontsize = 14)
    fig.set_size_inches(8, 6)
    fig.savefig("../outputs/plots/J_convergence_all_channels_eff90.pdf", dpi = 300)

def plot_m1_2_all_channel():
    channels = [3, 5, 9, 11, 13, 15, 17, 19]

    f1_all, f2_all = [], []

    fig, ax = plt.subplots(figsize = (8, 6))

    for ch in channels:
        f1, f2 = np.load(f"../outputs/Pileup_filter_functions/channel_{ch}/functions_eff90_V2.npy")
        f1_all.append(np.abs(f1))
        f2_all.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")

    # Convert to arrays
    f1_all = np.array(f1_all)
    f2_all = np.array(f2_all)

    # Median values
    f1_med = np.median(f1_all, axis = 0)
    f2_med = np.median(f2_all, axis = 0)

    # Positive frequencies only
    pos_freq = freq > 0

    ax.plot(freq[pos_freq], f1_med[pos_freq],
            color = "#2d508c", linewidth = 2, label = "Median $m_1$")
    ax.plot(freq[pos_freq], f2_med[pos_freq],
            color = "#fc342a", linewidth = 2, label = "Median $m_2$")

    # Axis styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize = 16)
    ax.set_ylabel("Gain", fontsize = 16)
    ax.set_title("Frequency vs $m_{1,opt}$ & $m_{2,opt}$", fontsize = 18)
    ax.tick_params(axis = "both", which = "major", labelsize = 14)
    ax.legend(fontsize = 14)
    ax.grid(True, which = "both", ls = "--", lw = 0.5)
    fig.tight_layout()
    fig.show()
    fig.savefig("../outputs/plots/Median_filters_channel_all_eff90.pdf", dpi=300)

def plot_m1_2_old_all_channel():
    import src.dataset as ds
    channels = [3, 5, 9, 11, 13, 15, 17, 19]

    f1_all, f2_all = [], []
    f1_all_old, f2_all_old = [], []
    fig, ax = plt.subplots(figsize = (8, 6))

    for ch in channels:
        f1, f2 = np.load(f"../outputs/Pileup_filter_functions/channel_{ch}/functions_eff90_V2.npy")
        f1_all.append(np.abs(f1))
        f2_all.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")
        H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(ch, n_deriv = 1, window_fct = np.hanning)
        f1, f2 = np.ones(2000), np.abs(H_unit * S)
        f1, f2  = f1 / np.mean(np.abs(S * f1 * H_unit)), f2 / np.mean(np.abs(S * f2 * H_unit))
        f1_all_old.append(np.abs(f1))
        f2_all_old.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")
    f1_all_old = np.array(f1_all_old )
    f2_all_old = np.array(f2_all_old)
    f1_med_old = np.median(f1_all_old, axis = 0)
    f2_med_old = np.median(f2_all_old, axis = 0)

    # Convert to arrays
    f1_all = np.array(f1_all)
    f2_all = np.array(f2_all)

    # Median values
    f1_med = np.median(f1_all, axis = 0)
    f2_med = np.median(f2_all, axis = 0)

    # Positive frequencies only
    pos_freq = freq > 0

    ax.plot(freq[pos_freq], f1_med[pos_freq],
            color = "#2d508c", linewidth = 2, label = "Median $m_1$")
    ax.plot(freq[pos_freq], f2_med[pos_freq],
            color = "#fc342a", linewidth = 2, label = "Median $m_2$")

    ax.plot(freq[pos_freq], f1_med_old[pos_freq],
            color = "deepskyblue", linewidth = 2, label = "Median $m_{1,old}$ ")
    ax.plot(freq[pos_freq], f2_med_old[pos_freq],
            color = "coral", linewidth = 2, label = "Median $m_{2,old}$")

    # Axis styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize = 16)
    ax.set_ylabel("Gain", fontsize = 16)
    ax.set_title("Frequency vs $m_{1,opt}$ & $m_{2,opt}$", fontsize = 18)
    ax.tick_params(axis = "both", which = "major", labelsize = 14)
    ax.legend(fontsize = 14)
    ax.yaxis.set_major_locator(LogLocator(base = 10.0))

    # Minor ticks: base-10 minor ticks (1–9)
    ax.yaxis.set_minor_locator(
        LogLocator(base = 10.0, subs = np.arange(1.0, 10.0), numticks = 1000)
    )
    ax.grid(True, which = "both", ls = "--", lw = 0.5)

    fig.tight_layout()
    fig.show()
    fig.savefig("../outputs/plots/Median_filters_channel_all_eff90_with_old.pdf", dpi=300)


def plot_H1_2_all_channel():
    import src.dataset as ds

    channels = [3, 5, 9, 11, 13, 15, 17, 19]

    f1_all, f2_all = [], []

    fig, ax = plt.subplots(figsize = (8, 6))

    for ch in channels:
        f1, f2 = np.load(f"../outputs/Pileup_filter_functions/channel_{ch}/functions_eff90_V2.npy")
        H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(ch, n_deriv = 1, window_fct = np.hanning)
        f1,f2 = f1 * H_unit * S, f2 * H_unit * S
        f1_all.append(np.abs(f1))
        f2_all.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")

    # Convert to arrays
    f1_all = np.array(f1_all)
    f2_all = np.array(f2_all)

    # Median values
    f1_med = np.median(f1_all, axis = 0)
    f2_med = np.median(f2_all, axis = 0)

    # Positive frequencies only
    pos_freq = freq > 0

    ax.plot(freq[pos_freq], f1_med[pos_freq],
            color = "#2d508c", linewidth = 2, label = "Median $|H_1 \hat{s}|$")
    ax.plot(freq[pos_freq], f2_med[pos_freq],
            color = "#fc342a", linewidth = 2, label = "Median $|H_2 \hat{s}|$")

    # Axis styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize = 16)
    ax.set_ylabel("Gain", fontsize = 16)
    ax.set_title("Frequency vs $|H_{1,opt}\hat{s}|$ & $|H_{2,opt}\hat{s}|$", fontsize = 18)
    ax.tick_params(axis = "both", which = "major", labelsize = 14)
    ax.legend(fontsize = 14)
    ax.grid(True, which = "both", ls = "--", lw = 0.5)
    fig.tight_layout()
    fig.show()
    fig.savefig("../outputs/plots/Median_filters_channel_all_eff90.pdf", dpi=300)
def plot_H1_2_old_all_channel():
    import src.dataset as ds

    channels = [3, 5, 9, 11, 13, 15, 17, 19]

    f1_all, f2_all = [], []
    f1_all_old, f2_all_old = [], []
    fig, ax = plt.subplots(figsize = (8, 6))

    for ch in channels:
        f1, f2 = np.load(f"../outputs/Pileup_filter_functions/channel_{ch}/functions_eff90_V2.npy")
        H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(ch, n_deriv = 1, window_fct = np.hanning)
        f1,f2 = f1 * H_unit * S, f2 * H_unit * S
        f1_all.append(np.abs(f1))
        f2_all.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")
        f1, f2 = np.ones(2000), np.abs(H_unit * S)
        f1, f2 = f1 / np.mean(np.abs(S * f1 * H_unit)), f2 / np.mean(np.abs(S * f2 * H_unit))
        f1,f2 = f1 * H_unit * S, f2 * H_unit * S
        f1_all_old.append(np.abs(f1))
        f2_all_old.append(np.abs(f2))

        freq = np.fft.fftfreq(len(f1), d = 1e-4)
        ax.plot(freq[:1000], f1[:1000], alpha = 0.1, color = "k")
        ax.plot(freq[:1000], f2[:1000], alpha = 0.1, color = "k")
    f1_all_old = np.array(f1_all_old)
    f2_all_old = np.array(f2_all_old)
    f1_med_old = np.median(f1_all_old, axis = 0)
    f2_med_old = np.median(f2_all_old, axis = 0)

    # Convert to arrays
    f1_all = np.array(f1_all)
    f2_all = np.array(f2_all)

    # Median values
    f1_med = np.median(f1_all, axis = 0)
    f2_med = np.median(f2_all, axis = 0)

    # Positive frequencies only
    pos_freq = freq > 0

    ax.plot(freq[pos_freq], f1_med[pos_freq],
            color = "#2d508c", linewidth = 2, label = "Median $|H_1 \hat{s}|$")
    ax.plot(freq[pos_freq], f2_med[pos_freq],
            color = "#fc342a", linewidth = 2, label = "Median $|H_2 \hat{s}|$")

    ax.plot(freq[pos_freq], f1_med_old[pos_freq],
            color = "deepskyblue", linewidth = 2, label = "Median $|H_{1,old} \hat{s}|$")
    ax.plot(freq[pos_freq], f2_med_old[pos_freq],
            color = "coral", linewidth = 2, label ="Median $|H_{1,old} \hat{s}|$")
    # Axis styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize = 16)
    ax.set_ylabel("Gain", fontsize = 16)
    ax.set_title("Frequency vs $|H_{1,opt}\hat{s}|$ & $|H_{2,opt}\hat{s}|$", fontsize = 18)
    ax.tick_params(axis = "both", which = "major", labelsize = 14)
    ax.legend(fontsize = 14)
    ax.grid(True, which = "both", ls = "--", lw = 0.5)
    fig.tight_layout()
    fig.show()
    fig.savefig("../outputs/plots/Median_filters_channel_all_eff90.pdf", dpi=300)

def Y_distrib_fixe_r_dt():
    from torch.utils.data import TensorDataset
    import src.dataset as ds
    import src.simulation as sim
    import src.analysis as an
    from torch import from_numpy
    n_deriv = 1
    channel = 15
    f1_opt, f2_opt = np.load(f"../outputs/Pileup_filter_functions/channel_{channel}/functions_eff90_V3.npy")

    H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(channel, n_deriv = n_deriv, window_fct = np.hanning)
    fpulses, signal_freq, noise_freq = sim.simulate_frequency_pulses_fixed_dt_r(S, nps, w, 3e-4, 0.5, nsim = 10000,
                                                                                seed = 15, signal_scale = 0.001874)
    dataset_simu = TensorDataset(from_numpy(np.fft.ifft(fpulses, axis = 1).real.astype(np.float32)))
    PSD_simu, _, _ = an.get_PSD_interpole(dataset_simu, H_unit, f1_opt, f2_opt)
    plt.hist(PSD_simu, bins = 100, zorder = 3, color = "#2d508c")
    plt.xlim(0.9955, 0.9998)
    plt.xlabel("PSD")
    ax = plt.gca()
    ax.grid(True, zorder = 0, which = 'both', linestyle = '--', linewidth = 0.5)
    plt.ylabel("Number of events")
    plt.title("PSD Distribution for simulated pile-up events r=0.5, dt=0.3 ms")


from matplotlib.colors import LogNorm
def plot_covariance_matrix(result):
    plt.figure(figsize=(8, 6))
    plt.imshow(result.covar, cmap='viridis', aspect='auto',norm = LogNorm(vmin=1e-12, vmax=np.max(result_2.covar)))
    plt.colorbar(label='Covariance')
    plt.title('Covariance Matrix of Fit Parameters (Bessel order 2)')
    plt.xticks(range(len(result.var_names)), result.var_names, rotation=45)
    plt.yticks(range(len(result.var_names)), result.var_names)
    plt.show()

import matplotlib as mpl

def plot_bessel_filters(channel,fcs):
    fcts = []
    for fc in fcs:
        f1_f2 = np.load(f"../outputs/Pileup_filter_functions/channel_{channel}/bessel_study/functions_fc_{fc}.npy")
        fcts.append((f1_f2[0], f1_f2[1]))
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    freq = np.fft.rfftfreq(len(fcts[0][0]), d=1e-4)
    for idx,(f1,f2 )in enumerate(fcts):
        ax.plot(freq,f1[:1001],c=cmap(idx/len(fcts)),alpha=0.7)
        ax.plot(freq,f2[:1001],c=cmap(idx/len(fcts)),alpha=0.7)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=min(fcs), vmax=max(fcs)))
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    cbar.set_label('Bessel Filter Cutoff Frequency (Hz)')
    ax.set_ylabel('Filter Amplitude')
    plt.show()

def plot_J_vs_bessel_fc():
    channels = [3, 5, 9, 11, 13, 15, 17, 19]
    channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        Js = np.loadtxt(f"../outputs/Pileup_filter_functions/channel_{channel}/bessel_study/Js_vs_fc")
        fcs = [500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
        plt.plot(fcs, 1 - np.array(Js), marker = 'o', label = f'detector ID {channel_ID[index_channel]}')
    plt.xlabel('Bessel Filter Cutoff Frequency (Hz)',fontdict = {'size':14})
    plt.ylabel('1-J aka "rejection power"',fontdict = {'size':14})
    plt.legend(fontsize = 12)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
    plt.show()

def plot_mp_freq_all_channel():
    import src.dataset as ds
    channels = [3, 5, 9, 11, 13, 15, 17, 19]
    channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
    plt.figure(figsize = (8, 6))
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        H_unit, S, nps, w, pulse_factor = ds.get_channel_specs(channel, n_deriv = 0, window_fct = np.hanning)
        plt.plot(w[:1000] / (2 * np.pi), np.abs(S)[:1000], label = f"detector ID {channel_ID[index_channel]}")
    plt.legend(fontsize = 14)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 14)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Hz)', fontdict = {'size': 14})
    plt.ylabel('Pulse Amplitude Spectrum |S(w)|', fontdict = {'size': 14})
    plt.show()
def plot_meanpulse_fit(meanpulse, y, t):
    fig,(ax1,ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True, gridspec_kw={'height_ratios': [3, 1],'hspace':0})
    ax1.plot(t, y, label="Transfer function model")
    ax1.plot(t, meanpulse)
    ax2.plot(t, meanpulse - y, label="Fit Residuals",marker='o',markersize=2,ls="")
    plt.legend()
def plot_meanpulse_fit_freq(meanpulse, y,axs=None, labels=None):
    labels = ["Original mean pulse", "Fitted mean pulse"] if labels is None else labels
    (ax1,ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True, gridspec_kw={'height_ratios': [3, 1],'hspace':0})[1] if axs is None else axs
    freq = np.fft.rfftfreq(len(meanpulse),1e-4)
    window = np.hanning(len(meanpulse))
    fmp = np.abs(np.fft.rfft(window*meanpulse))
    ax1.loglog(freq,fmp, label=labels[0],alpha=0.7)
    ffmp = np.abs(np.fft.rfft(window*y))
    ax1.loglog(freq,ffmp, label=labels[1],alpha=0.7)
    ax2.semilogx(freq,ffmp - fmp, label="Residuals",marker='o',markersize=2,ls="")
    ax2.set_ylim(-0.1*np.max(fmp),0.1*np.max(fmp))
    ax1.legend()
    ax1.set_ylim(1e-4,1e2)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude")
def plot_PSD_distribution(
    PSD_single, PSD_pileup,
    muY, sigmaY, ratio_distribution,
    N_sigma=3
):
    from scipy.stats import norm

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(16, 6),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}
    )

    # -----------------------
    # Single events
    # -----------------------
    cut = np.percentile(PSD_single.real, 10)

    hist_single, bins_single = np.histogram(
        PSD_single, bins=200, range=(0.99, 1.01), density=True
    )
    A_single = 0.5 * (bins_single[1:] + bins_single[:-1])

    mu = muY[0, 0].numpy()
    sigma = sigmaY[0, 0].numpy()
    pdf_single = norm.pdf(A_single, mu, sigma)

    ax1.plot(
        A_single, hist_single,
        drawstyle='steps-mid',
        color='tab:blue',
        alpha=0.6,
        lw=1.5,
        label='Single events (MC)'
    )
    ax1.plot(
        A_single, pdf_single,
        color='tab:blue',
        lw=2.5,
        label='Single events (model)'
    )


    # -----------------------
    # Pileup events
    # -----------------------
    hist_pu, bins_pu = np.histogram(
        PSD_pileup, bins=200, range=(0.96, 1.01), density=True
    )
    A_pu = 0.5 * (bins_pu[1:] + bins_pu[:-1])

    ax1.plot(
        A_pu, hist_pu,
        drawstyle='steps-mid',
        color='tab:orange',
        alpha=0.6,
        lw=1.5,
        label='Pileup events (MC)'
    )

    pdf_pu = np.sum(
        norm.pdf(A_pu[None, None, :], muY[:, :, None], sigmaY[:, :, None])
        * ratio_distribution[:, None, None],
        axis=(0, 1)
    ).real
    pdf_pu /= np.trapz(pdf_pu, A_pu)

    ax1.plot(
        A_pu, pdf_pu,
        color='tab:orange',
        lw=2.5,
        label='Pileup events (model)'
    )

    ax1.axvline(
        cut, color='tab:red', ls='--', lw=1.5,
        label='10th percentile cut'
    )
    ax1.axvline(
        mu - N_sigma * sigma,
        color='tab:green', ls='-.', lw=1.5,
        label=rf'$\mu - {N_sigma:.2f}\sigma$'
    )

    # -----------------------
    # Residuals
    # -----------------------
    ax2.axhline(0, color='k', lw=1, alpha=0.6)

    ax2.plot(
        A_single, pdf_single - hist_single,
        color='tab:blue',
        lw=1.5,
        label='Model − MC (single)'
    )
    ax2.plot(
        A_pu, pdf_pu - hist_pu,
        color='tab:orange',
        lw=1.5,
        label='Model − MC (pileup)'
    )

    res_max = max(
        np.abs(pdf_single - hist_single).max(),
        np.abs(pdf_pu - hist_pu).max()
    )
    ax2.set_ylim(-1.1 * res_max, 1.1 * res_max)

    # -----------------------
    # Final formatting
    # -----------------------
    ax1.set_xlim(0.96, 1.01)
    ax1.set_ylabel("Probability density")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("PSD")

    ax1.legend(frameon=False, ncol=3)
    ax2.legend(frameon=False)

    ax1.grid(axis='y', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)

    plt.show()

def injected_pulse_template_test():
    df = pd.read_csv("../outputs/Systematics/BI_meanpulse_test.csv")
    # Unique meanpulse types
    types = df["meanpulse_type"].unique()

    # Choose colors (3 types → 3 colors)
    colors = ["red", "blue", "green"]  # You can pick any colors you like
    markers = ['o', 's', '^']
    plt.figure(figsize = (8, 5))

    for t, c, m in zip(types, colors, markers):
        subset = df[df["meanpulse_type"] == t]
        subset["channel_ID"] = subset["channel"].apply(lambda x: channel_ID[channels.index(x)]).values

        subset.sort_values(by = "channel_ID", inplace = True)
        print(subset)
        plt.errorbar(
            subset["channel_ID"].apply(str),  # x values: channel indices
            subset["BI"] * 1e5,
            yerr = subset["BI_uncertainty"] * 1e5,
            ls = "",  # no line connecting points
            marker = m,
            color = c,
            label = (
                "single high energy pulse" if t == "raw" else "1 zero 4 poles fit" if t == "fit" else "1 zero 4 poles + bessel filter fit")
            # label for legend
        )
    for channel in df["channel"].unique():
        data_channel = df[df["channel"] == channel]
        min_bi = data_channel["BI"].min()
        max_bi = data_channel["BI"].max()
        print((max_bi - min_bi) / min_bi)
    plt.xlabel("Detector ID")
    plt.ylabel("BI [1e-5 cts/keV/kg/year]")
    plt.legend(title = "Injected Pulse template")
    plt.grid()
    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/pulse_template.pdf", dpi = 300)
    plt.show()

channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
channels = [3, 5, 9, 11, 13, 15, 17, 19]
def RT_A_fit_plot():
    import utility.functions as fn
    df = pd.read_csv("../outputs/Systematics/BI_pulsetimeamp_test.csv")
    from scipy.optimize import curve_fit
    fig, (ax, ax_res) = plt.subplots(2, 1, figsize = (12, 8), sharex = True,
                                     gridspec_kw = {'height_ratios': [3, 1], 'hspace': 0})
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        l, = ax.plot(data_channel["risetime"] ** 0.7514 / data_channel["amp_gain"] ** 0.5773,
                     data_channel["J"] * fn.K * 1e5, marker = 'o', ls = "", ms = 1)
        linear_f = lambda x, a, b: a * x + b
        popt, pcov = curve_fit(linear_f, data_channel["risetime"] ** 0.7514 / data_channel["amp_gain"] ** 0.5773,
                               data_channel["J"] * fn.K * 1e5)
        x_fit = np.linspace(1.5, 8, 100)
        ax.plot(x_fit, popt[0] * x_fit + popt[1], ls = "--", lw=0.7,
                c = l.get_color())
        residuals = data_channel["J"] * fn.K * 1e5 - linear_f(
            data_channel["risetime"] ** 0.7514 / data_channel["amp_gain"] ** 0.5773, *popt)
        ax_res.plot(data_channel["risetime"] ** 0.7514 / data_channel["amp_gain"] ** 0.5773, residuals, marker = 'o',
                    ls = "", c = l.get_color(), ms = 1)
        ax.plot([], [], c = l.get_color(), label = f"detector ID {channel_ID[index_channel]}",marker='s',ls="")

    ax.legend(fontsize = 12)
    ax.set_ylabel("Estimated BI (cts/keV/kg/year)",fontdict = {'size': 14})
    ax_res.axhline(0, color = 'black', ls = '--')
    ax_res.set_xlabel("$RT^{n}/A^{m}$ with n=0.751 and m=0.577",fontdict = {'size': 14})
    ax_res.set_ylabel("Residuals (cts/keV/kg/year)",fontdict = {'size': 14})
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax_res.tick_params(axis = 'both', which = 'major', labelsize = 14)
    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/RT_A_fit.pdf", dpi = 300)

    plt.show()

def grid_size_systematics_plot():
    import utility.functions as fn

    df = pd.read_csv("../outputs/Systematics/BI_varying_grid_size.csv")
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        # plt.errorbar(data_channel["grid_N"],data_channel["J"]*fn.K,marker='o',ls="")
        grid_N = data_channel["grid_N"].array
        BI = data_channel["J"].array * fn.K
        plt.plot(grid_N, (BI - np.min(BI)) / np.min(BI) * 100, marker = 'o', ls = "",
                 label = f"Detector ID {channel_ID[index_channel]}")

        print(grid_N[np.argwhere((BI - np.min(BI)) / np.min(BI) < 0.01)[0][0]])
    plt.axvline(90, color = 'black', ls = '--', label = 'threshold grid size \nto reach <1% variation')
    plt.xlabel("Grid size N")
    plt.ylabel("Relative difference to minimum BI (%)")
    plt.xticks(grid_N[::2])
    plt.xticks(grid_N[1::2], minor = True)
    plt.yticks(np.arange(0, 14, 2))
    plt.yticks(np.arange(1, 14, 2), minor = True)
    plt.grid()
    plt.grid(which = 'minor', ls = '--', alpha = 0.5)
    plt.legend()
    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/grid_size_systematics.pdf", dpi = 300)
    plt.show()

def pulse_pos_systematic():
    df = pd.read_csv("../outputs/Systematics/BI_frac_4Pole_Bessel.csv")
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        data_channel.loc[data_channel["BI_uncertainty"] > 3e-6, "BI_uncertainty"] = 0
        plt.errorbar(data_channel["pulse_center_ratio"], data_channel["BI"] * 1e5,
                     yerr = data_channel["BI_uncertainty"] * 1e5, label = f"detector ID {channel_ID[index_channel]}",
                     marker = 'o', ls = "")
        # print((np.max(data_channel["J"]) - np.min(data_channel["J"]))/np.mean(data_channel["J"]))
    plt.xlabel("Pulse center position within window")
    plt.ylabel("BI (cts/keV/kg/year)")
    plt.legend(ncols = 3, bbox_to_anchor = (0.49, 0.19, 0.5, 0.5), shadow = True)
    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/pulse_position_systematics.pdf", dpi = 300)
    plt.grid()
    plt.show()
def window_size_systematic():
    df = pd.read_csv("../outputs/Systematics/BI_window_size_4Pole_Bessel.csv")
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        data_channel.loc[data_channel["BI_uncertainty"] > 2e-6, "BI_uncertainty"] = 0
        plt.errorbar(data_channel["window_size"]*1e-4, data_channel["BI"] * 1e5,
                     yerr = data_channel["BI_uncertainty"] * 1e5, label = f"detector ID {channel_ID[index_channel]}",
                     marker = 'o', ls = "")
        # print((np.max(data_channel["J"]) - np.min(data_channel["J"]))/np.mean(data_channel["J"]))
    plt.xlabel("Window size (s)")
    plt.ylabel("BI (cts/keV/kg/year)")
    plt.legend(ncols = 3, shadow = True)
    plt.grid()
    plt.ylim(4.5, 9)
    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/window_size_systematics.pdf", dpi = 300)
    plt.show()
def simulated_analytical_BI():
    import utility.functions as fn
    df = pd.read_csv("../outputs/Systematics/BI_pulsetime_test.csv")
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        l = plt.errorbar(data_channel["BI"] * 1e5,
                         (data_channel["J"] * fn.K - data_channel["BI"]) / (data_channel["J"] * fn.K) * 100,
                         yerr = data_channel["BI_uncertainty"] / (data_channel["J"] * fn.K) * 100, marker = 'o',
                         ls = "", markersize = 3,
                         elinewidth = .8).lines[0]
        plt.plot([], [], label = f"detector ID {channel_ID[index_channel]}", marker = 's', ls = "", c = l.get_color())
    plt.axhline(0, color = 'black', ls = '--')
    plt.xlabel("BI (1e-5 cts/keV/kg/year)")
    plt.ylabel("Relative difference simulated - analytical BI (%)")
    plt.title("Oberved bias in BI estimation from pulse risetime variation test")
    plt.grid()
    plt.ylim(-5, 10.1)
    plt.legend()
    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/simulated_analytical_BI.pdf", dpi = 300)
    plt.show()
if __name__ == "__main__":
    simulated_analytical_BI()
