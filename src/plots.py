import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from matplotlib.ticker import LogLocator
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
    dt_pileup_sel = dt_pileup[np.abs(PSD_pileup - 1) < 0.02]
    PSD_double_sel = PSD_pileup[np.abs(PSD_pileup - 1) < 0.02]
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
        range=(-0.003, 0.015),
        density=True,
        alpha=0.7,
        label="Single Simulated Pulses",
        color="#ff9200",
        orientation="horizontal",
    )
    ax2.hist(
        1 - PSD_pileup,
        bins=100,
        range=(-0.003, 0.015),
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

if __name__ == "__main__":
    plot_m1_2_all_channel()
