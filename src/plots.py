import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic

import utility.functions as fn
import pandas as pd
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
#color theme, Colorblind-safe palette (Okabe-Ito)
colors = [
        "#F0E442",  # yellow
        "#009E73",  # bluish green
        "#56B4E9",  # sky blue
        "#0072B2",  # blue
        "#CC79A7",  # reddish purple
        "#D55E00",  # vermillion
        "#E69F00",  # orange
        "#8C564B"  # brown
    ]
channels = [3,5,9,11,13,15,17,19]
channel_ID = [5, 9, 11, 10, 4, 2, 12, 3]
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
    dt_pileup_sel = dt_pileup[np.abs(PSD_pileup - 1) < 0.1]
    PSD_double_sel = PSD_pileup[np.abs(PSD_pileup - 1) < 0.1]
    bin_means, _, _ = binned_statistic(
        dt_pileup_sel * 1e-1, PSD_double_sel, np.mean, bins
    )
    bin_high, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, PSD_double_sel, lambda x: np.percentile(x, 84.1), bins
    )
    bin_low, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, PSD_double_sel, lambda x:np.percentile(x,15.9), bins
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hlines(cut, 0, 0.8, colors="#fc342a", label="90% acceptance cut")
    ax.errorbar(
        bin_edges[:-1],
        bin_means,
        yerr = [bin_means - bin_low, bin_high - bin_means],
        fmt = "o",
        label = "Pile-up Simulated Pulses",
        color = "#D55E00",
    )
    ax.set_xlim(-0.01, 0.8)
    ax.set_xlabel("$\Delta$t [ms]")
    ax.set_ylabel("PSD")
    ax2 = ax.inset_axes([1.0, 0, 1, 1], sharey=ax)
    hist,_,_ = ax2.hist(
        PSD_single,
        bins=100,
        range=(PSD_min,PSD_max),
        alpha=0.9,
        label="Single Simulated Pulses",
        color="#0072B2",
        orientation="horizontal",
    )
    ax2.hist(
        PSD_pileup,
        bins=100,
        range=(PSD_min,PSD_max),
        alpha=0.9,
        label="Pile-up Simulated Pulses",
        color="#D55E00",
        orientation="horizontal",
    )

    ax2.hlines(cut, 0, hist.max(), colors="#fc342a", label="90% acceptance cut")
    ax2.tick_params(left=False, labelleft=False)
    # ax2.set_xticks([50, 100, 150, 200, 250, 300])
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
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
    ax.set_xlabel("dt [ms]")
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
def plot_computed_mu_sigma():
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

def plot_Trial_vs_J():
    """
    Plot the decrease of the M (misidentification rate) parameter with the train iteration across the 8 channels studied
    """
    fig, ax = plt.subplots(figsize=(8.4, 7))

    all_J = []


    for i, idx in enumerate(np.argsort(channel_ID)):
        channel = channels[idx]
        M_values = np.load(f"../outputs/training_Js/channel_{channel}/J_90_base.npy")
        all_J.append(M_values)
        ax.plot(M_values*100, color=colors[i], alpha=0.7)
        ax.plot([], [], color=colors[i], label=f"Detector {channel_ID[idx]}",marker="s", markersize=15, linestyle="")

    mean_J = np.median(all_J, axis=0)
    ax.plot(mean_J*100, color="black", lw=2,ls="--")

    ax.plot([], color="black", lw=2.5, label="Median",marker="s", markersize=15, linestyle="")

    ax.set_xlabel("Training iteration", fontsize=20)
    ax.set_ylabel(
        r"Global pile-up misidentification rate $\langle \mathcal{M} \rangle$ [%]",
        fontsize = 18
    )


    ax.tick_params(axis='both', labelsize=20)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=20, ncol=2)

    fig.tight_layout()
    #fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/J_convergence_all_channels_eff90.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{BASE}/outputs/plots/J_convergence_all_channels_eff90.pdf", dpi=300, bbox_inches="tight")
    plt.show()

def plot_m1_2_all_channel():
    """
    Plot the resulting m reweighting functions after training the the 8 channels
    """

    m1_all, m2_all = [], []

    fig, ax = plt.subplots(figsize = (8, 6))

    for ch in channels:
        m1, m2 = np.load(f"../outputs/Pileup_filter_functions/channel_{ch}/functions_eff90.npy")
        m1_all.append(np.abs(m1))
        m2_all.append(np.abs(m2))

        freq = np.fft.fftfreq(len(m1), d = 1e-4)
        pos_freq = freq > 0
        ax.plot(freq[pos_freq], m1[pos_freq], alpha = 0.1, color = "k")
        ax.plot(freq[pos_freq], m2[pos_freq], alpha = 0.1, color = "k")

    # Convert to arrays
    m1_all = np.array(m1_all)
    m2_all = np.array(m2_all)

    # Median values
    m1_med = np.median(m1_all, axis = 0)
    m2_med = np.median(m2_all, axis = 0)

    # Positive frequencies only
    pos_freq = freq > 0

    ax.plot(freq[pos_freq], m1_med[pos_freq],
            color = "#2d508c", linewidth = 2, label = "Median $m_1$")
    ax.plot(freq[pos_freq], m2_med[pos_freq],
            color = "#fc342a", linewidth = 2, label = "Median $m_2$")

    # Axis styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize = 20)
    ax.set_ylabel("Gain", fontsize = 20)
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
    ax.legend(fontsize = 20)
    ax.grid(True, which = "both", ls = "--", lw = 0.5)
    fig.tight_layout()
    ax.set_ylim(1e-2, 30)
    # fig.savefig("../outputs/plots/Median_filters_channel_all_eff90.pdf", dpi=300)
    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/Median_filters_channel_all_eff90.png", dpi=300)
    plt.show()


def pulse_template_injection():
    """
    Plot the BI resulting of different pulse template injection for the 8 channels.
    """
    df_1 = pd.read_csv("../outputs/Systematics/BI_meanpulse_test.csv")
    df_2 = pd.read_csv("../outputs/Systematics/BI_meanpulses_test_2.csv")
    df = pd.concat((df_1, df_2))
    df = df[df["meanpulse_type"] != "fit"]
    energy_raw = [35.47, 87.59, 39.87, 29.4, 36.1, 101.5, 73.23, 122.69]
    types = ['single_40Q', 'single_70Q', 'single_100Q', 'raw', 'fit_bessel', 'mean']
    offsets = np.linspace(-0.3, 0.3, len(types))
    df["energy"] = df.apply(
        lambda x: energy_raw[channels.index(x["channel"])] if x["meanpulse_type"] == "raw"
        else float(x["meanpulse_type"].split("_")[1][:-1]) if "single" in x["meanpulse_type"]
        else 150 if x["meanpulse_type"] == "fit_bessel"
        else 200,
        axis = 1
    )

    def assign_offsets(group):
        group = group.sort_values("energy")
        group["offset"] = offsets
        return group

    df = df.groupby("channel", group_keys = False).apply(assign_offsets)

    # Colorblind-friendly palette
    color_map = {
        'raw': "#D55E00",
        'single_40Q': "#0072B2",
        'single_70Q': "#0072B2",
        'single_100Q': "#0072B2",
        'fit_bessel': "#E69F00",
        'mean': "#CC79A7"
    }

    markers = {
        'raw': 'X',
        'single_40Q':'P',
        'single_70Q': 'P',
        'single_100Q': 'P',
        'fit_bessel': 'D',
        'mean': 'o'
    }

    # Categorical x positions (0,1,2,3,...)
    unique_channels = sorted(
        df["channel"].unique(),
        key=lambda x: channel_ID[channels.index(x)]
    )

    x_positions = {ch: i for i, ch in enumerate(unique_channels)}

    # Jitter offsets
    df["x_offset"] = df.apply(lambda row: row["offset"] + x_positions[row["channel"]], axis=1)

    plt.figure(figsize=(8, 6))
    for t in types:
        subset = df[df["meanpulse_type"] == t].copy()
        subset["x"] = subset["channel"].map(x_positions)

        subset.sort_values(by="x", inplace=True)

        x = subset["x_offset"]

        plt.errorbar(
            x,
            subset["BI"] * 1e5,
            yerr=subset["BI_uncertainty"] * 1e5,
            ls="",
            marker=markers[t],
            color=color_map[t],
            markersize=6,
            capthick=2,
            elinewidth=2,
            capsize=5
        )

    # Grid BETWEEN channels
    plt.grid(axis='x', which='major', alpha=0)
    plt.grid(axis='y', alpha=0.5)

    for i in range(len(unique_channels) - 1):
        plt.axvline(i + 0.5, color='black', lw=0.8, zorder=0)
    plt.xlim(-0.5,len(unique_channels)-0.5)
    # X ticks = real detector IDs
    plt.xticks(
        range(len(unique_channels)),
        [str(channel_ID[channels.index(ch)]) for ch in unique_channels]
    )

    # Legend
    plt.plot([], [], marker = "X", ls = "", ms = 12, color = "#D55E00",
             label = "single high energy pulses\nused in [50]")
    plt.plot([], [], marker="P", ls="", ms=12, color="#0072B2",
             label="single high energy pulses\nselected at different amplitudes")
    plt.plot([], [], marker="D", ls="", ms=12, color="#E69F00",
             label="1 zero 4 poles + bessel fit")
    plt.plot([], [], marker="o", ls="", ms=12, color="#CC79A7",
             label="average pulse")

    plt.xlabel("Detector ID", fontsize=20)
    plt.ylabel("BI [1e-5 cts/keV/kg/year]", fontsize=20)
    plt.legend(title="Injected Pulse template", fontsize=16, title_fontsize=18, ncol=1)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.subplots_adjust(left = 0.12)
    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/pulse_template.pdf", dpi=300)
    plt.show()
def grid_size_systematics():
    """
    Plot the impact of the grid size of dt and r used for computation on the BI.

    The y axis chosen here is the relative difference between simulated and analytical BI as the grid size mainly
    impact the analytical BI and not so much the simulated BI.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import utility.functions as fn

    df = pd.read_csv("../outputs/Systematics/BI_varying_grid_size.csv")

    # Colorblind-safe palette (Okabe-Ito)

    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']

    plt.figure(figsize=(8.5, 6.5))

    for i, index_channel in enumerate(np.argsort(channel_ID)):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        data_channel = data_channel[data_channel["grid_N_t"] > 9]  # Focus on N ≤ 200
        grid_N = data_channel["grid_N_t"].to_numpy()
        BI = data_channel["BI"].to_numpy()

        rel_diff = (data_channel["BI"] - data_channel["M"] * fn.K) / (data_channel["BI"] + data_channel["M"] * fn.K) * 200
        plt.plot(
            grid_N,
            rel_diff,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            ls="-",
            lw=0.4,
            markersize=6,
            label=f"Detector {channel_ID[index_channel]}"
        )
    # Threshold line (neutral color)
    plt.axvline(
        100,
        color="black",
        ls="--",
        linewidth=1.5,
        label="Stability threshold \n(<1% variation)"
    )

    plt.xlabel("Grid size N", fontsize=20)
    plt.ylabel("Relative difference\n(simulated − analytical BI) [%]", fontsize=20)

    plt.xticks(grid_N[::2])
    plt.xticks(grid_N[1::2], minor=True)
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Grid styling
    plt.grid(which='major', alpha=0.4)
    plt.grid(which='minor', ls='--', alpha=0.3)

    plt.legend(fontsize=16, ncol=2,framealpha=1)
    plt.tight_layout()

    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/grid_size_systematics.pdf", dpi=300)
    plt.savefig(f"{BASE}/outputs/plots/grid_size_systematics.pdf", dpi=300)

    plt.show()
def pulse_pos_systematic():
    """
        Plot the impact of the pulse position in the window on the BI.
    """
    df = pd.read_csv("../outputs/Systematics/BI_frac_4Pole_Bessel.csv")
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        data_channel.loc[data_channel["BI_uncertainty"] > 3e-6, "BI_uncertainty"] = 0
        plt.errorbar(data_channel["pulse_center_ratio"], data_channel["BI"] * 1e5,
                     yerr = data_channel["BI_uncertainty"] * 1e5, label = f"detector ID {channel_ID[index_channel]}",
                     marker = 'o', ls = "")
    plt.xlabel("Pulse center position within window")
    plt.ylabel("BI [1e-5 cts/keV/kg/year]")
    plt.legend(ncols = 3, bbox_to_anchor = (0.49, 0.19, 0.5, 0.5), shadow = True)
    plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/pulse_position_systematics.pdf", dpi = 300)
    plt.grid()
    plt.show()
def window_size_systematic():
    """
        Plot the impact of the window size on the BI.
    """
    df = pd.read_csv("../outputs/Systematics/BI_window_size.csv")
    for index_channel in np.argsort(channel_ID):
        channel = channels[index_channel]
        data_channel = df[df["channel"] == channel]
        data_channel.loc[data_channel["BI_uncertainty"] > 2e-6, "BI_uncertainty"] = 0
        plt.errorbar(data_channel["window_size"]*1e-4, data_channel["BI"] * 1e5,
                     yerr = data_channel["BI_uncertainty"] * 1e5, label = f"detector ID {channel_ID[index_channel]}",
                     marker = 'o', ls = "")
    plt.xlabel("Window size [s]")
    plt.ylabel("BI [1e-5 cts/keV/kg/year]")
    plt.legend(ncols = 3, shadow = True)
    plt.grid()
    plt.ylim(4.1, 7)
    #plt.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/window_size_systematics.pdf", dpi = 300)
    plt.show()

def compare_simulated_analytical_BI():
    """
    Plot the relative difference of the BI between the analysis applyed on simulated dataset and the analytical
    prediction
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import utility.functions as fn

    # Load data
    # df = pd.read_csv("../outputs/BI_results_base_bis.csv")
    df_1 = pd.read_csv("../outputs/Systematics/BI_meanpulse_test.csv")
    df_2 = pd.read_csv("../outputs/Systematics/BI_meanpulses_test_2.csv")
    df_injection = pd.concat((df_1, df_2))
    df_injection = df_injection.loc[df_injection["meanpulse_type"] != "fit"]

    meanpulse_type = "raw"  # "mean" or "raw"
    df = df_injection.loc[df_injection["meanpulse_type"] == meanpulse_type]
    # Detector IDs
    df["detector_id"] = df["channel"].apply(
        lambda x: channel_ID[channels.index(x)]
    )
    df.sort_values("detector_id", inplace=True)

    analytical_BI_same_mp = [7.21E-05,6.84E-05,4.41E-05,4.59E-05,5.58E-05,4.73E-05,5.38E-05,5.17E-05]
    analytical_BI = df["M"] * fn.K

    simulated_BI = df["BI"]
    rel_diff = (
        (simulated_BI - analytical_BI)
        / (simulated_BI + analytical_BI)
        * 200
    )
    rel_diff_same_mp = (
        (simulated_BI - analytical_BI_same_mp)
        / (simulated_BI + analytical_BI_same_mp)
        * 200
    )

    rel_diff_unc = (
        df["BI_uncertainty"]
        / (simulated_BI + analytical_BI)
        * 200
    )
    rel_diff_unc_same_mp = (
        df["BI_uncertainty"]
        / (simulated_BI + analytical_BI_same_mp)
        * 200
    )
    df_mp_raw = pd.read_csv("../outputs/BI_results_raw_mp.csv")
    df_mp_raw["detector_id"] = df_mp_raw["channel"].apply(
        lambda x: channel_ID[channels.index(x)]
    )
    df_mp_raw.sort_values("detector_id", inplace = True)
    simulated_BI_raw_mp = df_mp_raw["BI"]
    analytical_BI_raw_mp = df_mp_raw["M"] * fn.K
    rel_diff_raw_mp = (
        (simulated_BI_raw_mp - analytical_BI_raw_mp)
        / (simulated_BI_raw_mp + analytical_BI_raw_mp)
        * 200
    )
    rel_diff_unc_raw_mp = (
        df_mp_raw["BI_uncertainty"]
        / (simulated_BI_raw_mp + analytical_BI_raw_mp)
        * 200
    )

    diff_raw_mean = (simulated_BI_raw_mp.values - simulated_BI.values) / (simulated_BI_raw_mp.values + simulated_BI.values) * 200
    diff_raw_mean_unc = np.sqrt(df_mp_raw["BI_uncertainty"].values**2 + df["BI_uncertainty"].values**2) / (simulated_BI_raw_mp.values + simulated_BI.values) * 200
    detector_ids = df["detector_id"].values
    x = np.arange(len(detector_ids))

    median = np.median(rel_diff)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Symmetric y-range
    ylim = 1.15 * np.max(np.abs(rel_diff) + rel_diff_unc)

    # Light band around zero

    # Zero reference
    ax.axhline(0, color="0.4", ls="--", lw=1.5)

    # Error bars
    ax.errorbar(
        x - 0.15,
        rel_diff,
        yerr = rel_diff_unc,
        fmt = "o",
        markersize = 7,
        color = "b",  # colors[i],
        ecolor = "b",  # colors[i],
        elinewidth = 1.8,
        capsize = 4,
        capthick = 1.8,
        markeredgewidth = 1.5,
        zorder = 3,
        ls="-",
        lw=0.4
    )
    ax.errorbar(
        x - 0.05,
        rel_diff_same_mp,
        yerr = rel_diff_unc_same_mp,
        fmt = "o",
        markersize = 7,
        color = "r",  # colors[i],
        ecolor = "r",  # colors[i],
        elinewidth = 1.8,
        capsize = 4,
        capthick = 1.8,
        markeredgewidth = 1.5,
        zorder = 3,
        ls="-",
        lw=0.4
    )
    ax.errorbar(
        x + 0.05,
        rel_diff_raw_mp,
        yerr = rel_diff_unc_raw_mp,
        fmt = "o",
        markersize = 7,
        color = "g",  # colors[i],
        ecolor = "g",  # colors[i],
        elinewidth = 1.8,
        capsize = 4,
        capthick = 1.8,
        markeredgewidth = 1.5,
        zorder = 3,
        ls="-",
        lw=0.4
    )
    ax.errorbar(
        x + 0.15,
        diff_raw_mean,
        yerr = diff_raw_mean_unc,
        fmt = "o",
        markersize = 7,
        color = "magenta",  # colors[i],
        ecolor = "magenta",  # colors[i],
        elinewidth = 1.8,
        capsize = 4,
        capthick = 1.8,
        markeredgewidth = 1.5,
        zorder = 3,
        ls="-",
        lw=0.4
    )

    ax.set_ylim(-ylim, ylim)


    ax.plot([], [], color = "b", marker = "o", ls = "",
            label = "Simulation - Analytical mean pulse\nFilter: mean-pulse training")

    ax.plot([], [], color = "r", marker = "o", ls = "",
            label = "Simulation - Analytical injected pulse\nFilter: mean-pulse training")

    ax.plot([], [], color = "g", marker = "o", ls = "",
            label = "Simulation - Analytical injected pulse\nFilter: injected-pulse training")

    ax.plot([], [], color = "magenta", marker = "o", ls = "",
            label = "Simulation (injected-pulse training) - Simulation(mean-pulse training)")

    ax.legend(fontsize = 14, frameon = True, loc = "lower right")
    ax.set_xticks(x)
    ax.set_xticklabels(detector_ids.astype(str))

    ax.tick_params(axis="both", labelsize=18)

    ax.grid(axis="y", alpha=0.3)

    # Clean journal style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Detector ID", fontsize = 20)
    ax.set_ylabel("Relative difference\n(simulated − analytical) BI [%]", fontsize = 20)
    plt.tight_layout()

    plt.savefig(
        "/local/home/mp274748/Documents/paper/pile_up_cupid/figures/simulated_analytical_BI.pdf",
        dpi=300,
    )
    plt.savefig(
        f"{BASE}/outputs/plots/simulated_analytical_BI.pdf",
        dpi = 300,
    )

    plt.show()

def WP_test():
    """
    Plot the BI resulting of the study for different Working Point (WP) i.e. the biais applied to the NTD.

    This has only been done only on 2 channels for which working point exploration has been done during CROSS RUN9.
    """

    df = pd.read_csv("../outputs/Systematics/BI_WP.csv")

    fig, ax1 = plt.subplots(figsize=(8.4, 7))

    # Colorblind-friendly palette (Okabe-Ito)
    color_ch1 = "#0072B2"  # blue
    color_ch2 = "#CC79A7"  # purple

    primary_ticks = np.concatenate(([0.279], np.arange(2, 11, 2)))

    # -----------------------
    # Channel 3
    # -----------------------
    channel = 3
    data_ch1 = df[df["channel"] == channel].copy()
    data_ch1 = data_ch1.sort_values("biais_current")

    ax1.plot(
        data_ch1["biais_current"],
        data_ch1["J_value"] * 100,
        marker='o',
        ls='-',
        lw=1,
        markersize=10,
        color=color_ch1,
        label=f"Detector ID {channel_ID[channels.index(channel)]}"
    )

    factor1 = np.mean(data_ch1["resistance"] * data_ch1["biais_current"])

    ax_bis = ax1.secondary_xaxis(
        'top',
        functions=(lambda x: factor1 / x, lambda x: factor1 / x)
    )

    ax_bis.set_xlabel(
        f"Bias resistance detector {channel_ID[channels.index(channel)]} [M$\Omega$]",
        fontsize = 16,
        color = color_ch1
    )

    ax_bis.set_xticks(np.round(factor1 / primary_ticks, 2))

    ax_bis.tick_params(axis='x', colors=color_ch1, labelsize=14)
    ax_bis.spines['top'].set_color(color_ch1)

    # -----------------------
    # Channel 5 (offset top axis)
    # -----------------------
    channel = 5
    data_ch2 = df[df["channel"] == channel].copy()
    data_ch2 = data_ch2.sort_values("biais_current")

    ax1.plot(
        data_ch2["biais_current"],
        data_ch2["J_value"] * 100,
        marker='s',
        ls='-',
        lw=1,
        markersize=10,
        color=color_ch2,
        label=f"Detector ID {channel_ID[channels.index(channel)]}"
    )

    factor2 = np.mean(data_ch2["resistance"] * data_ch2["biais_current"])

    ax_ter = ax1.secondary_xaxis(
        'top',
        functions=(lambda x: factor2 / x, lambda x: factor2 / x)
    )

    ax_ter.spines["top"].set_position(("outward", 45))

    ax_ter.set_xlabel(
        f"Bias resistance detector {channel_ID[channels.index(channel)]} [M$\Omega$]",
        fontsize = 16,
        color = color_ch2
    )
    ax_ter.set_xticks( np.round(factor2 / primary_ticks, 2))

    ax_ter.tick_params(axis='x', colors=color_ch2, labelsize=14)
    ax_ter.spines['top'].set_color(color_ch2)

    # -----------------------
    # Main axis styling
    # -----------------------
    ax1.set_xlabel("Bias current [nA]", fontsize=20)

    ax1.set_ylabel(
        r"Residual unidentified pile-up rate $\langle \mathcal{M} \rangle$ [%]",
        fontsize=20
    )

    ax1.set_xticks(primary_ticks)
    ax1.set_xticklabels([f"{primary_ticks[0]:.3f}"]+[f"{int(tick)}" for tick in primary_ticks[1:]])

    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(alpha=0.25)

    ax1.yaxis.get_offset_text().set_fontsize(14)

    # Legend
    ax1.legend(fontsize=16, frameon=False)

    fig.tight_layout()

    plt.savefig(
        f"{BASE}/outputs/plots/WP_test.pdf",
        dpi=300
    )
    # plt.savefig(
    #    "/local/home/mp274748/Documents/paper/pile_up_cupid/figures/WP_test.pdf",
    #    dpi=300
    # )
    plt.show()


def meanpulse_with_dt_range():
    """
    Figure of a Light Detector (LD) pulse template with the highlighted dt range explored for the pileup analysis.
    This range is set a being the threshold above which all the pileup are detected.
    """
    from scipy.interpolate import interp1d
    mp = np.load("../outputs/meanpulses_build/channel3_meanpulse_raw.npy")
    mp = mp / np.max(mp)
    mp_interp = interp1d(np.arange(len(mp)), mp, kind = 'cubic')
    mp = mp_interp(np.linspace(0, len(mp) - 1, 10000))
    fig, ax = plt.subplots(1, 1, figsize = (8, 6))
    ax.plot(np.arange(- np.argwhere(mp > 1e-2)[0], len(mp) - np.argwhere(mp > 1e-2)[0]) * 4096 / 10000 * 1e-1, mp,
            label = 'Pulse impulse response s(t)', color = "black")
    ax.fill_betweenx(np.array([-0.11, 1.1]), -0e-1, 8e-1, color = 'green', alpha = 0.3,
                     label = '$\Delta t$ window: 0.8 ms')
    ax.set_xlim(-5, 20)
    ax.set_ylim(-0.04, 1.1)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 16)
    ax.set_xlabel("Time (ms)", fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.xticks(np.arange(-5, 20, 1), minor = True)
    plt.yticks(np.arange(-0.1, 1, 0.1), minor = True)
    ax.grid(linestyle = '--', linewidth = 1)
    ax.grid(which = 'minor', linestyle = ':', linewidth = 0.5)
    fig.set_tight_layout(True)
    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/meanpulse_with_dt_range.pdf")
    plt.show()


def r_PDF():
    """
    Plot the probablilty distribution of the relative amplitude r=E2/(E1+E2) of a pileup event from the 2v2b spectrum.
    """
    from utility.double_beta_spectrum import pdf_ratio2b
    fig, ax = plt.subplots(figsize = (8, 6))
    r = np.linspace(0, 1, 1000)
    PDF = pdf_ratio2b(r)
    PDF_norm = np.sum(PDF) * (r[1] - r[0])
    PDF /= PDF_norm
    ax.plot(r, PDF, color = 'red')
    ax.fill_between(r, np.zeros_like(r), PDF, color = 'red', alpha = 0.2)
    ax.set_xlabel('Relative amplitude r', fontsize = 20)
    ax.set_ylabel('Probability Density', fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.grid(linestyle = '--', linewidth = 1)
    fig.set_tight_layout(True)
    fig.savefig("/local/home/mp274748/Documents/paper/pile_up_cupid/figures/r_pdf.pdf", dpi = 300)
    plt.show()

def bb_spectrum():
    """
    Plot the 2v2b spectrum.
    """
    from utility.double_beta_spectrum import g
    fig, ax = plt.subplots(figsize=(8, 6))

    # Single-event energy distribution
    E_max = 3.034
    E = np.linspace(0, E_max, 1000)
    spectrum = g(E)

    # Energy bin width
    dE = E[1] - E[0]

    # Sum of two independent events:
    # f_sum(E) = integral f(E') f(E-E') dE'
    spectrum_sum = np.convolve(spectrum, spectrum) * dE

    # Energy axis for the convolution
    E_sum = np.linspace(0, 2 * E_max, len(spectrum_sum))

    # Plot single-event spectrum
    ax.plot(
        E,
        spectrum,
        color='blue',
        label='Single event'
    )

    # Plot 2-event sum spectrum
    ax.plot(
        E_sum,
        spectrum_sum,
        color='red',
        label='Sum of 2 events'
    )

    # Optional: fill the pile-up spectrum
    ax.fill_between(
        E_sum,
        np.zeros_like(E_sum),
        spectrum_sum,
        color='red',
        alpha=0.15
    )

    ax.set_xlabel('Energy [MeV]', fontsize=20)
    ax.set_ylabel('Energy distribution [Counts/keV]', fontsize=20)

    ax.tick_params(axis='both', labelsize=20)

    ax.grid(linestyle='--', linewidth=1)
    ax.legend(fontsize=16)

    fig.set_tight_layout(True)

    fig.savefig(
        "/local/home/mp274748/Documents/paper/pile_up_cupid/figures/bb_spectrum.pdf",
        dpi=300
    )

    plt.show()
def bb_spectrum_with_PU():
    """
    Plot of the 2v2b spectrum with the spectrum of the sum of two 2v2b event.
    """
    from utility.double_beta_spectrum import g

    fig, ax = plt.subplots(figsize = (8, 6))

    # Single-event spectrum
    E_max = 3.034
    E = np.linspace(0, E_max, 1000)
    spectrum = g(E)
    spectrum /= 1000
    # Energy step
    dE = E[1] - E[0]

    # Convolution: sum of two independent events
    spectrum_sum = np.convolve(spectrum, spectrum) * dE
    E_sum = np.linspace(0, 2 * E_max, len(spectrum_sum))

    # Normalize the 2-event spectrum to the same area
    norm_single = np.trapz(spectrum, E)
    norm_sum = np.trapz(spectrum_sum, E_sum)

    spectrum_sum *= norm_single / norm_sum
    # Plot
    ax.plot(
        E,
        spectrum,
        color = 'blue'
    )

    ax.fill_between(
        E,
        0,
        spectrum,
        color = 'blue',
        alpha = 0.2
    )

    ax.plot(
        [], [],
        ls = "",
        marker = "s",
        markerfacecolor = (0, 0, 1, 0.2),
        markeredgecolor = "blue",
        markeredgewidth = 1,
        markersize = 15,
        label = r"$2\nu\beta\beta$"
    )
    # Convert keV limits to MeV since E_sum is in MeV
    E_low = 3.019
    E_high = 3.049

    # Full spectrum
    ax.plot(
       E_sum,
       spectrum_sum,
       color = 'deepskyblue'
    )

    # Normal region
    mask_normal = (E_sum < E_low) | (E_sum > E_high)

    ax.fill_between(
        E_sum,
        0,
        spectrum_sum,
        where = mask_normal,
        color = 'deepskyblue',
        alpha = 0.2
    )

    # Highlighted region: 3019–3049 keV
    mask_roi = (E_sum >= E_low) & (E_sum <= E_high)
    ax.plot(
        E_sum[mask_roi],
        spectrum_sum[mask_roi],
        color = 'red'
    )
    ax.fill_between(
        E_sum,
        0,
        spectrum_sum,
        where = mask_roi,
        color = 'red',
        alpha = 0.5
    )
    ax.plot(
        [], [],
        ls = "",
        marker = "s",
        markerfacecolor = (0, 191/255, 1, 0.2),
        markeredgecolor = "deepskyblue",
        markeredgewidth = 1,
        markersize = 15,
        label = r"$2\nu\beta\beta$ pile-up"
    )
    ax.plot(
        [], [],
        ls = "",
        marker = "s",
        markerfacecolor = (1, 0, 0, 0.5),
        markeredgecolor = "red",
        markeredgewidth = 1,
        markersize = 15,
        label = r"CUPID-$0\nu\beta\beta$ ROI"
    )

    ax.set_xlabel('Energy [MeV]', fontsize = 20)
    ax.set_ylabel(r'Energy density [keV$^{-1}$]', fontsize=20)

    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.set_xlim(-0.1, 4.5)
    #ax.grid(linestyle = '--', linewidth = 1)
    ax.legend(fontsize = 16)
    #no axis:
    # Hide top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Keep only left and bottom axes
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    fig.set_tight_layout(True)

    fig.savefig(
        "/local/home/mp274748/Documents/presentation/thesis_defense/bb_spectrum_bis.png",
        dpi = 300
    )

    plt.show()

def bb_spectrum_with_Q_value_peak():
    """
    Plot of the 2v2b spectrum with Q-value highlighted by a peak
    """
    from utility.double_beta_spectrum import g
    fig, ax = plt.subplots(figsize = (8, 6))
    E = np.linspace(0, 3.100, 1000)
    spectrum = g(E)
    ax.plot(E, spectrum, color = 'blue')
    ax.fill_between(E, np.zeros_like(E), spectrum, color = 'blue', alpha = 0.2, label = '2νββ spectrum')
    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    E_peak = np.linspace(3.0, 3.100, 1000)
    peak = gaussian(E_peak, 3.034, 0.01) * np.max(spectrum) * 0.01
    #ax.plot(E_peak, peak, color = 'red')
    #ax.fill_between(E_peak, np.zeros_like(E_peak), peak, color = 'red', alpha = 0.2,label='0νββ')
    ax.set_xticks([3.034], labels = [r'$Q_{\beta\beta}$'], fontsize = 20)
    ax.set_yticks([])
    ax.set_xlabel('Energy', fontsize = 20)
    ax.set_ylabel('Events', fontsize = 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    ax.set_xlim(0, 3.1)
    ax.set_ylim(0,  np.max(spectrum))
    fig.legend(fontsize = 16)
    fig.set_tight_layout(True)
    fig.savefig(f"{BASE}/outputs/plots/bb_spectrum_without_Q_value_peak.pdf", dpi = 300)
    plt.show()
def denoise_NPS_comparaison():
    """
    Plot of the comparaison between raw and denoised data (with the multichannel encoder denoiser).
    """
    RMS_reductions = [0.96, 0.78, 0.43, 0.61, 0.3, 0.5, 0.88, 0.34]
    StdCuts = [0.0003, 0.0004, 0.0008, 0.0006, 0.0012, 0.0007, 0.00035, 0.0025]

    fig, axes = plt.subplots(2, 4, figsize = (18, 9), sharex = True, sharey = True)

    freq = np.fft.rfftfreq(2048, d = 1 / 10000)
    window_size = 2048
    meas_name = "000813_20230628T161508"
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20
    })
    print(np.argsort(channel_ID))
    for idx, channel_arg in enumerate(np.argsort(channel_ID)):
        ax = axes.flatten()[idx ]

        channel = channels[channel_arg]
        RMS_reduction = RMS_reductions[channel_arg]

        nps_raw = np.load(
            f"{BASE}/outputs/NPS_study/"
            f"NPS_channel{channel}_win{window_size}_{meas_name}.npy"
        )[:1025]

        nps_denoised = np.load(
            f"{BASE}/outputs/NPS_study/"
            f"NPS_channel{channel}_win{window_size}_{meas_name}_denoised.npy"
        )[:1025]

        # Plot styling
        ax.plot(freq, nps_raw, label = "Raw", lw = 1.8, alpha = 0.8, color = "#D55E00")
        ax.plot(freq, nps_denoised, label = "Denoised", lw = 2.2, alpha = 0.9, color = "#0072B2")

        ax.set_xscale("log")
        ax.set_yscale("log")
        # Channel title instead of legend duplication
        ax.set_title(f"Detector ID {channel_ID[channel_arg]}", fontsize = 20)
        RMS_reduction = (np.sum(nps_raw) - np.sum(nps_denoised)) / np.sum(nps_raw)
        # Subtle annotation
        ax.text(
            0.05, 0.1,
            f"ΔRMS: {RMS_reduction * 100:.1f}%",
            transform = ax.transAxes,
            fontsize = 20,
            bbox = dict(facecolor = 'white', alpha = 0.7, edgecolor = 'none')
        )

        # Grid (important for log plots)
        ax.grid(True, which = "both", ls = "--", alpha = 0.3)

    # Global labels
    fig.supxlabel("Frequency [Hz]", fontsize = 20)
    fig.supylabel(r"NPS [ADU/$\sqrt{\mathrm{Hz}}$]", fontsize = 20)

    # Single legend for all subplots
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc = "upper center", ncol = 2, fontsize = 20)

    plt.tight_layout(rect = [0, 0, 1, 0.95])
    plt.savefig(f"{BASE}/outputs/plots/NPS_comparison_denoising.pdf", dpi = 300)


def meanpulse_fig():
    """
    plot an pulse exemple with the decay and risetime highlighted
    """
    from scipy.interpolate import interp1d
    rise_c = '#428cdcff'  # Color for rise region (blue)
    decay_c = '#c3438a'  # Color for decay region (pink)
    mp = np.load(f"{BASE}/outputs/meanpulses_build/channel3_meanpulse_raw.npy")
    mp = mp / np.max(mp)

    mp_interp = interp1d(np.arange(len(mp)), mp, kind='cubic')
    mp = mp_interp(np.linspace(0, len(mp) - 1, 10000))

    # Time axis
    t = np.arange(len(mp))
    t0 = np.argwhere(mp > 1e-2)[0][0]
    t = (t - t0) * 4096 / 10000 * 1e-1  # ms

    # --- Thresholds ---
    y10, y90, y30 = 0.1, 0.9, 0.3

    # --- Find indices ---
    i10 = np.argmax(mp > y10)
    i90 = np.argmax(mp > y90)

    peak_idx = np.argmax(mp)

    # decay: after peak
    decay_region = mp[peak_idx:]
    i90_decay = peak_idx + np.argmax(decay_region < y90)
    i30_decay = peak_idx + np.argmax(decay_region < y30)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Pulse
    ax.plot(t, mp, label='Pulse impulse response s(t)', color="black",alpha=0.7)

    ax.plot(t[i10:i90], mp[i10:i90], label='Pulse impulse response s(t)', color=rise_c)
    ax.plot(t[i90_decay:i30_decay], mp[i90_decay:i30_decay], label='Pulse impulse response s(t)', color=decay_c)

    # --- Horizontal dashed lines ---
    ax.hlines(y10, xmin=t[0], xmax=t[np.argmax(mp > y10)],
              linestyles='dashed', colors=rise_c, linewidth=1)


    ax.hlines(y30, xmin=t[0], xmax=t[i30_decay],
              linestyles='dashed', colors=decay_c, linewidth=1)
    ax.hlines(y90, xmin=t[0], xmax=t[i90_decay],
              linestyles='dashed', colors='black', linewidth=1)
    ax.text(-4, 0.91, '90% ', color = 'black', fontsize = 12)
    ax.text(-4, 0.31, '30%', color = decay_c, fontsize = 12)
    ax.text(-4, 0.11, '10%', color = rise_c, fontsize = 12)
    ax.hlines(1, xmin=t[0], xmax=t[np.argmax(mp)],
              linestyles='dashed', colors='red', linewidth=1)
    ax.text(-4, 1.01, 'Amplitude', color = 'red', fontsize = 12)

    # --- Fill rise area (10% → 90%) ---
    ax.fill_between(t[i10:i90], mp[i10:i90],
                    color=rise_c, alpha=0.3, label='Rise region')

    # --- Fill decay area (90% → 30%) ---
    ax.fill_between(t[i90_decay:i30_decay], mp[i90_decay:i30_decay],
                    color=decay_c, alpha=0.3, label='Decay region')

    # --- Label rise region ---
    t_rise_center = 0.5 * (t[i10] + t[i90])
    y_rise_center = -0.05

    ax.text(t_rise_center, y_rise_center,
            "Rise time",
            color=rise_c, fontsize=13,
            ha='center', va='center',
            alpha=0.9)

    # --- Label decay region ---
    t_decay_center = 0.5 * (t[i90_decay] + t[i30_decay])
    y_decay_center = -0.05

    ax.text(t_decay_center, y_decay_center,
            "Decay time",
            color=decay_c, fontsize=13,
            ha='center', va='center',
            alpha=1)
    # --- Formatting ---
    ax.set_xlim(-5, 25)
    ax.set_ylim(-0.1, 1.1)

    ax.set_xlabel("Time (ms)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # --- Remove box, keep only axes ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional: make axes slightly thicker (nicer for thesis)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)


    fig.set_tight_layout(True)

    fig.savefig(f"{BASE}/outputs/plots/meanpulse.png", dpi=300)
    plt.show()

def NPS_figure():
    """
    Plot a noise power spectrum exemple
    """
    nps = np.load("../outputs/NPS_study/NPS_channel3_win4096_000813_20230628T161508.npy")
    freq = np.fft.rfftfreq(len(nps), d=1/10000)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(freq, nps[:len(nps)//2+1], color = 'k')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize=20)
    ax.set_ylabel(r"NPS [ADU/$\sqrt{\mathrm{Hz}}$]", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    # --- Remove box, keep only axes ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional: make axes slightly thicker (nicer for thesis)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    fig.set_tight_layout(True)
    fig.savefig(f"{BASE}/outputs/plots/NPS.pdf", dpi=300)


def raw_vs_denoise_muY_M():
    """
    Plot the comparaison of the Y (PSD) and M (pileup misidentification rate) for the raw and denoised data.

    The take away message here is that beside having a redused sigma the denoise data doesnt have a better
    misidentification rate.
    """
    det = 5
    channel = channels[channel_ID.index(det)]
    acceptance = 0.9
    mu_raw, sigma_raw = np.load(f"{BASE}/outputs/mu_sigmas/channel{channel}_mu_sigma_raw_test.npy")
    mu_corr, sigma_corr = np.load(f"{BASE}/outputs/mu_sigmas/channel{channel}_mu_sigma_denoise_test.npy")
    import torch
    mu_raw, sigma_raw = torch.from_numpy(mu_raw), torch.from_numpy(sigma_raw)
    mu_corr, sigma_corr = torch.from_numpy(mu_corr), torch.from_numpy(sigma_corr)
    t_min, t_max, N_t = 0, 8e-4, 100
    t_torch = np.linspace(t_min, t_max, N_t)
    from src.analysis import compute_A
    fig, ax1 = plt.subplots(figsize = (9, 7))
    ax2 = ax1.twinx()
    mu_raw = mu_raw.real
    sigma_raw = sigma_raw.real
    mu_corr = mu_corr.real
    sigma_corr = sigma_corr.real
    t_torch = t_torch.real
    color_raw = "#41639b"
    color_corr = "#a64d79"
    # --- AXIS 1 (left) ---
    l1, = ax1.plot(t_torch * 1e3, mu_raw[-1], color = color_raw, label = "Raw μ")
    f1 = ax1.fill_between(
        t_torch * 1e3,
        (mu_raw[-1] - sigma_raw[-1]).cpu(),
        (mu_raw[-1] + sigma_raw[-1]).cpu(),
        alpha = 0.3, color = color_raw, label = "Raw ±σ"
    )

    l2, = ax1.plot(t_torch * 1e3, mu_corr[-1], color = color_corr, label = "Denoised μ")
    f2 = ax1.fill_between(
        t_torch * 1e3,
        (mu_corr[-1] - sigma_corr[-1]).cpu(),
        (mu_corr[-1] + sigma_corr[-1]).cpu(),
        alpha = 0.3, color = color_corr, label = "Denoised ±σ"
    )

    ax1.set_xlabel("Δt [s]", fontsize = 16)
    ax1.set_ylabel("Y", fontsize = 16,rotation=0,labelpad=20)

    # --- AXIS 2 (right) ---
    l3, = ax2.plot(
        t_torch * 1e3,
        compute_A(mu_raw, sigma_raw, N_sigma = 1.28)[-1],
        color = color_raw, ls = "--", label = "Raw $\mathcal{M}$"
    )

    l4, = ax2.plot(
        t_torch * 1e3,
        compute_A(mu_corr, sigma_corr, N_sigma = 1.28)[-1],
        color = color_corr, ls = "-.", label = "Denoised $\mathcal{M}$"
    )

    ax2.set_ylabel(r"$\mathcal{M}$", fontsize = 16,rotation=0,labelpad=20)
    ax2.yaxis.set_label_coords(1.1, 0.551)
    ax2.tick_params(axis = 'both', which = 'major', labelsize = 14)
    ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
    # --- Separate legends ---
    # --- SHARED LEGEND ---
    handles = [l1, f1, l2, f2, l3, l4]
    labels = [h.get_label() for h in handles]
    ticks2 = [0, 0.2, 0.4, 0.6, 0.8, 1]
    ticks1 = np.linspace(0.99, 1.0, len(ticks2))
    ax1.set_yticks(ticks1)
    ax2.set_yticks(ticks2)
    ax1.set_ylim(0.989, 1.001)
    # map limits to ax2
    t1_min, t1_max = ticks1[0], ticks1[-1]
    t2_min, t2_max = ticks2[0], ticks2[-1]
    y1_min, y1_max = ax1.get_ylim()
    y2_min = t2_min + (y1_min - t1_min) * (t2_max - t2_min) / (t1_max - t1_min)
    y2_max = t2_min + (y1_max - t1_min) * (t2_max - t2_min) / (t1_max - t1_min)

    ax2.set_ylim(y2_min, y2_max)
    ax1.grid(linestyle = '--', linewidth = 0.5)
    ax1.legend(handles, labels, loc = "upper right", fontsize = 16)
    fig.savefig(f"{BASE}/outputs/plots/mu_sigma_M_raw_denoise.png", dpi = 300)
    plt.show()
def diff_acceptance_plot():
    """
    Plot of the difference in BI resulting from a training at specific acceptance vs a fixe training at 0.9 acceptance.
    The takeaway message here is that the acceptance has little impact at the training level.
    """
    acceptances = [0.7, 0.8, 0.84, 0.87, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.99]
    data_acc = pd.read_csv(f"{BASE}/outputs/BI_acceptance_study.csv")
    window_size = 4096
    meas_name = "000813_20230628T161508"
    BIs_channels = []
    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
    })

    fig, (ax, ax_diff) = plt.subplots(
        2,
        figsize = (11, 7),
        sharex = True,
        gridspec_kw = {"height_ratios": [3.5, 1], "hspace": 0.05}
    )

    for idx, channel_arg in enumerate(np.argsort(channel_ID)):
        channel = channels[channel_arg]
        BIs = []

        PSD_pileup, PSD_single = np.load(
            f"{BASE}/outputs/PSDs/PSD_channel{channel}_win{window_size}_{meas_name}_acc90.npy"
        )

        for acceptance in acceptances:
            cut = np.percentile(PSD_single, (1 - acceptance) * 100)
            rp = np.sum(PSD_pileup < cut) / len(PSD_pileup)
            BI = fn.K * (1.0 - rp)
            BIs.append(BI)

        BIs = np.array(BIs)
        BI_ref = data_acc.loc[data_acc["channel"] == channel, "BI"].values

        color = colors[idx % len(colors)]

        # Main curve
        ax.plot(
            acceptances,
            BIs * 1e5,
            marker = "o",
            markersize = 5,
            color = color,
            label = f"Channel ID {channel_ID[channel_arg]}"
        )

        # Reference points
        ax.scatter(
            data_acc.loc[data_acc["channel"] == channel, "acceptance"],
            BI_ref * 1e5,
            marker = "x",
            s = 60,
            linewidth = 2,
            color = color,
            alpha = 0.8
        )

        # Residuals
        diff = (BIs - BI_ref) * 1e5
        ax_diff.plot(
            acceptances,
            diff,
            marker = "o",
            markersize = 4,
            color = color,
            alpha = 0.85
        )
    # Dummy handles for legend explaining marker meaning
    ax.plot([], [], marker = 'x', ls = '', markersize = 8,
            label = 'BI from specific acceptance study', color = 'k')

    ax.plot([], [], marker = 'o', ls = '', markersize = 6,
            label = 'BI from 0.9 acceptance study', color = 'k')
    # Styling main panel
    ax.set_ylabel(r"BI [$10^{-5}$ cts / keV / kg / yr]")
    ax.set_title("Background Index vs Single-Pulse Acceptance")
    ax.grid(True, alpha = 0.3)
    #ax.legend(ncol = 4, frameon = False, loc = "upper center")
    ax.legend(ncol = 2, frameon = False, loc = 'upper center')
    # Residual panel
    ax_diff.axhline(0, linestyle = "--", linewidth = 1)
    ax_diff.grid(True, alpha = 0.3)
    ax_diff.set_ylabel("ΔBI")
    ax_diff.set_xlabel("Single pulse acceptance")

    plt.tight_layout()
    plt.show()
def ROC_curve():
    """
    Plot of the Receiver Operating Characteristic (ROC), which show the evolution of the True Positive Rate (pileup
    rejected) with the False Positive Rate (single pulse rejected).
    """
    window_size = 4096
    meas_name = "000813_20230628T161508"

    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    all_tpr = []
    all_fpr = []
    for idx, channel_arg in enumerate(np.argsort(channel_ID)):
        channel = channels[channel_arg]

        PSD_pileup, PSD_single = np.load(
            f"{BASE}/outputs/PSDs/"
            f"PSD_channel{channel}_win{window_size}_{meas_name}_acc90.npy"
        )

        # Reduce thresholds for speed
        thresholds = np.linspace(
            min(PSD_pileup.min(), PSD_single.min()),
            max(PSD_pileup.max(), PSD_single.max()),
            500
        )

        tpr = []
        fpr = []

        for cut in thresholds:
            tp = np.mean(PSD_pileup < cut)
            fp = np.mean(PSD_single < cut)

            tpr.append(tp)
            fpr.append(fp)

        ax.plot(
            fpr,
            tpr,
            color=colors[idx],
        )
        ax.plot(
            [],
            [],
            color = colors[idx],
            marker = "s",
            markersize = 10,
            ls="",
            label = f"Det. {channel_ID[channel_arg]}"
        )
        all_tpr.append(tpr)
        all_fpr.append(fpr)
    all_tpr = np.array(all_tpr)
    all_fpr = np.array(all_fpr)

    median_tpr = np.median(all_tpr, axis = 0)
    median_fpr = np.median(all_fpr, axis = 0)

    ax.plot(
        median_fpr,
        median_tpr,
        color = "black",
        lw = 2,
        label = "Median",
        ls="--"
    )
    # Random classifier line (ONLY ONCE)
    ax.plot([0, 1], [0, 1], '-.', color='gray', label='Random')
    ax.plot()
    ax.set_xlabel("False Positive Rate (single pulse rejected)", fontsize=16)
    ax.set_ylabel("True Positive Rate (pileup rejected)", fontsize=16)
    #ax.set_title("ROC Curve")

    def forward(y):
        return fn.K * (1 - y) * 1e5

    def inverse(y_sec):
        return 1 - y_sec / fn.K * 1e-5

    secax = ax.secondary_yaxis('right', functions = (forward, inverse))
    secax.set_ylabel("Background Index [1e-5 cts/keV/kg/year] ", fontsize=16)
    # Reference lines
    y_goal = inverse(5.0)
    x_goal = 1 - 0.9

    ax.axhline(y_goal, color = 'gray', linestyle = '--', alpha = 0.6)
    ax.axvline(x_goal, color = 'gray', linestyle = '--', alpha = 0.6)

    # Labels along the lines
    ax.text(
        0.55, y_goal-0.005 ,
        "CUPID TDR\n"+r"BI = $5 \times 10^{-5}$",
        color = "gray",
        ha = "center",
        va = "center",
        fontsize = 12,
        linespacing=1.5
    )

    ax.text(
        x_goal+0.001, 0.4,
        "CUPID TDR\n 90% acceptance",
        color = "gray",
        rotation = 90,
        ha = "center",
        va = "center",
        fontsize = 12,
        linespacing=2
    )
    ax.tick_params(axis='both', which='major', labelsize=14)
    secax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2,fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{BASE}/outputs/plots/ROC_curve.pdf", dpi=300)
    plt.show()

def scaling_factor():
    """
    Plot of the BI with the scaling factor applyed to the SNR of the data.
    We show also scaling factor lines corresponding to different scenario presented in M.Pageot thesis.
    """
    df1 = pd.read_csv(f"{BASE}/outputs/Systematics/BI_scaling_factor.csv")
    df2 = pd.read_csv(f"{BASE}/outputs/Systematics/BI_scaling_factor_bis.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    plt.style.use("default")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
    })

    fig, ax = plt.subplots(figsize = (8, 6))
    median_BI = df.groupby("scaling_factor")["BI"].median() * 1e5


    # ---- channel-wise fits ----
    for i, channel_arg in enumerate(np.argsort(channel_ID)):
        channel = channels[channel_arg]
        df_sub = df[df["channel"] == channel]

        x = df_sub["scaling_factor"].values
        x_log = np.log(x)
        y = df_sub["BI"].values * 1e5
        y_log = np.log(y)

        coeff = np.polyfit(x_log, y_log, 1)
        b = coeff[0]
        a = np.exp(coeff[1])

        print(f"Channel {channel}: a={a:.3e}, b={b:.3f}")

        color = colors[i % len(colors)]
        y_err = df_sub["BI_uncertainty"].values * 1e5

        ax.errorbar(x, y, yerr = y_err, color = color, alpha = 0.7,
                    label = f"Det {channel_ID[channel_arg]} (b={b:.2f})",
                    fmt = 'o', capsize = 3, markersize = 5, ls = "")

        xfit = np.linspace(0.6, x.max(), 200)
        ax.plot(xfit, a*xfit**b, color = color)

    ax.plot(
        median_BI.index,
        median_BI.values,
        marker = "x",
        linestyle = "--",
        color = "black",
        linewidth = 2
    )
    from scipy.interpolate import interp1d
    f_inter = interp1d(median_BI.index, median_BI.values, kind = "linear", fill_value = "extrapolate")
    # ---- scenario lines ----
    scenarios = [
        ("(a)+(c)", 0.625),
        ("(a)+(e)", 1.06),
        ("(b)+(d)", 1.00),
        ("(b)+(e)", 1.25),
        ("(b)+(e)+(f)", 1.56),
        ("(b)+(e)+(f)+(g)", 1.96)
    ]


    for label, x in scenarios:
        # print(f"Scenario {label}: BI = {f_inter(x):.2f} x 10^-5")
        print(f_inter(x),",")
        ax.axvline(
            x,
            color = "gray",
            linestyle = ":",
            linewidth = 1.9,
            alpha = 0.7,
        )
        ax.text(
            x+0.011,
            11.1,  # slightly above lower y-limit
            label,
            rotation = 90,
            va = "top",
            ha = "left",
            fontsize = 12,
            color = "k",
        )
    for i in range(5):
        ax.plot([], [], ls = "", label = " ")
    ax.plot(
        [],
        [],
        marker = "x",
        linestyle = "--",
        color = "black",
        linewidth = 2,
        label = "Median BI"
    )

    # ---- target line ----
    ax.axhline(
        5,
        color = "gray",
        linestyle = "--",
        linewidth = 1.5,
        label = r"Target BI = $5\times10^{-5}$"
    )
    ax.plot([], [], c = "k", label = "Fit: $BI = a \cdot (scaling\_factor)^b$")

    # ---- formatting ----
    ax.set_xlabel("Scaling Factor")
    ax.set_ylabel(r"BI [$10^{-5}$ cts / keV / kg / yr]")
    ax.grid(alpha = 0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.6,0.7,0.8, 0.9, 1, 1.2, 1.4, 1.7, 2], labels = [0.6,0.7,0.8, 0.9, 1, 1.2, 1.4, 1.7, 2])
    ax.set_yticks([4, 5, 6, 7, 8, 9,10], labels = [4, 5, 6, 7, 8, 9,10])
    ax.set_ylim(3.3, 11.15)

    plt.tight_layout()
    artist,labels = ax.get_legend_handles_labels()
    leg = plt.legend(artist[8:]+artist[1:8], labels[8:]+labels[1:8],
        ncol = 2, fontsize = 12, framealpha = 0, loc = "lower left")
    for txt in leg.get_texts():
        if txt.get_text() != " ":

            txt.set_bbox(dict(
                facecolor = "white",
                edgecolor = "none",
                alpha =1,
                pad = 4
            ))
    plt.savefig(f"{BASE}/outputs/plots/scaling_factor.pdf", dpi = 300)
    plt.show()


def BI_vs_bias_current_RUN121():
    """
    Plot of the BI vs working point (ie bias_current applied to the NTD) for Ulysse RUN121.
    """
    df = pd.read_csv(f"{BASE}/outputs/Systematics/BI_WP_RUN121_Ulysse.csv")
    df_info = pd.read_csv("/media/mp274748/Transcend2/data/Ulysse/RUN121/rms_amp_SNR_RUN121_bis.csv")

    fig, axs = plt.subplots(1,3,figsize = (14, 5), sharey = True, gridspec_kw = {"width_ratios": [1, 1, 1], "wspace": 0.})

    biais = np.array([0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 29])
    color_SNR = "r"
    color_rt= "b"
    color_J = "#6620a6"
    names = ["edgeSi2", "edgeSi1", "bingoSi3"]
    for idx, channel in enumerate([5,6,7]):

        ax1 = axs[idx]
        ax_SNR = ax1.twinx()
        ax_rt = ax1.twinx()
        ax_SNR.set_ylim(0, 0.2)
        ax_rt.set_ylim(0.2, 1.2)
        df_sub = df.loc[df["channel"] == channel]
        df_info_sub = df_info.loc[df_info["channel"] == channel]
        ax_SNR.plot(biais[df_info_sub["meas"].astype(int) - 1], 1 / df_info_sub["SNR"], c = color_SNR, marker = '^',
                 alpha = 0.6, label = f'1/SNR')
        ax1.plot([], [], c = color_SNR, marker = '^', alpha = 0.6, label = f'1/SNR')
        ax_rt.plot(biais[df_info_sub["meas"].astype(int) - 1], df_info_sub["rt"] * 1000, c = color_rt, marker = 'x',
                 alpha = 0.6,label = f'Rise time (ms)' )
        ax1.plot([], [], c = color_rt, marker = 'x',
                 alpha = 0.6,label = f'Rise time (ms)' )
        l, = ax1.plot(biais[df_sub["measurement"].astype(int) - 1], df_sub["J_value"] * 100, marker = 'o', lw = 3,
                      c = color_J, label = r'$\langle \mathcal{M} \rangle$')
        ax1.set_xlabel("biais [nA]", fontsize = 14)
        ax1.set_xticks(biais.astype(int))
        ax1.legend()
        ax1.set_ylim(0, 20)
        ax1.set_yticks(np.arange(0, 21, 2), labels = [f"{x:.0f}" for x in np.arange(0, 21, 2)], fontsize = 12)
        ax1.set_xticks(np.arange(0,31,4), labels = [f"{x:.0f}" for x in np.arange(0,31,4)], fontsize = 12)
        ax_SNR.set_yticks([])
        ax_rt.set_yticks([])
        ax1.grid()
        ax1.set_title(names[idx], fontsize = 14)
    ax_rt.spines["right"].set_position(("axes", 1.3))
    ax_rt.tick_params(axis = 'y', colors = color_rt, labelsize = 14)
    ax_rt.spines['right'].set_color(color_rt)
    ax_SNR.set_ylabel("1/SNR", fontsize = 14, color = color_SNR)
    ax_SNR.tick_params(axis = 'y', colors = color_SNR, labelsize = 14)
    ax_SNR.spines['right'].set_color(color_SNR)
    ax_rt.set_ylabel("Rise time [ms]", fontsize = 14, color = color_rt)
    axs[0].set_ylabel(r"Residual unidentified pile-up rate $\langle \mathcal{M} \rangle$ [%]", fontsize = 14,color=color_J)
    axs[0].tick_params(axis = 'y', labelsize = 14,colors=color_J)
    axs[0].spines['left'].set_color(color_J)
    ax_SNR.set_yticks(np.arange(0, 0.21, 0.02), labels = [f"{x:.2f}" for x in np.arange(0, 0.21, 0.02)], fontsize = 12)
    ax_rt.set_yticks(np.arange(0.2, 1.21, 0.1), labels = [f"{x:.1f}" for x in np.arange(0.2, 1.21, 0.1)], fontsize = 12)
    fig.subplots_adjust(right = 0.85)
    fig.subplots_adjust(left = 0.08)
    # fig.tight_layout()
    fig.savefig(f"{BASE}/outputs/plots/BI_vs_bias_current_RUN121.pdf", dpi = 300)
    plt.show()



if __name__ == "__main__":
    pass
    # diff_acceptance_plot()