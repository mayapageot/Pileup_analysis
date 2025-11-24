import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
def plot_psd_vs_dt(PSD_single, PSD_pileup, dt_pileup, cut=0.9, bins=40):
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
    dt_pileup_sel = dt_pileup[np.abs(PSD_pileup - 1) < 0.02]
    PSD_double_sel = PSD_pileup[np.abs(PSD_pileup - 1) < 0.02]

    bin_means, _, _ = binned_statistic(
        dt_pileup_sel * 1e-1, PSD_double_sel, np.mean, bins
    )
    bin_std, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, PSD_double_sel, np.std, bins
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hlines(cut, 0, 0.8, colors="#fc342a", label="90% acceptance cut")
    ax.errorbar(
        bin_edges[:-1],
        bin_means,
        yerr=bin_std,
        fmt="o",
        label="Pile-up Simulated Pulses",
        color="#2d508c",
    )
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("PSD")

    ax2 = ax.inset_axes([1.0, 0, 1, 1], sharey=ax)
    ax2.hlines(cut, 0, 300, colors="#fc342a", label="90% acceptance cut")
    ax2.hist(
        PSD_single,
        bins=100,
        range=(0.97, 1.006),
        density=True,
        alpha=0.7,
        label="Single Simulated Pulses",
        color="#ff9200",
        orientation="horizontal",
    )
    ax2.hist(
        PSD_pileup,
        bins=100,
        range=(0.97, 1.006),
        density=True,
        alpha=0.7,
        label="Pile-up Simulated Pulses",
        color="#2d508c",
        orientation="horizontal",
    )
    ax2.tick_params(left=False, labelleft=False)
    ax2.set_xticks([50, 100, 150, 200, 250, 300])
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
    bin_std, bin_edges, binnumber = binned_statistic(
        dt_pileup_sel * 1e-1, Biais, np.std, bins
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hlines(cut, 0, 0.8, colors="#fc342a", label="90% acceptance cut")
    ax.errorbar(
        bin_edges[:-1],
        bin_means,
        yerr=bin_std,
        fmt="o",
        label="Pile-up Simulated Pulses",
        color="#2d508c",
    )
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("Pileup Bias (1 - PSD)")

    ax2 = ax.inset_axes([1.0, 0, 1, 1], sharey=ax)
    ax2.hlines(cut, 0, 300, colors="#fc342a", label="90% acceptance cut")
    ax2.hist(
        1 - PSD_single,
        bins=100,
        range=(-0.002, 0.015),
        density=True,
        alpha=0.7,
        label="Single Simulated Pulses",
        color="#ff9200",
        orientation="horizontal",
    )
    ax2.hist(
        1 - PSD_pileup,
        bins=100,
        range=(-0.002, 0.015),
        density=True,
        alpha=0.7,
        label="Pile-up Simulated Pulses",
        color="#2d508c",
        orientation="horizontal",
    )
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

def plot_all_channel_m1_2():
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
            color = "red", linewidth = 2, label = "Median $m_1$")
    ax.plot(freq[pos_freq], f2_med[pos_freq],
            color = "blue", linewidth = 2, label = "Median $m_2$")

    # Axis styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]", fontsize = 16)
    ax.set_ylabel("Gain", fontsize = 16)
    ax.set_title("Frequency vs $m_{1,opt}$ & $m_{2,opt}$", fontsize = 18)
    ax.tick_params(axis = "both", which = "major", labelsize = 14)
    ax.legend(fontsize = 14)

    fig.tight_layout()
    # fig.savefig("../outputs/plots/Median_filters_channel_all_eff90.pdf", dpi=300)

if __name__ == "__main__":
    J_vs_rp()
