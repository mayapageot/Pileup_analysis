import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
def plot_psd_vs_dt(PSD_single, PSD_pileup, dt_pileup):
    dt_pileup_sel = dt_pileup[np.abs(PSD_pileup - 1) < 1]
    PSD_double_sel = PSD_pileup[np.abs(PSD_pileup - 1) < 1]
    bin_means, bin_edges, binnumber = binned_statistic(dt_pileup_sel * 1e-4 * 1e3, PSD_double_sel, np.mean, 40)
    bin_std, bin_edges, binnumber = binned_statistic(dt_pileup_sel * 1e-4 * 1e3, PSD_double_sel, np.std, 40)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.hlines(cut,0,.8,colors='#fc342a',label="90% acceptance cut")
    ax.errorbar(bin_edges[:-1],bin_means,yerr=bin_std, fmt='o',label="Pile-up Simulated Pulses",color="#2d508c")
    ax.set_xlim(0,.8)
    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("PSD")
    # ax.legend(loc = 'lower right')
    ax2 = ax.inset_axes([1.0, 0, 1, 1], sharey=ax)
    ax2.hlines(cut,0,300,colors='#fc342a',label="90% acceptance cut")
    ax2.hist(PSD_single,bins=100,range=(0.97,1.006),density=True,alpha=0.7,label="Single Simulated Pulses",color="#ff9200", orientation='horizontal')
    ax2.hist(PSD_pileup,bins=100,range=(0.97,1.006),density=True,alpha=0.7,label="Pile-up Simulated Pulses",color="#2d508c", orientation='horizontal')
    ax2.tick_params(left=False, labelleft=False)
    ax2.set_xticks([50,100,150,200,250,300])
    ax2.set_xlabel("Counts")
    ax.set_xticks(np.arange(0,.8,.1))
    ax2.legend()
    ax.legend(loc="lower right")
    plt.show()