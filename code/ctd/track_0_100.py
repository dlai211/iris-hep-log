# import modules
import uproot, sys, time, random, argparse, copy, ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from tqdm import tqdm
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
from collections import Counter
from scipy.stats import norm
from math import *

# import trackingError function
sys.path.append('/data/jlai/iris-hep-log/code/')
from trackingerror import Detector, inputfromfile

# Set up plot defaults
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14.0,10.0  # Roughly 11 cm wde by 8 cm high
mpl.rcParams['font.size'] = 20.0 # Use 14 point font
sns.set(style="whitegrid")

font_size = {
    "xlabel": 17,
    "ylabel": 17,
    "xticks": 15,
    "yticks": 15,
    "legend": 13,
    "title": 13,
}

plt.rcParams.update({
    "axes.labelsize": font_size["xlabel"],  # X and Y axis labels
    "xtick.labelsize": font_size["xticks"],  # X ticks
    "ytick.labelsize": font_size["yticks"],  # Y ticks
    "legend.fontsize": font_size["legend"]  # Legend
})


root_path = "/data/jlai/iris-hep/OutputPT2/output_pt_0/tracksummary_ckf.root"
tree_name = "tracksummary"

pt_edges = np.array([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])

file = uproot.open(root_path)
tree = file[tree_name]

branches = ["t_p", "t_pT", "t_theta", "res_eLOC1_fit"]  # z0 residual in mm
available = set(tree.keys())

arrays = tree.arrays(branches, library="ak")

pt_truth = ak.to_numpy(ak.flatten(arrays["t_pT"]))  # [GeV]
z0_res = ak.to_numpy(ak.flatten(arrays["res_eLOC1_fit"]))  # [mm]

valid = ~np.isnan(pt_truth) & ~np.isnan(z0_res)
pt_truth = pt_truth[valid]
z0_res = z0_res[valid]

# --- binning and per-bin Gaussian sigma ---
sigmas = []
sigma_errs = []
counts = []

for lo, hi in zip(pt_edges[:-1], pt_edges[1:]):
    m = (pt_truth >= lo) & (pt_truth < hi)
    data = z0_res[m]
    N = len(data)

    _, s = norm.fit(data)
    sigmas.append(s)  # mm
    sigma_errs.append(s / np.sqrt(2*max(N-1, 1)))
    counts.append(N)

sigmas = np.array(sigmas)
sigma_errs = np.array(sigma_errs)

# --- plot: σ(z0) [mm] vs pT [GeV] ---
plt.figure(figsize=(7.2, 4.2), dpi=150)
plt.errorbar(pt_centers, sigmas, yerr=sigma_errs, fmt='o', capsize=2, lw=1, ms=4)
plt.xlabel(r'$p_T$ [GeV]')
plt.ylabel(r'$\sigma_{z_0}$ [mm]')
plt.title('ODD Simulation — single particles, ⟨μ⟩=0 (muons)')
plt.grid(True, alpha=0.3)
# Optional: annotate points with entries
# for x, y, n in zip(pt_centers, sigmas, counts):
#     if np.isfinite(y):
#         plt.text(x, y, f'n={n}', fontsize=6, ha='center', va='bottom')
plt.tight_layout()
plt.savefig("z0_res_vs_pt.png")
# plt.show()
