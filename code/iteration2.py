# iteration_shortstrip.py
# Iterate ONLY short-strip parameters; pixel is fixed to your iteration-1 best.

import uproot, sys, time, random, argparse, copy, csv, json, math, os
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

from trackingerror import Detector, inputfromfile

# ---- Plot defaults ----
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 14.0,10.0
mpl.rcParams['font.size'] = 20.0
sns.set(style="whitegrid")

font_size = {"xlabel": 17,"ylabel": 17,"xticks": 15,"yticks": 15,"legend": 13,"title": 13}
plt.rcParams.update({
    "axes.labelsize": font_size["xlabel"],
    "xtick.labelsize": font_size["xticks"],
    "ytick.labelsize": font_size["yticks"],
    "legend.fontsize": font_size["legend"]
})

# ========= Load ACTS results (unchanged) =========
print("Loading ACTS results...")
path = '/data/jlai/iris-hep/OutputPT_pixel_only/'  # your current reference
var_labels = ['sigma(d)', 'sigma(z)', 'sigma(phi)', 'sigma(theta)', 'sigma(pt)/pt']

y_acts = {label: [] for label in var_labels}
y_acts_err = {label: [] for label in var_labels}

pT_values = np.arange(10, 100, 10)
for pT_value in pT_values:
    pT_value = int(pT_value)
    print(f'Saving ACTS track resol with pT = {pT_value} GeV')
    file = uproot.open(path + f'output_pt_{pT_value}' + '/tracksummary_ckf.root')
    tree = file['tracksummary']

    arrays = tree.arrays(["t_d0", "eLOC0_fit", "res_eLOC0_fit",
                          "t_z0", "eLOC1_fit", "res_eLOC1_fit",
                          "t_phi", "ePHI_fit", "res_ePHI_fit",
                          "t_theta", "eTHETA_fit", "res_eTHETA_fit",
                          "t_p", "t_pT", "eQOP_fit", "res_eQOP_fit",
                          "t_charge"], library='ak')

    pT_truth = arrays['t_p'] * np.sin(arrays['t_theta'])
    pT_reco  = np.abs(1 / arrays['eQOP_fit']) * np.sin(arrays['t_theta'])

    labels = {
        'sigma(d)'     : ak.flatten(arrays['res_eLOC0_fit']) * 1e3,
        'sigma(z)'     : ak.flatten(arrays['res_eLOC1_fit']) * 1e3,
        'sigma(phi)'   : ak.flatten(arrays['res_ePHI_fit']),
        'sigma(theta)' : ak.flatten(arrays['res_eTHETA_fit']),
        'sigma(pt)/pt' : ak.flatten((pT_reco - pT_truth)) / ak.flatten(pT_reco)
    }

    for key, data in labels.items():
        data = ak.to_numpy(data)
        data = data[~np.isnan(data)]
        N = len(data)
        _, sigma = norm.fit(data)
        y_acts[key].append(sigma)
        y_acts_err[key].append(sigma / np.sqrt(2*max(N-1,1)) if N > 1 else 0.0)

# ========= Fitting setup (iterate SHORT STRIP only) =========
VAR_LABELS = ['sigma(d)', 'sigma(z)', 'sigma(phi)', 'sigma(theta)']

# ---- Toggles for SS only ----
SS_WIDTHS_TIED = True   # True => w1=w2=w3=w4 for short strip
SS_RES_TIED    = True   # True => resxy==resz for short strip

OPTIMIZE_SS_WIDTHS = True
OPTIMIZE_SS_RES    = True
OPTIMIZE_SS_POS    = True

# ---- Fixed beam pipe & PIXEL (from your iteration-1 best) ----
beam_pipe = (0.00227, 9999.0, 9999.0, 0.024)  # width, resxy, resz, pos (m)

pixel_width   = 0.014944
pixel_res_xyz = 1.328531e-05
pixel_positions = np.array([0.032754, 0.070560, 0.110162, 0.165020], dtype=float)

# ---- SHORT STRIP base (your provided best as starting point) ----
ss_positions_base = np.array([0.260, 0.360, 0.500, 0.600], dtype=float)
ss_width_base     = 0.01475
ss_resxy_base     = 0.043e-3
ss_resz_base      = 1.2e-3

ss_widths_base = np.full(4, ss_width_base, dtype=float)
ss_pos_off_base = np.zeros(4, dtype=float)

# ---- Bounds & steps (SS only) ----
SS_WIDTH_MIN, SS_WIDTH_MAX = 0.0, 0.05
SS_RES_MIN,   SS_RES_MAX   = 5e-8, ss_resxy_base + 0.01e-3
SS_POS_MIN,   SS_POS_MAX   = -0.02, 0.02  # offsets around base positions

w_step_init  = 5e-4
r_step_init  = 5e-7
p_step_init  = 5e-4

w_eps, r_eps, p_eps = 2e-4, 1e-7, 1e-4
shrink, grow = 0.5, 1.5
max_rounds, patience = 16, 3

# ---- I/O ----
odd_txt_out = "/data/jlai/iris-hep-log/TrackingResolution-3.0/TrackingResolution-3.0/myODD_test.txt"
RUN_DIR = os.path.dirname(odd_txt_out) or "."
LOG_CSV = os.path.join(RUN_DIR, "shortstrip_fit_log.csv")
BEST_JSON = os.path.join(RUN_DIR, "shortstrip_fit_best.json")
BEST_TXT  = os.path.join(RUN_DIR, "myODD_best.txt")

OBJECTIVE = "chi2"  # or "diff"

# ========= Helpers =========
def clamp(x, lo, hi): return max(lo, min(hi, x))

def write_config_with_pixel_and_ss(path, ss_widths, ss_resxy, ss_resz, ss_pos_offsets):
    """Write: beam pipe + fixed pixel (your best) + variable short-strip (4 layers)."""
    ss_widths = np.asarray(ss_widths, dtype=float)
    ss_pos    = ss_positions_base + np.asarray(ss_pos_offsets, dtype=float)

    with open(path, "w") as f:
        f.write("# width(x/X0)    resolutionxy(m)    resolutionz(m)    position (m)\n")
        f.write("# beam pipe\n")
        f.write(f"{beam_pipe[0]} {beam_pipe[1]} {beam_pipe[2]} {beam_pipe[3]}\n")

        f.write("# pixel (fixed)\n")
        for i in range(4):
            f.write(f"{pixel_width} {pixel_res_xyz} {pixel_res_xyz} {pixel_positions[i]}\n")

        f.write("# short strip (variable)\n")
        for i in range(4):
            f.write(f"{max(ss_widths[i],1e-9)} {max(float(ss_resxy),1e-9)} {max(float(ss_resz),1e-9)} {ss_pos[i]}\n")

def write_final_txt(path, ss_widths, ss_resxy, ss_resz, ss_pos_offsets):
    """Pretty file with the same sections."""
    ss_widths = np.asarray(ss_widths, dtype=float)
    ss_pos    = ss_positions_base + np.asarray(ss_pos_offsets, dtype=float)

    with open(path, "w") as f:
        f.write("#width(x/x0) resolutionxy(mm) resolutionz(mm) position (m)\n")
        f.write("#beam pipe\n")
        f.write("0.00227 9999 9999 0.024\n")
        f.write("#pixel \n")
        for i in range(4):
            f.write(f"{pixel_width:.6f} {pixel_res_xyz:.6e} {pixel_res_xyz:.6e} {pixel_positions[i]:.6f}\n")
        f.write("#short strip \n")
        for i in range(4):
            f.write(f"{ss_widths[i]:.6f} {ss_resxy:.6e} {ss_resz:.6e} {ss_pos[i]:.6f}\n")

def cal(inputfile):
    y_calc = {label: [] for label in VAR_LABELS}
    for pT in pT_values:
        p, eta = float(pT), 0.0
        B, m = 2.6, 0.106  # your choice earlier
        det = inputfromfile(inputfile, 0)
        out = det.errorcalculation(p, B, eta, m)
        for lbl in VAR_LABELS:
            y_calc[lbl].append(out[lbl])
    return y_calc

def metrics(y_calc):
    diff_acc, chi2_acc = 0.0, 0.0
    for lbl in VAR_LABELS:
        acts_vals = np.array(y_acts[lbl], dtype=float)
        acts_errs = np.array(y_acts_err[lbl], dtype=float)
        calc_vals = np.array(y_calc[lbl], dtype=float)
        norm = max(float(np.max(acts_vals)), 1e-12)
        a = acts_vals / norm
        ae = np.maximum(acts_errs / norm, 1e-12)
        c = calc_vals / norm
        diff_acc += float(np.sum(np.abs(c - a)))
        chi2_acc += float(np.sum(((c - a) / ae)**2))
    return diff_acc, chi2_acc

def log_csv_init(path):
    head = ["timestamp","ss_w1","ss_w2","ss_w3","ss_w4","ss_resxy","ss_resz",
            "ss_pos1","ss_pos2","ss_pos3","ss_pos4",
            "diff","chi2","obj","phase","note"]
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(head)

def log_csv_row(path, rec, phase, note=""):
    w = rec["ss_widths"]; p = rec["ss_pos"]
    row = [time.strftime("%F_%T"), w[0], w[1], w[2], w[3], rec["ss_resxy"], rec["ss_resz"],
           p[0], p[1], p[2], p[3], rec["diff"], rec["chi2"], rec["obj"], phase, note]
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

def objective_value(ss_widths, ss_rxy, ss_rz, ss_pos, cache, key_round=9):
    """Compute & cache objective for SHORT STRIP params (pixel fixed)."""
    ss_widths = np.asarray(ss_widths, dtype=float)
    ss_pos    = np.asarray(ss_pos, dtype=float)

    key = (tuple(round(float(w), key_round) for w in ss_widths),
           round(float(ss_rxy), key_round),
           round(float(ss_rz),  key_round),
           tuple(round(float(x), key_round) for x in ss_pos))
    if key in cache:
        return cache[key]

    ss_w_c = np.clip(ss_widths, SS_WIDTH_MIN, SS_WIDTH_MAX)
    ss_rx_c = clamp(float(ss_rxy), SS_RES_MIN, SS_RES_MAX)
    ss_rz_c = clamp(float(ss_rz),  SS_RES_MIN, SS_RES_MAX)
    ss_pos_c = np.clip(ss_pos, SS_POS_MIN, SS_POS_MAX)

    write_config_with_pixel_and_ss(odd_txt_out, ss_w_c, ss_rx_c, ss_rz_c, ss_pos_c)
    try:
        y_calc = cal(odd_txt_out)
        diff_sum, chi2_sum = metrics(y_calc)
        obj = chi2_sum if OBJECTIVE == "chi2" else diff_sum
        if not np.isfinite(obj): obj = float('inf')
    except Exception:
        diff_sum = chi2_sum = float('inf'); obj = float('inf')

    rec = {"ss_widths": list(map(float, ss_w_c)),
           "ss_resxy": float(ss_rx_c),
           "ss_resz":  float(ss_rz_c),
           "ss_pos":   list(map(float, ss_pos_c)),
           "diff": float(diff_sum), "chi2": float(chi2_sum), "obj": float(obj)}
    cache[key] = rec
    return rec

# ========= Search phases (SS only) =========
def random_warmup(n, cache):
    best = {"obj": float('inf')}
    for _ in tqdm(range(n), desc="Warm-up (random, SS)"):
        # widths
        if OPTIMIZE_SS_WIDTHS:
            if SS_WIDTHS_TIED:
                w = np.random.uniform(SS_WIDTH_MIN, SS_WIDTH_MAX)
                ss_w = np.full(4, w)
            else:
                ss_w = np.random.uniform(SS_WIDTH_MIN, SS_WIDTH_MAX, size=4)
        else:
            ss_w = ss_widths_base.copy()
        # resolutions
        if OPTIMIZE_SS_RES:
            if SS_RES_TIED:
                r = np.random.uniform(SS_RES_MIN, SS_RES_MAX)
                ss_rxy, ss_rz = r, r
            else:
                ss_rxy = np.random.uniform(SS_RES_MIN, SS_RES_MAX)
                ss_rz  = np.random.uniform(SS_RES_MIN, SS_RES_MAX)
        else:
            ss_rxy, ss_rz = ss_resxy_base, ss_resz_base
        # positions
        if OPTIMIZE_SS_POS:
            ss_p = np.random.uniform(SS_POS_MIN, SS_POS_MAX, size=4)
        else:
            ss_p = ss_pos_off_base.copy()

        rec = objective_value(ss_w, ss_rxy, ss_rz, ss_p, cache)
        log_csv_row(LOG_CSV, rec, "warmup", "random")
        if rec["obj"] < best["obj"]: best = rec
    return best

def coarse_grid(cache):
    """Small pattern sweep on SS variables only."""
    # Width patterns
    if OPTIMIZE_SS_WIDTHS:
        dw = 5e-4
        if SS_WIDTHS_TIED:
            w0 = float(ss_widths_base[0])
            w_list = [w0, w0+dw, w0-dw, w0+2*dw, w0-2*dw]
            width_patterns = [np.full(4, clamp(w, SS_WIDTH_MIN, SS_WIDTH_MAX)) for w in w_list]
        else:
            width_patterns = [
                ss_widths_base,
                ss_widths_base + dw,
                ss_widths_base - dw,
                ss_widths_base + np.array([dw,0,0,dw]),
                ss_widths_base - np.array([dw,0,0,dw]),
                ss_widths_base + np.array([0,dw,dw,0]),
                ss_widths_base - np.array([0,dw,dw,0]),
            ]
    else:
        width_patterns = [ss_widths_base]

    # Resolution patterns
    if OPTIMIZE_SS_RES:
        drs = np.linspace(max(SS_RES_MIN, ss_resxy_base-0.001e-3),
                          min(SS_RES_MAX, ss_resxy_base+0.001e-3), 5)
        if SS_RES_TIED:
            res_patterns = [(r, r) for r in drs]
        else:
            res_patterns = [(rx, rz) for rx in drs for rz in drs]
    else:
        res_patterns = [(ss_resxy_base, ss_resz_base)]

    # Position patterns
    if OPTIMIZE_SS_POS:
        dp = 5e-4
        pos_patterns = [
            np.zeros(4),
            np.full(4, +dp),
            np.full(4, -dp),
            np.array([+dp, 0, 0, -dp]),
            np.array([0, +dp, -dp, 0]),
        ]
    else:
        pos_patterns = [ss_pos_off_base.copy()]

    best = {"obj": float('inf')}
    total = len(width_patterns)*len(res_patterns)*len(pos_patterns)
    pbar = tqdm(total=total, desc="Coarse grid (SS)")
    for widths in width_patterns:
        for (rxy, rz) in res_patterns:
            for pos in pos_patterns:
                rec = objective_value(widths, rxy, rz, pos, cache)
                log_csv_row(LOG_CSV, rec, "grid", "coarse")
                if rec["obj"] < best["obj"]:
                    best = rec
                pbar.update(1)
    pbar.close()
    return best

def explore_axis(center, step, lo, hi, plug, cache, label, get_val):
    c = clamp(float(center), lo, hi)
    candidates = [c, clamp(c + step, lo, hi), clamp(c - step, lo, hi)]
    best_local = {"obj": float("inf")}
    for x in candidates:
        rec = objective_value(*plug(x), cache=cache)
        log_csv_row(LOG_CSV, rec, "search", label)
        if rec["obj"] < best_local["obj"]:
            best_local = rec
    if best_local["obj"] < float("inf"):
        best_coord = float(get_val(best_local))
        if abs(best_coord - c) > 1e-18:
            direction = math.copysign(1.0, best_coord - c)
            x2 = clamp(c + 2.0*direction*step, lo, hi)
            rec2 = objective_value(*plug(x2), cache=cache)
            log_csv_row(LOG_CSV, rec2, "search", f"{label} 2-step")
            if rec2["obj"] < best_local["obj"]:
                best_local = rec2
    return best_local

def optimize(start, cache):
    ss_w   = np.array(start["ss_widths"], dtype=float)
    ss_rxy = float(start["ss_resxy"])
    ss_rz  = float(start["ss_resz"])
    ss_pos = np.array(start["ss_pos"], dtype=float)

    best = objective_value(ss_w, ss_rxy, ss_rz, ss_pos, cache)
    log_csv_row(LOG_CSV, best, "init", "start")

    w_step = w_step_init if OPTIMIZE_SS_WIDTHS else 0.0
    r_step = r_step_init if OPTIMIZE_SS_RES    else 0.0
    p_step = p_step_init if OPTIMIZE_SS_POS    else 0.0

    no_improve = 0
    for rnd in range(1, max_rounds+1):
        improved = False

        # widths
        if OPTIMIZE_SS_WIDTHS:
            if SS_WIDTHS_TIED:
                w0 = float(ss_w[0])
                rec_w = explore_axis(
                    w0, w_step, SS_WIDTH_MIN, SS_WIDTH_MAX,
                    plug=lambda x: (np.full(4, x), ss_rxy, ss_rz, ss_pos),
                    cache=cache, label="ss_width(all)", get_val=lambda rec: rec["ss_widths"][0],
                )
                if rec_w["obj"] + 1e-12 < best["obj"]:
                    ss_w[:] = rec_w["ss_widths"][0]
                    best, improved = rec_w, True
                    w_step *= grow
                else:
                    w_step *= shrink
            else:
                for i in range(4):
                    rec_wi = explore_axis(
                        ss_w[i], w_step, SS_WIDTH_MIN, SS_WIDTH_MAX,
                        plug=lambda x, i=i: (np.array([x if j==i else ss_w[j] for j in range(4)]), ss_rxy, ss_rz, ss_pos),
                        cache=cache, label=f"ss_width[{i}]",
                        get_val=lambda rec, i=i: rec["ss_widths"][i],
                    )
                    if rec_wi["obj"] + 1e-12 < best["obj"]:
                        ss_w[i] = rec_wi["ss_widths"][i]; best, improved = rec_wi, True; w_step *= grow
                    else:
                        w_step *= shrink

        # resolutions
        if OPTIMIZE_SS_RES:
            if SS_RES_TIED:
                r0 = float(ss_rxy)
                rec_r = explore_axis(
                    r0, r_step, SS_RES_MIN, SS_RES_MAX,
                    plug=lambda x: (ss_w, x, x, ss_pos),
                    cache=cache, label="ss_res(tied)", get_val=lambda rec: rec["ss_resxy"],
                )
                if rec_r["obj"] + 1e-12 < best["obj"]:
                    ss_rxy = ss_rz = rec_r["ss_resxy"]; best, improved = rec_r, True; r_step *= grow
                else:
                    r_step *= shrink
            else:
                rec_rx = explore_axis(
                    ss_rxy, r_step, SS_RES_MIN, SS_RES_MAX,
                    plug=lambda x: (ss_w, x, ss_rz, ss_pos),
                    cache=cache, label="ss_resxy", get_val=lambda rec: rec["ss_resxy"],
                )
                if rec_rx["obj"] + 1e-12 < best["obj"]:
                    ss_rxy, best, improved = rec_rx["ss_resxy"], rec_rx, True; r_step *= grow
                else:
                    r_step *= shrink
                rec_rz = explore_axis(
                    ss_rz, r_step, SS_RES_MIN, SS_RES_MAX,
                    plug=lambda x: (ss_w, ss_rxy, x, ss_pos),
                    cache=cache, label="ss_resz", get_val=lambda rec: rec["ss_resz"],
                )
                if rec_rz["obj"] + 1e-12 < best["obj"]:
                    ss_rz, best, improved = rec_rz["ss_resz"], rec_rz, True; r_step *= grow
                else:
                    r_step *= shrink

        # positions
        if OPTIMIZE_SS_POS:
            for i in range(4):
                rec_pi = explore_axis(
                    ss_pos[i], p_step, SS_POS_MIN, SS_POS_MAX,
                    plug=lambda x, i=i: (ss_w, ss_rxy, ss_rz, np.array([x if j==i else ss_pos[j] for j in range(4)])),
                    cache=cache, label=f"ss_pos[{i}]", get_val=lambda rec, i=i: rec["ss_pos"][i],
                )
                if rec_pi["obj"] + 1e-12 < best["obj"]]:
                    ss_pos[i] = rec_pi["ss_pos"][i]; best, improved = rec_pi, True; p_step *= grow
                else:
                    p_step *= shrink

        log_csv_row(LOG_CSV, best, "round",
                    f"r={rnd}, steps=(w={w_step:.2g}, r={r_step:.2g}, p={p_step:.2g})")

        small = ((not OPTIMIZE_SS_WIDTHS or w_step < w_eps) and
                 (not OPTIMIZE_SS_RES    or r_step < r_eps) and
                 (not OPTIMIZE_SS_POS    or p_step < p_eps))
        if small: break
        if not improved:
            no_improve += 1
            if no_improve >= patience: break
        else:
            no_improve = 0

    return best

# ========= Driver =========
np.random.seed(42)
os.makedirs(RUN_DIR, exist_ok=True)
log_csv_init(LOG_CSV)
cache = {}

start0 = dict(ss_widths=ss_widths_base, ss_resxy=ss_resxy_base, ss_resz=ss_resz_base, ss_pos=ss_pos_off_base)

best = random_warmup(n=80, cache=cache)
best_grid = coarse_grid(cache=cache)
if best_grid["obj"] < best["obj"]: best = best_grid
best_local = optimize(best, cache)
if best_local["obj"] < best["obj"]: best = best_local

if SS_WIDTHS_TIED: best["ss_widths"] = [best["ss_widths"][0]]*4
if SS_RES_TIED:    best["ss_resz"]   = best["ss_resxy"]

with open(BEST_JSON, "w") as f:
    json.dump({"best": best, "objective": OBJECTIVE,
               "modes": {"SS_WIDTHS_TIED": SS_WIDTHS_TIED, "SS_RES_TIED": SS_RES_TIED,
                         "OPTIMIZE_SS_WIDTHS": OPTIMIZE_SS_WIDTHS,
                         "OPTIMIZE_SS_RES": OPTIMIZE_SS_RES,
                         "OPTIMIZE_SS_POS": OPTIMIZE_SS_POS}}, f, indent=2)

write_final_txt(BEST_TXT, best["ss_widths"], best["ss_resxy"], best["ss_resz"], best["ss_pos"])

print("\n=== BEST ({}) ===".format(OBJECTIVE))
print(json.dumps(best, indent=2))
print(f"\nLog CSV:   {LOG_CSV}")
print(f"Best JSON: {BEST_JSON}")
print(f"Best TXT:  {BEST_TXT}")

# ========= Plotting (comparison) =========
def cal_for_plot(inputfile):
    y_calc = {label: [] for label in var_labels}
    for pT_value in pT_values:
        pT_value = int(pT_value)
        p, eta = pT_value, 0
        B, m = 2.6, 0.105658
        mydetector = inputfromfile(inputfile, 0)
        calc_result = mydetector.errorcalculation(p, B, eta, m)
        for lab in var_labels:
            y_calc[lab].append(calc_result[lab])
    return y_calc

base_dir = '/data/jlai/iris-hep-log/TrackingResolution-3.0/TrackingResolution-3.0/'
y_calc_best = cal_for_plot(os.path.join(base_dir, 'myODD_test.txt'))

plt.figure(figsize=(20, 10))
for var_label in var_labels:
    idx = var_labels.index(var_label)
    plt.subplot(231 + idx)
    plt.plot(pT_values, np.array(y_calc_best[var_label]), 'o--', label=f"My result (pixel fixed + SS tuned)")
    plt.errorbar(pT_values, y_acts[var_label], yerr=y_acts_err[var_label], fmt='x--', capsize=2, label="ACTS Fit σ ± Δσ")
    plt.xlabel(r"$p_T$ [GeV]")
    plt.ylabel(var_label)
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig('tracking_resolution_comparison_shortstrip.png', dpi=300)
plt.close()
