#!/usr/bin/env python3
# ========= 1. imports =========
import os, json, time, csv, math, random, copy
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import uproot
from tqdm import tqdm
from scipy.stats import norm

# ---- trackingError
from trackingerror import inputfromfile

# ========= 2. Load ACTS references (pixel-only outputs you already produced) =========
print("Loading ACTS results...")
ACTS_DIR = '/data/jlai/iris-hep/OutputPT/'
VAR_LABELS_ALL = ['sigma(d)', 'sigma(z)', 'sigma(phi)', 'sigma(theta)', 'sigma(pt)/pt']
VAR_LABELS = ['sigma(d)', 'sigma(z)', 'sigma(phi)', 'sigma(theta)']  # optimize targets

y_acts = {k: [] for k in VAR_LABELS_ALL}
y_acts_err = {k: [] for k in VAR_LABELS_ALL}

pT_values = np.arange(10, 100, 10, dtype=int)
for pT in pT_values:
    print(f"  reading ACTS pT={pT} GeV")
    with uproot.open(os.path.join(ACTS_DIR, f'output_pt_{pT}', 'tracksummary_ckf.root')) as f:
        tree = f['tracksummary']
        arr = tree.arrays(["t_d0","eLOC0_fit","res_eLOC0_fit",
                           "t_z0","eLOC1_fit","res_eLOC1_fit",
                           "t_phi","ePHI_fit","res_ePHI_fit",
                           "t_theta","eTHETA_fit","res_eTHETA_fit",
                           "t_p","t_pT","eQOP_fit","res_eQOP_fit","t_charge"], library='ak')
    pT_truth = arr['t_p'] * np.sin(arr['t_theta'])
    pT_reco  = np.abs(1.0/arr['eQOP_fit']) * np.sin(arr['t_theta'])

    labels = {
        'sigma(d)':      ak.flatten(arr['res_eLOC0_fit']) * 1e3,
        'sigma(z)':      ak.flatten(arr['res_eLOC1_fit']) * 1e3,
        'sigma(phi)':    ak.flatten(arr['res_ePHI_fit']),
        'sigma(theta)':  ak.flatten(arr['res_eTHETA_fit']),
        'sigma(pt)/pt':  ak.flatten((pT_reco - pT_truth)/pT_reco)
    }

    for key, data in labels.items():
        x = ak.to_numpy(data)
        x = x[~np.isnan(x)]
        N = len(x)
        if N == 0:
            y_acts[key].append(np.nan)
            y_acts_err[key].append(np.nan)
            continue
        mu, sigma = norm.fit(x)
        y_acts[key].append(float(sigma))
        # conservative error on sigma
        y_acts_err[key].append(float(sigma/np.sqrt(2*max(N-1,1)) if N > 1 else 0.0))

# ========= 3. Model / geometry definition =========

# Beam pipe (fixed)
BEAM_PIPE = (0.00227, 9999.0, 9999.0, 0.024)  # width, resxy, resz, pos (m)

# ---- Pixel (use your fixed iteration-1 result as base)
pix_pos_base = np.array([0.032754, 0.070560, 0.110162, 0.165020], dtype=float)
pix_w_base   = np.full(4, 0.014944, dtype=float)
pix_rxy_base = 1.328531e-05
pix_rz_base  = 1.328531e-05

# ---- Short strip (your “best stats” as base)
sst_pos_base = np.array([0.260, 0.360, 0.500, 0.600], dtype=float)
sst_w_base   = np.full(4, 0.01475, dtype=float)
sst_rxy_base = 0.043e-3
sst_rz_base  = 1.2e-3

# ---- Long strip (typical ODD values)
lst_pos_base = np.array([0.820, 1.020], dtype=float)
lst_w_base   = np.full(2, 0.10, dtype=float)
lst_rxy_base = 0.072e-3
lst_rz_base  = 9999.0  # effectively “no z-meas”

# ========= 4. Search toggles =========
# You said pixels are fixed now; only iterate short strip by default.
PIXEL_WIDTHS_TIED = True
PIXEL_RES_TIED    = True
PIXEL_OPT_W = False
PIXEL_OPT_R = False
PIXEL_OPT_P = False

SHORT_WIDTHS_TIED = True     # w1=w2=w3=w4 for short strip
SHORT_RES_TIED    = False    # resxy == resz (set False if you want them independent)
SHORT_OPT_W = True           # <— iterate widths for short strip
SHORT_OPT_R = True           # set True to also tune resolutions
SHORT_OPT_P = True           # set True to also tune positions

LONG_WIDTHS_TIED  = True
LONG_RES_TIED     = False
LONG_OPT_W = True
LONG_OPT_R = True
LONG_OPT_P = True

# ========= 5. Bounds & steps (per-detector, sensible defaults) =========
# Widths (x/X0)
PIX_W_MIN,  PIX_W_MAX  = 0.0, 0.05
SST_W_MIN,  SST_W_MAX  = 0.0, 0.05
LST_W_MIN,  LST_W_MAX  = 0.0, 0.20

# Resolutions (m)
PIX_R_MIN,  PIX_R_MAX  = max(5e-8, pix_rxy_base-0.005e-3), pix_rxy_base+0.005e-3
SST_RX_MIN, SST_RX_MAX = max(5e-8, sst_rxy_base-0.03e-3), sst_rxy_base+0.03e-3
SST_RZ_MIN, SST_RZ_MAX = max(5e-8, sst_rz_base -0.60e-3), sst_rz_base +0.60e-3
LST_RX_MIN, LST_RX_MAX = max(5e-8, lst_rxy_base-0.03e-3), lst_rxy_base+0.03e-3
# long-strip rz stays ~9999; we won’t vary it

# Positions (m)
PIX_P_MIN,  PIX_P_MAX  = -0.005, 0.005
SST_P_MIN,  SST_P_MAX  = -0.20,  0.20
LST_P_MIN,  LST_P_MAX  = -0.20,  0.20

# Initial step sizes
w_step_init = 8e-4
r_step_init = 8e-7
p_step_init = 8e-4

# Stop thresholds
w_eps, r_eps, p_eps = 1e-5, 1e-8, 8e-6
shrink, grow = 0.5, 1.5
max_rounds, patience = 30, 5

# ========= 6. I/O =========
ODD_TXT_OUT = "/data/jlai/iris-hep-log/TrackingResolution-3.0/TrackingResolution-3.0/myODD_test.txt"
RUN_DIR = os.path.dirname(ODD_TXT_OUT) or "."
LOG_CSV = os.path.join(RUN_DIR, "detectors_fit_log.csv")
BEST_JSON = os.path.join(RUN_DIR, "detectors_fit_best.json")
BEST_TXT  = os.path.join(RUN_DIR, "myODD_best.txt")
OBJECTIVE = "chi2"  # or "diff"

os.makedirs(RUN_DIR, exist_ok=True)

# ========= 7. Helpers =========
def clamp(x, lo, hi): return max(lo, min(hi, x))

def write_full_config(path, P):
    """Write beam pipe + pixel(4) + short(4) + long(2)."""
    with open(path, "w") as f:
        f.write("# width(x/X0)    resolutionxy(m)    resolutionz(m)    position (m)\n")
        f.write("# beam pipe\n")
        f.write(f"{BEAM_PIPE[0]} {BEAM_PIPE[1]} {BEAM_PIPE[2]} {BEAM_PIPE[3]}\n")

        f.write("# pixel\n")
        for i in range(4):
            f.write(f"{P['pix']['w'][i]} {P['pix']['rxy']} {P['pix']['rz']} {P['pix']['pos'][i]}\n")

        f.write("# short strip\n")
        for i in range(4):
            f.write(f"{P['sst']['w'][i]} {P['sst']['rxy']} {P['sst']['rz']} {P['sst']['pos'][i]}\n")

        f.write("# long strip\n")
        for i in range(2):
            f.write(f"{P['lst']['w'][i]} {P['lst']['rxy']} {P['lst']['rz']} {P['lst']['pos'][i]}\n")

def write_final_txt(path, P):
    with open(path, "w") as f:
        f.write("#width(x/x0) resolutionxy(mm) resolutionz(mm) position (m)\n")
        f.write("#beam pipe\n")
        f.write("0.00227 9999 9999 0.024\n")

        f.write("#pixel \n")
        for i in range(4):
            f.write(f"{P['pix']['w'][i]:.6f} {P['pix']['rxy']:.6e} {P['pix']['rz']:.6e} {P['pix']['pos'][i]:.6f}\n")

        f.write("#short strip \n")
        for i in range(4):
            f.write(f"{P['sst']['w'][i]:.6f} {P['sst']['rxy']:.6e} {P['sst']['rz']:.6e} {P['sst']['pos'][i]:.6f}\n")

        f.write("#long strip \n")
        for i in range(2):
            f.write(f"{P['lst']['w'][i]:.6f} {P['lst']['rxy']:.6e} {P['lst']['rz']:.6e} {P['lst']['pos'][i]:.6f}\n")

def base_params():
    return {
        "pix": {"w": pix_w_base.copy(), "rxy": float(pix_rxy_base), "rz": float(pix_rz_base), "pos": np.zeros(4)},
        "sst": {"w": sst_w_base.copy(), "rxy": float(sst_rxy_base), "rz": float(sst_rz_base), "pos": np.zeros(4)},
        "lst": {"w": lst_w_base.copy(), "rxy": float(lst_rxy_base), "rz": float(lst_rz_base), "pos": np.zeros(2)},
    }

def realized_positions(P):
    """Convert offsets to absolute positions using the bases."""
    R = {
        "pix": {"w": P["pix"]["w"].copy(), "rxy": P["pix"]["rxy"], "rz": P["pix"]["rz"],
                "pos": pix_pos_base + P["pix"]["pos"]},
        "sst": {"w": P["sst"]["w"].copy(), "rxy": P["sst"]["rxy"], "rz": P["sst"]["rz"],
                "pos": sst_pos_base + P["sst"]["pos"]},
        "lst": {"w": P["lst"]["w"].copy(), "rxy": P["lst"]["rxy"], "rz": P["lst"]["rz"],
                "pos": lst_pos_base + P["lst"]["pos"]},
    }
    return R

def cal(inputfile):
    y_calc = {label: [] for label in VAR_LABELS}
    for pT in pT_values:
        det = inputfromfile(inputfile, 0)
        # Use B consistent with your recent settings
        out = det.errorcalculation(float(pT), 2, 0.0, 0.105658)  # muon mass in GeV
        for lbl in VAR_LABELS:
            y_calc[lbl].append(out[lbl])
    return y_calc

def metrics(y_calc):
    diff_acc, chi2_acc = 0.0, 0.0
    for lbl in VAR_LABELS:
        acts_vals = np.asarray(y_acts[lbl], dtype=float)
        acts_errs = np.asarray(y_acts_err[lbl], dtype=float)
        calc_vals = np.asarray(y_calc[lbl], dtype=float)
        norm = max(float(np.nanmax(acts_vals)), 1e-12)
        a  = acts_vals / norm
        ae = np.maximum(acts_errs / norm, 1e-12)
        c  = calc_vals / norm
        diff_acc += float(np.sum(np.abs(c - a)))
        chi2_acc += float(np.sum(((c - a)/ae)**2))
    return diff_acc, chi2_acc

def params_key(P, rnd=9):
    def r(x): return round(float(x), rnd)
    tup = []
    for det in ("pix","sst","lst"):
        tup += [r(v) for v in P[det]["w"]]
        tup += [r(P[det]["rxy"]), r(P[det]["rz"])]
        tup += [r(v) for v in P[det]["pos"]]
    return tuple(tup)

def log_csv_init(path):
    hdr = ["timestamp",
           # pixel
           "pix_w1","pix_w2","pix_w3","pix_w4","pix_rxy","pix_rz","pix_pos1","pix_pos2","pix_pos3","pix_pos4",
           # short
           "sst_w1","sst_w2","sst_w3","sst_w4","sst_rxy","sst_rz","sst_pos1","sst_pos2","sst_pos3","sst_pos4",
           # long
           "lst_w1","lst_w2","lst_rxy","lst_rz","lst_pos1","lst_pos2",
           "diff","chi2","obj","phase","note"]
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(hdr)

def log_csv_row(path, rec, phase, note=""):
    P = rec["params"]
    R = realized_positions(P)
    row = [time.strftime("%F_%T"),
           # pix
           *R["pix"]["w"], R["pix"]["rxy"], R["pix"]["rz"], *R["pix"]["pos"],
           # sst
           *R["sst"]["w"], R["sst"]["rxy"], R["sst"]["rz"], *R["sst"]["pos"],
           # lst
           *R["lst"]["w"], R["lst"]["rxy"], R["lst"]["rz"], *R["lst"]["pos"],
           rec["diff"], rec["chi2"], rec["obj"], phase, note]
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)

# ========= 8. Objective wrapper with caching =========
_cache = {}

def clamp_params(P):
    Q = copy.deepcopy(P)

    Q["pix"]["w"] = np.clip(P["pix"]["w"], PIX_W_MIN, PIX_W_MAX)
    Q["sst"]["w"] = np.clip(P["sst"]["w"], SST_W_MIN, SST_W_MAX)
    Q["lst"]["w"] = np.clip(P["lst"]["w"], LST_W_MIN, LST_W_MAX)

    Q["pix"]["rxy"] = clamp(P["pix"]["rxy"], PIX_R_MIN, PIX_R_MAX)
    Q["pix"]["rz"]  = clamp(P["pix"]["rz"],  PIX_R_MIN, PIX_R_MAX)

    Q["sst"]["rxy"] = clamp(P["sst"]["rxy"], SST_RX_MIN, SST_RX_MAX)
    Q["sst"]["rz"]  = clamp(P["sst"]["rz"],  SST_RZ_MIN, SST_RZ_MAX)

    Q["lst"]["rxy"] = clamp(P["lst"]["rxy"], LST_RX_MIN, LST_RX_MAX)
    # lst rz stays as-is (9999)

    Q["pix"]["pos"] = np.clip(P["pix"]["pos"], PIX_P_MIN, PIX_P_MAX)
    Q["sst"]["pos"] = np.clip(P["sst"]["pos"], SST_P_MIN, SST_P_MAX)
    Q["lst"]["pos"] = np.clip(P["lst"]["pos"], LST_P_MIN, LST_P_MAX)
    return Q

def objective(P):
    # enforce ties
    PP = clamp_params(P)
    if PIXEL_RES_TIED:
        PP["pix"]["rz"] = PP["pix"]["rxy"]
    if SHORT_RES_TIED:
        PP["sst"]["rz"] = PP["sst"]["rxy"]
    if LONG_RES_TIED:
        PP["lst"]["rz"] = PP["lst"]["rxy"]

    if PIXEL_WIDTHS_TIED:
        PP["pix"]["w"][:] = PP["pix"]["w"][0]
    if SHORT_WIDTHS_TIED:
        PP["sst"]["w"][:] = PP["sst"]["w"][0]
    if LONG_WIDTHS_TIED:
        PP["lst"]["w"][:] = PP["lst"]["w"][0]

    key = params_key(PP)
    if key in _cache:
        return _cache[key]

    R = realized_positions(PP)
    write_full_config(ODD_TXT_OUT, R)

    try:
        y_calc = cal(ODD_TXT_OUT)
        diff_sum, chi2_sum = metrics(y_calc)
        obj = chi2_sum if OBJECTIVE == "chi2" else diff_sum
        if not np.isfinite(obj): obj = float('inf')
    except Exception:
        diff_sum = chi2_sum = float('inf')
        obj = float('inf')

    rec = {"params": PP, "diff": float(diff_sum), "chi2": float(chi2_sum), "obj": float(obj)}
    _cache[key] = rec
    return rec

# ========= 9. Search phases =========
def rand_widths(minv, maxv, n):
    return np.random.uniform(minv, maxv, size=n)

def random_warmup(n=64):
    best = {"obj": float('inf')}
    pbar = tqdm(range(n), desc="Warmup")
    for _ in pbar:
        P = base_params()

        # PIXEL
        if PIXEL_OPT_W:
            if PIXEL_WIDTHS_TIED:
                w = np.random.uniform(PIX_W_MIN, PIX_W_MAX)
                P["pix"]["w"][:] = w
            else:
                P["pix"]["w"] = rand_widths(PIX_W_MIN, PIX_W_MAX, 4)
        if PIXEL_OPT_R:
            if PIXEL_RES_TIED:
                r = np.random.uniform(PIX_R_MIN, PIX_R_MAX)
                P["pix"]["rxy"] = r; P["pix"]["rz"] = r
            else:
                P["pix"]["rxy"] = np.random.uniform(PIX_R_MIN, PIX_R_MAX)
                P["pix"]["rz"]  = np.random.uniform(PIX_R_MIN, PIX_R_MAX)
        if PIXEL_OPT_P:
            P["pix"]["pos"] = np.random.uniform(PIX_P_MIN, PIX_P_MAX, size=4)

        # SHORT STRIP
        if SHORT_OPT_W:
            if SHORT_WIDTHS_TIED:
                w = np.random.uniform(SST_W_MIN, SST_W_MAX)
                P["sst"]["w"][:] = w
            else:
                P["sst"]["w"] = rand_widths(SST_W_MIN, SST_W_MAX, 4)
        if SHORT_OPT_R:
            if SHORT_RES_TIED:
                r = np.random.uniform(SST_RX_MIN, SST_RX_MAX)
                P["sst"]["rxy"] = r; P["sst"]["rz"] = r
            else:
                P["sst"]["rxy"] = np.random.uniform(SST_RX_MIN, SST_RX_MAX)
                P["sst"]["rz"]  = np.random.uniform(SST_RZ_MIN, SST_RZ_MAX)
        if SHORT_OPT_P:
            P["sst"]["pos"] = np.random.uniform(SST_P_MIN, SST_P_MAX, size=4)

        # LONG STRIP
        if LONG_OPT_W:
            if LONG_WIDTHS_TIED:
                w = np.random.uniform(LST_W_MIN, LST_W_MAX)
                P["lst"]["w"][:] = w
            else:
                P["lst"]["w"] = rand_widths(LST_W_MIN, LST_W_MAX, 2)
        if LONG_OPT_R:
            if LONG_RES_TIED:
                r = np.random.uniform(LST_RX_MIN, LST_RX_MAX)
                P["lst"]["rxy"] = r; P["lst"]["rz"] = r
            else:
                P["lst"]["rxy"] = np.random.uniform(LST_RX_MIN, LST_RX_MAX)
                # P["lst"]["rz"] vary only if you really want (not common)
        if LONG_OPT_P:
            P["lst"]["pos"] = np.random.uniform(LST_P_MIN, LST_P_MAX, size=2)

        rec = objective(P)
        log_csv_row(LOG_CSV, rec, "warmup", "")
        if rec["obj"] < best["obj"]:
            best = rec
    return best

def coarse_grid():
    # keep combinatorics small
    dw = 5e-4; dp_pix = 5e-4; dp_sst = 5e-3; dp_lst = 5e-3
    grids = {}

    # widths patterns per detector
    def tied_width_grid(w0, dw, n=5):
        return [np.full_like(w0*np.ones(1), clamp(w, 0, 1e9))[0] for w in
                [w0, w0+dw, w0-dw, w0+2*dw, w0-2*dw]]

    # PIXEL
    grids["pix_w"] = [pix_w_base] if not PIXEL_OPT_W else (
        [np.full(4, tied_width_grid(pix_w_base[0], dw)[k]) for k in range(5)]
        if PIXEL_WIDTHS_TIED else [
            pix_w_base,
            pix_w_base + dw, pix_w_base - dw,
            pix_w_base + np.array([dw,0,0,dw]),
            pix_w_base - np.array([dw,0,0,dw]),
        ]
    )
    grids["pix_r"] = [(pix_rxy_base, pix_rz_base)] if not PIXEL_OPT_R else (
        [(r, r) for r in np.linspace(PIX_R_MIN, PIX_R_MAX, 3)]
        if PIXEL_RES_TIED else
        [(rx, rz) for rx in np.linspace(PIX_R_MIN, PIX_R_MAX, 3)
                  for rz in np.linspace(PIX_R_MIN, PIX_R_MAX, 3)]
    )
    grids["pix_p"] = [np.zeros(4)] if not PIXEL_OPT_P else [
        np.zeros(4), np.full(4, +dp_pix), np.full(4, -dp_pix)
    ]

    # SHORT
    grids["sst_w"] = [sst_w_base] if not SHORT_OPT_W else (
        [np.full(4, tied_width_grid(sst_w_base[0], dw)[k]) for k in range(5)]
        if SHORT_WIDTHS_TIED else [
            sst_w_base, sst_w_base + dw, sst_w_base - dw
        ]
    )
    grids["sst_r"] = [(sst_rxy_base, sst_rz_base)] if not SHORT_OPT_R else (
        [(r, r) for r in np.linspace(SST_RX_MIN, SST_RX_MAX, 3)]
        if SHORT_RES_TIED else
        [(rx, rz) for rx in np.linspace(SST_RX_MIN, SST_RX_MAX, 3)
                  for rz in np.linspace(SST_RZ_MIN, SST_RZ_MAX, 3)]
    )
    grids["sst_p"] = [np.zeros(4)] if not SHORT_OPT_P else [
        np.zeros(4), np.full(4, +dp_sst), np.full(4, -dp_sst)
    ]

    # LONG
    grids["lst_w"] = [lst_w_base] if not LONG_OPT_W else (
        [np.full(2, tied_width_grid(lst_w_base[0], dw)[k]) for k in range(5)]
        if LONG_WIDTHS_TIED else [
            lst_w_base, lst_w_base + dw, lst_w_base - dw
        ]
    )
    grids["lst_r"] = [(lst_rxy_base, lst_rz_base)] if not LONG_OPT_R else (
        [(r, r) for r in np.linspace(LST_RX_MIN, LST_RX_MAX, 3)]
        if LONG_RES_TIED else
        [(rx, lst_rz_base) for rx in np.linspace(LST_RX_MIN, LST_RX_MAX, 3)]
    )
    grids["lst_p"] = [np.zeros(2)] if not LONG_OPT_P else [
        np.zeros(2), np.full(2, +dp_lst), np.full(2, -dp_lst)
    ]

    best = {"obj": float('inf')}
    total = (len(grids["pix_w"])*len(grids["pix_r"])*len(grids["pix_p"])*
             len(grids["sst_w"])*len(grids["sst_r"])*len(grids["sst_p"])*
             len(grids["lst_w"])*len(grids["lst_r"])*len(grids["lst_p"]))
    pbar = tqdm(total=total, desc="Coarse grid")
    for pw in grids["pix_w"]:
        for prxy, prz in grids["pix_r"]:
            for pp in grids["pix_p"]:
                for sw in grids["sst_w"]:
                    for srxy, srz in grids["sst_r"]:
                        for sp in grids["sst_p"]:
                            for lw in grids["lst_w"]:
                                for lrxy, lrz in grids["lst_r"]:
                                    for lp in grids["lst_p"]:
                                        P = {
                                            "pix": {"w": np.array(pw, float), "rxy": float(prxy), "rz": float(prz), "pos": np.array(pp, float)},
                                            "sst": {"w": np.array(sw, float), "rxy": float(srxy), "rz": float(srz), "pos": np.array(sp, float)},
                                            "lst": {"w": np.array(lw, float), "rxy": float(lrxy), "rz": float(lrz), "pos": np.array(lp, float)},
                                        }
                                        rec = objective(P)
                                        log_csv_row(LOG_CSV, rec, "grid", "")
                                        if rec["obj"] < best["obj"]:
                                            best = rec
                                        pbar.update(1)
    pbar.close()
    return best

def explore_axis_scalar(P, getter, setter, step, lo, hi, label):
    """Generic 1D line search with {center, center±step, optional 2nd step}."""
    c = float(getter(P))
    cand = [clamp(c, lo, hi), clamp(c + step, lo, hi), clamp(c - step, lo, hi)]
    best = {"obj": float("inf")}
    for x in cand:
        Q  = copy.deepcopy(P)
        # fix numpy arrays:
        for d in ("pix","sst","lst"):
            Q[d]["w"]  = np.array(P[d]["w"],  dtype=float)
            Q[d]["pos"]= np.array(P[d]["pos"],dtype=float)
        setter(Q, float(x))
        rec = objective(Q)
        log_csv_row(LOG_CSV, rec, "search", label)
        if rec["obj"] < best["obj"]:
            best = rec
    # 2nd step in winning direction
    c_best = float(getter(best["params"]))
    if abs(c_best - c) > 1e-18:
        direction = math.copysign(1.0, c_best - c)
        x2 = clamp(c + 2.0*direction*step, lo, hi)
        Q2 = copy.deepcopy(best["params"])
        for d in ("pix","sst","lst"):
            Q2[d]["w"]  = np.array(best["params"][d]["w"],  dtype=float)
            Q2[d]["pos"]= np.array(best["params"][d]["pos"],dtype=float)
        setter(Q2, float(x2))
        rec2 = objective(Q2)
        log_csv_row(LOG_CSV, rec2, "search", f"{label} (2-step)")
        if rec2["obj"] < best["obj"]:
            best = rec2
    return best

def optimize(start_rec):
    best = start_rec
    w_step, r_step, p_step = w_step_init, r_step_init, p_step_init
    no_improve = 0

    for rnd in range(1, max_rounds+1):
        P = best["params"]
        improved = False

        # --- widths (tied as configured) ---
        if SHORT_OPT_W:
            if SHORT_WIDTHS_TIED:
                # single scalar for all 4 short strips
                def g(P): return P["sst"]["w"][0]
                def s(P, x): P["sst"]["w"][:] = x
                rec = explore_axis_scalar(P, g, s, w_step, SST_W_MIN, SST_W_MAX, "sst width (all)")
                if rec["obj"] + 1e-12 < best["obj"]:
                    best, improved, w_step = rec, True, w_step*grow
                else:
                    w_step *= shrink
            else:
                # per-layer (rarely needed)
                for i in range(4):
                    def gi(P, i=i): return P["sst"]["w"][i]
                    def si(P, x, i=i): P["sst"]["w"][i] = x
                    rec = explore_axis_scalar(P, gi, si, w_step, SST_W_MIN, SST_W_MAX, f"sst w[{i}]")
                    if rec["obj"] + 1e-12 < best["obj"]:
                        best, improved, w_step = rec, True, w_step*grow
                    else:
                        w_step *= shrink

        # (Optional) pixel/long widths if enabled
        if PIXEL_OPT_W:
            if PIXEL_WIDTHS_TIED:
                def g(P): return P["pix"]["w"][0]
                def s(P, x): P["pix"]["w"][:] = x
                rec = explore_axis_scalar(P, g, s, w_step, PIX_W_MIN, PIX_W_MAX, "pix width (all)")
                if rec["obj"] + 1e-12 < best["obj"]:
                    best, improved, w_step = rec, True, w_step*grow
                else:
                    w_step *= shrink
        if LONG_OPT_W:
            if LONG_WIDTHS_TIED:
                def g(P): return P["lst"]["w"][0]
                def s(P, x): P["lst"]["w"][:] = x
                rec = explore_axis_scalar(P, g, s, w_step, LST_W_MIN, LST_W_MAX, "lst width (all)")
                if rec["obj"] + 1e-12 < best["obj"]:
                    best, improved, w_step = rec, True, w_step*grow
                else:
                    w_step *= shrink

        # --- resolutions (enable if you want) ---
        if SHORT_OPT_R:
            if SHORT_RES_TIED:
                def g(P): return P["sst"]["rxy"]
                def s(P, x): P["sst"]["rxy"]=x; P["sst"]["rz"]=x
                rec = explore_axis_scalar(P, g, s, r_step, SST_RX_MIN, SST_RX_MAX, "sst res (tied)")
                if rec["obj"] + 1e-12 < best["obj"]:
                    best, improved, r_step = rec, True, r_step*grow
                else:
                    r_step *= shrink
            else:
                def g1(P): return P["sst"]["rxy"]
                def s1(P, x): P["sst"]["rxy"]=x
                rec = explore_axis_scalar(P, g1, s1, r_step, SST_RX_MIN, SST_RX_MAX, "sst resxy")
                if rec["obj"] + 1e-12 < best["obj"]:
                    best, improved, r_step = rec, True, r_step*grow
                else:
                    r_step *= shrink
                def g2(P): return P["sst"]["rz"]
                def s2(P, x): P["sst"]["rz"]=x
                rec = explore_axis_scalar(best["params"], g2, s2, r_step, SST_RZ_MIN, SST_RZ_MAX, "sst resz")
                if rec["obj"] + 1e-12 < best["obj"]:
                    best, improved, r_step = rec, True, r_step*grow
                else:
                    r_step *= shrink

        # --- positions (enable if you want) ---
        if SHORT_OPT_P:
            for i in range(4):
                def gi(P, i=i): return P["sst"]["pos"][i]
                def si(P, x, i=i): P["sst"]["pos"][i]=x
                rec = explore_axis_scalar(best["params"], gi, si, p_step, SST_P_MIN, SST_P_MAX, f"sst pos[{i}]")
                if rec["obj"] + 1e-12 < best["obj"]:
                    best, improved, p_step = rec, True, p_step*grow
                else:
                    p_step *= shrink

        # (pixel/long pos/res if toggled True)
        # ... you can mirror the blocks above as needed

        log_csv_row(LOG_CSV, best, "round", f"r={rnd}, steps=(w={w_step:.2g}, r={r_step:.2g}, p={p_step:.2g})")

        small = (w_step < w_eps) and (r_step < r_eps) and (p_step < p_eps)
        if small: break
        if not improved:
            no_improve += 1
            if no_improve >= patience: break
        else:
            no_improve = 0

    return best

# ========= 10. Driver =========
log_csv_init(LOG_CSV)

start = {"params": base_params(), "diff": 0.0, "chi2": 0.0, "obj": float('inf')}
start = objective(start["params"])  # evaluate base
log_csv_row(LOG_CSV, start, "init", "base")

best = random_warmup(n=200)
if best["obj"] > start["obj"]:
    best = start

best_grid = coarse_grid()
if best_grid["obj"] < best["obj"]:
    best = best_grid

best_local = optimize(best)
if best_local["obj"] < best["obj"]:
    best = best_local

# Save
def params_to_serializable(P):
    return {
        "pix": {
            "w":   np.asarray(P["pix"]["w"]).tolist(),
            "rxy": float(P["pix"]["rxy"]),
            "rz":  float(P["pix"]["rz"]),
            "pos": np.asarray(P["pix"]["pos"]).tolist(),
        },
        "sst": {
            "w":   np.asarray(P["sst"]["w"]).tolist(),
            "rxy": float(P["sst"]["rxy"]),
            "rz":  float(P["sst"]["rz"]),
            "pos": np.asarray(P["sst"]["pos"]).tolist(),
        },
        "lst": {
            "w":   np.asarray(P["lst"]["w"]).tolist(),
            "rxy": float(P["lst"]["rxy"]),
            "rz":  float(P["lst"]["rz"]),
            "pos": np.asarray(P["lst"]["pos"]).tolist(),
        },
    }

payload = {
    "best": {
        "params": params_to_serializable(best["params"]),
        "diff": float(best["diff"]),
        "chi2": float(best["chi2"]),
        "obj":  float(best["obj"]),
    },
    "objective": OBJECTIVE,
}
with open(BEST_JSON, "w") as f:
    # json.dump({"best": best, "objective": OBJECTIVE}, f, indent=2)
    json.dump(payload, f, indent=2)

write_final_txt(BEST_TXT, realized_positions(best["params"]))

print("\n=== BEST ({}) ===".format(OBJECTIVE))
print(json.dumps(payload["best"], indent=2)) 
print(f"\nLog CSV:   {LOG_CSV}")
print(f"Best JSON: {BEST_JSON}")
print(f"Best TXT:  {BEST_TXT}")

# ========= 11. Plot prediction vs ACTS (final best) =========
def cal_vec_from_params(P):
    write_full_config(ODD_TXT_OUT, realized_positions(P))
    return cal(ODD_TXT_OUT)

pred = cal_vec_from_params(best["params"])

plt.figure(figsize=(20, 10))
labels_plot = VAR_LABELS_ALL  # show all, with ACTS errbars where available
for idx, var in enumerate(labels_plot):
    plt.subplot(231 + idx)
    if var in pred:
        plt.plot(pT_values, np.array(pred[var]), 'o--', label="Model (best)")
    if var in y_acts:
        y = np.array(y_acts[var], float); yerr = np.array(y_acts_err[var], float)
        plt.errorbar(pT_values, y, yerr=yerr, fmt='x--', capsize=2, label="ACTS fit σ ± Δσ")
    plt.xlabel(r"$p_T$ [GeV]"); plt.ylabel(var); plt.yscale('log'); plt.grid(True); plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, 'tracking_resolution_comparison.png'), dpi=300)
plt.close()
