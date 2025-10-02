#!/usr/bin/env python3
# PyROOT plots for track-parameter resolutions vs pT (muon, eta=0)
# Saves one PNG+PDF per parameter.

import os, math, array, sys
import numpy as np
import uproot
import awkward as ak
from scipy.stats import norm

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTitleFont(42, "XYZ")
ROOT.gStyle.SetLabelFont(42, "XYZ")
ROOT.gStyle.SetLegendTextSize(0.030)
ROOT.gStyle.SetEndErrorSize(5)   # bigger little caps on error bars
ROOT.gStyle.SetErrorX(0.)        # no horizontal error caps (optional)
ROOT.TGaxis.SetMaxDigits(3) 

# ---- Config ----
OUTDIR = "/data/jlai/iris-hep-log/track_parameter_plots"
BASEDIR = "/data/jlai/iris-hep/OutputPT"
MAT_PATH = "/data/jlai/iris-hep/material_composition.root"
os.makedirs(OUTDIR, exist_ok=True)

# pT grid (match your previous script)
pT_values = np.concatenate((np.linspace(1, 15, 15), np.linspace(20, 100, 9))).astype(int)
# pT_values = np.arange(10, 90, 20).astype(int)

# Var labels / titles
var_labels  = ['sigma(d)', 'sigma(z)', 'sigma(phi)', 'sigma(theta)', 'sigma(pt)/pt']
y_titles    = ['σ(d₀) [μm]', 'σ(z₀) [μm]', 'σ(ϕ)', 'σ(θ)', 'σ(pₜ)/pₜ']
y_titles = ['#sigma(d_{0}) [#mum]', '#sigma(z_{0}) [#mum]', '#sigma(#phi)', '#sigma(#theta)', '#sigma(p_{T})/p_{T}']
file_tags   = ['d0', 'z0', 'phi', 'theta', 'ptOverpt']

# Colors
steelblue = ROOT.TColor.GetColor("#4682B4")
orange    = ROOT.kOrange+1

# ===== Tracking Resolution Calculator bits =====
sys.path.append('/data/jlai/iris-hep-log/code/')
from trackingerror import Detector

def _vals_edges_any(h):
    try:
        vals, edges = h.to_numpy()
        return np.asarray(vals), np.asarray(edges)
    except Exception:
        pass
    try:
        edges = np.asarray(h.axis().edges())
    except Exception:
        edges = None
    vals = np.asarray(h.values())
    if edges is None:
        edges = np.linspace(-5,5,len(vals)+1)
    return vals, edges

def _x0_interp(mfile, key, eta):
    vals, edges = _vals_edges_any(mfile[key])
    centers = 0.5*(edges[:-1] + edges[1:])
    eta_c = np.clip(eta, centers.min(), centers.max())
    return float(np.interp(eta_c, centers, vals))

def _vol_to_tech(vid: int) -> str:
    PIXEL, SSTRIP, LSTRIP = {16,17,18}, {23,24,25}, {28,29,30}
    if vid in PIXEL:  return "pixel"
    if vid in SSTRIP: return "sstrip"
    if vid in LSTRIP: return "lstrip"
    return "pixel"

def build_detector_for(pT, eta):
    MEAS = f"{BASEDIR}/output_pt_{pT}/measurements.root"

    # read layer medians
    cols = ["volume_id","layer_id","true_x","true_y","true_z",
            "var_loc0","var_loc1","residual_loc0","residual_loc1"]
    with uproot.open(MEAS) as f:
        df = f["measurements"].arrays(cols, library="pd")

    df["R_m"] = np.sqrt(df.true_x**2 + df.true_y**2) * 1e-3
    def _std_meters(s): return np.nanstd(s, ddof=0) * 1e-3
    dfL = (
        df.groupby(["volume_id","layer_id"], as_index=False)
          .agg(R_m=("R_m","median"),
               res_loc0_std_m=("residual_loc0", _std_meters),
               res_loc1_std_m=("residual_loc1", _std_meters))
          .sort_values(["volume_id","layer_id"])
          .reset_index(drop=True)
    )
    dfL["tech"] = dfL["volume_id"].map(_vol_to_tech)

    # x/X0 per technology at given eta (split total roughly per #layers)
    with uproot.open(MAT_PATH) as mf:
        x0_beam  = _x0_interp(mf, "beampipe_x0_vs_eta_all", eta)
        x0_pix   = _x0_interp(mf, "pixel_x0_vs_eta_all",     eta)
        x0_sstr  = _x0_interp(mf, "sstrips_x0_vs_eta_all",   eta)
        x0_lstr  = _x0_interp(mf, "lstrips_x0_vs_eta_all",   eta)

    tech_total = {"pixel": x0_pix, "sstrip": x0_sstr, "lstrip": x0_lstr}
    n_by_tech  = dfL.groupby("tech").size().to_dict()
    dfL["x_over_X0"] = dfL.apply(
        lambda r: tech_total[r["tech"]] / max(n_by_tech.get(r["tech"],1),1), axis=1
    )
    dfL["sigma_loc1_for_add"] = dfL["res_loc1_std_m"].fillna(9999.0)

    det = Detector()
    det.addlayer(x0_beam, 9999.0, 9999.0, 0.024)  # beampipe

    # skip lstrip if that's what you want (matches your current script)
    for _, row in dfL.sort_values("R_m").iterrows():
        if row["tech"] == "lstrip":
            continue
        det.addlayer(float(row["x_over_X0"]),
                     float(row["res_loc0_std_m"]),
                     float(row["sigma_loc1_for_add"]),
                     float(row["R_m"]))
    return det

def calc_resolution_curve():
    B = 2.0                # Tesla
    m_mu = 0.105658        # GeV
    eta = 0.0
    out = {k: [] for k in var_labels}
    for pT in pT_values:
        p = float(pT)  # for eta=0, p ≈ pT
        det = build_detector_for(pT, eta)
        res = det.errorcalculation(p, B, eta, m_mu)  # expects dict with our keys
        for k in var_labels:
            out[k].append(res[k])
    return out

# ===== ACTS σ (fit) from files =====
def acts_resolution_curve():
    y = {k: [] for k in var_labels}
    yerr = {k: [] for k in var_labels}
    for pT in pT_values:
        fpath = f"{BASEDIR}/output_pt_{pT}/tracksummary_ckf.root"
        rf = uproot.open(fpath)
        arr = rf["tracksummary"].arrays(
            ["t_d0","eLOC0_fit","res_eLOC0_fit",
             "t_z0","eLOC1_fit","res_eLOC1_fit",
             "t_phi","ePHI_fit","res_ePHI_fit",
             "t_theta","eTHETA_fit","res_eTHETA_fit",
             "t_p","t_pT","eQOP_fit","res_eQOP_fit","t_charge"],
            library='ak'
        )

        pT_truth = arr['t_p'] * np.sin(arr['t_theta'])
        pT_reco  = np.abs(1/arr['eQOP_fit']) * np.sin(arr['t_theta'])

        labels = {
            'sigma(d)'    : ak.flatten(arr['res_eLOC0_fit']) * 1e3, # mm->μm
            'sigma(z)'    : ak.flatten(arr['res_eLOC1_fit']) * 1e3, # mm->μm
            'sigma(phi)'  : ak.flatten(arr['res_ePHI_fit']),
            'sigma(theta)': ak.flatten(arr['res_eTHETA_fit']),
            'sigma(pt)/pt': ak.flatten( (pT_reco - pT_truth) / pT_truth )
        }

        for k, data in labels.items():
            d = ak.to_numpy(data)
            d = d[~np.isnan(d)]
            N = len(d)
            if N < 2:
                y[k].append(float('nan'))
                yerr[k].append(0.0)
                continue
            mu, sig = norm.fit(d)
            y[k].append(sig)
            yerr[k].append(sig / math.sqrt(2*max(N-1,1)))
    return y, yerr

# ===== Make ROOT TGraphErrors and Draw =====
def make_graph(xvals, yvals, yerrs=None, marker=20, color=ROOT.kBlack):
    n = len(xvals)
    xv = array.array('d', [float(x) for x in xvals])
    yv = array.array('d', [float(y) for y in yvals])
    xe = array.array('d', [0.0]*n)
    ye = array.array('d', [float(e) for e in (yerrs if yerrs else [0.0]*n)])
    g = ROOT.TGraphErrors(n, xv, yv, xe, ye)
    g.SetMarkerStyle(marker)
    g.SetMarkerColor(color)
    g.SetLineColor(color)
    g.SetMarkerSize(1)
    g.SetLineWidth(2)
    return g

def draw_one(param_idx, x, calc_y, acts_y, acts_yerr, ytitle, tag):
    c = ROOT.TCanvas(f"c_{tag}", "", 850, 620)
    c.SetRightMargin(0.12)
    c.SetGridx(True) 
    c.SetGridy(True) 
    c.SetTicks(1, 1)   # ticks on top and right
    c.SetLeftMargin(0.16)  # give the y-axis/title more room

    frame = ROOT.TH1F("frame","", 100, 0, 105)
    frame.GetXaxis().SetTitle("p_{T} [GeV]")
    frame.GetYaxis().SetTitle(ytitle)
    # auto y range from data
    all_y = [v for v in calc_y + acts_y if np.isfinite(v)]
    if len(all_y)==0: ymin, ymax = 0.0, 1.0
    else:
        ymin = min(all_y); ymax = max(all_y)
        if ymin == ymax: ymax = ymin + 1.0
        pad = 0.15*(ymax - ymin)
        ymin, ymax = ymin - pad, ymax + pad
        if ymin < 0: ymin = 0
    frame.GetYaxis().SetRangeUser(0, ymax)


    axY = frame.GetYaxis()
    axY.SetTitleOffset(1.3)   # move the title away from the tick labels
    axY.SetLabelSize(0.035)   # slightly smaller numbers; 0.04 default-ish
    axY.SetLabelOffset(0.005) # a bit more space between ticks and numbers

    frame.Draw()
    c.Update()

    alpha = 0.80  # 0 = fully transparent, 1 = opaque

    g_calc = make_graph(x, calc_y, None, marker=20, color=steelblue)
    g_calc.SetLineWidth(0)
    g_calc.SetMarkerColor(ROOT.TColor.GetColorTransparent(steelblue, alpha))
    g_calc.Draw("P SAME")

    g_acts = make_graph(x, acts_y, acts_yerr, marker=21, color=orange)
    g_acts.SetMarkerColor(ROOT.TColor.GetColorTransparent(orange, alpha))
    g_acts.Draw("PE1 SAME")

    leg = ROOT.TLegend(0.55, 0.75, 0.88, 0.88)
    leg.AddEntry(g_calc, "Tracking Resolution Calculator", "lp")
    leg.AddEntry(g_acts, "ACTS fit #sigma #pm #Delta#sigma", "pe")
    leg.SetBorderSize(0)
    leg.SetTextSize(0.034)
    leg.Draw()

    # label like your example
    pave = ROOT.TPaveText(0.18, 0.75, 0.48, 0.88, "NDC")
    pave.SetFillStyle(0); pave.SetBorderSize(0)
    pave.SetTextFont(42)     # 42 = normal (not bold); 62 = bold
    pave.SetTextSize(0.030)  # control size
    pave.SetTextAlign(13)    # left-top
    pave.AddText("ODD Simulation")
    pave.AddText("single muons, #eta=0")
    pave.Draw()

    # right-hand axis mirroring left
    axL = frame.GetYaxis()
    right = ROOT.TGaxis(
        c.GetUxmax(), c.GetUymin(), c.GetUxmax(), c.GetUymax(),
        0, ymax, 510, "+L"
    )
    right.SetLabelSize(0)      # hides numbers on the right axis
    right.SetTitleSize(0)      # hides any title on the right axis
    right.SetTickSize(axL.GetTickLength())  # visible ticks
    right.Draw()

    # c.RedrawAxis() # make sure axes are on top of everything

    out_png = os.path.join(OUTDIR, f"{tag}_vs_pT_muon_eta0.png")
    out_pdf = os.path.join(OUTDIR, f"{tag}_vs_pT_muon_eta0.pdf")
    c.SaveAs(out_png)
    c.SaveAs(out_pdf)
    c.Close()

def main():
    print("Computing calculator curves…")
    y_calc = calc_resolution_curve()

    print("Fitting ACTS resolutions…")
    y_acts, y_acts_err = acts_resolution_curve()

    # per-parameter plots
    for i, key in enumerate(var_labels):
        draw_one(
            i,
            pT_values.tolist(),
            y_calc[key],
            y_acts[key],
            y_acts_err[key],
            y_titles[i],
            f"{file_tags[i]}"
        )
    print(f"Done. Plots saved in: {OUTDIR}")

if __name__ == "__main__":
    main()
