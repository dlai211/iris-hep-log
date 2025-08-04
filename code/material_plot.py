import ROOT
import os
import numpy as np

ROOT.gROOT.SetBatch(True)  # Don't open canvas GUI

f = ROOT.TFile("/data/jlai/iris-hep/acts/thirdparty/OpenDataDetector/data/odd-material-maps.root")
outdir = "material_plots"
os.makedirs(outdir, exist_ok=True)

# Loop over top-level keys (directories)
for key in f.GetListOfKeys():
    dirname = key.GetName()
    if not dirname.startswith("SurfaceMaterial_vol"):
        continue

    print(f"📁 Entering directory: {dirname}")
    subdir = f.Get(dirname)
    if not isinstance(subdir, ROOT.TDirectoryFile):
        continue

    hists = {}

    # Loop over histograms inside each directory
    for subkey in subdir.GetListOfKeys():
        obj = subkey.ReadObj()
        name = obj.GetName().replace(";1", "")

        if name not in ["t", "x0"]:
            continue

        print(f"  📊 Found: {name}")
        hists[name] = obj.Clone()


        c = ROOT.TCanvas("c", "", 800, 600)
        obj.SetStats(False)
        obj.Draw("COLZ")


        # if obj.InheritsFrom("TH2"):
        #     obj.Draw("COLZ")
        # elif obj.InheritsFrom("TH1"):
        #     obj.Draw("HIST")

        # Replace slashes and semicolons in names for safe filenames
        safe_name = name.replace("/", "_").replace(";", "")
        safe_dir = dirname.replace("/", "_").replace(";", "")
        outfile = f"{outdir}/{safe_dir}_{safe_name}.png"

        c.SaveAs(outfile)
        print(f"  ✅ Saved: {outfile}")

        # Plot t/x0 if both exist
        # if "t" in hists and "x0" in hists:
        #     h_t = hists["t"]
        #     h_x0 = hists["x0"]

        #     h_ratio = h_t.Clone("t_over_x0")
        #     h_ratio.Divide(h_x0)

        #     c = ROOT.TCanvas("c", "", 800, 600)
        #     h_ratio.SetTitle("t / x0")
        #     h_ratio.SetStats(False)
        #     h_ratio.Draw("COLZ")
        #     safe_dir = dirname.replace("/", "_").replace(";", "")
        #     outfile = f"{outdir}/{safe_dir}_t_over_x0.png"
        #     c.SaveAs(outfile)
        #     print(f"  ✅ Saved: {outfile}")


        # Plot t/x0 if both exist
        if "t" in hists and "x0" in hists:
            h_t = hists["t"]
            h_x0 = hists["x0"]

            h_ratio = h_t.Clone("t_over_x0")
            h_ratio.Divide(h_x0)

            # Save the 2D t/x0 plot
            c = ROOT.TCanvas("c", "", 800, 600)
            h_ratio.SetTitle("t / x0")
            h_ratio.SetStats(False)
            h_ratio.Draw("COLZ")
            outfile = f"{outdir}/{safe_dir}_t_over_x0.png"
            c.SaveAs(outfile)
            print(f"  ✅ Saved: {outfile}")

            # === Build 1D profile of t/x0 vs η ===
            eta_vals = []
            ratio_vals = []

            nbx = h_ratio.GetNbinsX()
            nby = h_ratio.GetNbinsY()

            for ix in range(1, nbx + 1):
                for iy in range(1, nby + 1):
                    val = h_ratio.GetBinContent(ix, iy)
                    if val <= 0:
                        continue

                    # Axis conversions: b0 = phi, b1 = theta
                    theta = h_ratio.GetYaxis().GetBinCenter(iy) * np.pi / 180.0  # deg → rad
                    eta = -np.log(np.tan(theta / 2.0))

                    eta_vals.append(eta)
                    ratio_vals.append(val)

            # Bin into 1D histogram
            import matplotlib.pyplot as plt

            bins = np.linspace(-4, 4, 80)
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            bin_sums = np.zeros_like(bin_centers)
            bin_counts = np.zeros_like(bin_centers)

            for e, v in zip(eta_vals, ratio_vals):
                idx = np.searchsorted(bins, e) - 1
                if 0 <= idx < len(bin_centers):
                    bin_sums[idx] += v
                    bin_counts[idx] += 1

            bin_means = np.divide(bin_sums, bin_counts, out=np.zeros_like(bin_sums), where=bin_counts > 0)

            plt.figure(figsize=(8, 5))
            plt.plot(bin_centers, bin_means, drawstyle="steps-mid")
            plt.xlabel("η")
            plt.ylabel("⟨t / X₀⟩")
            plt.title(f"{dirname} : Average material vs. η")
            plt.grid(True)
            plt.savefig(f"{outdir}/{safe_dir}_t_over_x0_vs_eta.png")
            print(f"  📈 Saved: {safe_dir}_t_over_x0_vs_eta.png")
