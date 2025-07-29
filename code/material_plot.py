import ROOT
import os

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
        if "t" in hists and "x0" in hists:
            h_t = hists["t"]
            h_x0 = hists["x0"]

            h_ratio = h_t.Clone("t_over_x0")
            h_ratio.Divide(h_x0)

            c = ROOT.TCanvas("c", "", 800, 600)
            h_ratio.SetTitle("t / x0")
            h_ratio.SetStats(False)
            h_ratio.Draw("COLZ")
            safe_dir = dirname.replace("/", "_").replace(";", "")
            outfile = f"{outdir}/{safe_dir}_t_over_x0.png"
            c.SaveAs(outfile)
            print(f"  ✅ Saved: {outfile}")
