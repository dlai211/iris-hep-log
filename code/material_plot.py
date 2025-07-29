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

    # Loop over histograms inside each directory
    for subkey in subdir.GetListOfKeys():
        obj = subkey.ReadObj()
        name = obj.GetName().replace(";1", "")
        print(f"  📊 Found: {name}")

        c = ROOT.TCanvas("c", "", 800, 600)
        obj.SetStats(False)

        if obj.InheritsFrom("TH2"):
            obj.Draw("COLZ")
        elif obj.InheritsFrom("TH1"):
            obj.Draw("HIST")

        # Replace slashes and semicolons in names for safe filenames
        safe_name = name.replace("/", "_").replace(";", "")
        safe_dir = dirname.replace("/", "_").replace(";", "")
        outfile = f"{outdir}/{safe_dir}_{safe_name}.png"

        c.SaveAs(outfile)
        print(f"  ✅ Saved: {outfile}")
