import ROOT
import os

ROOT.gROOT.SetBatch(True)  # Don't open canvas GUI

f = ROOT.TFile("/data/jlai/iris-hep/acts/thirdparty/OpenDataDetector/data/odd-material-maps.root")
outdir = "material_plots"
os.makedirs(outdir, exist_ok=True)

# Loop over keys (directories)
for key in f.GetListOfKeys():
    dirname = key.GetName()
    if not dirname.startswith("SurfaceMaterial_vol"):
        continue

    print(f"📁 Entering directory: {dirname}")
    subdir = f.Get(dirname)
    if not isinstance(subdir, ROOT.TDirectoryFile):
        continue

    # Loop over objects inside the directory
    for subkey in subdir.GetListOfKeys():
        obj = subkey.ReadObj()
        if obj.InheritsFrom("TH2"):
            name = obj.GetName()
            print(f"  📊 Found histogram: {name}")

            c = ROOT.TCanvas("c", "", 800, 600)
            obj.SetStats(False)
            obj.Draw("COLZ")

            outfile = f"{outdir}/{dirname}_{name}.png"
            c.SaveAs(outfile)
            print(f"  ✅ Saved: {outfile}")
