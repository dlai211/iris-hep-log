import ROOT

# Open ROOT file
f = ROOT.TFile("/data/jlai/iris-hep2/OpenDataDetector/ci/reference/material_composition.root")

# Mapping of detector regions to histogram names
names = {
    "beampipe": "beampipe_x0_vs_eta_all",
    "pixel": "pixel_x0_vs_eta_all",
    "sstrips": "sstrips_x0_vs_eta_all",
    "lstrips": "lstrips_x0_vs_eta_all",
}

# Create a THStack and legend
stack = ROOT.THStack("stack", "Stacked X_{0} vs #eta;#eta;X_{0} [mm]")
legend = ROOT.TLegend(0.65, 0.7, 0.88, 0.88)

# Custom colors
colors = {
    "beampipe": ROOT.kBlue + 1,
    "pixel": ROOT.kOrange - 3,
    "sstrips": ROOT.kGreen + 1,
    "lstrips": ROOT.kRed + 1,
}

# Process and add histograms
for label, key in names.items():
    h = f.Get(key)
    h_clone = h.Clone(f"{label}_clone")
    h_clone.SetFillColor(colors[label])
    h_clone.SetLineColor(ROOT.kBlack)
    h_clone.SetLineWidth(1)
    h_clone.SetTitle("")
    stack.Add(h_clone)
    legend.AddEntry(h_clone, label, "f")

# Create canvas and draw
c = ROOT.TCanvas("c", "c", 800, 600)
stack.Draw("HIST")
legend.Draw()
c.SetGrid()
c.SetTicks()
c.SetLeftMargin(0.12)
c.SetRightMargin(0.05)

# Save to file
c.SaveAs("x0_vs_eta_stacked_pyroot.png")
