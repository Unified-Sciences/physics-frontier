#!/usr/bin/env python3
"""make_figure.py — render the physics-frontier proof-point figure from the benchmark
JSON (no matplotlib; PIL only). Two panels (inversion-apply, PDE solve), each a bar
chart of wall-clock for true-fp32 / fp32-accumulate / fp16-accumulate, annotated with
relative error, so the speed/accuracy trade of the lever is visible at a glance.
"""
import json, argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--solve", default=str(HERE / "results_solve.json"))
ap.add_argument("--pde", default=str(HERE / "results_pde.json"))
ap.add_argument("--out", default=str(HERE / "assets" / "physics_lever.png"))
a = ap.parse_args()

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except Exception:
    font = fb = fs = ImageFont.load_default()

GRAY, BLUE, GREEN, INK = (150, 150, 150), (70, 120, 200), (40, 160, 90), (30, 30, 30)
PW, PH = 470, 430
pad = 20


def panel(title, sub, bars, speedup):
    """bars: list of (label, ms, relerr, color). Returns a PW x PH image."""
    im = Image.new("RGB", (PW, PH), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((pad, 12), title, font=fb, fill=INK)
    d.text((pad, 32), sub, font=fs, fill=(110, 110, 110))
    base_y, top_y = PH - 70, 80
    mx = max(b[1] for b in bars) * 1.18
    n = len(bars); bw = 90; gap = (PW - 2 * pad - n * bw) // (n + 1)
    for i, (lab, ms, err, col) in enumerate(bars):
        x0 = pad + gap + i * (bw + gap)
        h = (ms / mx) * (base_y - top_y)
        y0 = base_y - h
        d.rectangle([x0, y0, x0 + bw, base_y], fill=col)
        d.text((x0 + bw / 2 - d.textlength(f"{ms:.1f} ms", font=fb) / 2, y0 - 36), f"{ms:.1f} ms", font=fb, fill=INK)
        d.text((x0 + bw / 2 - d.textlength(f"err {err:.0e}", font=fs) / 2, y0 - 19), f"err {err:.0e}", font=fs, fill=(120, 120, 120))
        for j, ln in enumerate(lab.split("\n")):
            d.text((x0 + bw / 2 - d.textlength(ln, font=fs) / 2, base_y + 6 + 15 * j), ln, font=fs, fill=INK)
    d.line([pad, base_y, PW - pad, base_y], fill=(200, 200, 200), width=2)
    d.text((pad, PH - 26), speedup, font=fb, fill=GREEN)
    return im


sv = json.load(open(a.solve)); pd = json.load(open(a.pde))
p1 = panel("Matrix inversion: apply A⁻¹ to many RHS",
           f"dense GP/RBF kernel, n={sv['n']}, {sv['R']} right-hand sides, cond={sv['cond']:.0f}",
           [("true fp32\n(CUDA core)", sv["t_true32"], sv["relerr_true32"], GRAY),
            ("fp32-acc\ntensor", sv["t_fp32acc"], sv["relerr_fp32acc"], BLUE),
            ("fp16-acc\ntensor", sv["t_fp16acc"], sv["relerr_fp16acc"], GREEN)],
           f"lever {sv['speedup']:.2f}x  ({sv['speedup_vs_true32']:.1f}x vs naive fp32)")
p2 = panel("PDE: screened-Poisson, many sources",
           f"dense Green's operator, N={pd['N']}, {pd['R']} sources, cond={pd['cond']:.0f}",
           [("true fp32\n(CUDA core)", pd["t_true32"], pd["relerr_true32"], GRAY),
            ("fp32-acc\ntensor", pd["t_fp32acc"], pd["relerr_fp32acc"], BLUE),
            ("fp16-acc\ntensor", pd["t_fp16acc"], pd["relerr_fp16acc"], GREEN)],
           f"lever {pd['speedup']:.2f}x  ({pd['speedup_vs_true32']:.1f}x vs naive fp32)")

W = PW * 2 + pad
H = PH + 44
canvas = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(canvas)
dr.text((pad, 10), "physics-frontier — full-rate fp16-accumulate (RTX 3080, sm_86)",
        font=font, fill=INK)
canvas.paste(p1, (0, 40))
canvas.paste(p2, (PW + pad, 40))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
canvas.save(a.out)
print(f"saved {a.out} ({W}x{H})")
