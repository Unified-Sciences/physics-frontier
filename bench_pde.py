#!/usr/bin/env python3
"""bench_pde.py — many-source PDE solves via a dense solve operator on the full-rate
fp16-accumulate tensor path.

Operator: the screened-Poisson / modified-Helmholtz problem  (I - alpha Laplacian) u = f
on a 2D grid -- the implicit-diffusion step operator, the Yukawa/screened potential, and
the Helmholtz smoothing filter all share it. It is SPD and well-conditioned, so the
inverse-apply is numerically clean (unlike plain Poisson, whose inverse is ill-conditioned
and amplifies fp16 error -- we use the well-posed operator deliberately).

When the SAME operator is solved for many right-hand sides -- many source configurations,
parameter sweeps, ensemble/Monte-Carlo UQ -- you form the dense solve operator M = (I -
alpha L)^{-1} (the discrete Green's function) once and apply it as a GEMM X = M F. The
apply dominates and is exactly where the consumer-Ampere fp16-accumulate lever pays off.

FAIR comparison: fp16-accumulate vs fp32-accumulate, BOTH the fp16 tensor-core path
(same kernel, only the accumulator differs). True-fp32 (CUDA-core) is reported only for
context -- comparing against it would overstate the lever. Each solve is independent, so
there is no error accumulation; we report relative error vs an fp64 gold solve.
"""
import os, sys, json, time, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--grid", type=int, default=96, help="g; N = g*g unknowns")
ap.add_argument("--rhs", type=int, default=4096, help="number of source fields solved together")
ap.add_argument("--alpha", type=float, default=2e-3, help="screening (sets conditioning of I - alpha L)")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out", default=os.path.join(HERE, "results_pde.json"))
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
g = a.grid; N = g * g; R = a.rhs
h = 1.0 / (g + 1)


def laplacian(g):
    idx = torch.arange(g * g, device=dev)
    L = torch.zeros(g * g, g * g, device=dev, dtype=torch.float64)
    L[idx, idx] = 4.0
    r, c = idx // g, idx % g
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        m = (rr >= 0) & (rr < g) & (cc >= 0) & (cc < g)
        L[idx[m], (rr[m] * g + cc[m])] = -1.0
    return L / (h * h)


L = laplacian(g)
Bop = torch.eye(N, device=dev, dtype=torch.float64) + a.alpha * L      # (I - alpha*Lap), SPD
ev = torch.linalg.eigvalsh(Bop.float()); kappa = (ev[-1] / ev[0]).item()
M64 = torch.linalg.inv(Bop)                                            # dense Green's operator
M32, M16 = M64.float(), M64.half()

F64 = torch.randn(N, R, device=dev, dtype=torch.float64)
F64 /= F64.norm(dim=0, keepdim=True)
F32, F16 = F64.float(), F64.half()
Xgold = M64 @ F64


def timed(fn):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(a.reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def relerr(X):
    return ((X.double() - Xgold).norm() / Xgold.norm()).item()


# fp32-accumulate tensor path (fp16 inputs, fp32 accumulator) -- the FAIR baseline
torch.backends.cuda.matmul.allow_fp16_accumulation = False
X_f32acc = M16 @ F16
t_f32acc = timed(lambda: M16 @ F16)
# fp16-accumulate tensor path (full-rate)
torch.backends.cuda.matmul.allow_fp16_accumulation = True
X_f16acc = M16 @ F16
t_f16acc = timed(lambda: M16 @ F16)
torch.backends.cuda.matmul.allow_fp16_accumulation = False
# true fp32 (CUDA core) -- context only
torch.backends.cuda.matmul.allow_tf32 = False
X_true32 = M32 @ F32
t_true32 = timed(lambda: M32 @ F32)

gflop = 2.0 * N * N * R / 1e9
print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}")
print(f"screened Poisson (I - {a.alpha}*Lap): g={g} N={N} R={R} cond={kappa:.1f}")
print(f"apply M @ F  ({R} sources, {gflop:.1f} GFLOP):")
print(f"  fp32-acc tensor (fair baseline): {t_f32acc:7.3f} ms  {gflop/t_f32acc:6.1f} TFLOP/s  rel-err {relerr(X_f32acc):.2e}")
print(f"  fp16-acc tensor (full-rate)    : {t_f16acc:7.3f} ms  {gflop/t_f16acc:6.1f} TFLOP/s  rel-err {relerr(X_f16acc):.2e}")
print(f"  LEVER speedup (fp32acc/fp16acc): {t_f32acc/t_f16acc:.2f}x   |   vs naive true-fp32: {t_true32/t_f16acc:.2f}x")
print(f"  true fp32 (CUDA core, context) : {t_true32:7.3f} ms  {gflop/t_true32:6.1f} TFLOP/s  rel-err {relerr(X_true32):.2e}")

json.dump(dict(g=g, N=N, R=R, alpha=a.alpha, cond=kappa, gflop=gflop,
               t_fp32acc=t_f32acc, t_fp16acc=t_f16acc, t_true32=t_true32,
               speedup=t_f32acc / t_f16acc, speedup_vs_true32=t_true32 / t_f16acc,
               tflops_fp16acc=gflop / t_f16acc, tflops_fp32acc=gflop / t_f32acc,
               relerr_fp32acc=relerr(X_f32acc), relerr_fp16acc=relerr(X_f16acc),
               relerr_true32=relerr(X_true32)),
          open(a.out, "w"), indent=2)
print(f"wrote {a.out}")
