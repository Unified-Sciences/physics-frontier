#!/usr/bin/env python3
"""bench_solve.py — matrix inversion in practice: invert once, apply to many RHS.

A single matrix inverse is almost never the goal; you invert to solve A X = B for many
right-hand sides (many sources, many targets, many parameter samples). The inverse /
factorization is computed once (fp32, cuSOLVER); the recurring cost is the apply
X = A^{-1} B -- a dense GEMM, and exactly where the consumer-Ampere fp16-accumulate
lever (full-rate, ~1.8x the fp32-accumulate tensor path on sm_86) pays off.

Test system: a dense Gaussian-process / RBF kernel matrix A = K + lambda I (K_ij =
exp(-||x_i-x_j||^2 / 2l^2)), the workhorse of kernel ridge regression, RBF field
interpolation, and data assimilation -- genuinely dense and inversion-bound, made
SPD/well-posed by the regularizer.

Reports, for the apply: fp32 vs fp16-accumulate wall-clock + the relative error vs an
fp64 gold solve, so the accuracy traded for ~1.8x is explicit. Also quantifies
iterative refinement honestly: for a *dense* A each refinement step is itself a GEMM,
so it buys accuracy at a measured cost -- the win is the raw apply when ~1e-3 is
acceptable (surrogates, UQ ensembles, ML-for-physics), which is very often.
"""
import os, sys, json, time, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8192, help="matrix dimension (number of kernel centers)")
ap.add_argument("--rhs", type=int, default=2048, help="number of right-hand sides")
ap.add_argument("--lengthscale", type=float, default=0.6)
ap.add_argument("--reg", type=float, default=1e-2, help="ridge lambda relative to max kernel eigenvalue")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--out", default=os.path.join(HERE, "results_solve.json"))
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(0)
n, R = a.n, a.rhs


def rbf_system(n, d=3, ell=0.6, reg=1e-2):
    x = torch.rand(n, d, device=dev, dtype=torch.float64)
    d2 = torch.cdist(x, x) ** 2
    K = torch.exp(-d2 / (2 * ell * ell))
    evK = torch.linalg.eigvalsh(K.float())               # SPD; fp32 is plenty for sizing
    lam = reg * evK[-1].item()                           # ridge relative to spectral radius
    A = K + lam * torch.eye(n, device=dev, dtype=torch.float64)
    kap = ((evK[-1] + lam) / (evK[evK > 0][0] + lam)).item()
    return A, kap


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}")
A64, kappa = rbf_system(n, ell=a.lengthscale, reg=a.reg)
B64 = torch.randn(n, R, device=dev, dtype=torch.float64)
B64 /= B64.norm(dim=0, keepdim=True)

# ---- invert once (fp32, the one-time cost) ----
torch.backends.cuda.matmul.allow_tf32 = False
t0 = time.time(); Ainv32 = torch.linalg.inv(A64.float()); torch.cuda.synchronize()
t_inv = (time.time() - t0) * 1e3
Ainv16 = Ainv32.half()
A32 = A64.float()
B32 = B64.float(); B16 = B64.half()

# ---- gold solve (fp64) ----
Xgold = torch.linalg.solve(A64, B64)


def relerr(X):
    return ((X.double() - Xgold).norm() / Xgold.norm()).item()


def resid(X):
    return ((A64 @ X.double() - B64).norm() / B64.norm()).item()


# ---- apply: fp32-accumulate (fair baseline) vs fp16-accumulate (full-rate) ----
# both are the fp16 tensor-core path; only the accumulator differs. true-fp32 is
# CUDA-core, reported for context only (comparing against it overstates the lever).
torch.backends.cuda.matmul.allow_fp16_accumulation = False
X_f32acc = Ainv16 @ B16
t_f32acc = timed(lambda: Ainv16 @ B16, a.reps)

torch.backends.cuda.matmul.allow_fp16_accumulation = True
X_fp16 = Ainv16 @ B16
t_fp16 = timed(lambda: Ainv16 @ B16, a.reps)
torch.backends.cuda.matmul.allow_fp16_accumulation = False

torch.backends.cuda.matmul.allow_tf32 = False
X_fp32 = Ainv32 @ B32
t_fp32 = timed(lambda: Ainv32 @ B32, a.reps)

gflop_apply = 2.0 * n * n * R / 1e9
print(f"system: dense RBF A = K + lambda I,  n={n}  R={R}  cond={kappa:.1f}")
print(f"one-time inverse (fp32): {t_inv:.1f} ms")
print(f"apply A^-1 @ B  ({R} RHS, {gflop_apply:.1f} GFLOP):")
print(f"  fp32-acc tensor (fair baseline): {t_f32acc:7.3f} ms  {gflop_apply/t_f32acc:6.1f} TFLOP/s  rel-err {relerr(X_f32acc):.2e}  resid {resid(X_f32acc):.2e}")
print(f"  fp16-acc tensor (full-rate)    : {t_fp16:7.3f} ms  {gflop_apply/t_fp16:6.1f} TFLOP/s  rel-err {relerr(X_fp16):.2e}  resid {resid(X_fp16):.2e}")
print(f"  true fp32 (CUDA core, context) : {t_fp32:7.3f} ms  {gflop_apply/t_fp32:6.1f} TFLOP/s  rel-err {relerr(X_fp32):.2e}  resid {resid(X_fp32):.2e}")
print(f"  LEVER (fp32acc/fp16acc): {t_f32acc/t_fp16:.2f}x   |   vs naive true-fp32: {t_fp32/t_fp16:.2f}x  at rel-err {relerr(X_fp16):.1e}")
print(f"note: fp16 *storage* of the inverse floors accuracy at ~1e-3, so iterative")
print(f"      refinement cannot pass it for a dense operator; ~1e-3 is ample for GP")
print(f"      surrogates, UQ ensembles, and ML-for-physics where inputs are noisier.")

json.dump(dict(n=n, R=R, cond=kappa, t_inv=t_inv,
               t_fp32acc=t_f32acc, t_fp16acc=t_fp16, t_true32=t_fp32,
               speedup=t_f32acc / t_fp16, speedup_vs_true32=t_fp32 / t_fp16,
               tflops_fp32acc=gflop_apply / t_f32acc, tflops_fp16acc=gflop_apply / t_fp16,
               relerr_fp32acc=relerr(X_f32acc), relerr_fp16acc=relerr(X_fp16),
               relerr_true32=relerr(X_fp32),
               resid_fp16acc=resid(X_fp16), resid_true32=resid(X_fp32)),
          open(a.out, "w"), indent=2)
print(f"wrote {a.out}")
