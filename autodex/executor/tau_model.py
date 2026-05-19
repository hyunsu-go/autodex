"""
fit_tau_model.py — Train MLP to predict τ_motor(q, q̇) from dynamic_data.npz.

At runtime, τ_ext = τ_hat(q, q̇) − τ_motor replaces g_mujoco + b. This is a
pure-torch workflow: save with torch.save, load with torch.load.

Usage:
    python fit_tau_model.py --source stream --out results/tau_model_stream.pt
    python fit_tau_model.py --source rpc    --out results/tau_model_rpc.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn


N_JOINTS = 6
# Motor torque sustained limits (Nm) from xArm6-1305 URDF effort limits.
# Values above these are current spikes / sensor artifacts, not real torque.
TAU_LIMITS = np.array([50.0, 50.0, 32.0, 32.0, 32.0, 20.0], dtype=np.float32)


def encode_q(q, use_sincos=True):
    """Encode joint angles. If use_sincos=True, returns [sin(q), cos(q)] (2N),
    otherwise raw q (N)."""
    if use_sincos:
        return np.concatenate([np.sin(q), np.cos(q)], axis=-1)
    return q


def build_input(q, qdot, use_sincos=True, use_qdot=True, use_sign_qdot=False):
    """Build model input from q (N,6) and qdot (N,6).
    use_sincos, use_qdot, use_sign_qdot combine flexibly:
      q only           = 6 dims  (quasi-static)
      sincos(q)        = 12 dims (quasi-static + wrap-safe)
      q + qdot         = 12 dims
      sincos(q) + qdot = 18 dims
      + sign(qdot)     = +6 dims (helps capture friction hysteresis at direction
                         reversal — same q,|q̇| can have different τ depending
                         on sign of q̇ due to Coulomb friction)"""
    q_enc = encode_q(q, use_sincos)
    parts = [q_enc]
    if use_qdot:
        parts.append(qdot)
        if use_sign_qdot:
            # tanh(k·q̇) smooths sign() near q̇=0 so MLP doesn't see a hard step.
            # k=50 → ramp width ~0.02 rad/s (~1 deg/s) which matches deadband
            # between stick and slip.
            parts.append(np.tanh(50.0 * qdot))
    return np.concatenate(parts, axis=-1)


class TauMLP(nn.Module):
    """MLP: encoded q (+optional q̇) → τ_hat. Stores encoding flags.

    If `output_joints` is set, the model only predicts those joints. At
    inference, results are scattered into a 6-vector with zeros elsewhere.
    """

    def __init__(self, in_dim, hidden=64, n_hidden=2, out_dim=N_JOINTS,
                 use_sincos=True, use_qdot=True, use_sign_qdot=False,
                 output_joints=None):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.use_sincos = use_sincos
        self.use_qdot = use_qdot
        self.use_sign_qdot = use_sign_qdot
        self.output_joints = output_joints    # None or list of 0-indexed joints
        self.out_dim = out_dim
        # Buffers for normalization (saved with state_dict)
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(out_dim))
        self.register_buffer("y_std", torch.ones(out_dim))

    def forward(self, x):
        x = (x - self.x_mean) / self.x_std
        y = self.net(x)
        return y * self.y_std + self.y_mean

    def predict_full(self, x):
        """Return 6-vector τ_hat. For trimmed models (output_joints set),
        fills non-predicted joints with 0."""
        y_trim = self.forward(x)   # (out_dim,) or (N, out_dim)
        if self.output_joints is None:
            return y_trim
        # Scatter into 6-D
        import torch as _t
        full = _t.zeros(*y_trim.shape[:-1], N_JOINTS,
                        dtype=y_trim.dtype, device=y_trim.device)
        idx = _t.tensor(self.output_joints, dtype=_t.long, device=y_trim.device)
        full[..., idx] = y_trim
        return full


def _load_one(path, source, use_sincos, use_qdot, smooth_qdot, max_qddot, use_sign_qdot=False):
    d = np.load(path)
    if "qdot" in d:
        # Unified schema: per-sample q, qdot, [qddot], tau_stream, tau_rpc
        q = d["q"].astype(np.float32)
        qdot = d["qdot"].astype(np.float32)
        tau = d[f"tau_{source}"].astype(np.float32)
        qddot = d["qddot"].astype(np.float32) if "qddot" in d else None
        if smooth_qdot > 0 and use_qdot:
            smoothed = np.zeros_like(qdot)
            s = qdot[0].copy()
            for i in range(len(qdot)):
                s = smooth_qdot * qdot[i] + (1 - smooth_qdot) * s
                smoothed[i] = s
            qdot = smoothed
        X = build_input(q, qdot, use_sincos=use_sincos, use_qdot=use_qdot,
                        use_sign_qdot=use_sign_qdot).astype(np.float32)
        # Outlier reject: drop high-|q̈| samples (sensor transients, gearbox lash)
        keep = np.ones(len(X), dtype=bool)
        if max_qddot is not None and qddot is not None:
            keep &= np.all(np.abs(qddot) < max_qddot, axis=1)
        return X[keep], tau[keep]
    else:
        # Legacy static schema (aggregated): q_meas + tau_{source}_mean
        q = d["q_meas"].astype(np.float32)
        tau = d[f"tau_{source}_mean"].astype(np.float32)
        X = encode_q(q, use_sincos).astype(np.float32)
        return X, tau


def load_data(paths, source="stream", use_sincos=True, use_qdot=True,
              smooth_qdot=0.1, max_qddot=None, use_sign_qdot=False):
    """Load one or more .npz files (comma-separated string or list), concat.
    max_qddot (rad/s²): drop samples where any joint |q̈| exceeds it. Removes
    sensor transients and current spikes that poison the τ target."""
    if isinstance(paths, str):
        paths = [p.strip() for p in paths.split(",") if p.strip()]
    Xs, Ys = [], []
    for p in paths:
        X, tau = _load_one(p, source, use_sincos, use_qdot, smooth_qdot, max_qddot,
                           use_sign_qdot=use_sign_qdot)
        print(f"  loaded {p}: {len(X)} samples")
        Xs.append(X); Ys.append(tau)
    X = np.concatenate(Xs, axis=0)
    tau = np.concatenate(Ys, axis=0)
    valid = ~(np.any(np.isnan(X), axis=1) | np.any(np.isnan(tau), axis=1))
    return X[valid], tau[valid]


def train(X, y, hidden=128, n_hidden=3, epochs=3000, lr=1e-3, val_frac=0.1, seed=42,
          clip_tau=True, huber_delta=5.0, use_sincos=True, use_qdot=True,
          use_sign_qdot=False, output_joints=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Clip target to physical actuator limits — removes current-spike outliers
    if clip_tau:
        n_clipped = np.sum(np.any(np.abs(y) > TAU_LIMITS, axis=1))
        y = np.clip(y, -TAU_LIMITS, TAU_LIMITS)
        print(f"  clipped {n_clipped}/{len(y)} samples to actuator limits ±{TAU_LIMITS.tolist()}")

    # Trim output to selected joints (e.g., just J2/J3 for targeted compliance)
    if output_joints is not None:
        y = y[:, output_joints]
        print(f"  output trimmed to joints {[j+1 for j in output_joints]} → y.shape={y.shape}")

    N = len(X)
    idx = np.random.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    out_dim = y.shape[1] if y.ndim == 2 else 1
    model = TauMLP(in_dim=X.shape[1], out_dim=out_dim,
                   hidden=hidden, n_hidden=n_hidden,
                   use_sincos=use_sincos, use_qdot=use_qdot,
                   use_sign_qdot=use_sign_qdot,
                   output_joints=output_joints)

    # Robust normalization using 1st/99th percentiles of training split
    def _robust_stats(a):
        lo = np.quantile(a, 0.01, axis=0)
        hi = np.quantile(a, 0.99, axis=0)
        center = ((hi + lo) / 2).astype(np.float32)
        scale = ((hi - lo) / 2).astype(np.float32).clip(min=1e-6)
        return center, scale

    x_mean, x_std = _robust_stats(X[train_idx])
    y_mean, y_std = _robust_stats(y[train_idx])
    model.x_mean.copy_(torch.from_numpy(x_mean))
    model.x_std.copy_(torch.from_numpy(x_std))
    model.y_mean.copy_(torch.from_numpy(y_mean))
    model.y_std.copy_(torch.from_numpy(y_std))

    # Move to GPU if available (full-batch GD is much faster there)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    Xt = torch.from_numpy(X[train_idx]).to(device)
    yt = torch.from_numpy(y[train_idx]).to(device)
    Xv = torch.from_numpy(X[val_idx]).to(device)
    yv = torch.from_numpy(y[val_idx]).to(device)
    print(f"  device: {device}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # Huber loss: quadratic for |err| < delta, linear beyond. Outlier-robust.
    loss_fn = nn.HuberLoss(delta=huber_delta)
    print(f"  loss=Huber(δ={huber_delta} Nm)  normalization=q1/q99 robust")

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if (epoch + 1) % 200 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_loss = nn.functional.mse_loss(model(Xv), yv).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  epoch {epoch+1:>5d}  train={loss.item():.4f} "
                  f"val={val_loss:.4f}  best={best_val:.4f} (MSE Nm²)")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Validation error report (over predicted joints only)
    with torch.no_grad():
        err = (model(Xv) - yv).cpu().numpy()
    # Move model back to CPU for saving + inference test
    model = model.cpu()
    joints = output_joints if output_joints is not None else list(range(N_JOINTS))
    print(f"\nValidation error (Nm):")
    for k, j in enumerate(joints):
        print(f"  J{j+1}: mean={err[:, k].mean():+.3f}  "
              f"std={err[:, k].std():.3f}  "
              f"max|err|={np.abs(err[:, k]).max():.3f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/dynamic_data.npz")
    parser.add_argument("--out", default="results/tau_model.pt")
    parser.add_argument("--source", choices=["stream", "rpc"], default="stream")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--sincos", action="store_true",
                        help="encode q as [sin(q), cos(q)] — only useful if joints "
                             "cross ±π (e.g. J6 sampled in ±180°)")
    parser.add_argument("--no-qdot", action="store_true",
                        help="quasi-static: drop q̇ from input, predict τ from q only")
    parser.add_argument("--sign-qdot", action="store_true",
                        help="add tanh(50·q̇) features (6 dims) — helps capture "
                             "Coulomb friction hysteresis at direction reversal")
    parser.add_argument("--smooth-qdot", type=float, default=0.1,
                        help="EMA alpha for qdot (0=no smoothing)")
    parser.add_argument("--max-qddot", type=float, default=None,
                        help="drop samples where any joint |q̈| > this (rad/s²). "
                             "Typical 500-1000 to kill transients; None = keep all.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-joints", type=str, default=None,
                        help="comma-separated 1-indexed joints to predict, e.g. '2,3'. "
                             "Other joints get 0 at inference. Shrinks output dim for "
                             "per-joint capacity.")
    args = parser.parse_args()

    output_joints = None
    if args.output_joints:
        output_joints = [int(j) - 1 for j in args.output_joints.split(",")]
        print(f"Output joints (1-indexed): {[j+1 for j in output_joints]}")

    use_sincos = args.sincos
    use_qdot = not args.no_qdot
    use_sign_qdot = args.sign_qdot and use_qdot
    print(f"Loading {args.data} (source={args.source}, sincos={use_sincos}, "
          f"qdot={use_qdot}, sign_qdot={use_sign_qdot}, smooth_qdot={args.smooth_qdot})")
    X, y = load_data(args.data, source=args.source,
                     use_sincos=use_sincos, use_qdot=use_qdot,
                     use_sign_qdot=use_sign_qdot,
                     smooth_qdot=args.smooth_qdot,
                     max_qddot=args.max_qddot)
    in_dim = X.shape[1]
    print(f"  {len(X)} samples, input={in_dim}D, output={y.shape[1]}D")
    print(f"  y range (Nm): {y.min(0).round(2)} → {y.max(0).round(2)}")

    print(f"\nTraining MLP: {in_dim} → {args.hidden} × {args.layers} → {N_JOINTS}")
    model = train(X, y,
                  hidden=args.hidden, n_hidden=args.layers,
                  epochs=args.epochs, lr=args.lr, seed=args.seed,
                  use_sincos=use_sincos, use_qdot=use_qdot,
                  use_sign_qdot=use_sign_qdot,
                  output_joints=output_joints)

    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "hidden": args.hidden,
        "n_hidden": args.layers,
        "source": args.source,
        "use_sincos": use_sincos,
        "use_qdot": use_qdot,
        "use_sign_qdot": use_sign_qdot,
        "smooth_qdot": args.smooth_qdot,
        "output_joints": output_joints,
        "out_dim": model.out_dim,
    }, args.out)
    print(f"\nSaved {args.out}")

    # Quick inference speed test
    import time
    x_test = torch.from_numpy(X[0:1])
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10000):
            _ = model(x_test)
    elapsed = (time.time() - t0) / 10000
    print(f"Inference: {elapsed*1e6:.1f} μs/call ({1/elapsed:.0f} Hz)")


def load_model(path):
    """Helper for run_mcc.py to load and return a ready-to-call model.
    Call model.predict_full(x) to always get a 6-vector τ (zeros on untrained joints)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    output_joints = ckpt.get("output_joints", None)
    out_dim = ckpt.get("out_dim", N_JOINTS) if output_joints is None else len(output_joints)
    model = TauMLP(
        in_dim=ckpt["in_dim"],
        hidden=ckpt["hidden"],
        n_hidden=ckpt["n_hidden"],
        out_dim=out_dim,
        use_sincos=ckpt.get("use_sincos", False),
        use_qdot=ckpt.get("use_qdot", True),
        use_sign_qdot=ckpt.get("use_sign_qdot", False),
        output_joints=output_joints,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    main()
