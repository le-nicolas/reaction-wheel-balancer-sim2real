import argparse
from pathlib import Path

import numpy as np


FEATURE_DIM = 15
ACTION_DIM = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train residual MLP and export checkpoint for final/residual_model.py.")
    parser.add_argument("--dataset", type=str, required=True, help="Input dataset .npz path from build_residual_dataset.py.")
    parser.add_argument("--out", type=str, required=True, help="Output checkpoint path (.pt).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def _build_mlp(nn, input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int):
    layers = []
    in_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception as exc:
        raise RuntimeError("PyTorch is required. Install with: pip install torch") from exc

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=True)
    x = np.asarray(data["features"], dtype=np.float32)
    y = np.asarray(data["targets"], dtype=np.float32)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Dataset arrays must be 2D.")
    if x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {x.shape[1]}.")
    if y.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected target dim {ACTION_DIM}, got {y.shape[1]}.")
    if x.shape[0] < 64:
        raise ValueError("Dataset is too small for training (<64 samples).")

    n = x.shape[0]
    order = np.random.permutation(n)
    x = x[order]
    y = y[order]

    n_val = int(np.clip(round(args.val_frac * n), 1, max(1, n - 1)))
    n_train = n - n_val
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    input_mean = np.mean(x_train, axis=0, dtype=np.float64)
    input_std = np.std(x_train, axis=0, dtype=np.float64)
    input_std = np.where(np.abs(input_std) < 1e-6, 1.0, input_std)

    output_scale = np.std(y_train, axis=0, dtype=np.float64)
    output_scale = np.where(np.abs(output_scale) < 1e-6, 1.0, output_scale)

    x_train_n = ((x_train - input_mean) / input_std).astype(np.float32)
    x_val_n = ((x_val - input_mean) / input_std).astype(np.float32)
    y_train_n = (y_train / output_scale).astype(np.float32)
    y_val_n = (y_val / output_scale).astype(np.float32)

    device = torch.device("cpu")
    model = _build_mlp(nn, FEATURE_DIM, int(args.hidden_dim), int(args.hidden_layers), ACTION_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.MSELoss()

    x_train_t = torch.from_numpy(x_train_n).to(device)
    y_train_t = torch.from_numpy(y_train_n).to(device)
    x_val_t = torch.from_numpy(x_val_n).to(device)
    y_val_t = torch.from_numpy(y_val_n).to(device)

    batch_size = max(1, int(args.batch_size))
    best_val = float("inf")
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        perm = torch.randperm(x_train_t.shape[0], device=device)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, x_train_t.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            bsz = int(xb.shape[0])
            epoch_loss += float(loss.item()) * bsz
            seen += bsz

        train_mse = epoch_loss / max(seen, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_mse = float(loss_fn(val_pred, y_val_t).item())

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch == int(args.epochs) or epoch % 10 == 0:
            print(f"epoch={epoch:03d} train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint.")

    checkpoint = {
        "state_dict": best_state,
        "input_dim": FEATURE_DIM,
        "output_dim": ACTION_DIM,
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "output_scale": output_scale.astype(np.float32),
        "best_val_mse": float(best_val),
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "source_dataset": str(dataset_path),
    }
    torch.save(checkpoint, out_path)

    print(f"Wrote checkpoint: {out_path}")
    print(f"train_samples={n_train} val_samples={n_val} best_val_mse={best_val:.6f}")


if __name__ == "__main__":
    main()
