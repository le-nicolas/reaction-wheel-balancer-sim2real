from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runtime_config import RuntimeConfig


FEATURE_DIM = 15
ACTION_DIM = 3


@dataclass(frozen=True)
class ResidualStep:
    delta_u: np.ndarray
    applied: bool
    clipped: bool
    gate_blocked: bool


class ResidualPolicy:
    """Optional offline residual policy: u_total = u_nominal + delta_u."""

    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.enabled = False
        self.model_path = cfg.residual_model_path
        self.status = "disabled (set --residual-model and --residual-scale > 0 to enable)."
        self.input_dim = FEATURE_DIM
        self.output_dim = ACTION_DIM
        self.hidden_dim = 0
        self.hidden_layers = 0

        self._torch = None
        self._model = None
        self._input_mean = np.zeros(FEATURE_DIM, dtype=float)
        self._input_std = np.ones(FEATURE_DIM, dtype=float)
        self._output_scale = np.ones(ACTION_DIM, dtype=float)

        if not cfg.residual_model_path or cfg.residual_scale <= 0.0:
            return
        self._load_model(cfg.residual_model_path)
        self.enabled = True

    @staticmethod
    def _as_vector(value, dim: int, default: float) -> np.ndarray:
        if value is None:
            return np.full(dim, default, dtype=float)
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1:
            arr = np.full(dim, float(arr[0]), dtype=float)
        if arr.size != dim:
            raise ValueError(f"Expected vector of size {dim}, got {arr.size}.")
        return arr.astype(float, copy=False)

    @staticmethod
    def _build_mlp(nn, input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int):
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _strip_prefix(state_dict: dict, prefix: str) -> dict:
        if not state_dict:
            return state_dict
        keys = list(state_dict.keys())
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix) :]: v for k, v in state_dict.items()}
        return state_dict

    def _load_model(self, model_path: str):
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError(
                "Residual model requested but PyTorch could not be imported. Install with: pip install torch"
            ) from exc

        ckpt_path = Path(model_path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Residual model checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        meta = {}
        state_dict = None
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                meta = checkpoint
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                meta = checkpoint
            else:
                state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format; expected a dict-like torch checkpoint.")

        input_dim = int(meta.get("input_dim", FEATURE_DIM))
        output_dim = int(meta.get("output_dim", ACTION_DIM))
        hidden_dim = int(meta.get("hidden_dim", 32))
        hidden_layers = int(meta.get("hidden_layers", 2))
        if input_dim != FEATURE_DIM:
            raise ValueError(
                f"Unsupported residual input_dim={input_dim}; expected {FEATURE_DIM} features "
                "(x_est[9], u_eff_applied[3], u_nominal[3])."
            )
        if output_dim != ACTION_DIM:
            raise ValueError(f"Unsupported residual output_dim={output_dim}; expected {ACTION_DIM}.")

        model = self._build_mlp(nn, input_dim, hidden_dim, hidden_layers, output_dim)
        state_dict = self._strip_prefix(state_dict, "model.")
        state_dict = self._strip_prefix(state_dict, "net.")
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self._input_mean = self._as_vector(meta.get("input_mean"), FEATURE_DIM, 0.0)
        self._input_std = self._as_vector(meta.get("input_std"), FEATURE_DIM, 1.0)
        self._input_std = np.where(np.abs(self._input_std) < 1e-6, 1.0, self._input_std)
        self._output_scale = self._as_vector(meta.get("output_scale"), ACTION_DIM, 1.0)

        self._torch = torch
        self._model = model
        self.model_path = str(ckpt_path)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.status = f"enabled ({ckpt_path.name})"

    @staticmethod
    def _disabled_step(gate_blocked: bool = False) -> ResidualStep:
        return ResidualStep(
            delta_u=np.zeros(ACTION_DIM, dtype=float),
            applied=False,
            clipped=False,
            gate_blocked=gate_blocked,
        )

    def step(
        self,
        x_est: np.ndarray,
        x_true: np.ndarray,
        u_nominal: np.ndarray,
        u_eff_applied: np.ndarray,
    ) -> ResidualStep:
        if not self.enabled or self._model is None or self._torch is None:
            return self._disabled_step()

        tilt_mag = max(abs(float(x_true[0])), abs(float(x_true[1])))
        rate_mag = max(abs(float(x_true[2])), abs(float(x_true[3])))
        if tilt_mag > self.cfg.residual_gate_tilt_rad or rate_mag > self.cfg.residual_gate_rate_rad_s:
            return self._disabled_step(gate_blocked=True)

        features = np.concatenate([x_est, u_eff_applied, u_nominal]).astype(float, copy=False)
        if features.size != FEATURE_DIM or not np.all(np.isfinite(features)):
            return self._disabled_step()

        features = (features - self._input_mean) / self._input_std
        with self._torch.no_grad():
            x = self._torch.as_tensor(features, dtype=self._torch.float32).unsqueeze(0)
            raw = self._model(x).cpu().numpy().reshape(-1)

        delta_u = raw * self._output_scale * float(self.cfg.residual_scale)
        pre_clip = delta_u.copy()
        delta_u = np.clip(delta_u, -self.cfg.residual_max_abs_u, self.cfg.residual_max_abs_u)
        clipped = bool(np.any(np.abs(delta_u - pre_clip) > 1e-12))
        applied = bool(np.any(np.abs(delta_u) > 1e-9))
        return ResidualStep(delta_u=delta_u.astype(float, copy=False), applied=applied, clipped=clipped, gate_blocked=False)
