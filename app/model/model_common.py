"""Inference-only architecture compatible with PV_RUL_Mamba PEFT-1403 weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 6
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
MAMBA_D_STATE = 16
MAMBA_D_CONV = 4
MAMBA_EXPAND = 2
WINDOW_SIZE = 60


class MyGLU(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, output_size)
        self.linear_2 = nn.Linear(input_size, output_size)
        self.glu = nn.GLU()

    def forward(self, x):
        a = self.linear_1(x)
        b = self.linear_2(x)
        return self.glu(torch.cat([a, b], dim=-1))


class GRN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float, external: bool = False):
        super().__init__()
        self.shortcut = nn.Identity()
        body_input_size = 2 * input_size if external else input_size
        self.body = nn.Sequential(
            nn.Linear(body_input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout),
            MyGLU(input_size, input_size),
        )
        self.norm = nn.LayerNorm(input_size)

    def forward(self, x, e=None):
        shortcut = self.shortcut(x)
        if e is not None:
            x = torch.cat([x, e], dim=-1)
        return self.norm(shortcut + self.body(x))


class MambaBlock(nn.Module):
    """Minimal pure-PyTorch S6/Mamba block used by the supplied checkpoint."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        a_hippo = torch.arange(1, d_state + 1).float().unsqueeze(0)
        self.register_buffer("A_log", -torch.exp(a_hippo).log())

    def forward(self, x):
        batch, length, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[:, :, :length]
        u = F.silu(u.transpose(1, 2))
        proj = self.x_proj(u)
        delta, b_state, c_state = proj.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta)
        a = -torch.exp(self.A_log.float()).unsqueeze(0)
        a_bar = torch.exp(delta.unsqueeze(-1) * a)
        b_bar = (a_bar - 1.0) / a * b_state.unsqueeze(2)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(length):
            h = a_bar[:, t] * h + b_bar[:, t]
            outputs.append((h * c_state[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(outputs, dim=1) * F.silu(z)
        return self.out_proj(y)


class PV_RUL_Mamba(nn.Module):
    """Three-head training architecture: RUL, internal physics trajectory, annual degradation rate.

    The trajectory head is retained for checkpoint compatibility and training-time
    physics/smoothness/alignment constraints. It is not exposed as a direct 60-day
    business forecast in the demo.
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mamba = nn.Sequential(*[
            MambaBlock(hidden_dim, MAMBA_D_STATE, MAMBA_D_CONV, MAMBA_EXPAND)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(GRN(hidden_dim, hidden_dim, dropout), nn.Linear(hidden_dim, 1))
        self.trajectory_head = nn.Sequential(
            GRN(hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, WINDOW_SIZE),
        )
        self.rate_head = nn.Sequential(GRN(hidden_dim, hidden_dim, dropout), nn.Linear(hidden_dim, 1))
        # Present in the supplied checkpoint; inference does not directly use it.
        self.log_A = nn.Parameter(torch.tensor(-12.110762))

    def forward(self, x):
        h = self.mamba(self.input_proj(x))[:, -1, :]
        return self.head(h), self.trajectory_head(h), self.rate_head(h)
