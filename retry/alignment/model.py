from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None


TORCH_AVAILABLE = torch is not None


@dataclass(frozen=True)
class ModelArgs:
    embedding_dim: int = 768
    dropout: float = 0.3
    gat_num: int = 1
    center_norm: bool = False
    neighbor_norm: bool = True
    emb_norm: bool = True
    combine: bool = True
    multi_head_dim: int = 1


if TORCH_AVAILABLE:
    class BatchMultiHeadGraphAttention(nn.Module):
        def __init__(self, device: "torch.device", args: ModelArgs, n_head: int, f_in: int, f_out: int, bias: bool = True) -> None:
            super().__init__()
            self.device = device
            self.w = nn.Parameter(torch.empty(n_head, f_in, f_out))
            self.a_src = nn.Parameter(torch.empty(n_head, f_out, 1))
            self.a_dst = nn.Parameter(torch.empty(n_head, f_out, 1))

            self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(args.dropout)
            if bias:
                self.bias = nn.Parameter(torch.empty(f_out))
                nn.init.constant_(self.bias, 0)
            else:
                self.register_parameter("bias", None)

            nn.init.xavier_uniform_(self.w)
            nn.init.xavier_uniform_(self.a_src)
            nn.init.xavier_uniform_(self.a_dst)

        def forward(self, h: "torch.Tensor", adj: "torch.Tensor") -> "torch.Tensor":
            _, n = h.size()[:2]
            h_prime = torch.matmul(h.unsqueeze(1), self.w)
            attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)
            attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)
            attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)

            attn = self.leaky_relu(attn)
            mask = ~(adj.unsqueeze(1) | torch.eye(adj.shape[-1], dtype=torch.bool, device=self.device))
            attn = attn.masked_fill(mask, float("-inf"))
            attn = self.softmax(attn)
            attn = self.dropout(attn)
            output = torch.matmul(attn, h_prime)
            if self.bias is None:
                return output
            return output + self.bias


    class GraphAlignmentModel(nn.Module):
        def __init__(self, args: ModelArgs, device: "torch.device") -> None:
            super().__init__()
            self.args = args
            self.device_obj = device
            self.embedding_dim = args.embedding_dim
            self.attn = BatchMultiHeadGraphAttention(
                device=self.device_obj,
                args=self.args,
                n_head=self.args.multi_head_dim,
                f_in=self.embedding_dim,
                f_out=self.embedding_dim,
            )
            self.attn_mlp = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            )

        def forward(self, batch: "torch.Tensor") -> "torch.Tensor":
            batch = batch.to(self.device_obj)
            batch_in = batch[:, :, : self.embedding_dim]
            adj = batch[:, :, self.embedding_dim :]

            center = batch_in[:, 0].to(self.device_obj)
            center_neigh = batch_in.to(self.device_obj)

            for _ in range(self.args.gat_num):
                center_neigh = self.attn(center_neigh, adj.bool()).squeeze(1)

            center_neigh = center_neigh[:, 0]

            if self.args.center_norm:
                center = F.normalize(center, p=2, dim=1)
            if self.args.neighbor_norm:
                center_neigh = F.normalize(center_neigh, p=2, dim=1)
            if self.args.combine:
                out_hat = torch.cat((center, center_neigh), dim=1)
                out_hat = self.attn_mlp(out_hat)
                if self.args.emb_norm:
                    out_hat = F.normalize(out_hat, p=2, dim=1)
            else:
                out_hat = center_neigh

            return out_hat


def create_alignment_model(device: str = "cpu", args: ModelArgs | None = None):
    if not TORCH_AVAILABLE:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for final_model evaluation")
    runtime_args = args or ModelArgs()
    device_obj = torch.device(device)
    model = GraphAlignmentModel(runtime_args, device=device_obj).to(device_obj)
    return model


def load_checkpoint(model, model_path: str | Path, device: str = "cpu") -> None:
    if not TORCH_AVAILABLE:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for final_model evaluation")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch versions
        checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif hasattr(checkpoint, "state_dict"):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(checkpoint).__name__}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()
