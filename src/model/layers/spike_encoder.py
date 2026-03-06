import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
except ImportError as exc:
    raise ImportError(
        "spikingjelly is required for SNNModelForecastV1. "
        "Please install it in your environment."
    ) from exc


class SpikeInputProjector(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class SpikingResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        mlp_ratio=2.0,
        dropout=0.1,
        tau=2.0,
        v_threshold=1.0,
        detach_reset=True,
        backend="torch",
        collect_aux_losses=False,
    ):
        super().__init__()
        self.collect_aux_losses = collect_aux_losses
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.lif1 = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
            store_v_seq=collect_aux_losses,
        )
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.lif2 = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=detach_reset,
            step_mode="m",
            backend=backend,
            store_v_seq=collect_aux_losses,
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _membrane_stability(v_seq):
        if v_seq is None:
            return None
        if v_seq.dim() < 2 or v_seq.size(0) < 2:
            return v_seq.new_zeros(())
        return (v_seq[1:] - v_seq[:-1]).abs().mean()

    def forward(self, x, return_aux=False):
        residual = x
        aux = {}
        x = self.fc1(x)
        x = self.norm1(x)
        x = x.transpose(0, 1).contiguous()  # [T, B, C]
        functional.reset_net(self.lif1)
        x = self.lif1(x)
        if return_aux and self.collect_aux_losses:
            aux["spike_sparsity_loss"] = x.abs().mean()
            aux["membrane_stability_loss"] = self._membrane_stability(
                getattr(self.lif1, "v_seq", None)
            )
        x = x.transpose(0, 1).contiguous()
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = x.transpose(0, 1).contiguous()  # [T, B, C]
        functional.reset_net(self.lif2)
        x = self.lif2(x)
        if return_aux and self.collect_aux_losses:
            cur_sparsity = x.abs().mean()
            cur_stability = self._membrane_stability(getattr(self.lif2, "v_seq", None))
            aux["spike_sparsity_loss"] = aux["spike_sparsity_loss"] + cur_sparsity
            if aux["membrane_stability_loss"] is None:
                aux["membrane_stability_loss"] = cur_stability
            elif cur_stability is not None:
                aux["membrane_stability_loss"] = aux["membrane_stability_loss"] + cur_stability
        x = x.transpose(0, 1).contiguous()
        x = self.dropout(x)
        out = residual + x
        if return_aux:
            return out, aux
        return out


class SpikingTemporalEncoder(nn.Module):
    def __init__(
        self,
        in_dim=4,
        embed_dim=128,
        depth=4,
        mlp_ratio=2.0,
        dropout=0.1,
        tau=2.0,
        v_threshold=1.0,
        detach_reset=True,
        hist_downsample=1,
        spike_steps=None,
        pooling="last",
        backend="torch",
        collect_aux_losses=False,
    ):
        super().__init__()
        self.hist_downsample = max(1, int(hist_downsample))
        self.spike_steps = spike_steps
        self.pooling = pooling
        self.backend = self._resolve_backend(backend)
        self.collect_aux_losses = collect_aux_losses
        self.latest_aux_losses = None
        self.projector = SpikeInputProjector(
            in_dim=in_dim,
            hidden_dim=64,
            embed_dim=embed_dim,
        )
        self.blocks = nn.ModuleList(
            [
                SpikingResidualTemporalBlock(
                    embed_dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    tau=tau,
                    v_threshold=v_threshold,
                    detach_reset=detach_reset,
                    backend=self.backend,
                    collect_aux_losses=collect_aux_losses,
                )
                for _ in range(depth)
            ]
        )

    @staticmethod
    def _resolve_backend(backend):
        backend = str(backend).lower()
        if backend == "torch":
            return backend
        if not torch.cuda.is_available():
            warnings.warn(
                f"Requested spike backend '{backend}' but CUDA is unavailable. Falling back to 'torch'."
            )
            return "torch"
        try:
            test_node = neuron.LIFNode(step_mode="m", backend=backend)
            test_x = torch.randn(4, 2, 8, device="cuda")
            functional.reset_net(test_node)
            _ = test_node(test_x)
            torch.cuda.synchronize()
            return backend
        except Exception as exc:
            warnings.warn(
                f"Requested spike backend '{backend}' is not available ({exc}). "
                "Falling back to 'torch'."
            )
            return "torch"

    @staticmethod
    def _masked_mean(x, mask):
        if mask is None:
            return x.mean(dim=1)
        valid = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / denom

    def _compress_time(self, x, mask=None):
        # 1) stride-based downsample on raw history steps
        if self.hist_downsample > 1:
            x = x[:, :: self.hist_downsample]
            if mask is not None:
                mask = mask[:, :: self.hist_downsample]

        # 2) optional fixed spike steps via adaptive average pooling
        if self.spike_steps is not None and x.size(1) > self.spike_steps:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.spike_steps).transpose(1, 2)
            if mask is not None:
                pooled_mask = F.adaptive_avg_pool1d(
                    mask.float().unsqueeze(1), self.spike_steps
                ).squeeze(1)
                mask = pooled_mask > 0.5

        return x, mask

    def forward(self, hist_feat, hist_valid_mask=None):
        x, hist_valid_mask = self._compress_time(hist_feat, hist_valid_mask)
        x = self.projector(x)
        spike_sparsity_terms = []
        membrane_stability_terms = []
        for blk in self.blocks:
            if self.collect_aux_losses:
                x, aux = blk(x, return_aux=True)
                if aux.get("spike_sparsity_loss") is not None:
                    spike_sparsity_terms.append(aux["spike_sparsity_loss"])
                if aux.get("membrane_stability_loss") is not None:
                    membrane_stability_terms.append(aux["membrane_stability_loss"])
            else:
                x = blk(x, return_aux=False)

        # Keep losses as tensors on the right device for trainer aggregation.
        zero = x.new_zeros(())
        self.latest_aux_losses = {
            "spike_sparsity_loss": (
                torch.stack(spike_sparsity_terms).mean() if spike_sparsity_terms else zero
            ),
            "membrane_stability_loss": (
                torch.stack(membrane_stability_terms).mean()
                if membrane_stability_terms
                else zero
            ),
        }

        if hist_valid_mask is None:
            return x[:, -1]

        if self.pooling == "mean":
            return self._masked_mean(x, hist_valid_mask)

        valid_lengths = hist_valid_mask.long().sum(dim=1).clamp(min=1)
        last_valid_idx = valid_lengths - 1
        batch_idx = torch.arange(x.size(0), device=x.device)
        return x[batch_idx, last_valid_idx]
