import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
except ImportError as exc:
    raise ImportError(
        "spikingjelly is required for TemporalConvSNNEncoder."
    ) from exc


class TemporalConvSNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim=4,
        embed_dim=128,
        conv_hidden=64,
        compressed_steps=8,
        spike_depth=2,
        spike_tau=2.0,
        spike_v_threshold=1.0,
        spike_detach_reset=True,
        spike_backend="torch",
        collect_aux_losses=True,
    ):
        super().__init__()
        self.compressed_steps = int(compressed_steps)
        self.collect_aux_losses = bool(collect_aux_losses)
        self.spike_backend = self._resolve_backend(spike_backend)
        self.latest_aux_losses = {
            "spike_sparsity_loss": None,
            "membrane_stability_loss": None,
        }

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_dim, conv_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_hidden, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)

        self.spike_layers = nn.ModuleList(
            [
                neuron.LIFNode(
                    tau=spike_tau,
                    v_threshold=spike_v_threshold,
                    surrogate_function=surrogate.ATan(),
                    detach_reset=spike_detach_reset,
                    step_mode="m",
                    backend=self.spike_backend,
                    store_v_seq=self.collect_aux_losses,
                )
                for _ in range(spike_depth)
            ]
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def _test_backend(backend):
        test_node = neuron.LIFNode(step_mode="m", backend=backend)
        device = "cuda"
        test_x = torch.randn(4, 2, 8, device=device)
        test_node = test_node.to(device)
        functional.reset_net(test_node)
        _ = test_node(test_x)
        torch.cuda.synchronize()

    @classmethod
    def _resolve_backend(cls, backend):
        backend = str(backend).lower()
        if backend == "auto":
            if not torch.cuda.is_available():
                return "torch"
            for candidate in ("triton", "cupy", "torch"):
                try:
                    cls._test_backend(candidate)
                    return candidate
                except Exception:
                    continue
            return "torch"
        if backend == "torch":
            return backend
        if not torch.cuda.is_available():
            warnings.warn(
                f"Requested spike backend '{backend}' but CUDA is unavailable. Falling back to 'torch'."
            )
            return "torch"
        try:
            cls._test_backend(backend)
            return backend
        except Exception as exc:
            warnings.warn(
                f"Requested spike backend '{backend}' is unavailable ({exc}). Falling back to 'torch'."
            )
            return "torch"

    @staticmethod
    def _membrane_stability(v_seq):
        if v_seq is None:
            return None
        if v_seq.dim() < 2 or v_seq.size(0) < 2:
            return v_seq.new_zeros(())
        return (v_seq[1:] - v_seq[:-1]).abs().mean()

    def forward(self, hist_feat, hist_valid_mask=None):
        # hist_feat: [B, L, C]
        x = hist_feat.transpose(1, 2).contiguous()
        x = self.temporal_conv(x)
        x = F.adaptive_avg_pool1d(x, self.compressed_steps)
        x = x.transpose(1, 2).contiguous()  # [B, T, C]
        x = self.temporal_norm(x)

        x = x.transpose(0, 1).contiguous()  # [T, B, C]
        spike_sparsity_terms = []
        membrane_stability_terms = []
        for lif in self.spike_layers:
            functional.reset_net(lif)
            x = lif(x)
            if self.collect_aux_losses:
                spike_sparsity_terms.append(x.abs().mean())
                cur_stability = self._membrane_stability(getattr(lif, "v_seq", None))
                if cur_stability is not None:
                    membrane_stability_terms.append(cur_stability)
        spike_rate = x.abs().mean(dim=0).mean(dim=-1)  # [B]
        x = x.transpose(0, 1).contiguous()  # [B, T, C]
        x = self.out_proj(x)

        zero = x.new_zeros(())
        self.latest_aux_losses = {
            "spike_sparsity_loss": (
                torch.stack(spike_sparsity_terms).mean() if spike_sparsity_terms else zero
            ),
            "membrane_stability_loss": (
                torch.stack(membrane_stability_terms).mean() if membrane_stability_terms else zero
            ),
        }

        if hist_valid_mask is None:
            actor_feat = x[:, -1]
        else:
            valid_lengths = hist_valid_mask.long().sum(dim=1).clamp(min=1)
            valid_lengths = torch.clamp(valid_lengths, max=self.compressed_steps)
            last_idx = valid_lengths - 1
            batch_idx = torch.arange(x.size(0), device=x.device)
            actor_feat = x[batch_idx, last_idx]

        return actor_feat, spike_rate
