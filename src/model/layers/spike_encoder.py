import torch
import torch.nn as nn

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
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.lif1 = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=detach_reset,
            step_mode="s",
        )
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.lif2 = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=detach_reset,
            step_mode="s",
        )
        self.dropout = nn.Dropout(dropout)

    def _lif_multi_step(self, lif_node, x):
        functional.reset_net(lif_node)
        outs = []
        for t in range(x.size(1)):
            outs.append(lif_node(x[:, t]))
        return torch.stack(outs, dim=1)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.norm1(x)
        x = self._lif_multi_step(self.lif1, x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self._lif_multi_step(self.lif2, x)
        x = self.dropout(x)
        return residual + x


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
    ):
        super().__init__()
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
                )
                for _ in range(depth)
            ]
        )

    def forward(self, hist_feat, hist_valid_mask=None):
        x = self.projector(hist_feat)
        for blk in self.blocks:
            x = blk(x)

        if hist_valid_mask is None:
            return x[:, -1]

        valid_lengths = hist_valid_mask.long().sum(dim=1).clamp(min=1)
        last_valid_idx = valid_lengths - 1
        batch_idx = torch.arange(x.size(0), device=x.device)
        return x[batch_idx, last_valid_idx]
