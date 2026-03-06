import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
except ImportError as exc:
    raise ImportError(
        "spikingjelly is required for SNN scene interaction. "
        "Please install it in your environment."
    ) from exc


class SpikingInteractionBlock(nn.Module):
    def __init__(
        self,
        dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        attn_drop=0.1,
        drop=0.1,
        tau=2.0,
        v_threshold=1.0,
        detach_reset=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.lif1 = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=detach_reset,
            step_mode="s",
        )
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.lif2 = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=detach_reset,
            step_mode="s",
        )
        self.drop = nn.Dropout(drop)

    @staticmethod
    def _to_valid_mask(key_padding_mask):
        if key_padding_mask is None:
            return None
        return ~key_padding_mask

    def _lif_multi_step(self, lif_node, x, valid_mask=None):
        functional.reset_net(lif_node)
        outs = []
        for t in range(x.size(1)):
            xt = x[:, t]
            if valid_mask is not None:
                xt = xt * valid_mask[:, t:t + 1].to(dtype=xt.dtype)
            outs.append(lif_node(xt))
        return torch.stack(outs, dim=1)

    def forward(self, src, mask=None, key_padding_mask=None):
        x = self.norm1(src)
        x = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop(x)

        valid_mask = self._to_valid_mask(key_padding_mask)
        x = self.norm2(src)
        x = self.fc1(x)
        x = self._lif_multi_step(self.lif1, x, valid_mask=valid_mask)
        x = self.drop(x)
        x = self.fc2(x)
        x = self._lif_multi_step(self.lif2, x, valid_mask=valid_mask)
        x = self.drop(x)
        return src + x
