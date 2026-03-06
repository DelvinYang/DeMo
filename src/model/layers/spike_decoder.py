import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.activation_based import functional, neuron, surrogate
except ImportError as exc:
    raise ImportError(
        "spikingjelly is required for SNN V3 decoder. "
        "Please install it in your environment."
    ) from exc


class MultiStepSpikeFFN(nn.Module):
    def __init__(
        self,
        dim=128,
        hidden_ratio=2.0,
        tau=2.0,
        v_threshold=1.0,
        detach_reset=True,
        dropout=0.1,
    ):
        super().__init__()
        hidden_dim = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.lif1 = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=detach_reset,
            step_mode="m",
        )
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.lif2 = neuron.LIFNode(
            tau=tau,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=detach_reset,
            step_mode="m",
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        h = x.transpose(0, 1).contiguous()  # [T, B, C]
        functional.reset_net(self.lif1)
        h = self.fc1(h)
        h = self.lif1(h)
        h = self.dropout(h)

        functional.reset_net(self.lif2)
        h = self.fc2(h)
        h = self.lif2(h)
        h = self.dropout(h)
        h = h.transpose(0, 1).contiguous()
        return self.norm(x + h)


class SpikingModeInteraction(nn.Module):
    def __init__(
        self,
        dim=128,
        num_heads=8,
        hidden_ratio=2.0,
        attn_drop=0.1,
        dropout=0.1,
        tau=2.0,
        v_threshold=1.0,
        detach_reset=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=attn_drop, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = MultiStepSpikeFFN(
            dim=dim,
            hidden_ratio=hidden_ratio,
            tau=tau,
            v_threshold=v_threshold,
            detach_reset=detach_reset,
            dropout=dropout,
        )

    def forward(self, mode, encoding, key_padding_mask=None):
        q = self.norm1(mode)
        cross = self.attn(
            query=q,
            key=encoding,
            value=encoding,
            key_padding_mask=key_padding_mask,
        )[0]
        mode = mode + self.dropout(cross)
        mode = self.ffn(mode)
        return mode


class SpikingGMMPredictor(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super().__init__()
        self.future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, future_len * 2),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, future_len * 2),
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, mode):
        # mode: [B, M, C]
        b, m, _ = mode.shape
        y_hat = self.gaussian(mode).view(b, m, self.future_len, 2)
        scal = F.elu_(self.scale(mode), alpha=1.0) + 1.0 + 1e-4
        scal = scal.view(b, m, self.future_len, 2)
        pi = self.score(mode).squeeze(-1)
        return y_hat, pi, scal


class SpikingDensePredictor(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        # x: [B, T, C]
        return self.head(x)


class SpikingDecoupledDecoder(nn.Module):
    def __init__(
        self,
        future_len=60,
        dim=128,
        num_modes=6,
        num_heads=8,
        mode_depth=2,
        state_depth=2,
        hybrid_depth=2,
        tau=2.0,
        v_threshold=1.0,
        detach_reset=True,
        dropout=0.1,
    ):
        super().__init__()
        self.future_len = future_len
        self.num_modes = num_modes

        self.mode_query_embedding = nn.Embedding(num_modes, dim)
        self.register_buffer("modal", torch.arange(num_modes).long())

        self.mode_blocks = nn.ModuleList(
            [
                SpikingModeInteraction(
                    dim=dim,
                    num_heads=num_heads,
                    hidden_ratio=2.0,
                    tau=tau,
                    v_threshold=v_threshold,
                    detach_reset=detach_reset,
                    dropout=dropout,
                )
                for _ in range(mode_depth)
            ]
        )
        self.state_blocks = nn.ModuleList(
            [
                MultiStepSpikeFFN(
                    dim=dim,
                    hidden_ratio=2.0,
                    tau=tau,
                    v_threshold=v_threshold,
                    detach_reset=detach_reset,
                    dropout=dropout,
                )
                for _ in range(state_depth)
            ]
        )
        self.hybrid_blocks = nn.ModuleList(
            [
                MultiStepSpikeFFN(
                    dim=dim,
                    hidden_ratio=2.0,
                    tau=tau,
                    v_threshold=v_threshold,
                    detach_reset=detach_reset,
                    dropout=dropout,
                )
                for _ in range(hybrid_depth)
            ]
        )

        self.mode_predictor = SpikingGMMPredictor(future_len=future_len, dim=dim)
        self.refine_predictor = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )
        self.refine_scale = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )
        self.refine_score = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.dense_predictor = SpikingDensePredictor(dim=dim)

    @staticmethod
    def _masked_mean(x, key_padding_mask=None):
        if key_padding_mask is None:
            return x.mean(dim=1)
        valid = (~key_padding_mask).to(dtype=x.dtype).unsqueeze(-1)
        summed = (x * valid).sum(dim=1)
        count = valid.sum(dim=1).clamp(min=1.0)
        return summed / count

    def forward(self, mode, encoding, mask=None):
        # mode: [B, T, C], encoding: [B, N, C], mask: [B, N] with True as invalid
        scene_global = self._masked_mean(encoding, key_padding_mask=mask)

        # state branch
        state = mode + scene_global.unsqueeze(1)
        for blk in self.state_blocks:
            state = blk(state)
        dense_predict = self.dense_predictor(state)

        # intention branch
        mode_query = encoding[:, 0]
        multi_modal_query = self.mode_query_embedding(self.modal)
        mode_feat = mode_query[:, None] + multi_modal_query
        for blk in self.mode_blocks:
            mode_feat = blk(mode_feat, encoding, key_padding_mask=mask)
        y_hat, pi, scal = self.mode_predictor(mode_feat)

        # hybrid refine branch
        mode_dense = mode_feat[:, :, None, :] + state[:, None, :, :]
        b, m, t, c = mode_dense.shape
        mode_dense = mode_dense.view(b * m, t, c)
        for blk in self.hybrid_blocks:
            mode_dense = blk(mode_dense)
        mode_dense = mode_dense.view(b, m, t, c)

        delta = self.refine_predictor(mode_dense)
        new_y_hat = y_hat + delta
        scal_new = F.elu_(self.refine_scale(mode_dense), alpha=1.0) + 1.0 + 1e-4
        new_pi = self.refine_score(mode_dense.mean(dim=2)).squeeze(-1)

        return dense_predict, y_hat, pi, mode_feat, new_y_hat, new_pi, mode_dense, scal, scal_new
