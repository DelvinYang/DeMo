import torch
import torch.nn as nn


class LightGlobalContext(nn.Module):
    """Cheap global context correction branch for hybrid scene encoding."""

    def __init__(self, dim=128, hidden=128):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.actor_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.lane_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    @staticmethod
    def _masked_mean(x, mask):
        valid = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / denom

    def forward(self, actor_feat, lane_feat, actor_valid, lane_valid):
        actor_ctx = self._masked_mean(actor_feat, actor_valid)
        lane_ctx = self._masked_mean(lane_feat, lane_valid)
        global_ctx = self.global_mlp(torch.cat([actor_ctx, lane_ctx], dim=-1))

        actor_delta = self.actor_gate(actor_feat) * global_ctx.unsqueeze(1)
        lane_delta = self.lane_gate(lane_feat) * global_ctx.unsqueeze(1)
        return actor_feat + actor_delta, lane_feat + lane_delta, global_ctx

