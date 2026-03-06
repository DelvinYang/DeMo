import torch
import torch.nn as nn
import torch.nn.functional as F


class FastDecoder(nn.Module):
    def __init__(self, dim=128, future_steps=60, num_modes=6):
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes

        self.mode_embed = nn.Embedding(num_modes, dim)
        self.mode_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.traj_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, future_steps * 2),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, future_steps * 2),
        )
        self.refine_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, future_steps * 2),
        )
        self.refine_scale_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, future_steps * 2),
        )
        self.dense_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, future_steps * 2),
        )

    @staticmethod
    def _masked_mean(x, valid_mask):
        valid = valid_mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / denom

    def forward(self, x_encoder, valid_mask):
        # x_encoder: [B, N, C], valid_mask: [B, N] with True as valid
        bsz = x_encoder.size(0)
        scene_feat = self._masked_mean(x_encoder, valid_mask)

        mode_feat = scene_feat[:, None, :] + self.mode_embed.weight[None, :, :]
        pi = self.mode_logits(mode_feat).squeeze(-1)
        y_hat = self.traj_head(mode_feat).view(bsz, self.num_modes, self.future_steps, 2)
        scal = F.elu_(self.scale_head(mode_feat), alpha=1.0) + 1.0 + 1e-4
        scal = scal.view(bsz, self.num_modes, self.future_steps, 2)

        refine = self.refine_head(mode_feat).view(bsz, self.num_modes, self.future_steps, 2)
        new_y_hat = y_hat + refine
        scal_new = F.elu_(self.refine_scale_head(mode_feat), alpha=1.0) + 1.0 + 1e-4
        scal_new = scal_new.view(bsz, self.num_modes, self.future_steps, 2)
        new_pi = pi

        dense_predict = self.dense_head(scene_feat).view(bsz, self.future_steps, 2)
        return dense_predict, y_hat, pi, mode_feat, new_y_hat, new_pi, mode_feat, scal, scal_new
