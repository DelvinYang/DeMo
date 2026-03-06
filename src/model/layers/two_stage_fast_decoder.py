import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStageFastDecoder(nn.Module):
    """Coarse-to-fine lightweight multi-modal decoder."""

    def __init__(self, dim=128, future_steps=60, num_modes=6, coarse_steps=6):
        super().__init__()
        self.future_steps = int(future_steps)
        self.num_modes = int(num_modes)
        self.coarse_steps = int(coarse_steps)

        self.mode_embed = nn.Embedding(self.num_modes, dim)
        self.mode_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        self.coarse_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, self.coarse_steps * 2),
        )
        self.refine_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, self.future_steps * 2),
        )
        self.refine_head_stage2 = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, self.future_steps * 2),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, self.future_steps * 2),
        )
        self.refine_scale_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, self.future_steps * 2),
        )
        self.dense_head = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, self.future_steps * 2),
        )

    @staticmethod
    def _masked_mean(x, valid_mask):
        valid = valid_mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1.0)
        return (x * valid).sum(dim=1) / denom

    def _coarse_to_full(self, coarse):
        # coarse: [B, K, coarse_steps, 2] -> [B, K, future_steps, 2]
        bsz, kmode, _, coord = coarse.shape
        x = coarse.view(bsz * kmode, self.coarse_steps, coord).transpose(1, 2).contiguous()
        x = F.interpolate(
            x,
            size=self.future_steps,
            mode="linear",
            align_corners=False,
        )
        return x.transpose(1, 2).contiguous().view(bsz, kmode, self.future_steps, coord)

    def forward(self, x_encoder, valid_mask, global_ctx=None):
        bsz = x_encoder.size(0)
        scene_feat = self._masked_mean(x_encoder, valid_mask)
        if global_ctx is not None:
            scene_feat = scene_feat + global_ctx

        mode_feat = scene_feat[:, None, :] + self.mode_embed.weight[None, :, :]
        pi = self.mode_logits(mode_feat).squeeze(-1)

        coarse = self.coarse_head(mode_feat).view(bsz, self.num_modes, self.coarse_steps, 2)
        coarse_full = self._coarse_to_full(coarse)
        refine = self.refine_head(mode_feat).view(bsz, self.num_modes, self.future_steps, 2)
        y_hat = coarse_full + refine

        scal = F.elu_(self.scale_head(mode_feat), alpha=1.0) + 1.0 + 1e-4
        scal = scal.view(bsz, self.num_modes, self.future_steps, 2)

        refine_delta = self.refine_head_stage2(mode_feat).view(
            bsz, self.num_modes, self.future_steps, 2
        )
        new_y_hat = y_hat + 0.5 * refine_delta

        scal_new = F.elu_(self.refine_scale_head(mode_feat), alpha=1.0) + 1.0 + 1e-4
        scal_new = scal_new.view(bsz, self.num_modes, self.future_steps, 2)

        dense_predict = self.dense_head(scene_feat).view(bsz, self.future_steps, 2)
        new_pi = pi
        return dense_predict, y_hat, pi, mode_feat, new_y_hat, new_pi, mode_feat, scal, scal_new
