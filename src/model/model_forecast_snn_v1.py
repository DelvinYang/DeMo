import torch
import torch.nn as nn

from .model_forecast import ModelForecast
from .layers.spike_encoder import SpikingTemporalEncoder


class SNNModelForecastV1(ModelForecast):
    """V1: only replace agent temporal encoder with a spiking encoder."""

    def __init__(
        self,
        embed_dim=128,
        spike_depth=4,
        spike_mlp_ratio=2.0,
        spike_dropout=0.1,
        spike_tau=2.0,
        spike_v_threshold=1.0,
        spike_detach_reset=True,
        spike_backend="torch",
        hist_downsample=1,
        spike_steps=None,
        spike_pooling="last",
        **kwargs,
    ):
        super().__init__(embed_dim=embed_dim, use_temporal_mamba=False, **kwargs)
        # V1 fully replaces the original temporal encoder path.
        # Remove unused base projector to avoid DDP unused-parameter errors.
        del self.hist_embed_mlp

        self.spike_temporal_encoder = SpikingTemporalEncoder(
            in_dim=4,
            embed_dim=embed_dim,
            depth=spike_depth,
            mlp_ratio=spike_mlp_ratio,
            dropout=spike_dropout,
            tau=spike_tau,
            v_threshold=spike_v_threshold,
            detach_reset=spike_detach_reset,
            backend=spike_backend,
            hist_downsample=hist_downsample,
            spike_steps=spike_steps,
            pooling=spike_pooling,
        )
        self.actor_out_norm = nn.LayerNorm(embed_dim)
        self._latest_snn_aux_losses = {
            "spike_sparsity_loss": None,
            "membrane_stability_loss": None,
        }

    def _encode_actor_history(self, hist_feat, hist_feat_key_valid, B, N):
        valid_hist_feat = hist_feat[hist_feat_key_valid].contiguous()
        valid_hist_mask = valid_hist_feat[..., -1] > 0.5
        actor_feat = self.spike_temporal_encoder(valid_hist_feat, valid_hist_mask)
        self._latest_snn_aux_losses = self.spike_temporal_encoder.latest_aux_losses
        actor_feat = self.actor_out_norm(actor_feat)

        actor_feat_tmp = torch.zeros(
            B * N,
            actor_feat.shape[-1],
            device=actor_feat.device,
            dtype=actor_feat.dtype,
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        return actor_feat_tmp.view(B, N, actor_feat.shape[-1])

    def forward(self, data):
        out = super().forward(data)
        if self._latest_snn_aux_losses is not None:
            out.update(self._latest_snn_aux_losses)
        return out
