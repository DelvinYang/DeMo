import torch
import torch.nn as nn

from .model_forecast import ModelForecast
from .layers.temporal_conv_snn import TemporalConvSNNEncoder
from .layers.event_scene_graph import EventSceneGraph
from .layers.fast_decoder import FastDecoder


class SNNModelForecastFast(ModelForecast):
    """
    SNNModelForecastFastV2 (refine_snn):
    SNNTemporalEncoder -> EventSceneGraph -> Light/FastDecoder
    """

    def __init__(
        self,
        embed_dim=128,
        future_steps=60,
        active_agents=16,
        lane_tokens=16,
        graph_depth=2,
        compressed_steps=8,
        spike_depth=2,
        spike_tau=2.0,
        spike_v_threshold=1.0,
        spike_detach_reset=True,
        spike_backend="torch",
        max_lane_tokens=16,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            future_steps=future_steps,
            scene_depth=0,
            max_lane_tokens=max_lane_tokens,
            use_temporal_mamba=False,
            use_legacy_time_decoder=False,
            **kwargs,
        )
        del self.hist_embed_mlp
        del self.hist_embed_mamba
        del self.norm_f
        del self.drop_path
        del self.blocks
        del self.norm
        del self.time_embedding_mlp
        del self.time_decoder

        self.temporal_encoder = TemporalConvSNNEncoder(
            in_dim=4,
            embed_dim=embed_dim,
            compressed_steps=compressed_steps,
            spike_depth=spike_depth,
            spike_tau=spike_tau,
            spike_v_threshold=spike_v_threshold,
            spike_detach_reset=spike_detach_reset,
            spike_backend=spike_backend,
            collect_aux_losses=True,
        )
        self.event_scene_graph = EventSceneGraph(
            dim=embed_dim,
            active_agents=active_agents,
            lane_tokens=lane_tokens,
            depth=graph_depth,
        )
        self.fast_decoder = FastDecoder(
            dim=embed_dim,
            future_steps=future_steps,
            num_modes=6,
        )

    def _encode_actor_history(self, hist_feat, hist_feat_key_valid, B, N):
        valid_hist_feat = hist_feat[hist_feat_key_valid].contiguous()
        valid_hist_mask = valid_hist_feat[..., -1] > 0.5
        actor_feat_valid, spike_rate_valid = self.temporal_encoder(valid_hist_feat, valid_hist_mask)

        actor_feat = torch.zeros(
            B * N,
            actor_feat_valid.size(-1),
            device=actor_feat_valid.device,
            dtype=actor_feat_valid.dtype,
        )
        spike_rate = torch.zeros(B * N, device=actor_feat_valid.device, dtype=actor_feat_valid.dtype)
        actor_feat[hist_feat_key_valid] = actor_feat_valid
        spike_rate[hist_feat_key_valid] = spike_rate_valid
        actor_feat = actor_feat.view(B, N, -1)
        spike_rate = spike_rate.view(B, N)
        return actor_feat, spike_rate, self.temporal_encoder.latest_aux_losses

    def forward(self, data):
        hist_valid_mask = data["x_valid_mask"]
        hist_key_valid_mask = data["x_key_valid_mask"]
        hist_feat = torch.cat(
            [
                data["x_positions_diff"],
                data["x_velocity_diff"][..., None],
                hist_valid_mask[..., None],
            ],
            dim=-1,
        )
        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)
        actor_feat, spike_rate, aux_losses = self._encode_actor_history(
            hist_feat, hist_feat_key_valid, B, N
        )

        lane_valid_mask = data["lane_valid_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat([lane_normalized, lane_valid_mask[..., None]], dim=-1)
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)
        lane_feat, lane_centers, lane_angles, lane_key_valid_mask, lane_attr = \
            self._select_topk_lanes(lane_feat, data)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed[lane_attr[..., 0].long()]
        actor_feat = actor_feat + actor_type_embed
        lane_feat = lane_feat + lane_type_embed

        # Sparse event scene graph update
        graph_data = {
            "x_key_valid_mask": data["x_key_valid_mask"],
            "x_centers": data["x_centers"],
            "lane_key_valid_mask": lane_key_valid_mask,
            "lane_centers": lane_centers,
        }
        actor_feat, lane_feat = self.event_scene_graph(actor_feat, lane_feat, graph_data, spike_rate)

        x_centers = torch.cat([data["x_centers"], lane_centers], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], lane_angles], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_valid_mask = torch.cat([data["x_key_valid_mask"], lane_key_valid_mask], dim=1)
        x_encoder = x_encoder + pos_embed

        dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, _, scal, scal_new = \
            self.fast_decoder(x_encoder, key_valid_mask)

        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2)

        return {
            "y_hat": y_hat,
            "pi": pi,
            "scal": scal,
            "dense_predict": dense_predict,
            "y_hat_others": y_hat_others,
            "new_y_hat": new_y_hat,
            "new_pi": new_pi,
            "scal_new": scal_new,
            "spike_sparsity_loss": aux_losses["spike_sparsity_loss"],
            "membrane_stability_loss": aux_losses["membrane_stability_loss"],
        }
