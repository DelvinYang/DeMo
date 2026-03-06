import torch
import torch.nn as nn

from .model_forecast import ModelForecast
from .layers.temporal_conv_snn import TemporalConvSNNEncoder
from .layers.event_scene_graph import EventSceneGraph
from .layers.light_global_context import LightGlobalContext
from .layers.two_stage_fast_decoder import TwoStageFastDecoder
from .layers.transformer_blocks import Block


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
        hybrid_scene_depth=1,
        hybrid_num_heads=4,
        hybrid_mlp_ratio=2.0,
        compressed_steps=8,
        spike_depth=2,
        spike_tau=2.0,
        spike_v_threshold=1.0,
        spike_detach_reset=True,
        spike_backend="torch",
        recent_frames=4,
        recent_residual_weight=0.2,
        max_lane_tokens=24,
        lane_near_ego=12,
        lane_near_goal=8,
        lane_diverse=4,
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
        self.global_context = LightGlobalContext(dim=embed_dim, hidden=embed_dim)
        self.hybrid_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=hybrid_num_heads,
                    mlp_ratio=hybrid_mlp_ratio,
                    qkv_bias=False,
                    drop_path=0.1,
                )
                for _ in range(hybrid_scene_depth)
            ]
        )
        self.hybrid_norm = nn.LayerNorm(embed_dim)
        self.fast_decoder = TwoStageFastDecoder(
            dim=embed_dim,
            future_steps=future_steps,
            num_modes=6,
        )
        self.lane_near_ego = int(lane_near_ego)
        self.lane_near_goal = int(lane_near_goal)
        self.lane_diverse = int(lane_diverse)
        self.max_lane_tokens = int(max_lane_tokens)

        # update temporal encoder settings after creation for clarity in checkpoints
        self.temporal_encoder.recent_frames = max(1, int(recent_frames))
        self.temporal_encoder.recent_residual_weight = float(recent_residual_weight)

    @staticmethod
    def _gather_tokens(x, idx):
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))

    @staticmethod
    def _gather_mask(x, idx):
        return torch.gather(x, 1, idx)

    def _select_structured_lanes(self, lane_feat, data):
        bsz, total_lanes, _ = lane_feat.shape
        k_total = min(self.max_lane_tokens, total_lanes)
        if k_total >= total_lanes:
            return (
                lane_feat,
                data["lane_centers"],
                data["lane_angles"],
                data["lane_key_valid_mask"],
                data["lane_attr"],
            )

        lane_valid = data["lane_key_valid_mask"]
        lane_centers = data["lane_centers"]
        ego_center = data["x_centers"][:, 0:1]  # [B,1,2]
        ego_heading = data["x_angles"][:, 0, -1]  # [B]
        heading_vec = torch.stack([ego_heading.cos(), ego_heading.sin()], dim=-1).unsqueeze(1)
        forward_goal = ego_center + 30.0 * heading_vec

        dist_ego = torch.norm(lane_centers - ego_center, dim=-1).masked_fill(~lane_valid, float("inf"))
        dist_goal = torch.norm(lane_centers - forward_goal, dim=-1).masked_fill(
            ~lane_valid, float("inf")
        )

        k_ego = min(self.lane_near_ego, k_total)
        ego_idx = torch.topk(dist_ego, k=k_ego, dim=1, largest=False).indices

        selected_mask = torch.zeros_like(lane_valid)
        selected_mask.scatter_(1, ego_idx, True)

        remain = k_total - k_ego
        goal_masked = dist_goal.masked_fill(selected_mask, float("inf"))
        k_goal = min(self.lane_near_goal, max(remain, 0))
        goal_idx = (
            torch.topk(goal_masked, k=k_goal, dim=1, largest=False).indices
            if k_goal > 0
            else goal_masked.new_zeros((bsz, 0), dtype=torch.long)
        )
        if k_goal > 0:
            selected_mask.scatter_(1, goal_idx, True)
        remain = k_total - k_ego - k_goal

        # diverse lanes by heading difference (avoid only nearest lanes)
        lane_angles = data["lane_angles"]
        angle_diff = torch.abs(
            (lane_angles - ego_heading[:, None] + torch.pi) % (2 * torch.pi) - torch.pi
        )
        diverse_score = angle_diff.masked_fill(~lane_valid, -1.0).masked_fill(selected_mask, -1.0)
        k_div = min(self.lane_diverse, max(remain, 0))
        div_idx = (
            torch.topk(diverse_score, k=k_div, dim=1, largest=True).indices
            if k_div > 0
            else diverse_score.new_zeros((bsz, 0), dtype=torch.long)
        )
        if k_div > 0:
            selected_mask.scatter_(1, div_idx, True)
        remain = k_total - k_ego - k_goal - k_div

        fill_idx = None
        if remain > 0:
            fill_score = dist_ego.masked_fill(selected_mask, float("inf"))
            fill_idx = torch.topk(fill_score, k=remain, dim=1, largest=False).indices

        parts = [ego_idx]
        if k_goal > 0:
            parts.append(goal_idx)
        if k_div > 0:
            parts.append(div_idx)
        if fill_idx is not None:
            parts.append(fill_idx)
        lane_idx = torch.cat(parts, dim=1)

        lane_feat = self._gather_tokens(lane_feat, lane_idx)
        lane_centers = self._gather_tokens(data["lane_centers"], lane_idx)
        lane_angles = self._gather_mask(data["lane_angles"], lane_idx)
        lane_key_valid_mask = self._gather_mask(data["lane_key_valid_mask"], lane_idx)
        lane_attr = self._gather_tokens(data["lane_attr"], lane_idx)
        return lane_feat, lane_centers, lane_angles, lane_key_valid_mask, lane_attr

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
        spike_rate[hist_feat_key_valid] = spike_rate_valid.to(dtype=spike_rate.dtype)
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
            self._select_structured_lanes(lane_feat, data)

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
        actor_feat, lane_feat, global_ctx = self.global_context(
            actor_feat, lane_feat, data["x_key_valid_mask"], lane_key_valid_mask
        )

        x_centers = torch.cat([data["x_centers"], lane_centers], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], lane_angles], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_valid_mask = torch.cat([data["x_key_valid_mask"], lane_key_valid_mask], dim=1)
        x_encoder = x_encoder + pos_embed
        for blk in self.hybrid_blocks:
            x_encoder = blk(x_encoder, key_padding_mask=~key_valid_mask)
        x_encoder = self.hybrid_norm(x_encoder)

        dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, _, scal, scal_new = \
            self.fast_decoder(x_encoder, key_valid_mask, global_ctx=global_ctx)

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
