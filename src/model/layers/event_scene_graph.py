import torch
import torch.nn as nn


class EventSceneGraph(nn.Module):
    def __init__(
        self,
        dim=128,
        active_agents=16,
        lane_tokens=16,
        depth=2,
    ):
        super().__init__()
        self.active_agents = int(active_agents)
        self.lane_tokens = int(lane_tokens)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, dim),
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _topk_valid(scores, valid_mask, k):
        # scores: [B, N], valid_mask: [B, N]
        masked = scores.masked_fill(~valid_mask, float("-inf"))
        k = min(k, scores.size(1))
        idx = torch.topk(masked, k=k, dim=1, largest=True).indices
        return idx

    def _select_lanes(self, data, active_agent_idx):
        lane_centers = data["lane_centers"]   # [B, M, 2]
        lane_valid = data["lane_key_valid_mask"]  # [B, M]
        x_centers = data["x_centers"]  # [B, N, 2]
        active_centers = torch.gather(
            x_centers,
            1,
            active_agent_idx.unsqueeze(-1).expand(-1, -1, x_centers.size(-1)),
        )  # [B, K, 2]
        pair_dist = torch.cdist(active_centers, lane_centers)  # [B, K, M]
        lane_dist = pair_dist.min(dim=1).values  # [B, M]
        lane_dist = lane_dist.masked_fill(~lane_valid, float("inf"))
        k_lane = min(self.lane_tokens, lane_centers.size(1))
        lane_idx = torch.topk(lane_dist, k=k_lane, dim=1, largest=False).indices
        return lane_idx

    def forward(self, actor_feat, lane_feat, data, spike_rate):
        # actor_feat: [B, N, C], lane_feat: [B, M, C], spike_rate: [B, N]
        actor_valid = data["x_key_valid_mask"]
        active_k = min(self.active_agents, actor_feat.size(1))
        active_agent_idx = self._topk_valid(spike_rate, actor_valid, active_k)
        lane_idx = self._select_lanes(data, active_agent_idx)

        actor_nodes = torch.gather(
            actor_feat, 1, active_agent_idx.unsqueeze(-1).expand(-1, -1, actor_feat.size(-1))
        )
        lane_nodes = torch.gather(
            lane_feat, 1, lane_idx.unsqueeze(-1).expand(-1, -1, lane_feat.size(-1))
        )
        nodes = torch.cat([actor_nodes, lane_nodes], dim=1)

        for layer in self.layers:
            nodes = self.norm(nodes + layer(nodes))

        updated_actor_nodes = nodes[:, : active_agent_idx.size(1)]
        updated_lane_nodes = nodes[:, active_agent_idx.size(1):]

        actor_out = actor_feat.clone()
        lane_out = lane_feat.clone()
        actor_out.scatter_(
            1, active_agent_idx.unsqueeze(-1).expand_as(updated_actor_nodes), updated_actor_nodes
        )
        lane_out.scatter_(
            1, lane_idx.unsqueeze(-1).expand_as(updated_lane_nodes), updated_lane_nodes
        )

        return actor_out, lane_out
