import torch.nn as nn

from .model_forecast_snn_v1 import SNNModelForecastV1
from .layers.spike_interaction import SpikingInteractionBlock
from .layers.transformer_blocks import Block


class SNNModelForecastV2(SNNModelForecastV1):
    """V2: V1 + replace scene context blocks with spiking interaction blocks."""

    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        interaction_type="spiking",
        interaction_depth=5,
        interaction_tau=2.0,
        interaction_v_threshold=1.0,
        interaction_detach_reset=True,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            **kwargs,
        )

        valid_types = {"spiking", "hybrid", "transformer"}
        if interaction_type not in valid_types:
            raise ValueError(f"interaction_type must be one of {valid_types}, got {interaction_type}")

        if interaction_type == "transformer":
            self.blocks = nn.ModuleList(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=0.2,
                )
                for _ in range(interaction_depth)
            )
        elif interaction_type == "hybrid":
            blocks = []
            for i in range(interaction_depth):
                if i % 2 == 0:
                    blocks.append(
                        SpikingInteractionBlock(
                            dim=embed_dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            tau=interaction_tau,
                            v_threshold=interaction_v_threshold,
                            detach_reset=interaction_detach_reset,
                        )
                    )
                else:
                    blocks.append(
                        Block(
                            dim=embed_dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            drop_path=0.2,
                        )
                    )
            self.blocks = nn.ModuleList(blocks)
        else:
            self.blocks = nn.ModuleList(
                SpikingInteractionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    tau=interaction_tau,
                    v_threshold=interaction_v_threshold,
                    detach_reset=interaction_detach_reset,
                )
                for _ in range(interaction_depth)
            )
