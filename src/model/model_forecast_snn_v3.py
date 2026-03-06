import torch.nn as nn

from .model_forecast_snn_v2 import SNNModelForecastV2
from .layers.spike_decoder import SpikingDecoupledDecoder


class SNNModelForecastV3(SNNModelForecastV2):
    """V3: V2 + spiking decoupled decoder."""

    def __init__(
        self,
        embed_dim=128,
        future_steps=60,
        num_heads=8,
        decoder_mode_depth=2,
        decoder_state_depth=2,
        decoder_hybrid_depth=2,
        decoder_tau=2.0,
        decoder_v_threshold=1.0,
        decoder_detach_reset=True,
        decoder_dropout=0.1,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            future_steps=future_steps,
            num_heads=num_heads,
            use_legacy_time_decoder=False,
            **kwargs,
        )

        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
        )
        self.time_decoder = SpikingDecoupledDecoder(
            future_len=future_steps,
            dim=embed_dim,
            num_modes=6,
            num_heads=num_heads,
            mode_depth=decoder_mode_depth,
            state_depth=decoder_state_depth,
            hybrid_depth=decoder_hybrid_depth,
            tau=decoder_tau,
            v_threshold=decoder_v_threshold,
            detach_reset=decoder_detach_reset,
            dropout=decoder_dropout,
        )
