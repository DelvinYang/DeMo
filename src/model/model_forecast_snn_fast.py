from .model_forecast_snn_v1 import SNNModelForecastV1


class SNNModelForecastFast(SNNModelForecastV1):
    """
    Hybrid-SNN-Fast:
    - SNN only for agent temporal encoder
    - ANN scene interaction + ANN decoder kept from original DeMo
    """

    def __init__(self, **kwargs):
        # Keep the original continuous decoder path for accuracy stability.
        kwargs["use_legacy_time_decoder"] = True
        super().__init__(**kwargs)
