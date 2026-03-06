import os
import sys
import warnings
import time
import numpy as np

# Compatibility shim for old packages (e.g. older wandb) under NumPy>=2.0.
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.custom_(fwd|bwd)\(args\.\.\.\)` is deprecated\..*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module=r"torchmetrics\.utilities\.imports",
)

import hydra
import torch
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Performance hint for Tensor Core GPUs (e.g., A100): use TF32 matmul kernels.
torch.set_float32_matmul_precision("high")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir
    strategy = conf.strategy
    precision = conf.precision
    model_type = conf.model.target.model.type
    log_every_n_steps = conf.log_every_n_steps
    if model_type == "SNNModelForecastV3":
        strategy = "ddp_find_unused_parameters_true"
    if model_type in {
        "SNNModelForecastFast",
        "SNNModelForecastV1",
        "SNNModelForecastV2",
        "SNNModelForecastV3",
    } and precision == "32-true":
        precision = "bf16-mixed"
    if model_type == "SNNModelForecastFast":
        log_every_n_steps = max(int(log_every_n_steps), 200)

    logger = TensorBoardLogger(save_dir=output_dir, name="logs")

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        precision=precision,
        accelerator="gpu",
        devices=conf.gpus,
        strategy=strategy,
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
        log_every_n_steps=log_every_n_steps,
        enable_model_summary=False,
    )

    model = instantiate(conf.model.target)
    os.system('cp -a %s %s' % ('conf', output_dir))
    os.system('cp -a %s %s' % ('src', output_dir))
    with open(f'{output_dir}/model.txt', 'w') as f:
        original_stdout = sys.stdout  
        sys.stdout = f  
        print(model)  
        sys.stdout = original_stdout  
    datamodule = instantiate(conf.datamodule.target)
    train_start = time.perf_counter()
    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)
    train_seconds = time.perf_counter() - train_start
    if trainer.is_global_zero:
        elapsed = int(train_seconds)
        hh = elapsed // 3600
        mm = (elapsed % 3600) // 60
        ss = elapsed % 60
        print(f"Total training time: {hh:02d}:{mm:02d}:{ss:02d} ({train_seconds:.2f}s)")
        if trainer.global_step > 0 and train_seconds > 0:
            steps_per_sec = trainer.global_step / train_seconds
            samples_per_sec = steps_per_sec * conf.batch_size * conf.gpus
            print(f"Training throughput: {steps_per_sec:.2f} steps/s, {samples_per_sec:.2f} samples/s")
    trainer.validate(model, datamodule.val_dataloader())


if __name__ == "__main__":
    main()
