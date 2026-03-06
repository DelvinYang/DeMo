from pathlib import Path
from typing import Optional
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader
from .av2_dataset import Av2Dataset, collate_fn


def _worker_init_fn(_worker_id: int) -> None:
    # Avoid CPU thread oversubscription inside each dataloader worker.
    torch.set_num_threads(1)


class Av2DataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        dataset: dict = {},
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        multiprocessing_context: str = "spawn",
        prefetch_factor: int = 2,
        test: bool = False,
    ):
        super(Av2DataModule, self).__init__()
        self.data_root = Path(data_root)
        self.dataset_cfg = dataset
        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context
        self.prefetch_factor = prefetch_factor
        self.test = test

    def _loader_kwargs(self) -> dict:
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "worker_init_fn": _worker_init_fn,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["multiprocessing_context"] = self.multiprocessing_context
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = Av2Dataset(
                data_root=self.data_root, split="train", **self.dataset_cfg
            )
            self.val_dataset = Av2Dataset(
                data_root=self.data_root, split="val", **self.dataset_cfg
            )
        else:
            self.test_dataset = Av2Dataset(
                data_root=self.data_root, split="test", **self.dataset_cfg
            )

    def train_dataloader(self):
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_fn,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **self._loader_kwargs(),
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **self._loader_kwargs(),
        )
    
