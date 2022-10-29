from vilt.datasets import SNLIVEDataset
from .datamodule_base import BaseDataModule


class SNLIVEDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SNLIVEDataset

    @property
    def dataset_name(self):
        return "snli_ve"
