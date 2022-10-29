import os
import copy
import torch
import pytorch_lightning as pl

from pytorch_lightning.plugins import environments as pl_env
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.distributed import rank_zero_info
from pytorch_lightning.plugins.training_type import DDPPlugin

from vilt.config import ex
from vilt.modules import ViLTransformerSS, TwoTowerViLTransformerSS, DstTwoTowerViLTransformerSS
from vilt.datamodules.multitask_datamodule import MTDataModule

class OMPIClusterEnvironment(pl_env.ClusterEnvironment):
    def __init__(self):
        super().__init__()

    def creates_children(self) -> bool:
        # return True if the cluster is managed (you don't launch processes yourself)
        assert (
            "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ
        )  # this cluster is managed
        return True

    def world_size(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])

    def set_world_size(self, size: int):
        pass
        # raise RuntimeError("this cluster is managed")

    def global_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])

    def set_global_rank(self, rank: int):
        pass
        # raise RuntimeError("this cluster is managed")

    def local_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    def node_rank(self) -> int:
        # mpi doesn't set node rank and it cannot be deduced
        # global_rank = local_rank + node_rank * numprocesses
        if "NODE_RANK" in os.environ:
            return int(os.environ["NODE_RANK"])
        else:
            return 0

    def master_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    def master_port(self) -> int:
        return int(os.environ["MASTER_PORT"])

def get_cluster_plugin(num_gpus=1, num_nodes=1):
    if num_nodes > 1 or (
        num_nodes == 1 and "OMPI_COMM_WORLD_SIZE" in os.environ
    ):
        rank_zero_info("ClusterPlugin: using OMPI Cluster Environment")
        return OMPIClusterEnvironment()
    # ITP also allows MPI jobs on one node
    # either way, the azure_runner will set the environment variables accordingly
    # WORLD_SIZE, RANK, LOCAL_RANK
    if num_gpus >= 1:
        rank_zero_info("ClusterPlugin: using Lightning Cluster Environment")
        return pl_env.LightningEnvironment()
    return None

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    dm = MTDataModule(_config, dist=True)
    if _config['use_dst'] and _config['use_two_tower']: # both used
        model = DstTwoTowerViLTransformerSS(_config)
    elif _config['use_two_tower']:
        model = TwoTowerViLTransformerSS(_config)
    else:
        model = ViLTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=_config['save_top_k'],
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )

    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print("grad_steps: {}".format(grad_steps))

    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    cluster_plugin = get_cluster_plugin(
        _config["num_gpus"], _config["num_nodes"]
    )

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        check_val_every_n_epoch=int(_config["check_val_every_n_epoch"]),
        plugins=[cluster_plugin, DDPPlugin(find_unused_parameters=True)],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
