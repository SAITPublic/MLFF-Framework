import os

from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask

from src.common.utils import bm_logging


@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self):
        # checkpoint is loaded at BaseTask.setup()
        bm_logging.info(f"Loading validation dataset : {self.config['validate_data']}")
        loader = self.get_dataloader(self.config['validate_data'])

        # start to valiate the data
        table = self.trainer.create_metric_table(
            dataloaders={"validate-data": loader},
        )
        bm_logging.info(f"\n{table}")

    def get_dataloader(self, datapath):
        # for now, we assume the dataset is .lmdb.
        dataset_class = registry.get_dataset_class("lmdb")

        # connect lmdb
        dataset = dataset_class({"src": datapath})

        # sampler
        sampler = self.trainer.get_sampler(
            dataset=dataset,
            batch_size=self.config.get(
                "validate_batch_size", 
                self.config["optim"].get(
                    "eval_batch_size", 
                    self.config["optim"].get("batch_size")
                )
            ),
            shuffle=False,
        )

        # loader
        loader = self.trainer.get_dataloader(
            dataset=dataset,
            sampler=sampler,
            collater=self.trainer.parallel_collater,
        )
        return loader