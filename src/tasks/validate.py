"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

from pathlib import Path

from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask

from src.common.utils import bm_logging


@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self):
        # checkpoint is loaded at BaseTask.setup()
        bm_logging.info(f"Loading dataset path to be validated: {self.config['validate_data']}")

        path = Path(self.config['validate_data'])
        if not path.is_file():
            self.trainer.config['data_config_style'] = "SAIT"

        # prepare dataloaders
        loaders = {}
        if (self.trainer.config['data_config_style'] == "SAIT"
            and self.config["separate_evaluation"]
            and not path.is_file()
        ):
            paths = sorted(path.glob("*.lmdb"))
            for file_path in paths:
                dataset = self.get_dataset(file_path)
                loaders[f"{file_path.name}"] = self.get_dataloader(dataset)
        else:
            dataset = self.get_dataset(path)
            loaders[f"{path.name}"] = self.get_dataloader(dataset)

        # valiate the dataset(s)
        table = self.trainer.create_metric_table(dataloaders=loaders)
        bm_logging.info(f"\n{table}")

    def get_dataset(self, datapath):
        # for now, we assume the dataset is .lmdb.
        if self.trainer.config['data_config_style'] == "OCP":
            dataset_class = registry.get_dataset_class("lmdb")
            db_config = {"src": datapath}
        elif self.trainer.config['data_config_style'] == "SAIT":
            dataset_class = registry.get_dataset_class("lmdb_sait")
            db_config = [{"src": datapath}]
       
        # connect lmdb
        dataset = dataset_class(db_config)
        return dataset

    def get_dataloader(self, dataset):
        # sampler
        if self.config["validate_batch_size"] is None:
            bs = self.config["optim"].get("eval_batch_size", self.config["optim"].get("batch_size"))
        else:
            bs = self.config["validate_batch_size"]
        sampler = self.trainer.get_sampler(
            dataset=dataset,
            batch_size=bs,
            shuffle=self.config.get("shuffle", False),
        )

        # collater
        collater = self.trainer.parallel_collater

        # loader
        loader = self.trainer.get_dataloader(
            dataset=dataset,
            sampler=sampler,
            collater=collater,
        )
        return loader