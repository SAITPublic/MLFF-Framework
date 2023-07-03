"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask

from src.common.utils import bm_logging


@registry.register_task("train")
class TrainTask(BaseTask):
    def run(self):
        try:
            self.trainer.train()

        except RuntimeError as e:
            e_str = str(e)
            if (
                "find_unused_parameters" in e_str
                and "torch.nn.parallel.DistributedDataParallel" in e_str
            ):
                for name, parameter in self.trainer.model.named_parameters():
                    if parameter.requires_grad and parameter.grad is None:
                        bm_logging.warning(
                            f"Parameter {name} has no gradient. Consider removing it from the model."
                        )
            raise e