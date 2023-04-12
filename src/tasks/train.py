from ocpmodels.common.registry import registry
from ocpmodels.tasks.task import BaseTask

from src.common.utils import bm_logging


@registry.register_task("train")
class TrainTask(BaseTask):
    def run(self):
        try:
            self.trainer.train()

            # save the model as a class with trained parameters
            # self.trainer.save_model_as_class()

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