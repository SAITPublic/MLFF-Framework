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


@registry.register_task("evaluate")
class EvaluateTask(BaseTask):
    def setup(self, evaluator):
        self.evaluator = evaluator

    def run(self):
        self.evaluator.evaluate() 
        