"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import numpy as np

def calc_error_metric(f_predict, f_target, metric_name):
    metric_name_lower = metric_name.lower()
    if metric_name_lower == "mae":
        return np.mean(np.absolute(f_predict - f_target))
    elif metric_name_lower == "rmse":
        return np.sqrt(np.mean((f_predict - f_target)**2))
    else:
        raise Exception(f"Provided metric name '{metric_name}' is not supported!")