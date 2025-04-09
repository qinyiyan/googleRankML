# Copyright 2024 RecML authors <recommendations-ml@google.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for metrics."""

# pylint: disable=g-importing-member

from recml.core.metrics.base_metrics import Metric
from recml.core.metrics.base_metrics import scalar
from recml.core.metrics.confusion_metrics import aucpr
from recml.core.metrics.confusion_metrics import aucroc
from recml.core.metrics.confusion_metrics import estimate_confusion_matrix
from recml.core.metrics.confusion_metrics import f1_score
from recml.core.metrics.confusion_metrics import fbeta_score
from recml.core.metrics.confusion_metrics import precision
from recml.core.metrics.confusion_metrics import precision_at_recall
from recml.core.metrics.confusion_metrics import recall
from recml.core.metrics.mean_metrics import accuracy
from recml.core.metrics.mean_metrics import binary_accuracy
from recml.core.metrics.mean_metrics import mean_squared_error
from recml.core.metrics.mean_metrics import top_k_accuracy
from recml.core.metrics.reduction_metrics import mean
from recml.core.metrics.reduction_metrics import sum  # pylint: disable=redefined-builtin
from recml.core.metrics.tools import MetricAccumulator
