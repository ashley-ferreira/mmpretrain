from typing import List, Optional

from mmengine.evaluator import BaseMetric

from mmpretrain.registry import METRICS

import torch

######## INCOMPLETE #########
@METRICS.register_module()
class MSELoss(BaseMetric):
    """
    """
    default_prefix = 'MSELoss'

    def __init__(self,
                 threshold: float = 0.5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.threshold = threshold

    def process(self, data_batch, data_samples) -> None:
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for sample in data_samples:
            result = {
                'prediction': None
                'truth': None
            }

            self.results.append(result)

    def compute_metrics(self, results: List) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        total_score = torch.nn.MSELoss(results['prediction'], results['truth'])
        return {'MSELoss': total_score}