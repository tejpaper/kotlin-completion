from tunekit.eval import TopKAccuracy

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from transformers import EvalPrediction, Trainer
from transformers.tokenization_utils_base import BatchEncoding


def metrics_fn(preds: EvalPrediction) -> dict[str, float]:
    logits, labels = map(torch.tensor, preds)
    return dict(
        top_1_accuracy=TopKAccuracy(1).update(logits, labels),
        top_5_accuracy=TopKAccuracy(5).update(logits, labels),
        top_20_accuracy=TopKAccuracy(20).update(logits, labels),
        top_100_accuracy=TopKAccuracy(100).update(logits, labels),
    )


class CodeCompletionTrainer(Trainer):
    def compute_loss(self, model: nn.Module, inputs: BatchEncoding, return_outputs: bool = False) -> torch.Tensor:
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = cross_entropy(outputs.logits, labels.flatten())
        return (loss, outputs) if return_outputs else loss
