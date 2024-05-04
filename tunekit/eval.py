import abc
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from datasets.arrow_dataset import Dataset
from tqdm.autonotebook import tqdm


class Metric(abc.ABC):
    @property
    @abc.abstractmethod
    def value(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.value:.04f})'

    __str__ = __repr__


class TopKAccuracy(Metric):
    def __init__(self, k: int) -> None:
        self.k = k
        self.tp_plus_tn = 0
        self.n_samples = 0

    @property
    def value(self) -> float:
        return self.tp_plus_tn / self.n_samples

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        self.n_samples += logits.shape[0]
        self.tp_plus_tn += (logits.topk(self.k, dim=-1).indices == labels).any(-1).sum().item()
        return self.value

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.value:.04f}, k={self.k})'


class MeanCrossEntropy(Metric):
    def __init__(self) -> None:
        self.loss = 0
        self.n_samples = 0

    @property
    def value(self) -> float:
        return self.loss / self.n_samples

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        self.n_samples += logits.shape[0]
        self.loss += cross_entropy(logits, labels.flatten(), reduction='sum').item()
        return self.value


@dataclass
class EvaluatorState:
    index: int
    metrics: list[Metric]


@torch.no_grad()
def evaluate(model: nn.Module,
             dataset: Dataset,
             batch_size: int = 64,
             cache_file: str | None = None,
             verbose: bool = True,
             ) -> list[Metric]:
    training = model.training
    model.eval()

    n_iters = math.ceil(len(dataset) / batch_size)

    if cache_file is None or not os.path.exists(cache_file):
        state = EvaluatorState(
            index=-1,
            metrics=[MeanCrossEntropy(),
                     TopKAccuracy(1),
                     TopKAccuracy(5),
                     TopKAccuracy(20),
                     TopKAccuracy(100)])
    else:
        state = torch.load(cache_file)
    metrics = state.metrics

    if state.index != n_iters - 1:
        pbar = tqdm(dataset.iter(batch_size),
                    total=n_iters,
                    desc='Evaluating',
                    disable=not verbose)

        for i, batch in enumerate(pbar):
            if i <= state.index:
                continue

            labels = batch.pop('labels')
            logits = model(**batch).logits

            for metric in metrics:
                metric.update(logits, labels)

            pbar.set_description(f'Evaluating; loss={metrics[0].value:.04f}; '
                                 f'top_20_accuracy={metrics[3].value:.04f}')

            if cache_file is not None:
                state.index = i
                torch.save(state, cache_file)

    model.train(training)
    return metrics
