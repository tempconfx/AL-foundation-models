"""Uncertainty sampling query strategy."""


from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from ALFM.src.clustering.kmeans import cluster_features
from ALFM.src.query_strategies.base_query import BaseQuery


class Uncertainty(BaseQuery):
    """Select samples with highest softmax uncertainty."""

    def __init__(
        self,
        enable_dropout: bool,
        typical_features: bool,
        cluster_features: bool,
        oversample: int,
        **params: Any,
    ) -> None:
        """Call the superclass constructor.

        Args:
            enable_dropout (bool): flag to enable dropout at inference.
        """
        super().__init__(**params)
        self.enable_dropout = enable_dropout
        self.typical_features = typical_features
        self.cluster_features = cluster_features
        self.oversample = oversample if cluster_features else 1

    def rank_features(self, probs: torch.Tensor) -> torch.Tensor:
        max_probs = probs.max(dim=1)[0]
        return max_probs.argsort()

    def query(self, num_samples: int) -> NDArray[np.bool_]:
        """Select a new set of datapoints to be labeled.

        Args:
            num_samples (int): The number of samples to select.

        Returns:
            NDArray[np.bool_]: A boolean mask for the selected samples.
        """
        mask = np.zeros(len(self.features), dtype=bool)
        unlabeled_indices = np.flatnonzero(~self.labeled_pool)

        if num_samples > len(unlabeled_indices):
            raise ValueError(
                f"num_samples ({num_samples}) is greater than unlabeled pool size ({len(unlabeled_indices)})"
            )

        features = self.features[unlabeled_indices]
        softmax_probs = self.model.get_probs(features, dropout=self.enable_dropout)
        indices = self.rank_features(softmax_probs)
        budget = min(self.oversample * num_samples, len(unlabeled_indices))

        if self.typical_features:  # which points to select
            start = len(features) // 2 - budget // 2  # median centered points
            indices = indices[start : start + budget]
        else:
            indices = indices[:budget]  # select top B points

        if budget > num_samples:  # cluster the diverse samples
            vectors = F.normalize(torch.from_numpy(features[indices]))
            centroids, _ = cluster_features(vectors.numpy(), num_samples)
            indices = indices[centroids]  # pick the points closest to centroids

        mask[unlabeled_indices[indices]] = True
        return mask
