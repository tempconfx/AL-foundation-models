"""Base class for classifiers."""


from typing import Dict
from typing import List
from typing import Literal
from typing import Tuple
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Optimizer
from torchmetrics import Metric


PredType = Literal["probs", "embed", "grad"]


class BaseClassifier(LightningModule):
    """This class provides the blueprint for different classifiers."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_p: float,
        lr: float,
        weight_decay: float,
        metrics: List[Metric],
    ) -> None:
        """Intialize the model parameters."""
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.dropout = nn.Dropout(p=dropout_p)  # for MC sampling
        self.feature_extractor: nn.Module  # define this in subclasses
        self.linear = nn.Linear(input_dim, num_classes)

        self.lr = lr
        self.weight_decay = weight_decay

        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.metrics = nn.ModuleDict(
            {
                stage: nn.ModuleList([metric.clone() for metric in metrics])
                for stage in ["TRAIN", "VAL", "TEST"]
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.feature_extractor(x)
        x = self.linear(x)
        return x

    def step(
        self, batch: torch.Tensor, stage: Literal["TRAIN", "VAL", "TEST"]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        y = y.squeeze()

        y_pred = self(x)  # logits
        y_prob = y_pred.softmax(dim=-1)  # class probabilities
        y_true = y.argmax(dim=1) if y.ndim == 2 else y

        for metric in self.metrics[stage]:
            metric(y_prob, y_true)
            self.log(
                f"{stage}_{type(metric).__name__}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return y_pred, y

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        y_pred, y = self.step(batch, "TRAIN")

        if y.ndim == 2:
            y = y / y.sum(dim=1, keepdim=True)
            weight = 1 - (-y * torch.log(y + 1e-8)).sum(dim=-1) / np.log(y.shape[1])
            loss = (self.loss(y_pred, y.argmax(dim=1)) * weight).mean()

        else:
            loss = self.loss(y_pred, y).mean()

        self.log("CELoss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return cast(torch.Tensor, loss)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.step(batch, "VAL")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.step(batch, "TEST")

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def set_pred_mode(self, mode: PredType | List[PredType]) -> None:
        self._pred_mode = mode if isinstance(mode, list) else [mode]

    def set_dropout(self, flag: bool) -> None:
        self._enable_dropout = flag

    def predict_step(  # type: ignore[override]
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, _ = batch

        self.dropout.train(self._enable_dropout)
        x = self.dropout(x)  # does nothing if dropout is disabled
        embedding = self.feature_extractor(x)

        with torch.inference_mode(False), torch.autocast("cuda", enabled=False):
            embedding = embedding.clone().requires_grad_()
            logits = self.linear(embedding)
            probs = logits.softmax(dim=1)
            tensors = {"embed": embedding, "probs": probs}

            if "grad" in self._pred_mode:
                loss = F.cross_entropy(logits, logits.argmax(dim=1), reduction="sum")
                grad = torch.autograd.grad(loss, embedding)[0]
                tensors["grad"] = grad

        return {k: tensors[k].detach().float().cpu() for k in self._pred_mode}
