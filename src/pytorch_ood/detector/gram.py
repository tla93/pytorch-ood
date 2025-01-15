# Parts of this code are taken from https://github.com/Jingkang50/OpenOOD/blob/main/openood/postprocessors/gram_postprocessor.py
"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-brightred?style=flat-square
   :alt: segmentation badge

..  autoclass:: pytorch_ood.detector.Gram
    :members:
    :exclude-members: fit, fit_features
"""
from typing import Optional, TypeVar, List

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

import numpy as np
from ..api import Detector, ModelNotSetException, RequiresFittingException

import torch.nn.functional as F

Self = TypeVar("Self")


class Gram(Detector):
    """
    Implements the on Gram matrices based Method from the paper *Detecting Out-of-Distribution Examples with
    In-distribution Examples and Gram Matrices*.

    The GRAM Detector identifies OOD examples by analyzing feature correlations within the layers of a neural network using Gram matrices. These matrices capture pairwise correlations between feature maps. For enhanced sensitivity, higher-order Gram matrices are computed as:

    .. math :: G^p_l = \\left(F_l^p F_l^{pT}\\right)^{\\frac{1}{p}}


    During training, class-specific minimum *Mins* and maximum *Maxs* bounds are calculated for each entry in the Gram matrices.
    For a test input :math:`D`, deviations are calculated layer-wise by comparing the Gram matrix values against the stored bounds.
    The total deviation across all layers is normalized using the expected deviation :math:`E[\\delta_l]`:

    .. math :: \\Delta(D) = \\sum_{l} \\frac{\\delta_l(D)}{E[\\delta_l]}

    This method detects OOD examples by identifying deviations from learned in-distribution patterns at multiple network layers.



    :see Implementation: `GitHub <https://github.com/VectorInstitute/gram-ood-detection>`__
    :see Paper: `ArXiv <https://arxiv.org/abs/1912.12510>`__
    """

    def __init__(
        self,
        head: Module,
        feature_layers: List[Module],
        num_classes: int,
        num_poles_list: List[int] = None,
    ):
        """
        :param head: the head of the model
        :param feature_layers: the layers of the model to be used for feature extraction
        :param num_classes: the number of classes in the dataset
        :param num_poles_list: the list of poles to be used for higher-order Gram matrices
        """
        super(Gram, self).__init__()
        self.head = head
        self.feature_layers = feature_layers
        self.num_layer = len(feature_layers)
        self.num_classes = num_classes
        if num_poles_list is None:
            self.num_poles_list = range(1, len(self.feature_layers) + 1)
        else:
            self.num_poles_list = num_poles_list
        self.feature_min, self.feature_max = None, None

    def _create_feature_list(self, data: Tensor):
        """
        :param data: input tensor
        :return: feature list
        """
        with torch.no_grad():
            feature_list = []
            data_tmp = data.clone()
            for idx in range(self.num_layer):
                data_tmp = self.feature_layers[idx](data_tmp)
                feature_list.append(data_tmp.clone())

            # calculate logits
            logits = self.head(data_tmp)
            return logits, feature_list

    def fit(self: Self, data_loader: DataLoader, device: str = None) -> Self:
        """
        Calculate the minimum and maximum values for the Gram matrices of the training data.
        :param data_loader: data loader for training data
        :param device: device to run the model on
        :return: self
        """
        num_poles = len(self.num_poles_list)
        feature_class = [
            [[None for x in range(num_poles)] for y in range(self.num_layer)]
            for z in range(self.num_classes)
        ]
        label_list = []
        mins = [
            [[None for x in range(num_poles)] for y in range(self.num_layer)]
            for z in range(self.num_classes)
        ]
        maxs = [
            [[None for x in range(num_poles)] for y in range(self.num_layer)]
            for z in range(self.num_classes)
        ]

        with torch.no_grad():
            # collect features and compute gram metrix
            for n, (x, y) in enumerate(data_loader):
                data = x.to(device)
                label = y.to(device)
                _, feature_list = self._create_feature_list(data)
                label_list = label.tolist()
                for layer_idx in range(self.num_layer):

                    for pole_idx, p in enumerate(self.num_poles_list):
                        temp = feature_list[layer_idx].detach()

                        temp = temp**p
                        temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
                        temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1)))).sum(dim=2)
                        temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(
                            temp.shape[0], -1
                        )

                        temp = temp.tolist()
                        for feature, label in zip(temp, label_list):
                            if isinstance(feature_class[label][layer_idx][pole_idx], type(None)):
                                feature_class[label][layer_idx][pole_idx] = feature
                            else:
                                feature_class[label][layer_idx][pole_idx].extend(feature)
                # print update steps
                if n % 100 == 0:
                    print(f"Step {n}/{len(data_loader)}")

            for label in range(self.num_classes):
                for layer_idx in range(self.num_layer):
                    for poles_idx in range(num_poles):
                        feature = torch.tensor(
                            np.array(feature_class[label][layer_idx][poles_idx])
                        )
                        current_min = feature.min(dim=0, keepdim=True)[0]
                        current_max = feature.max(dim=0, keepdim=True)[0]

                        if mins[label][layer_idx][poles_idx] is None:
                            mins[label][layer_idx][poles_idx] = current_min
                            maxs[label][layer_idx][poles_idx] = current_max
                        else:
                            mins[label][layer_idx][poles_idx] = torch.min(
                                current_min, mins[label][layer_idx][poles_idx]
                            )
                            maxs[label][layer_idx][poles_idx] = torch.max(
                                current_min, maxs[label][layer_idx][poles_idx]
                            )
            self.feature_min = mins
            self.feature_max = maxs
            return self

    def fit_features(self: Self, *args, **kwargs) -> Self:
        """
        Not required.
        """
        return self

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate deviation for inputs

        :param x: input tensor, will be passed through model

        :return: Gram based Deviations
        """
        if self.head is None:
            raise ModelNotSetException

        if self.feature_min is None:
            raise RequiresFittingException

        logits, feature_list = self._create_feature_list(x)

        return self._score(logits, feature_list)

    def predict_features(self, logits: Tensor, feature_list: List[Tensor]) -> Tensor:
        """
        :param logits: logits given by your model
        :param feature_list: list of features extracted from the model
        :return: Gram based Deviations
        """
        return self._score(logits, feature_list)

    @torch.no_grad()
    def _score(self, logits: Tensor, feature_list: List[Tensor]) -> Tensor:
        """
        Calculate deviation for inputs
        :param logits: logits of input
        :param feature_list: list of features extracted from the model
        :return: Gram based Deviations
        """
        if self.feature_min is None or self.feature_max is None:
            raise RequiresFittingException("Fit the detector first.")

        exist = 1
        pred_list = []
        dev = [0 for x in range(logits.shape[0])]

        preds = torch.argmax(logits, dim=1)

        for pred in preds:
            exist = 1
            if len(pred_list) == 0:
                pred_list.extend([pred])
            else:
                for pred_now in pred_list:
                    if pred_now == pred:
                        exist = 0
                if exist == 1:
                    pred_list.extend([pred])
        # compute sample level deviation
        for layer_idx in range(self.num_layer):
            for pole_idx, p in enumerate(self.num_poles_list):
                # get gram metirx
                temp = feature_list[layer_idx].detach()
                temp = temp**p
                temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
                temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1)))).sum(dim=2)
                temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)
                temp = temp.tolist()

                # compute the deviations with train data
                for idx in range(len(temp)):
                    dev[idx] += (
                        F.relu(self.feature_min[preds[idx]][layer_idx][pole_idx] - sum(temp[idx]))
                        / torch.abs(self.feature_min[preds[idx]][layer_idx][pole_idx] + 10**-6)
                    ).sum()
                    dev[idx] += (
                        F.relu(sum(temp[idx]) - self.feature_max[preds[idx]][layer_idx][pole_idx])
                        / torch.abs(self.feature_max[preds[idx]][layer_idx][pole_idx] + 10**-6)
                    ).sum()

        conf = [i / 50 for i in dev]

        return -torch.tensor(conf)
