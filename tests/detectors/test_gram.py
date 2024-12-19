import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.pytorch_ood.api import RequiresFittingException
from src.pytorch_ood.detector import Gram
from tests.helpers import ClassificationModel
from tests.helpers.model import ConvClassifier
import numpy as np


class GramTest(unittest.TestCase):
    """
    Test gram matrix based method
    """

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_something(self):
        nn = ConvClassifier(in_channels=3, out_channels=16)
        model = Gram(nn.classifier, [nn.layer1, nn.pool, nn.dropout], 16, [1, 2, 3])

        # use integers as labels
        # y= torch.cat([torch.zeros(size=(10,),dtype=torch.int), torch.ones(size=(10,))])
        y = torch.cat(
            [torch.zeros(size=(10,), dtype=torch.int), torch.ones(size=(10,), dtype=torch.int)]
        )
        x = torch.randn(size=(20, 3, 16, 16))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)
        logits = nn(x)

        model.fit(loader)

        scores = model(x)
        print(f"Scores: {scores}")

        self.assertIsNotNone(scores)
        self.assertEqual(scores.shape[0], 20)

    def test_nofit(self):
        nn = ConvClassifier(in_channels=3, out_channels=16)
        model = Gram(nn.classifier, [nn.layer1, nn.pool], 16, [1, 2])
        x = torch.randn(size=(20, 10))

        with self.assertRaises(RequiresFittingException):
            model(x)
