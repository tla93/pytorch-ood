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
        nn = ConvClassifier(in_channels=3, num_outputs=2)

        class MyHead(torch.nn.Module):
            def __init__(self, classifier, pool):
                super(MyHead, self).__init__()
                self.classifier = classifier
                self.pool = pool
                self.flatten = torch.nn.Flatten()

            def forward(self, x):
                x = self.pool(x)
                x = self.flatten(x)
                x = self.classifier(x)
                return x

        model = Gram(MyHead(nn.classifier, nn.pool), [nn.layer1], 2, [1])

        # use integers as labels
        # y= torch.cat([torch.zeros(size=(10,),dtype=torch.int), torch.ones(size=(10,))])
        y = torch.cat(
            [torch.zeros(size=(10,), dtype=torch.int), torch.ones(size=(10,), dtype=torch.int)]
        )
        x = torch.randn(size=(20, 3, 16, 16))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset)

        model.fit(loader)

        scores = model(x)
        print(f"Scores: {scores}")

        self.assertIsNotNone(scores)
        self.assertEqual(scores.shape[0], 20)

    def test_nofit(self):
        nn = ConvClassifier(in_channels=3, num_outputs=2)

        class MyHead(torch.nn.Module):
            def __init__(self, classifier, pool):
                super(MyHead, self).__init__()
                self.classifier = classifier
                self.pool = pool
                self.flatten = torch.nn.Flatten()

            def forward(self, x):
                x = self.pool(x)
                x = self.flatten(x)
                x = self.classifier(x)
                return x

        model = Gram(MyHead(nn.classifier, nn.pool), [nn.layer1], 2, [1])
        x = torch.randn(size=(20, 10))

        with self.assertRaises(RequiresFittingException):
            model(x)
