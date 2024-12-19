"""
Gram
==============================

Running :class:`Gram <pytorch_ood.detector.Gram>` on CIFAR 10.

"""
import logging

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import torch.nn.functional as F
from pytorch_ood.dataset.img import Textures
from pytorch_ood.detector import Gram
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown, fix_random_seed

logging.basicConfig(level=logging.INFO)

fix_random_seed(123)

device = "cuda"

# %%
# Setup preprocessing and data
trans = WideResNet.transform_for("cifar10-pt")

dataset_train = CIFAR10(root="data", train=True, download=True, transform=trans)
dataset_in_test = CIFAR10(root="data", train=False, download=True, transform=trans)
dataset_out_test = Textures(
    root="data", download=True, transform=trans, target_transform=ToUnknown()
)

train_loader = DataLoader(dataset_train, batch_size=128, shuffle=True)

# create data loaders
test_loader = DataLoader(dataset_in_test + dataset_out_test, batch_size=128)

# %%
# Stage 1: Create DNN pre-trained on CIFAR 10
model = WideResNet(num_classes=10, pretrained="cifar10-pt").to(device).eval()

layer1 = model.conv1
layer2 = model.block1
layer3 = model.block2
layer4 = model.block3


class MyLayer(nn.Module):
    def __init__(self, bn1, relu):
        super(MyLayer, self).__init__()
        self.bn1 = bn1
        self.relu = relu

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        return x


layer5 = MyLayer(model.bn1, model.relu)


class MyHead(nn.Module):
    def __init__(self, nChannels, fc):
        super(MyHead, self).__init__()
        self.nChannels = nChannels
        self.fc = fc

    def forward(self, x):
        out = F.avg_pool2d(x, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


head = MyHead(model.nChannels, model.fc)

# %%
# Stage 2: Create and fit model
detector = Gram(
    head, [layer1, layer2, layer3, layer4, layer5], num_classes=10, num_poles_list=[1, 2, 3, 4]
)
# detector = MultiMahalanobis([layer1, layer2, layer3, layer4, layer5])

print("Fitting...")
detector.fit(train_loader, device=device)
# use msp detector


# %%
# Stage 3: Evaluate Detectors
print("Testing...")
metrics = OODMetrics()
for x, y in test_loader:
    metrics.update(detector(x.to(device)), y)

print(metrics.compute())

# %%
# This produces a table with the following output:

# {'AUROC': 0.8129576444625854, 'AUTC': 0.4571926361469207, 'AUPR-IN': 0.8341963291168213, 'AUPR-OUT': 0.7648203372955322, 'FPR95TPR': 0.830299973487854}
