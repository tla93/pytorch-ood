"""
Models
******************

Frequently used Neural Network based Models


Vision
==================

Wide ResNet
-------------

..  autoclass:: pytorch_ood.model.WideResNet
    :members:

Language
==============================

GRU Classifier
---------------------
..  autoclass:: pytorch_ood.model.GRUClassifier
    :members:


Modules
=================
Neural Network modules frequently used in OOD Detection.


Class Centers
---------------------
..  autoclass:: pytorch_ood.model.ClassCenters
    :members:


..  autoclass:: pytorch_ood.model.RunningCenters
    :members:


"""
from .centers import ClassCenters, RunningCenters
from .gru import GRUClassifier
from .wrn import WideResNet
