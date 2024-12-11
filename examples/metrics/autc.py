"""
AUTC
-------------------------

Historgram and Metrics for random scores with different delta.

"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch_ood.utils.metrics import binary_clf_curve
from pytorch_ood.utils import OODMetrics

# %%
# Parameters

# delta bewtween in and ood data
near_delta = 2
far_delta = 10

# split
in_sampels_num = 9
out_sampels_num = 1


# random torch tensors
offset = 10**3
in_scores = torch.rand(in_sampels_num * offset)
out_scores = torch.rand(out_sampels_num * offset)

# %%
# Define function


def metricsandplots(in_scores, out_scores, delta, name):

    metrics = OODMetrics()
    # concat all scores
    scores = torch.cat([in_scores, out_scores + delta])
    # create labels
    labels = torch.cat([torch.zeros_like(in_scores), torch.ones_like(out_scores)])
    metrics.update(scores, -labels)
    metric_dict = metrics.compute()
    print(name, metric_dict)

    # plot histograms
    plt.figure()
    plt.hist(in_scores.cpu().numpy(), bins=100, alpha=0.5, label="in-distribution")
    plt.hist(out_scores.cpu().numpy() + delta, bins=100, alpha=0.5, label="out-of-distribution")
    plt.legend(loc="upper right")
    plt.savefig(f"{name}_histogram.png")
    plt.title(f"{name} Histogram")
    plt.show()

    fpr, tpr, thresholds = binary_clf_curve(labels, scores)
    plt.figure()
    plt.plot(thresholds, fpr, label="FPR")
    plt.plot(thresholds, 1 - tpr, label="FNR")
    plt.legend()
    plt.title(f"{name} FPR and FNR")
    plt.savefig(f"{name}_fpr_fnr.png")
    plt.show()


# %%
# Plot and calculate metrics
metricsandplots(in_scores, out_scores, near_delta, "Near")
metricsandplots(in_scores, out_scores, far_delta, "Far")
