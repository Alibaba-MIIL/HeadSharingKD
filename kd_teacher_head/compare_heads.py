import torch
import numpy as np


def cos(v0, v1):
    n0 = torch.norm(v0)
    n1 = torch.norm(v1)
    return torch.sum(v0*v1) / (n0*n1)


def compare_weights(weights0, weights1):

    n = weights0.shape[0]
    ds = np.zeros([n, 1])
    for i in range(n):
        ds[i, 0] = cos(weights0[i, :], weights1[i, :]).detach().cpu().numpy()

    res = {"mean":np.mean(ds),
           "mean_abs": np.mean(np.abs(ds)),
           "std": np.std(ds),
           "max_abs": np.max(np.abs(ds))}
    str_ = " | ".join("{}: {:.6f}".format(k, v) for k, v in res.items())
    return res, str_


def compare_biases(bias0, bias1):
    n = bias0.shape[0]
    l2 = torch.linalg.norm(bias0-bias1).detach().cpu().numpy()
    abs_ = torch.sum(torch.abs(bias0-bias1))/n
    cos_ = cos(bias0, bias1)
    res = {
        "l2": l2,
        "abs": abs_,
        "cos": cos_
    }
    str_ = " | ".join("{}: {:.6f}".format(k, v) for k, v in res.items())
    return res, str_