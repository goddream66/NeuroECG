from typing import Collection, Optional, Union

import torch
import torch.nn as nn


Floats = Union[float, Collection[float]]


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def listify(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def bn_drop_lin(n_in, n_out, bn=True, p=0.0, actn=None):
    layers = []
    if bn:
        layers.append(nn.BatchNorm1d(n_in))
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


class AdaptiveConcatPool1d(nn.Module):
    """Concatenate adaptive max pooling and adaptive average pooling."""

    def __init__(self, sz: Optional[int] = None):
        super().__init__()
        sz = sz or 1
        self.ap = nn.AdaptiveAvgPool1d(sz)
        self.mp = nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def create_head1d(
    nf: int,
    nc: int,
    lin_ftrs: Optional[Collection[int]] = None,
    ps: Floats = 0.5,
    bn_final: bool = False,
    bn: bool = True,
    act="relu",
    concat_pooling=True,
):
    lin_ftrs = [2 * nf if concat_pooling else nf, nc] if lin_ftrs is None else [
        2 * nf if concat_pooling else nf,
        *lin_ftrs,
        nc,
    ]
    ps = listify(ps)
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps

    actns = [nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, bn, p, actn)
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)
