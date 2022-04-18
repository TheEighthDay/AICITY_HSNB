#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence
from torch import Tensor




class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight.view(-1,1) #weight parameter will act as the alpha parameter to balance class weights
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = torch.sum(-target * F.log_softmax(input, dim=-1), dim=-1)
        weight_ce_loss = self.weight * ce_loss
        pt = torch.exp(-weight_ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * weight_ce_loss)
        if self.reduction == "mean":
            focal_loss = focal_loss.mean()
        return focal_loss



class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError

def TarDisClusterLoss(epoch, output, target, softmax=True, em=True): #em=False
    if softmax:
        prob_p = F.softmax(output, dim=1)
    else:
        prob_p = output / output.sum(1, keepdim=True)
    if em:
        prob_q = prob_p
    else:
        prob_q1 = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
        prob_q1.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda()) # assigned pseudo labels
        if  epoch == 0:
            prob_q = prob_q1
        else:
            prob_q2 = prob_p / prob_p.sum(0, keepdim=True).pow(0.5)
            prob_q2 /= prob_q2.sum(1, keepdim=True)
            prob_q = 0.5 * prob_q1 + 0.5 * prob_q2
    
    if softmax:
        loss = - (prob_q * F.log_softmax(output, dim=1)).sum(1).mean()
    else:
        loss = - (prob_q * prob_p.log()).sum(1).mean()
    
    return loss



_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "focal": FocalLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """

    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]

