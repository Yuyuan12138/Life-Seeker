import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(CustomLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, predicted, target, attention_scores):
        # Assuming predicted and target are binary tensors indicating predicted and true methylated positions
        # attention_scores is a tensor representing attention scores for each position
        # Weighting factor for the loss
        weight = self.weight
        print(predicted.shape)
        # Calculate binary cross entropy loss for the main prediction
        bce_loss = F.cross_entropy(predicted, target, weight=self.weight, reduction=self.reduction)



        # Combine the losses
        loss = torch.mean(bce_loss * attention_scores)

        return loss
