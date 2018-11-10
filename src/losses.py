import torch.nn.functional as F
from torch import nn


class BCELoss(nn.Module):
    def __init__(self, pad_id, reduction='elementwise_mean'):
        super().__init__()
        self.reduction = reduction
        self.pad_id = pad_id

    def forward(self, input, target):
        log_probs = F.log_softmax(input.view(-1, input.size(-1)), dim=1)
        loss = F.nll_loss(
            input=log_probs,
            target=target.view(-1),
            ignore_index=self.pad_id,
            reduction=self.reduction)
        return loss
