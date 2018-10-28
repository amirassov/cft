import torch.nn.functional as F
from torch import nn


class BCELoss(nn.Module):
    def __init__(self, pad_id, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.pad_id = pad_id

    def forward(self, input, target, reduce=True):
        log_probs = F.log_softmax(input.view(-1, input.size(-1)), dim=1)
        loss = F.nll_loss(
            input=log_probs,
            target=target.view(-1),
            size_average=self.size_average,
            ignore_index=self.pad_id,
            reduce=reduce)
        return loss
