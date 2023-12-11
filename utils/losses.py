import torch
import torch.nn as nn
import torch.nn.functional as F


# cover 0 stego 1

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.25):  # margin=2
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = label.to(torch.float32)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive
