import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1.unsqueeze(0), output2.unsqueeze(0))
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class Similarity(torch.nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class InfoNCELoss(torch.nn.Module):
    """
    An unofficial implementation of InfoNCELoss
    By Mario 2022.04
    """

    def __init__(self, temp):
        super().__init__()
        self.sim = Similarity(temp=temp)

    def forward(self, anchor, positive, negative):
        eps = 1e-6
        negative_sim_sum = torch.zeros([1], dtype=torch.float).cuda()

        for i in range(negative.size(0)):
            negative_sim_sum += torch.exp(self.sim(anchor, negative[i]))
        # print("Positive cos similarity is:", torch.exp(self.sim(anchor, positive)))  # 0.04
        # print("Negative cos similarity is:", negative_sim_sum)  # 1.00

        return -torch.log(torch.exp(self.sim(anchor, positive)) / negative_sim_sum + eps)
