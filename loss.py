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
        self.temp = temp

    def forward(self, anchor, positive, negative):
        eps = 1e-6
        device = anchor.device
        pos_sim = torch.exp(torch.sum(anchor * positive, dim=-1) / self.temp)
        negative_sim_sum = torch.zeros([1], dtype=torch.float).to(device)

        for i in range(negative.size(0)):
            # negative_sim_sum += torch.exp(self.sim(anchor, negative[i]))
            negative_sim_sum += torch.exp(torch.sum(anchor*negative[i], dim=-1)/self.temp)

        # print("Positive cos similarity is:", torch.exp(self.sim(anchor, positive)))  # 0.04
        # print("Negative cos similarity is:", negative_sim_sum)  # 1.00

        return (-torch.log(pos_sim/(negative_sim_sum+pos_sim+eps))).mean()


class ConversationLoss(torch.nn.Module):
    """
    An unofficial implementation of InfoNCELoss
    By Mario 2022.04
    """

    def __init__(self, temp):
        super().__init__()
        self.sim = Similarity(temp=temp)

    def forward(self, feats, labels):
        device = feats.device
        length = len(labels)
        feats = feats[:length]
        mask = torch.zeros([length, length], dtype=torch.bool).to(device)
        for i in range(length):
            for j in range(i, length):
                if labels[i] == labels[j]:
                    mask[i][j] = True
                    mask[j][i] = True
        mask = ~mask

        return
