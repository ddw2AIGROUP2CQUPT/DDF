import torch
import torch.nn as nn
import torch.nn.functional as F


# new
class tver_loss(nn.Module):
    def __init__(self, delta=0.7):
        super(tver_loss, self).__init__()
        self.delta = delta
        # self.gamma = gamma

    def forward(self, output, label):
        smooth = 1e-6
        n, c, w, h = output.shape
        tversky_loss = 0
        tp = torch.sum(output * label)
        fp = torch.sum(output * (1 - label))
        fn = torch.sum((1 - output) * label)
        tversky_loss = (tp + smooth) / (tp + self.delta*fn + (1 - self.delta)*fp + smooth)

        softmax_m = - label * torch.log(tversky_loss)
        softmax_m[label == 0] = 0

        return softmax_m.sum()

class f_bce_loss(nn.Module):
    def __init__(self):
        super(f_bce_loss, self).__init__()
        self.m_loss = tver_loss()

    def forward(self, output, label):
        beta = 1 - torch.mean(label)
        weights = 1 - beta + (2 * beta - 1) * label
        bce_loss = F.binary_cross_entropy(output, label, weights, reduction='sum')
        m_loss = self.m_loss(output, label)
        return m_loss + (0.1 * bce_loss)