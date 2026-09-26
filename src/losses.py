
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)
        intersection = (probs * target).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (probs.sum(dim=1) + target.sum(dim=1) + self.smooth)
        return 1 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    def forward(self, logits, target):
        return 0.5 * self.bce(logits, target) + 0.5 * self.dice(logits, target)
