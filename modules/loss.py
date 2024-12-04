import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class ClassificationCriterion(nn.Module):
    def __init__(self):
        super(ClassificationCriterion, self).__init__()
        self.loss_fn = FocalLoss()

    def forward(self, pred, labels, mask):
        valid_pred = pred.view(-1, 14, 3).transpose(-1, -2) # 14 classes, 3-classification
        valid_labels = labels.transpose(-1, -2)
        # print("VVVV", valid_pred.shape, valid_labels.shape)
        loss = self.loss_fn(valid_pred, valid_labels)
        # print("loss:", loss)
        return loss
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss

def compute_loss(output, reports_ids, reports_masks, pred, labels, labels_mask):
    lm_criterion = LanguageModelCriterion()
    classification_criterion = ClassificationCriterion()
    
    lm_loss = lm_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    classification_loss = classification_criterion(pred, labels, labels_mask)
    
    return lm_loss, classification_loss