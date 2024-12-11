import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # print("loss/mask:", mask)
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        
        truncate_input = input[:, 1:]
        truncate_target = target[:, :-1]
        truncate_mask = mask[:, :-1]
        input = torch.log(input) # log softmax
        truncate_input = torch.log(1.0 - truncate_input)
        # print("loss/truncate_input:", truncate_input)
        
        output = - input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        repetition = - truncate_input.gather(2, truncate_target.long().unsqueeze(2)).squeeze(2) * truncate_mask
        # print("output", output)
        
        mle_loss = torch.sum(output) / torch.sum(mask)
        repetition_loss = torch.sum(repetition) / torch.sum(truncate_mask)

        return mle_loss, repetition_loss

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
    
    # print("report: ", reports_ids[0])
    lm_loss, rep_loss = lm_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])
    classification_loss = classification_criterion(pred, labels, labels_mask)
    
    return lm_loss, rep_loss, classification_loss