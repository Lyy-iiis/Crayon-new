import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # print("loss/input:", input.shape, mask.shape)
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        input = torch.log(input) # log softmax
        
        output = - input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        # print("output", output)
        
        mle_loss = torch.sum(output) / torch.sum(mask)

        return mle_loss

class RepetitionCriterion(nn.Module):
    def __init__(self):
        super(RepetitionCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        
        truncate_input = input[:, 1:]
        truncate_target = target[:, :-1]
        truncate_mask = mask[:, :-1]
        
        truncate_input = torch.log(1.0 - truncate_input)
        
        repetition = - truncate_input.gather(2, truncate_target.long().unsqueeze(2)).squeeze(2) * truncate_mask
        
        repetition_loss = torch.sum(repetition) / torch.sum(truncate_mask)

        return repetition_loss
    
class EntropyCriterion(nn.Module):
    def __init__(self):
        super(EntropyCriterion, self).__init__()

    def forward(self, input, mask):
        # print("entropy/input", input.shape, mask.shape)
        mask = mask[:, :input.size(1)]
        input = input * torch.log(input) # entropy
        output = input * mask.unsqueeze(2)
        entropy = torch.sum(output) / torch.sum(mask)

        return entropy

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

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, att_feats_0, att_feats_1):
        # print("att_feats_0", att_feats_0.shape, att_feats_1.shape)
        att_feats_0 = F.normalize(att_feats_0, p=2, dim=2) # 16x49x2048
        att_feats_1 = F.normalize(att_feats_1, p=2, dim=2) # 16x49x2048
        sim = torch.einsum('and,bnd->abn', att_feats_0, att_feats_1) # 16x16x49
        sim_1 = - torch.log_softmax(sim, dim=1)
        sim_1 = torch.einsum('aab->ab', sim_1).mean()
        sim_0 = - torch.log_softmax(sim, dim=0)
        sim_0 = torch.einsum('aab->ab', sim_0).mean()
        return sim_0 + sim_1

def compute_loss(output, reports_ids, reports_masks, pred, labels, labels_mask, att_feats_0, att_feats_1):
    lm_criterion = LanguageModelCriterion()
    # rep_criterion = RepetitionCriterion()
    entropy_criterion = EntropyCriterion()
    classification_criterion = ClassificationCriterion()
    contrastive_criterion = ContrastiveLoss()
    
    # print("report: ", reports_ids[0])
    lm_loss = lm_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])
    # rep_loss = rep_criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])
    entropy_loss = entropy_criterion(output, reports_masks[:, 1:])
    classification_loss = classification_criterion(pred, labels, labels_mask)
    contrastive_loss = contrastive_criterion(att_feats_0, att_feats_1)
    
    return lm_loss, entropy_loss, classification_loss, contrastive_loss