import torch
import torch.nn as nn
import torchvision.models as models
# Add the import for LoRA
import loraclip
from peft import get_peft_model, LoraConfig

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model, preprocess = loraclip.load('ViT-B/32', device='cuda', r=16, lora_mode='vision')
        self.model = model.visual
        loraclip.print_trainable_parameters(self.model)
        # model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        # modules = list(model.children())[:-2]
        # self.model = nn.Sequential(*modules)
        # self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)


    def forward(self, images):
        # images = images.half()
        patch_feats = self.model(images)
        # att_feats = self.avg_fnt(patch_feats)
        return patch_feats.unsqueeze(1), patch_feats
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        print(patch_feats.shape, avg_feats.shape)
        return patch_feats, avg_feats