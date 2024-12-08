import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.classifier import Classifier
from modules.decoder_lora import DecoderLoRA


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.fc_transform = nn.Linear(args.d_vf, args.d_model)
        self.encoder_decoder = DecoderLoRA(
            model_name=args.pretrained_model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        self.classifier = Classifier(args.feature_shape, args.num_pred_heads)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        att_feats = self.fc_transform(att_feats)
        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets)
        elif mode == 'sample':
            output = self.encoder_decoder(att_feats, mode='sample')
        else:
            raise ValueError
        classification_output = self.classifier(fc_feats)
        if mode == 'train':
            return output, classification_output
        else:
            return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        att_feats = self.fc_transform(att_feats)
        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets)
        elif mode == 'sample':
            output = self.encoder_decoder(att_feats)
        else:
            raise ValueError
        classification_output = self.classifier(fc_feats)
        if mode == 'train':
            return output, classification_output
        else:
            return output

