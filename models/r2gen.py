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
        self.method = args.method
        if self.method == 'pretrained':
            self.fc_transform = nn.Linear(args.d_vf, args.d_model)
        if self.method == 'pretrained':
            self.encoder_decoder = DecoderLoRA(
                model_name=args.pretrained_model_name,
                model_path=args.pretrained_model_path,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                max_seq_length=args.max_seq_length
            )
        elif self.method == 'r2gen':
            self.encoder_decoder = EncoderDecoder(args, tokenizer)
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
        # print(att_feats_0.mean().item(), att_feats_0.std().item())
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        # print("r2gen/fc_feats", fc_feats.shape)
        if self.method == 'pretrained':
            att_feats = self.fc_transform(att_feats)
        if mode == 'train':
            if self.method == 'pretrained':
                output = self.encoder_decoder(att_feats, targets, mode='forward')
            elif self.method == 'r2gen':
                output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            if self.method == 'pretrained':
                output = self.encoder_decoder(att_feats, mode='sample')
            elif self.method == 'r2gen':
                output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        classification_output = self.classifier(fc_feats)
        if mode == 'train':
            return output, classification_output, att_feats_0, att_feats_1
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

