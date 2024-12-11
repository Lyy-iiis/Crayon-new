import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import os

class DecoderLoRA(nn.Module):
    def __init__(self, model_name, lora_r, lora_alpha, lora_dropout, model_path = None):
        super(DecoderLoRA, self).__init__()
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        local_model_path = os.path.join(model_path, model_name)
        if os.path.exists(local_model_path):
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.model = get_peft_model(self.model, peft_config)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.embedding = nn.Embedding(self.model.config.vocab_size, self.model.config.hidden_size)  # Embedding layer

    def forward(self, att_feats, targets=None, mode='forward'):
        if mode == 'forward':
            assert targets is not None
            # print("ATT_FEATS: ", att_feats.shape) # (batch_size, -1, 1536)
            # print("TARGETS: ", targets.shape, targets)
            embedded_targets = self.embedding(targets)
            # print("EMBEDDED_TARGETS: ", embedded_targets.shape) # (batch_size, max_seq_length, 1536)
            input_ids = torch.cat((att_feats, embedded_targets), dim=1)
            attention_mask = torch.ones(input_ids.size()[:-1], dtype=torch.long, device=input_ids.device)
            logits = self.model(inputs_embeds=input_ids, attention_mask=attention_mask).logits
            output_logits = logits[:, att_feats.size(1):, :]
            return self.log_softmax(output_logits)
        elif mode == 'sample':
            assert targets is None
            embedded_att_feats = att_feats
            attention_mask = torch.ones(embedded_att_feats.size()[:-1], dtype=torch.long, device=embedded_att_feats.device)
            return self.model.generate(
                inputs_embeds=embedded_att_feats,
                attention_mask=attention_mask, 
                max_new_tokens=80, 
                pad_token_id=self.model.config.eos_token_id
            )
        else:
            raise ValueError("Invalid mode: choose 'forward' or 'sample'")
