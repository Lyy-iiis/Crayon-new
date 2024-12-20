import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.duplication_num = args.duplication_num

        if (self.split == 'train'):
            self.examples = self.augment_diseased_data(self.ann[self.split], self.duplication_num)
        else: 
            self.examples = self.ann[self.split]
        
        for i in range(len(self.examples)):
            if args.method == 'pretrained':
                encoding = tokenizer(self.examples[i]['report'], return_tensors='pt', truncation=True, max_length=self.max_seq_length)
                self.examples[i]['ids'] = encoding['input_ids'][0].tolist()  # Extract token IDs
            else:
                self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            # if not i: 
            #     print("REPORT: ", self.examples[i]['report'])
            #     print("IDS: ", self.examples[i]['ids'])
        
    def augment_diseased_data(self, data, duplication_num):
        augmented_data = []
        
        for example in data:
            labels = example['labels']
            has_disease = any(
                value == "1.0" for key, value in labels.items() if key != "No Finding"
            )
            augmented_data.append(example)
            
            if has_disease:
                num_duplicates = int(duplication_num)
                for _ in range(num_duplicates):
                    augmented_data.append(example)
                    
        return augmented_data

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        
        labels_tensor = torch.zeros(14, 3)
        labels_mask = torch.zeros(14)
        cnt = 0
        for key, value in example['labels'].items():
            if value == "":
                pass
            else:
                labels_tensor[cnt][int(float(value) + 1)] = 1.0
                labels_mask[cnt] = 1
            cnt += 1
        # print(labels_tensor, labels_mask)
        
        sample = (image_id, image, report_ids, report_masks, seq_length, labels_tensor, labels_mask)
        # print(sample)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
