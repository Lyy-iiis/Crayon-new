import torch
import argparse
import numpy as np
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=80, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=3072, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=512, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')
    parser.add_argument('--num_pred_heads', type=int, default=42, help='the number of prediction classes.')
    parser.add_argument('--feature_shape', type=tuple, default=(1024,), help='the shape of the feature of visual extractor.')
    parser.add_argument('--zeta', type=float, default=1.0, help='coefficient of classfication loss.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='../autodl-tmp/results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')
    parser.add_argument('--eval_period', type=int, default=2, help='the period of val & test.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=6e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    parser.add_argument('--pretrained_model_name', type=str, default='meta-llama/Llama-3.2-3B', help='the name of the pretrained model.')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='the path to the pretrained model.')
    parser.add_argument('--lora_r', type=int, default=16, help='the rank of the LoRA adaptation.')
    parser.add_argument('--lora_alpha', type=float, default=16, help='the alpha parameter for LoRA.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='the dropout rate for LoRA.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    
    print("[TEST] Starting to generate reports...")

    # create tokenizer
    local_model_path = os.path.join(args.pretrained_model_path, args.pretrained_model_name)
    if os.path.exists(local_model_path):
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)

    # create data loader
    dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    print("[TEST] Data loaded successfully")

    # load model directly from checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = R2GenModel(args, tokenizer)  # Initialize the model architecture
    model.load_state_dict(torch.load(args.resume, map_location=device))  # Load the state dictionary
    model = model.to(device)
    print("[TEST] Model loaded successfully to device: ", device)
    model.eval()

    # generate reports
    print("[TEST] Generating reports...")
    results = []
    with torch.no_grad():
        for batch_idx, (images_id, images, reports_ids, reports_masks, labels, labels_mask) in enumerate(dataloader):
            images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
            output = model(images, mode='sample')
            reports = model.tokenizer.decode_batch(output.cpu().numpy())
            results.extend(reports)

    # save results
    with open('generated_reports.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Report Impression'])
        for report in results:
            writer.writerow([report])
            
    print("[TEST] Reports generated successfully and saved to generated_reports.csv")

if __name__ == '__main__':
    main()
