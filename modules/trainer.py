import os
from abc import abstractmethod

import time
import torch
import pandas as pd
import json
from numpy import inf
from tqdm import tqdm
import numpy as np

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.eval_period = args.eval_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            if epoch % self.eval_period == 0:
                self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    # print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                    #     self.mnt_metric))
                    # self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()
        self._save_features()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        val_df = pd.DataFrame([self.best_recorder['val']])
        test_df = pd.DataFrame([self.best_recorder['test']])
        record_table = pd.concat([record_table, val_df, test_df], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))
    
    def _save_features(self):
        self.model.eval()
        # features = []
        # patient_ids = []
        features_dict = {}
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels, labels_mask) in tqdm(enumerate(self.test_dataloader)):
                # print(f"*** images_id *** {images_id}")
                images = images.to(self.device)
                batch_size, num_views, channels, height, width = images.shape
                images = images.view(batch_size * num_views, channels, height, width)  # [16, 2, 3, 224, 224] -> [32, 3, 224, 224]
                _, avg_feats = self.model.visual_extractor(images)
                avg_feats = avg_feats.cpu().numpy()
                # features.append(avg_feats.cpu().numpy())
                # patient_ids.extend(images_id.cpu().numpy().repeat(num_views))
                # print(f"*** features shape of batch_idx {batch_idx} *** {len(features)}")
                for i in range(batch_size):
                    img_id = images_id[i]
                    features_dict[f"{img_id}_view1"] = avg_feats[i * num_views]
                    features_dict[f"{img_id}_view2"] = avg_feats[i * num_views + 1]
        features_path = os.path.join(self.checkpoint_dir, 'features.npy')
        np.save(features_path, features_dict)
        print(f"Features saved to {features_path}")
        # features = np.concatenate(features, axis=0)
        # patient_ids = np.array(patient_ids)
        # print(f"*** features shape *** {features.shape}")
        # print(f"*** patient_ids shape *** {patient_ids.shape}")
        
        features_path = os.path.join(self.checkpoint_dir, 'features.npy')
        # patient_ids_path = os.path.join(self.checkpoint_dir, 'patient_ids.npy')
        # np.save(features_path, features)
        # np.save(patient_ids_path, patient_ids)
        # print(f"Features saved to {features_path}")
        # print(f"Patient IDs saved to {patient_ids_path}")
    


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.zeta = args.zeta
        self.zeta_entropy = args.zeta_entropy
        self.zeta_contrast = args.zeta_contrast
        self.enable_test = 1
        self.enable_val = 1
        self.val_iters = args.val_iters
        self.test_iters = args.test_iters
        self.method = args.method

    def _train_epoch(self, epoch):
        train_loss = 0
        total_pred_loss = 0
        total_entropy_loss = 0
        total_contrast_loss = 0
        self.model.train()
        with tqdm(total=len(self.train_dataloader), desc=f'Epoch {epoch}/{self.epochs}', unit='batch') as pbar:
            for batch_idx, (images_id, images, reports_ids, reports_masks, labels, labels_mask) in enumerate(self.train_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                    self.device)
                labels = labels.to(self.device)
                labels_mask = labels_mask.to(self.device)
                # print("trainer/reports_ids:", reports_ids)
                # print("trainer/reports_masks:", reports_masks)
                # assert torch.nonzero(reports_ids).size(0) == reports_masks.sum().item(), f"reports_ids and reports_masks do not match. Please check the dataloader. # of non-zero numbers: {torch.nonzero(reports_ids).size(0)} and {reports_masks.sum().item()}"
                output, pred, att_feats_0, att_feats_1 = self.model(images, reports_ids, mode='train')
                loss, entropy_loss, pred_loss, contrast_loss = self.criterion(output, reports_ids, reports_masks, pred, labels, labels_mask, att_feats_0, att_feats_1)
                train_loss += loss.item()
                total_pred_loss += pred_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_contrast_loss += contrast_loss.item()
                self.optimizer.zero_grad()
                (loss + self.zeta_entropy * entropy_loss + self.zeta * pred_loss + self.zeta_contrast * contrast_loss).backward() # zeta-R2Gen!!!
                # contrast_loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                
                pbar.set_postfix({'lm_loss': train_loss / (batch_idx + 1), 'pred_loss': total_pred_loss / (batch_idx + 1), 'ent_loss': total_entropy_loss / (batch_idx + 1), 'contrast_loss': total_contrast_loss / (batch_idx + 1)})
                pbar.update(1)
        
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        log['total_pred_loss'] = total_pred_loss / len(self.train_dataloader)
        log['total_ent_loss'] = total_entropy_loss / len(self.train_dataloader)

        if epoch % self.eval_period == 0:
            if self.enable_val == 1:
                self.model.eval()
                with torch.no_grad():
                    val_gts, val_res = [], []
                    ok = False
                    iters = 0
                    for batch_idx, (images_id, images, reports_ids, reports_masks, labels, labels_mask) in tqdm(enumerate(self.val_dataloader)):
                        iters += 1
                        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                            self.device), reports_masks.to(self.device)
                        output = self.model(images, mode='sample')
                        assert not isinstance(output, tuple)
                        if self.method == 'pretrained':
                            reports = [self.model.tokenizer.decode(ids, skip_special_tokens=True) for ids in output.cpu().numpy()]
                            ground_truths = [self.model.tokenizer.decode(ids, skip_special_tokens=True) for ids in reports_ids.cpu().numpy()]
                        elif self.method == 'r2gen':
                            reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                            ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                        if ok == False:
                            print("reports of first iter:", reports)
                            ok = True
                        
                        val_res.extend(reports)
                        val_gts.extend(ground_truths)
                        if iters >= self.val_iters:
                            print(f"Validation iter {iters} reached the limit of {self.val_iters}, breaking...")
                            break
                    val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                            {i: [re] for i, re in enumerate(val_res)})
                    log.update(**{'val_' + k: v for k, v in val_met.items()})
                    
            if self.enable_test == 1:
                self.model.eval()
                with torch.no_grad():
                    test_gts, test_res = [], []
                    iters = 0
                    for batch_idx, (images_id, images, reports_ids, reports_masks, labels, labels_mask) in tqdm(enumerate(self.test_dataloader)):
                        iters += 1
                        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                            self.device), reports_masks.to(self.device)
                        output = self.model(images, mode='sample')
                        if self.method == 'pretrained':
                            reports = [self.model.tokenizer.decode(ids, skip_special_tokens=True) for ids in output.cpu().numpy()]
                            ground_truths = [self.model.tokenizer.decode(ids, skip_special_tokens=True) for ids in reports_ids.cpu().numpy()]
                        elif self.method == 'r2gen':
                            reports = self.model.tokenizer.decode_batch(output.cpu().numpy())  # `reports` is a string of the generated report, output has shape (batch_size, sequence_length) which represent token ID
                            ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                        test_res.extend(reports)
                        test_gts.extend(ground_truths)
                        if iters >= self.test_iters:
                            print(f"Test iter {iters} reached the limit of {self.test_iters}, breaking...")
                            break
                    
                    current_dir = self.checkpoint_dir+'/'+'epoch_'+str(epoch)
                    if not os.path.exists(current_dir):
                        os.makedirs(current_dir)   
                    # print("test_gts: ", test_gts)
                    # print("test_res: ", test_res)
                    test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                                {i: [re] for i, re in enumerate(test_res)})
                    
                    log.update(**{'test_' + k: v for k, v in test_met.items()})
                    
                    test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
                    test_res.to_csv(current_dir+'/test_res.csv', index=False, header=["Report Impression"])
                    test_gts.to_csv(current_dir+'/test_gts.csv', index=False, header=["Report Impression"])
                    
                    # print("*** 0 *** current_dir: ", current_dir)
                    os.system(f'python {current_dir}/../../../external/CheXbert/src/label.py -d={current_dir}/test_res.csv -o={current_dir} -c={current_dir}/../../../external/chexbert.pth')
                    os.system(f'mv {current_dir}/labeled_reports.csv {current_dir}/test_res_labeled.csv')
                    # print("*** 1 ***")
                    os.system(f'python {current_dir}/../../../external/CheXbert/src/label.py -d={current_dir}/test_gts.csv -o={current_dir} -c={current_dir}/../../../external/chexbert.pth')
                    os.system(f'mv {current_dir}/labeled_reports.csv {current_dir}/test_gts_labeled.csv')
                    # print("*** 2 ***")
                    os.system(f'python {current_dir}/../../../compute_ce.py --res_path={current_dir}/test_res_labeled.csv --gts_path={current_dir}/test_gts_labeled.csv')

        self.lr_scheduler.step()
        
        # print("HERE is log", log)
        
        score_1 = (log['test_BLEU_1'] * log['test_BLEU_2'] * log['test_BLEU_3'] * log['test_BLEU_4'] * log['test_METEOR'] * log['test_ROUGE_L']) ** (1/6)
        metric = json.load(open(f'{current_dir}/AURROC.json'))
        score_2 = (metric['F1_MICRO'] + metric['PRECISION_MICRO'] + metric['RECALL_MICRO']) / 3
        print("Language score: ", score_1)
        print("Medical score: ", score_2)
        print("Final score: ", (score_1 + score_2) * 0.5)

        return log
