import math
import sys
import os
import datetime
import json
from turtle import undo
from typing import Iterable
import random
from pathlib import Path 
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.utils.model_ema import ModelEmaV2
import copy
import utils
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix


class Engine():
    def __init__(self, model=None,device=None,class_mask=[], domain_list= [], args=None):
        
        self.current_task=0
        self.current_classes=[]
        #! distillation
        max_class_id = max([item for mask in class_mask for item in mask])
        self.class_group_size = len(class_mask[0])  # keep this as it is
        self.class_group_num = (max_class_id // self.class_group_size) + 1

        self.classifier_pool = [None for _ in range(self.class_group_num)]
        self.class_group_train_count = [0 for _ in range(self.class_group_num)]
        #changed
        self.visited_domains = set()

        #changed 
        self.replay_buffer = defaultdict(list)  # key: (domain_id, class_id) → list of samples
        self.buffer_size_per_key = args.replay_buffer_size_per_key  # new argument (explained below)
        self.buffer_size = args.replay_buffer_size  # Total buffer capacity
        self.replay_top_k_percent = args.replay_top_k_percent  # e.g., 0.2 (top 20%)

        self.task_num = len(class_mask)
        self.class_group_size = len(class_mask[0])
        self.model = model
        
        self.num_classes= max([item for mask in class_mask for item in mask])+1
        self.distill_head = None
        model.distill_head = nn.Linear(768, self.num_classes).to(device)        
        self.labels_in_head = np.arange(self.num_classes)
        self.added_classes_in_cur_task = set()
        self.head_timestamps = np.zeros_like(self.labels_in_head)
        self.args=args
        
        self.class_mask=class_mask
        self.domain_list=domain_list

        self.task_type="initial"
        self.args=args
        #changed
        self.final_all_targets = []
        self.final_all_preds = []

        #replay
        self.buffer_size = args.replay_buffer_size
        self.seen_classes = set()
        self.num_domains_per_class = defaultdict(lambda: 0)
        
        self.adapter_vec=[]
        self.task_type_list=[]
        self.class_group_list=[]
        self.adapter_vec_label=[]
        self.device=device
        self.global_class_stats = {k: {'total': 0, 'correct': 0} for k in range(5)}
        
        self.task5_true = []
        self.task5_pred = []
        self.global_confusion = {i: {j: 0 for j in range(5)} for i in range(5)}
        if self.args.d_threshold:
            self.acc_per_label = np.zeros((self.args.class_num, self.args.domain_num))
            self.label_train_count = np.zeros((self.args.class_num))
            self.tanh = torch.nn.Tanh()
            
        self.cs=torch.nn.CosineSimilarity(dim=1,eps=1e-6)
        
        # New variables to keep track of metrics
        #changed
        self.domain_class_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        self.task_class_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        self.domain_confusion = defaultdict(lambda: {i: {j: 0 for j in range(5)} for i in range(5)})
        self.task_confusion = defaultdict(lambda: {i: {j: 0 for j in range(5)} for i in range(5)})
        self.task_acc_history = defaultdict(lambda: defaultdict(list))  # For forgetting analysis
        self.all_task_targets = []
        self.all_task_preds = []
        self.all_task_domains = []
        self.domain_acc_history = defaultdict(lambda: defaultdict(list))  # For domain forgetting
        self.class_acc_history = defaultdict(lambda: defaultdict(list))   # For class forgetting

    #changed
    @torch.no_grad()
    def compute_sample_score(self, model, input, target):
        output = model(input.unsqueeze(0))
        prob = torch.softmax(output, dim=1)
        loss = torch.nn.functional.cross_entropy(output, target.unsqueeze(0))
        entropy = -(prob * torch.log(prob + 1e-8)).sum()
        return loss.item() + entropy.item()  # combine loss & entropy
    def _update_buffer_quota(self):
        """Update quota for each class to maintain fixed total buffer size."""
        self.num_classes_seen = len(self.seen_classes)  # set of all class_ids seen so far
        if self.num_classes_seen == 0:
            return
        self.buffer_per_class = self.buffer_size // self.num_classes_seen
    
    def _collect_buffer_samples(self, class_id, domain_id, samples):
        """Save samples to buffer for (class, domain)."""
        key = (class_id, domain_id)
        if key not in self.replay_buffer:
            self.replay_buffer[key] = []
        # How many slots for this (class, domain)?
        per_domain_quota = 400
        # Add new samples
        self.replay_buffer[key].extend(samples)
        # Trim to per-domain quota
        if len(self.replay_buffer[key]) > per_domain_quota:
            self.replay_buffer[key] = self.replay_buffer[key][:per_domain_quota]
    
    def _rebalance_buffer(self):
        # For each class, collect all (class, *) buffers, trim to buffer_per_class
        for class_id in self.seen_classes:
            all_keys = [k for k in self.replay_buffer if k[0] == class_id]
            all_samples = []
            for k in all_keys:
                all_samples.extend(self.replay_buffer[k])
            # Shuffle and trim
            random.shuffle(all_samples)
            all_samples = all_samples[:self.buffer_per_class]
            # Re-distribute among domains
            n_domains = self.num_domains_per_class[class_id]
            per_domain_quota = 400
            idx = 0
            for k in all_keys:
                self.replay_buffer[k] = all_samples[idx:idx+per_domain_quota]
                idx += per_domain_quota

    def kl_div(self,p,q):
        p=F.softmax(p,dim=1)
        q=F.softmax(q,dim=1)
        kl = torch.mean(torch.sum(p * torch.log(p / q),dim=1))
        return kl
    
    #changed
    def spectral_regularization(self, model, lambda_spec=1e-3, device='cpu'):
        spec_loss = 0.0
        device = self.device
        for name, param in model.named_parameters():
            if 'weight' in name and param.ndim >= 2:
                try:
                    weight = param.to(device)
                    with torch.no_grad():  
                        w = weight.view(weight.size(0), -1)
                        u = torch.randn(w.size(0), device=device)
                        for _ in range(3):  # Few iterations are enough
                            v = torch.nn.functional.normalize(torch.matmul(w.T, u), dim=0)
                            u = torch.nn.functional.normalize(torch.matmul(w, v), dim=0)
                        sigma = torch.dot(u, torch.matmul(w, v)).item()

                    spec_loss += (sigma - 1.0)**2  
                except RuntimeError:
                    continue
        return lambda_spec * spec_loss


    def set_new_head(self, model, labels_to_be_added,task_id):
        len_new_nodes = len(labels_to_be_added)
        self.labels_in_head = np.concatenate((self.labels_in_head, labels_to_be_added))
        self.added_classes_in_cur_task.update(labels_to_be_added)
        self.head_timestamps = np.concatenate((self.head_timestamps, [task_id]*len_new_nodes))
        prev_weight, prev_bias = model.head.weight, model.head.bias
        prev_shape = prev_weight.shape # (class, dim)
        new_head = torch.nn.Linear(prev_shape[-1], prev_shape[0] + len_new_nodes)
    
        new_head.weight[:prev_weight.shape[0]].data.copy_(prev_weight)
        new_head.weight[prev_weight.shape[0]:].data.copy_(prev_weight[labels_to_be_added])
        new_head.bias[:prev_weight.shape[0]].data.copy_(prev_bias)
        new_head.bias[prev_weight.shape[0]:].data.copy_(prev_bias[labels_to_be_added])
        
        print(f"Added {len_new_nodes} nodes with label ({labels_to_be_added})")
        return new_head
    
    
    def inference_acc(self,model,data_loader,device):
        print("Start detecting labels to be added...")
        accuracy_per_label = []
        correct_pred_per_label = [0 for i in range(len(self.current_classes))]
        num_instance_per_label = [0 for i in range(len(self.current_classes))]
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(data_loader):
                if self.args.develop:
                    if batch_idx>200:
                        break
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                output = model(input)
                
                if output.shape[-1] > self.num_classes: # there are already added nodes till now
                    output,_,_ = self.get_max_label_logits(output, self.current_classes) # there are added nodes previously, but not in current task -> get maximum value and use it
                mask = self.current_classes
                not_mask = np.setdiff1d(np.arange(self.num_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = output.index_fill(dim=1, index=not_mask, value=float('-inf'))
                _, pred = torch.max(logits, 1)
                
                correct_predictions = (pred == target)
                for i, label in enumerate(self.current_classes):
                    mask = (target == label)
                    num_correct_pred = torch.sum(correct_predictions[mask])
                    correct_pred_per_label[i] += num_correct_pred.item()
                    num_instance_per_label[i] += sum(mask).item()
        for correct, num in zip (correct_pred_per_label, num_instance_per_label):
            accuracy_per_label.append(round(correct/num,2))
        return accuracy_per_label
    
    def detect_labels_to_be_added(self,inference_acc, thresholds=[]):
        labels_with_low_accuracy = []
        
        if self.args.d_threshold:
            for label,acc,thre in zip(self.current_classes, inference_acc,thresholds):
                if acc <= thre:
                    labels_with_low_accuracy.append(label)
        else: # static threshold
            for label,acc in zip(self.current_classes, inference_acc):
                if acc <= self.args.thre:
                    labels_with_low_accuracy.append(label)
                
        print(f"Labels whose node to be increased: {labels_with_low_accuracy}")
        return labels_with_low_accuracy
    
    def find_same_cluster_items(self,vec):
        if self.kmeans.n_clusters == 1:
            other_cluster_vecs = self.adapter_vec_array
            other_cluster_vecs = torch.tensor(other_cluster_vecs,dtype=torch.float32).to(self.device)
            same_cluster_vecs = None
        else:
            device = self.device
            predicted_cluster = self.kmeans.predict(vec.unsqueeze(0).detach().cpu())[0]
            same_cluster_vecs = self.adapter_vec_array[self.cluster_assignments == predicted_cluster]
            other_cluster_vecs = self.adapter_vec_array[self.cluster_assignments != predicted_cluster]
            same_cluster_vecs = torch.tensor(same_cluster_vecs,dtype=torch.float32).to(self.device)
            other_cluster_vecs = torch.tensor(other_cluster_vecs,dtype=torch.float32).to(self.device)
        return same_cluster_vecs, other_cluster_vecs
    
    def calculate_l2_distance(self,diff_adapter, other):
        weights=[]
        for o in other:
            l2_distance = torch.norm(diff_adapter - o, p=2)
            weights.append(l2_distance.item())
        weights = torch.tensor(weights)
        weights = weights / torch.sum(weights) # summation-> 1
        return weights
        
    def train_one_epoch(self, model: torch.nn.Module, 
                        criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                        device: torch.device, epoch: int, max_norm: float = 0,
                        set_training_mode=True, task_id=-1, class_mask=None, ema_model = None, args = None,):
        
        torch.cuda.empty_cache()
        
        # Memory optimization settings
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        model.train(set_training_mode)
    
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
        
        for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            if self.args.develop:
                if batch_idx > 20:
                    break
            
            # Clear cache every 20 batches to prevent memory buildup
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
                
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
    
            output = model(input)  # (bs, class + n)
            distill_loss = 0
            
            if self.distill_head is not None:
                feature = model.forward_features(input)[:, 0]
                with torch.no_grad():
                    teacher_logits = self.distill_head(feature)
    
                # Masking to get distillation classes only (older classes)
                mask = torch.isin(torch.tensor(self.labels_in_head), torch.tensor(self.current_classes))
                cur_class_nodes = torch.where(mask)[0]
                m = torch.isin(torch.tensor(self.labels_in_head[cur_class_nodes]), torch.tensor(list(self.added_classes_in_cur_task)))
                distill_node_indices = self.labels_in_head[cur_class_nodes][~m]
    
                # Get student + teacher logits for distillation
                student_logits = output[:, distill_node_indices]
                teacher_logits = teacher_logits[:, distill_node_indices]
    
                # KL divergence for logits
                logit_distill_loss = torch.nn.functional.kl_div(
                    torch.log_softmax(student_logits / args.distill_temp, dim=1),
                    torch.softmax(teacher_logits / args.distill_temp, dim=1),
                    reduction='batchmean'
                ) * (args.distill_temp ** 2)
    
                # Feature distillation (cosine sim)
                student_feat = feature
                with torch.no_grad():
                    teacher_feat = model.forward_features(input)[:, 0]  # from previous model if needed
    
                feature_loss = 1 - self.cs(student_feat, teacher_feat.detach()).mean()
    
                # Combine losses
                distill_loss = args.alpha_feat * feature_loss + args.alpha_logit * logit_distill_loss
    
            if output.shape[-1] > self.num_classes:  # there are already added nodes till now 
                output, _, _ = self.get_max_label_logits(output, class_mask[task_id], slice=False)
                if len(self.added_classes_in_cur_task) > 0:  # there are added nodes in current task
                    for added_class in self.added_classes_in_cur_task:
                        cur_node = np.where(self.labels_in_head == added_class)[0][-1]  # the latest appended node
                        output[:, added_class] = output[:, cur_node]  # replace logit value of added label
                    
                output = output[:, :self.num_classes]
                
            # Here is the trick to mask out classes of non-current tasks
            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(5), mask)
                not_mask_tensor = torch.tensor(not_mask, dtype=torch.int64).to(output.device)
                logits = output.index_fill(dim=1, index=not_mask_tensor, value=float('-inf'))
    
            loss = criterion(logits, target)  # (bs, class), (bs)
            task_loss = loss  # keep original task loss separately
    
            if self.args.use_cast_loss:
                if len(self.adapter_vec) > args.k: 
                    cur_adapters = model.get_adapter()
                    self.cur_adapters = self.flatten_parameters(cur_adapters)
                    diff_adapter = self.cur_adapters - self.prev_adapters
                    _, other = self.find_same_cluster_items(diff_adapter)
                    sim = 0
                    
                    weights = self.calculate_l2_distance(diff_adapter, other)
                    for o, w in zip(other, weights):
                        if self.args.norm_cast:
                            sim += w * torch.matmul(diff_adapter, o) / (torch.norm(diff_adapter) * torch.norm(o))
                        else:
                            sim += w * torch.matmul(diff_adapter, o)
                            
                    orth_loss = args.beta * torch.abs(sim)
                    if self.args.use_cast_loss:  
                        if orth_loss > 0:
                            loss += orth_loss
    
            if self.args.IC:
                if distill_loss > 0:
                    loss += distill_loss
    
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
    
            if self.current_task == 0:
                for param in model.parameters():
                    param.requires_grad = True
    
            optimizer.zero_grad()
            
            # Functional Regularization directly here
            if args.use_spectral_reg:
                spec_loss = self.spectral_regularization(model, lambda_spec=args.lambda_spec, device=device)
                loss += spec_loss
                
            loss.backward(retain_graph=False)
            optimizer.step()
    
            # Replay - Add new samples to replay buffer
            try:
                for i in range(input.size(0)):
                    # Only add non-replay samples (from the current batch), not replayed ones
                    if i >= input.size(0) - getattr(self.args, "replay_batch_size", 0):
                        break  # skip the replayed samples
                        
                    class_id = target[i].item()
                    domain_id = task_id  # If you track domain per task, use task_id
                    
                    # Move to CPU immediately to save GPU memory
                    input_cpu = input[i].detach().cpu()
                    target_cpu = target[i].detach().cpu()
                    
                    score = self.compute_sample_score(model, input[i], target[i])
                    sample = (input_cpu, target_cpu, score)
                    
                    # Update seen_classes and buffer quota if new
                    if class_id not in self.seen_classes:
                        self.seen_classes.add(class_id)
                        self._update_buffer_quota()
                        
                    self._collect_buffer_samples(class_id, domain_id, [sample])
                
                self._rebalance_buffer()
            except Exception as e:
                print(f"Warning: Replay buffer update failed: {e}")
                # Continue training even if replay fails
    
            torch.cuda.synchronize()
            
            # Update metrics
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
    
            if ema_model is not None:
                ema_model.update(model.get_adapter())
            
            # Clear intermediate variables to save memory
            del output, logits
            if 'distill_loss' in locals() and distill_loss != 0:
                del feature, teacher_logits, student_logits
            if 'orth_loss' in locals():
                del cur_adapters, diff_adapter
                
            # Clear cache periodically during training
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        
        # Final cache clear
        torch.cuda.empty_cache()
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                            
    def get_max_label_logits(self,output, class_mask,task_id=None, slice=True,target=None):
        #! Get max value for each label output
        correct=0
        total=0
        for label in range(self.num_classes): 
            label_nodes = np.where(self.labels_in_head == label)[0]
            output[:,label],max_index = torch.max(output[:,label_nodes],dim=1)
        if slice:
            output = output[:, :self.num_classes] # discard logits of added nodes
            
        return output,correct,total
    # def update_acc_per_label(self, label_correct, label_total, output, target):
    #     preds = output.argmax(dim=1)
    #     for t, p in zip(target.cpu().numpy(), preds.cpu().numpy()):
    #         label_total[t] += 1
    #         if t == p:
    #             label_correct[t] += 1
    #     return label_correct, label_total

    #changed
    def print_final_results(self):
        import numpy as np
        from sklearn.metrics import confusion_matrix
    
        all_classes = list(range(5))  # 5 classes: 0, 1, 2, 3, 4
    
        y_true = np.array(self.final_all_targets)
        y_pred = np.array(self.final_all_preds)
    
        if len(y_true) == 0 or len(y_pred) == 0:
            print("No predictions available.")
            return
    

        cm = confusion_matrix(self.task5_true, self.task5_pred, labels=list(range(5)))
        print("\n=== FINAL CONFUSION MATRIX (rows: true, cols: pred) ===")
        print(cm)
    
        print("\n=== Per-class Accuracy ===")
        accs = []
        for idx in all_classes:
            total = np.sum(y_true == idx)
            correct = np.sum((y_true == idx) & (y_pred == idx))
            if total == 0:
                print(f"Class {idx}: NULL")
                accs.append(None)
            else:
                acc = correct / total
                print(f"Class {idx}: {acc:.2%} ({correct}/{total})")
                accs.append(acc)

    
        valid_accs = [a for a in accs if a is not None]
        if valid_accs:
            avg_acc = np.mean(valid_accs)
            print(f"\n=== Average Accuracy (used classes): {avg_acc:.2%} ===")
        else:
            print("No classes were used.")
    
    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data_loader, 
                device, task_id=-1, class_mask=None, ema_model=None, args=None, flag_t5=0):
        criterion = torch.nn.CrossEntropyLoss()
        all_targets = []
        all_preds = []
        all_domains = []
    
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test: [Task {}]'.format(task_id + 1)
                    
        # switch to evaluation mode
        model.eval()
    
        correct_sum, total_sum = 0, 0
        label_correct, label_total = np.zeros((self.class_group_size)), np.zeros((self.class_group_size))
        
        # Clear GPU cache before evaluation
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
                if args.develop:
                    if batch_idx > 20:
                        break
                
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                
                # Clear cache periodically during evaluation
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                
                output = model(input)
                
                output, correct, total = self.get_max_label_logits(output, class_mask[task_id], task_id=task_id, target=target, slice=True) 
                output_ema = [output.softmax(dim=1)]
                correct_sum += correct
                total_sum += total
                
                if ema_model is not None:
                    tmp_adapter = model.get_adapter()
                    model.put_adapter(ema_model.module)
                    output = model(input)
                    output, _, _ = self.get_max_label_logits(output, class_mask[task_id], slice=True) 
                    output_ema.append(output.softmax(dim=1))
                    model.put_adapter(tmp_adapter)
                
                output = torch.stack(output_ema, dim=-1).max(dim=-1)[0]
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                _, preds = torch.max(output, 1)
                
                # Move to CPU immediately to save GPU memory
                target_cpu = target.cpu().tolist()
                preds_cpu = preds.cpu().tolist()
                
                # Store predictions and targets
                all_targets.extend(target_cpu)
                all_preds.extend(preds_cpu)
                all_domains.extend([task_id] * len(target_cpu))
                
                # Update domain-class and task-class stats
                for t, p in zip(target_cpu, preds_cpu):
                    # Domain-class stats
                    self.domain_class_stats[task_id][t]['total'] += 1
                    if t == p:
                        self.domain_class_stats[task_id][t]['correct'] += 1
                    
                    # Task-class stats  
                    self.task_class_stats[task_id][t]['total'] += 1
                    if t == p:
                        self.task_class_stats[task_id][t]['correct'] += 1
                    
                    # Update confusion matrices
                    self.domain_confusion[task_id][t][p] += 1
                    self.task_confusion[task_id][t][p] += 1
                
                if flag_t5 == 1:
                    self.task5_true.extend(target_cpu)
                    self.task5_pred.extend(preds_cpu)
                
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
                
                # Clear variables to save memory
                del input, target, output
                if 'output_ema' in locals():
                    del output_ema
                
            if total_sum > 0:
                print(f"Max Pooling acc: {correct_sum/total_sum}")
                
            if self.args.d_threshold and task_id == self.current_task:
                domain_idx = int(self.label_train_count[self.current_classes][0])
                self.acc_per_label[self.current_classes, domain_idx] += np.round(label_correct / label_total, decimals=3)
                print(self.label_train_count)
                print(self.acc_per_label)
    
        # Store task accuracy for forgetting analysis
        task_acc = metric_logger.meters['Acc@1'].global_avg
        self.task_acc_history[task_id][self.current_task].append(task_acc)
        
        # Store domain accuracy for forgetting analysis  
        domain_correct = sum(self.domain_class_stats[task_id][c]['correct'] for c in range(5))
        domain_total = sum(self.domain_class_stats[task_id][c]['total'] for c in range(5))
        if domain_total > 0:
            domain_acc = domain_correct / domain_total
            self.domain_acc_history[task_id][self.current_task].append(domain_acc)
        
        # Store class accuracy for forgetting analysis
        for class_id in range(5):
            stats = self.domain_class_stats[task_id][class_id]
            if stats['total'] > 0:
                class_acc = stats['correct'] / stats['total']
                self.class_acc_history[class_id][self.current_task].append(class_acc)
        
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        all_domains = np.array(all_domains)
        
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
        
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1:.3f} Loss {loss:.3f} Precision {precision:.3f} Recall {recall:.3f} F1 {f1:.3f}'
              .format(top1=metric_logger.meters['Acc@1'].global_avg, 
                      loss=metric_logger.meters['Loss'].global_avg,
                      precision=precision, recall=recall, f1=f1))
        
        # Store for global analysis
        if flag_t5 == 1:
            self.final_all_targets.extend(all_targets.tolist())
            self.final_all_preds.extend(all_preds.tolist())
            self.all_task_targets.extend(all_targets.tolist())
            self.all_task_preds.extend(all_preds.tolist())
            self.all_task_domains.extend(all_domains.tolist())
        
        # Clear cache at the end
        torch.cuda.empty_cache()
                        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    def print_comprehensive_metrics_after_task(self, current_task_id):
        """Print comprehensive metrics after each task"""
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        import numpy as np
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE METRICS AFTER TASK {current_task_id + 1}")
        print(f"{'='*80}")
        
        # 1. Overall Confusion Matrix (all classes seen so far)
        if hasattr(self, 'all_task_targets') and len(self.all_task_targets) > 0:
            all_classes = list(range(5))
            cm = confusion_matrix(self.all_task_targets, self.all_task_preds, labels=all_classes)
            print("\n=== OVERALL CONFUSION MATRIX (All Classes) ===")
            print("Classes:", all_classes)
            print(cm)
            
            # Overall accuracy
            total_correct = sum(cm[i][i] for i in range(5))
            total_samples = cm.sum()
            overall_acc = total_correct / total_samples if total_samples > 0 else 0
            print(f"Overall Accuracy: {overall_acc:.4f}")
        
        # 2. Comprehensive Per-class metrics 
        print("\n=== COMPREHENSIVE PER-CLASS METRICS ===")
        if len(self.all_task_targets) > 0:
            precision, recall, f1, support = precision_recall_fscore_support(
                self.all_task_targets, self.all_task_preds, labels=list(range(5)), average=None, zero_division=0
            )
            
            # Calculate additional metrics manually from confusion matrix
            cm = confusion_matrix(self.all_task_targets, self.all_task_preds, labels=list(range(5)))
            
            for class_id in range(5):
                if support[class_id] > 0:
                    # Basic metrics
                    sensitivity = recall[class_id]  # Sensitivity = Recall = TP/(TP+FN)
                    
                    # Calculate specificity: TN/(TN+FP)
                    tp = cm[class_id, class_id]
                    fp = cm[:, class_id].sum() - tp  # False positives
                    fn = cm[class_id, :].sum() - tp  # False negatives
                    tn = cm.sum() - tp - fp - fn     # True negatives
                    
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    # NPV (Negative Predictive Value): TN/(TN+FN)
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                    
                    # PPV (Positive Predictive Value) = Precision
                    ppv = precision[class_id]
                    
                    print(f"Class {class_id}: Precision={precision[class_id]:.4f}, Recall={recall[class_id]:.4f}, "
                          f"F1={f1[class_id]:.4f}, Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}, "
                          f"PPV={ppv:.4f}, NPV={npv:.4f}, Support={support[class_id]}")
                else:
                    print(f"Class {class_id}: No samples yet")
        
        # 3. Domain-wise Class-wise Accuracies
        print("\n=== DOMAIN-WISE CLASS-WISE ACCURACIES ===")
        for domain_id in range(current_task_id + 1):
            print(f"\nDomain {domain_id} ({self.domain_list[domain_id]}):")
            for class_id in range(5):
                stats = self.domain_class_stats[domain_id][class_id]
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total']
                    print(f"  Class {class_id}: {acc:.4f} ({stats['correct']}/{stats['total']})")
                else:
                    print(f"  Class {class_id}: No samples")
        
        # 4. Domain-wise Confusion Matrix
        print("\n=== DOMAIN-WISE CONFUSION MATRICES ===")
        for domain_id in range(current_task_id + 1):
            print(f"\nDomain {domain_id} ({self.domain_list[domain_id]}) Confusion Matrix:")
            cm_domain = np.array([[self.domain_confusion[domain_id][i][j] for j in range(5)] for i in range(5)])
            print(cm_domain)
            
            # Domain accuracy
            domain_correct = sum(cm_domain[i][i] for i in range(5))
            domain_total = cm_domain.sum()
            domain_acc = domain_correct / domain_total if domain_total > 0 else 0
            print(f"Domain {domain_id} Accuracy: {domain_acc:.4f}")
        
        # 5. Task-wise metrics
        print("\n=== TASK-WISE ACCURACIES ===")
        for task_id in range(current_task_id + 1):
            task_correct = sum(self.task_class_stats[task_id][c]['correct'] for c in range(5))
            task_total = sum(self.task_class_stats[task_id][c]['total'] for c in range(5))
            task_acc = task_correct / task_total if task_total > 0 else 0
            print(f"Task {task_id}: {task_acc:.4f} ({task_correct}/{task_total})")
        
        # 6. Average per-class accuracy
        print("\n=== AVERAGE PER-CLASS ACCURACY ===")
        class_accs = []
        for class_id in range(5):
            total_correct = sum(self.domain_class_stats[d][class_id]['correct'] for d in range(current_task_id + 1))
            total_samples = sum(self.domain_class_stats[d][class_id]['total'] for d in range(current_task_id + 1))
            if total_samples > 0:
                class_acc = total_correct / total_samples
                class_accs.append(class_acc)
                print(f"Class {class_id}: {class_acc:.4f} ({total_correct}/{total_samples})")
            else:
                print(f"Class {class_id}: No samples")
        
        if class_accs:
            avg_per_class_acc = np.mean(class_accs)
            print(f"Average Per-Class Accuracy: {avg_per_class_acc:.4f}")
        
        # 7. Forward and Backward Transfer Analysis
        if current_task_id > 0:
            print("\n=== TRANSFER ANALYSIS ===")
            
            # Task-level Forward Transfer
            print("\n--- Task-level Forward Transfer ---")
            forward_transfers = []
            for task in range(1, current_task_id + 1):
                if task in self.task_acc_history and current_task_id in self.task_acc_history[task]:
                    if len(self.task_acc_history[task][current_task_id]) > 0:
                        current_perf = self.task_acc_history[task][current_task_id][-1]
                        # Compare with a baseline (could be random performance or first task performance)
                        baseline_perf = 0.2  # Assuming 5-class problem, random = 20%
                        forward_transfer = current_perf - baseline_perf
                        forward_transfers.append(forward_transfer)
                        print(f"Task {task} forward transfer: {forward_transfer:.4f}")
            
            if forward_transfers:
                avg_forward_transfer = np.mean(forward_transfers)
                print(f"Average Forward Transfer: {avg_forward_transfer:.4f}")
            
            # Task-level Backward Transfer  
            print("\n--- Task-level Backward Transfer ---")
            backward_transfers = []
            for prev_task in range(current_task_id):
                if (prev_task in self.task_acc_history and 
                    prev_task in self.task_acc_history[prev_task] and 
                    current_task_id in self.task_acc_history[prev_task]):
                    
                    task_accs = self.task_acc_history[prev_task]
                    if (len(task_accs[prev_task]) > 0 and len(task_accs[current_task_id]) > 0):
                        initial_perf = task_accs[prev_task][-1]  # Performance right after learning prev_task
                        current_perf = task_accs[current_task_id][-1]  # Performance after learning current task
                        backward_transfer = current_perf - initial_perf
                        backward_transfers.append(backward_transfer)
                        print(f"Task {prev_task} backward transfer: {backward_transfer:.4f}")
            
            if backward_transfers:
                avg_backward_transfer = np.mean(backward_transfers)
                print(f"Average Backward Transfer: {avg_backward_transfer:.4f}")
        
        # 8. Comprehensive Forgetting Analysis
        if current_task_id > 0:
            print("\n=== COMPREHENSIVE FORGETTING ANALYSIS ===")
            
            # Task-wise forgetting
            print("\n--- Task-wise Forgetting ---")
            task_forgettings = []
            for prev_task in range(current_task_id):
                if (prev_task in self.task_acc_history and 
                    len(self.task_acc_history[prev_task]) > 1):
                    
                    task_accs = self.task_acc_history[prev_task]
                    accs_list = []
                    for task_eval in sorted(task_accs.keys()):
                        if len(task_accs[task_eval]) > 0:
                            accs_list.append(task_accs[task_eval][-1])
                    
                    if len(accs_list) >= 2:
                        max_acc = max(accs_list)
                        current_acc = accs_list[-1]
                        forgetting = max_acc - current_acc
                        task_forgettings.append(forgetting)
                        print(f"Task {prev_task} forgetting: {forgetting:.4f} (max: {max_acc:.4f}, current: {current_acc:.4f})")
            
            if task_forgettings:
                avg_task_forgetting = np.mean(task_forgettings)
                print(f"Average Task Forgetting: {avg_task_forgetting:.4f}")
            
            # Domain-wise forgetting
            print("\n--- Domain-wise Forgetting ---")
            for domain_id in range(current_task_id):
                domain_accs = []
                for eval_task in range(domain_id, current_task_id + 1):
                    domain_correct = sum(self.domain_class_stats[domain_id][c]['correct'] for c in range(5))
                    domain_total = sum(self.domain_class_stats[domain_id][c]['total'] for c in range(5))
                    if domain_total > 0:
                        domain_accs.append(domain_correct / domain_total)
                
                if len(domain_accs) >= 2:
                    max_acc = max(domain_accs)
                    current_acc = domain_accs[-1]
                    domain_forgetting = max_acc - current_acc
                    print(f"Domain {domain_id} forgetting: {domain_forgetting:.4f}")
            
            # Class-wise forgetting
            print("\n--- Class-wise Forgetting ---")
            for class_id in range(5):
                class_accs_over_time = []
                for eval_task in range(current_task_id + 1):
                    total_correct = sum(self.domain_class_stats[d][class_id]['correct'] for d in range(eval_task + 1))
                    total_samples = sum(self.domain_class_stats[d][class_id]['total'] for d in range(eval_task + 1))
                    if total_samples > 0:
                        class_accs_over_time.append(total_correct / total_samples)
                
                if len(class_accs_over_time) >= 2:
                    max_acc = max(class_accs_over_time)
                    current_acc = class_accs_over_time[-1]
                    class_forgetting = max_acc - current_acc
                    print(f"Class {class_id} forgetting: {class_forgetting:.4f}")
        
        print(f"\n{'='*80}")
        print(f"END OF METRICS FOR TASK {current_task_id + 1}")
        print(f"{'='*80}\n")

    
    def evaluate_till_now(self, model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, ema_model=None, args=None):
        stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss
        flag_t5 = 0
        if task_id == 4:
            flag_t5 = 1
        print("======Flag Task 4 flag : ", flag_t5)
        
        # Reset cumulative stats for this evaluation
        if flag_t5 == 1:
            self.all_task_targets = []
            self.all_task_preds = []
            self.all_task_domains = []
        
        for i in range(task_id + 1):
            test_stats = self.evaluate(model=model, data_loader=data_loader[i]['val'], 
                                device=device, task_id=i, class_mask=class_mask, ema_model=ema_model, args=args, flag_t5=flag_t5)
            print(f"\nTesting on Task {i}:")
            print(f"Domain: {self.domain_list[i]}")
            print(f"Classes: {self.class_mask[i]}")
    
            stat_matrix[0, i] = test_stats['Acc@1']
            stat_matrix[2, i] = test_stats['Loss']
            acc_matrix[i, task_id] = test_stats['Acc@1']
        
        avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
        diagonal = np.diag(acc_matrix)
    
        result_str = "[Average accuracy till task{}]	Acc@1: {:.4f}	Acc@5: {:.4f}	Loss: {:.4f}".format(
            task_id + 1, avg_stat[0], avg_stat[1], avg_stat[2])
        
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])
            forward = np.mean((acc_matrix[:, task_id] - acc_matrix[:, 0])[1:task_id + 1])
        
            result_str += "	Forgetting: {:.4f}	Backward: {:.4f}	Forward: {:.4f}".format(
                forgetting, backward, forward)
    
        print(result_str)
        
        # Print comprehensive metrics after each task
        self.print_comprehensive_metrics_after_task(task_id)
        
        return test_stats
                        
    def flatten_parameters(self,modules):
        flattened_params = []
       
        for m in modules:
            params = list(m.parameters())
            flattened_params.extend(params) 
        return torch.cat([param.view(-1) for param in flattened_params])
    
    def cluster_adapters(self):
        k = self.args.k
        if len(self.adapter_vec) > k:
            device = self.device
            self.adapter_vec_array = torch.stack(self.adapter_vec).detach().cpu().numpy().astype(float)
            self.kmeans = KMeans(n_clusters=k,n_init=10)
            self.kmeans.fit(self.adapter_vec_array)
            self.cluster_assignments = self.kmeans.labels_
            print("Cluster(shifts) Assignments:", self.cluster_assignments)
    
    
    def pre_train_epoch(self, model: torch.nn.Module, epoch: int = 0, task_id: int = 0, args = None,):
        if task_id == 0 or args.num_freeze_epochs < 1:
            return model
        
        if epoch == 0:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = False
            print('Freezing adapter parameters for {} epochs'.format(args.num_freeze_epochs))

        if epoch == args.num_freeze_epochs:
            for n, p in model.named_parameters():
                if 'adapter' in n:
                    p.requires_grad = True
            print('Unfreezing adapter parameters')        
        return model
    
    
    def pre_train_task(self, model, data_loader, device, task_id, args):
        epsilon = 1e-8
        self.current_task += 1
        self.current_class_group = int(min(self.class_mask[task_id])/self.class_group_size)
        self.class_group_list.append(self.current_class_group)
        self.current_classes = self.class_mask[task_id]

        print()
        print(f"TASK : {task_id}")
        self.added_classes_in_cur_task = set()  
        #! distillation
        if self.class_group_train_count[self.current_class_group]==0:
            self.distill_head=None
        else: # already seen classes
            if self.args.IC:
                self.distill_head = self.classifier_pool[self.current_class_group]
                inf_acc = self.inference_acc(model, data_loader, device)
                thresholds=[]
                if self.args.d_threshold:
                    count = self.class_group_train_count[self.current_class_group]
                    if count > 0:
                        average_accs = np.sum(self.acc_per_label[self.current_classes, :count], axis=1) / count
                    thresholds = self.args.gamma*(average_accs - inf_acc) / (average_accs + epsilon)
                    thresholds = self.tanh(torch.tensor(thresholds)).tolist()
                    thresholds = [round(t,2) if t>self.args.thre else self.args.thre for t in thresholds]
                    print(f"Thresholds for class {self.current_classes[0]}~{self.current_classes[-1]} : {thresholds}")
                labels_to_be_added = self.detect_labels_to_be_added(inf_acc, thresholds)
                
                
                if len(labels_to_be_added) > 0: #! Add node to the classifier if needed
                    new_head = self.set_new_head(model, labels_to_be_added,task_id).to(device)
                    model.head = new_head
        optimizer = create_optimizer(args, model)

        with torch.no_grad():
            prev_adapters = model.get_adapter()
            self.prev_adapters = self.flatten_parameters(prev_adapters)
            self.prev_adapters.requires_grad=False
        self.cur_domain = self.domain_list[task_id]

        if task_id==0:
            self.task_type_list.append("Initial")
            self.visited_domains.add(self.cur_domain)
            return model, optimizer
        
        self.cur_domain = self.domain_list[task_id]
        
    
        if self.cur_domain not in self.visited_domains:
            # First time seeing this domain → DIL
            self.visited_domains.add(self.cur_domain)
            self.task_type = "DIL"
        else:
            self.task_type = "CIL"
        
        self.task_type_list.append(self.task_type)
        print(f"Current task : {self.task_type}")
        
        return model, optimizer


    def post_train_task(self,model: torch.nn.Module,task_id=-1):
        #! update classifier pool
        self.class_group_train_count[self.current_class_group]+=1
        self.classifier_pool[self.current_class_group]=copy.deepcopy(model.head)
        for c in self.classifier_pool:
                if c != None:
                    for p in c.parameters():
                        p.requires_grad=False
      
        cur_adapters = model.get_adapter()
        self.cur_adapters = self.flatten_parameters(cur_adapters)
        vector=self.cur_adapters - self.prev_adapters
        # if task_id>0: #? 1
        self.adapter_vec.append(vector)
        self.adapter_vec_label.append(self.task_type)
        self.cluster_adapters()
                 
    def train_and_evaluate(self, model: torch.nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                        lr_scheduler, device: torch.device, class_mask=None, args = None,):

        # create matrix to save end-of-task accuracies 
        acc_matrix = np.zeros((5, 5))
        
        ema_model = None
        # Each session = (domain_id, list_of_class_indices)

        for task_id in range(5):
            # Create new optimizer for each task to clear optimizer status
            if task_id > 0 and args.reinit_optimizer:
                optimizer = create_optimizer(args, model)
            
            if task_id == 1 and len(args.adapt_blocks) > 0:
                # ema_model = ModelEmaV2(model.adapter, decay=args.ema_decay).to(device)
                ema_model = ModelEmaV2(model.get_adapter(), decay=args.ema_decay, device=device)
            print()
            print()
            print(f"\n--- Task {task_id}: ---")
            print("Train: ")
            print(f"Domain: {self.domain_list[task_id]}")
            print(f"Classes: {self.class_mask[task_id]}")    
            print()
            print()
            model, optimizer = self.pre_train_task(model, data_loader[task_id]['train'], device, task_id,args)
            for epoch in range(args.epochs):
                model = self.pre_train_epoch(model=model, epoch=epoch, task_id=task_id, args=args,)
                train_stats = self.train_one_epoch(model=model, criterion=criterion, 
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask, ema_model=ema_model, args=args,)
              
                if lr_scheduler:
                    lr_scheduler.step(epoch)
                    
            self.post_train_task(model,task_id=task_id)
            if self.args.d_threshold:
                self.label_train_count[self.current_classes] += 1 
            test_stats = self.evaluate_till_now(model=model, data_loader=data_loader, device=device, 
                                        task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, ema_model=ema_model, args=args)
            if args.output_dir and utils.is_main_process():
                Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                
                checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
                state_dict = {
                        'model': model.state_dict(),
                        'ema_model': ema_model.state_dict() if ema_model is not None else None,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                if args.sched is not None and args.sched != 'constant':
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()
                
                utils.save_on_master(state_dict, checkpoint_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,}

            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                    f.write(json.dumps(log_stats) )
        self.print_final_results()
        
