import ast
import json
import shutil
from enum import Enum
from typing import Dict, Optional

import math
import os
import subprocess
import time

import numpy as np
import torch
import transformers
import wandb

from data_binding import TrainArgs, CustomSchedulerType, CustomIntervalStrategy, DataArguments
from sklearn.metrics import recall_score
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from warmup_scheduler import GradualWarmupScheduler

from . import trainer_utils
from .trainer import LabTrainer

class Summary(Enum):
  NONE = 0
  AVERAGE = 1
  SUM = 2
  COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class MMPlanLLMTrainer(LabTrainer):

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 optimizer: Optimizer = None,
                 args: TrainArgs = None,
                 data_args: DataArguments = None,
                 config: Dict = None,):
        self.config = config if config is not None else args.to_dict()
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.args = args
        self.optimizer = optimizer
        self.data_args = data_args
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)

        # Build Optimizer
        if self.optimizer is None:
            # params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer_cls = torch.optim.AdamW
            print('Using torch.optim.AdamW as the optimizer.')
            self.optimizer = optimizer_cls(model.parameters(), args.learning_rate,
                                      betas=(args.adam_beta1, args.adam_beta2),
                                      weight_decay=args.weight_decay,
                                      eps=args.adam_epsilon)

        # Compute Stats
        self.batches_per_epoch = (len(train_dataloader) // args.gradient_accumulation_steps)
        self.batches_per_epoch = max(self.batches_per_epoch, 1)

        if self.args.steps_per_epoch > 0:
            self.batches_per_epoch = self.args.steps_per_epoch

        if args.max_steps > 0:
            self.total_steps = args.max_steps
            self.num_train_epochs = args.max_steps // self.batches_per_epoch + int(
                args.max_steps % self.batches_per_epoch > 0
            )
        else:
            self.total_steps = math.ceil(args.num_train_epochs * self.batches_per_epoch)
            self.num_train_epochs = math.ceil(args.num_train_epochs)

        self.eval_batches = 0
        if self.args.evaluation_strategy != CustomIntervalStrategy.NO:
            self.eval_batches = len(eval_dataloader)

        self.warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(
            self.batches_per_epoch * self.args.warmup_ratio)

        # Build Scheduler
        if args.lr_scheduler_type == CustomSchedulerType.STEP_LR:
            """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
            scheduler_steplr = StepLR(self.optimizer, step_size=10 * self.batches_per_epoch, gamma=0.1)
            self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1.0, total_epoch=self.args.warmup_steps,
                                               after_scheduler=scheduler_steplr)
        elif args.lr_scheduler_type == CustomSchedulerType.CYCLICAL or args.lr_scheduler_type == "cyclical":
            self.scheduler = trainer_utils.get_cyclical_schedule(self.optimizer, args.learning_rate,
                                                                 args.lr_step_size)
        else:
            self.scheduler = transformers.get_scheduler(args.lr_scheduler_type, self.optimizer,
                                                        self.warmup_steps, self.total_steps)

        self.best_val_loss = float("inf")
        self.global_step = 0
        self.saved_checkpoints = []

        if self.args.infer_checkpoints:
            self.eval_file = os.path.join(self.data_args.data_path, self.data_args.dataset_name, self.data_args.dataset_name + "_eval.json")

        if args.report_to_wandb:
            self._init_wandb()

    def train(self):
        self.print(" +++++++++++++++++++++++ RUN STATISTICS +++++++++++++++++++++++")
        self.print(f"Epochs: {self.num_train_epochs}")
        self.print(f"Batch Size: {self.args.per_device_train_batch_size}")
        self.print(f"Batches per Epoch: {self.batches_per_epoch}")
        self.print(f"Evaluation Batches: {self.eval_batches}")
        self.print(f"Warmup Steps: {self.warmup_steps}")
        self.print(f"Total Train Steps: {self.total_steps}")
        self.print(" ++++++++++++++++++++++++++ TRAINING ++++++++++++++++++++++++++")

        for epoch in range(1, int(self.num_train_epochs) + 1):
            t0 = time.time()
            self.print(f"Epoch {epoch} starting...")
            self.print("--> entering train loop")
            train_loss = self.train_loop(epoch)

            self.print(f"--> train done")

            if self.args.evaluation_strategy == CustomIntervalStrategy.EPOCH:
                self.print(f"--> running validation for epoch {epoch}")
                eval_loss = self.validation_loop(f"checkpoint_{self.global_step}")
                if self.args.save_strategy == CustomIntervalStrategy.BEST_EVAL and self.best_val_loss > eval_loss:
                    self.best_val_loss = eval_loss
                    self.print(f"--> saving best model with loss: {self.best_val_loss}")
                    self.save_checkpoint()
            if self.args.save_strategy == CustomIntervalStrategy.EPOCH:
                self.print(f"--> saving on epoch {epoch}")
                ckpt_name = self.save_checkpoint()
                if self.args.infer_checkpoints and self.is_main_process():
                    subprocess.call(
                        ['sbatch', f'scripts/{self.args.infer_file}', os.path.join(self.args.output_dir, ckpt_name),
                         self.eval_file])
            self.print(f"--> epoch {epoch} completed")
            if self.global_step >= self.total_steps:
                break

        self.save_checkpoint(f"checkpoint_{self.global_step}")

    def train_loop(self, epoch, callback=None):
        """Main training loop."""
        ngpus_per_node = torch.cuda.device_count()
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        ce_losses = AverageMeter('CeLoss', ':.4e')
        cont_losses = AverageMeter('ContLoss', ':.4e')

        progress = ProgressMeter(
            self.batches_per_epoch,
            [batch_time, losses, ce_losses, cont_losses],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        self.model.train()

        end = time.time()

        for i, batch in enumerate(self.train_dataloader):
            supported_tasks = [m[0] for m in batch.pop("supported_tasks", [])]
            assert len(supported_tasks) > 0, "Supported tasks not provided in the batch."

            batch_size = batch['tgt_tokens'].size(0)

            loss = 0

            for model_mode in supported_tasks:
                batch['mode'] = model_mode
                outputs, task_loss = self.training_step(batch)

                loss += task_loss
                if model_mode in ['captioning', 'textgen']:
                    ce_losses.update(task_loss.item(), batch_size)
                elif model_mode == 'retrieval':
                    cont_losses.update(task_loss.item(), batch_size)
                else:
                    raise NotImplementedError

            loss = loss / self.args.gradient_accumulation_steps
            losses.update(loss.item(), batch_size)
            loss.backward()

            # Update weights
            if ((i + 1) % self.args.gradient_accumulation_steps == 0) or (i == self.batches_per_epoch - 1):
                self.global_step += 1
                self.gradient_accumulation()

            self.norm_grads()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if ((((i + 1) % self.args.gradient_accumulation_steps == 0) or (i == self.batches_per_epoch - 1)) and
                    (self.global_step == 1 or self.global_step % self.args.logging_steps == 0)):

                ex_per_sec = (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps) / batch_time.avg

                progress.display(self.global_step % self.batches_per_epoch)

                log_dict = {
                    'loss': losses.avg,
                    'ce_loss': ce_losses.avg,
                    'contrastive_loss': cont_losses.avg,
                    'total_secs_per_batch': batch_time.avg,
                    'examples_per_sec': ex_per_sec,
                    'lr': self.scheduler.get_last_lr()[0],
                }


                if self.args.report_to_wandb:
                    log_dict = {f"train/{k}": v for k, v in log_dict.items()}
                    wandb.log(log_dict, step=self.global_step)

                batch_time.reset()
                losses.reset()
                ce_losses.reset()
                cont_losses.reset()

                if self.args.save_strategy == CustomIntervalStrategy.STEPS and self.global_step % self.args.save_steps == 0:
                    self.print(f"--> saving model@{self.global_step}")
                    ckpt_name = self.save_checkpoint()
                    if self.args.infer_checkpoints and self.global_step >= self.args.warmup_before_inference and self.is_main_process():
                        # we use sbatch(SLURM) to run the inference script, change with whatever you use
                        subprocess.call(['sbatch', f'scripts/{self.args.infer_file}', os.path.join(self.args.output_dir, ckpt_name), self.eval_file])

                if self.args.evaluation_strategy == CustomIntervalStrategy.STEPS and self.global_step % self.args.eval_steps == 0:
                    self.print(f"--> running validation@{self.global_step}")
                    self.model.eval()
                    eval_score = self.validation_loop(f"checkpoint_{self.global_step}")
                    self.model.train()
                    if self.args.save_strategy == CustomIntervalStrategy.BEST_EVAL and self.best_val_loss > eval_score:
                        self.best_val_loss = eval_score
                        self.print(f"----> saving best model@{self.global_step} with weighted recall: {self.best_val_loss}")
                        self.save_checkpoint()

            if self.global_step >= self.total_steps:
                break

            self.scheduler.step()

    def training_step(self, batch):
        model_mode = batch.pop("mode")

        for key in batch.keys():
            batch[key] = batch[key].to(self.device)

        # compute output
        concat_captions = np.random.uniform(0, 1) < 0.5
        concat_captions = concat_captions and model_mode == 'captioning'

        (model_output, full_labels, last_embedding, _, visual_embs, task_loss) = self.model(
            **batch, mode=model_mode, concat_captions=concat_captions, inference=False)
        output = model_output.logits

        return output, task_loss

    def norm_grads(self):
        if self.model.model.args.freeze_emb:
            with torch.no_grad():
                # Normalize trainable embeddings.
                frozen_norm = torch.norm(self.model.model.input_embeddings.weight[:-1, :], dim=1).mean(0)
                trainable_weight = self.model.model.input_embeddings.weight[-1, :]
                self.model.model.input_embeddings.weight[-1, :].div_(torch.norm(trainable_weight) / frozen_norm)

                if not self.model.model.lm.config.tie_word_embeddings:
                    # do the same for the output embeddings
                    frozen_norm = torch.norm(self.model.model.lm.get_output_embeddings().weight[:-1, :], dim=1).mean(0)
                    trainable_weight = self.model.model.lm.get_output_embeddings().weight[-1, :]
                    self.model.model.lm.get_output_embeddings().weight[-1, :].div_(
                        torch.norm(trainable_weight) / frozen_norm)

    def gradient_accumulation(self):
        if self.model.model.args.freeze_emb:
            # Zero out gradients of the embedding matrix outside of [RET].
            for param in self.model.model.input_embeddings.parameters():
                assert param.grad.shape[0] == len(self.tokenizer)
                # Keep other embeddings frozen.
                mask = torch.arange(param.grad.shape[0]) != self.model.model.retrieval_token_idx
                param.grad[mask, :] = 0

            if not self.model.model.lm.config.tie_word_embeddings:
                # do the same for the output embeddings
                for param in self.model.model.lm.get_output_embeddings().parameters():
                    if param.grad is not None:
                        assert param.grad.shape[0] == len(self.tokenizer)
                        mask = torch.arange(param.grad.shape[0]) != self.model.model.retrieval_token_idx
                        try:
                            param.grad[mask, :] = 0
                        except:
                            param.grad[mask] = 0

        # compute gradient and do SGD step
        if self.args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def validation_loop(self, ckpt_name):
        raise NotImplementedError

    def _log(self, start_time, total_loss, total_ce_loss, total_ret_loss, elapsed_steps, phase="train", step_loss=None, step_ce_loss=None, step_ret_loss=None, ):
        avg_loss = total_loss / elapsed_steps
        avg_ce_loss = total_ce_loss / elapsed_steps
        avg_ret_loss = total_ret_loss / elapsed_steps
        loss = step_loss if step_loss is not None else avg_loss
        ce_loss = step_ce_loss if step_ce_loss is not None else avg_ce_loss
        ret_loss = step_ret_loss if step_ret_loss is not None else avg_ret_loss

        metrics = {"epoch": self.global_step / self.batches_per_epoch,
                   "sec_per_batch": (time.time() - start_time) / elapsed_steps,
                   "loss": loss,
                   "avg_loss": avg_loss,
                   "ce_loss": ce_loss,
                   "avg_ce_loss": avg_ce_loss,
                   "contrastive_loss": ret_loss,
                   "avg_ret_loss": avg_ret_loss,
                   }
        # this line occasionally causes an OverflowError: math range error
        # need to wrap it in a try/except

        metrics["PPL"] = math.exp(metrics["loss"])

        if not (phase.lower() == "eval" or self.scheduler is None):
            metrics["lr"] = self.scheduler.get_last_lr()[0]

        self.print(phase + " " + str(metrics))

        if self.args.report_to_wandb:
            def name(prefix, k):
                if k == "epoch":
                    return prefix.split("/")[0] + "/" + k
                return prefix + k

            prefix = self.args.run_type + "/" + phase + "_"
            prefix = self.args.run_type + "/"

            self._wandb_log({name(prefix, k): v for k, v in metrics.items()})

    def save_checkpoint(self, checkpoint_name=None):
        self.print(f"----> entering save model state")

        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{self.global_step}"

        if len(self.saved_checkpoints) > 0 and self.saved_checkpoints[-1] == checkpoint_name:
            self.print(f"----> model with name {checkpoint_name} already exists")
            return checkpoint_name

        self.print(f"----> saving model with name {checkpoint_name}")
        # Check if number of save surpasses the limit
        if len(self.saved_checkpoints) >= self.args.save_total_limit:
            # Get the oldest folder
            dir_to_delete = os.path.join(self.args.output_dir, self.saved_checkpoints[0])
            # Delete the folder
            print(f"----> deleting folder {dir_to_delete}")
            shutil.rmtree(dir_to_delete)
            # Update checkpoint list
            self.saved_checkpoints = self.saved_checkpoints[1:]


        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        # create the checkpoint_name folder
        os.makedirs(os.path.join(self.args.output_dir, checkpoint_name), exist_ok=True)
        # save the model and tokenizer
        torch.save(state, os.path.join(self.args.output_dir, checkpoint_name, 'pretrained_model.pth.tar'))
        # save the tokenizer
        self.tokenizer.save_pretrained(os.path.join(self.args.output_dir, checkpoint_name))
        # save the model args


        self.saved_checkpoints.append(checkpoint_name)

        self.print(f"----> finished saving model and tokenizer")
        return checkpoint_name

