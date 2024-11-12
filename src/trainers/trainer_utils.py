import os
import re
import shutil

import torch
import transformers
from accelerate import init_empty_weights
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import CyclicLR
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    LlamaConfig, AutoConfig, BitsAndBytesConfig

from peft import PeftConfig, PeftModel
import gc


# If this fails, try this;
# https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html#save-and-load-the-model
def save_model(model: PreTrainedModel, save_dir: str, optimizer: Optimizer = None,
               tokenizer: PreTrainedTokenizer = None, state_dict=None, save_fn=None):
    if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    # check if model has attribute config
    if hasattr(model, "config"):
        model.config.save_pretrained(save_dir)

    model.save_pretrained(save_dir, state_dict=state_dict, is_main_process=True, save_fn=save_fn)

    if not (tokenizer is None):
        tokenizer.save_pretrained(save_dir)

    if not (optimizer is None):
        optim_state_dict = {'optim_state_dict': optimizer.state_dict()}
        torch.save(optim_state_dict, save_dir + "/optim_state.pkl")


def get_cyclical_schedule(optimizer: Optimizer, learning_rate: float, step_size: int = -1):
    return CyclicLR(optimizer,
                    base_lr=learning_rate,
                    max_lr=learning_rate * 20,
                    mode="triangular2",
                    gamma=0.999750031,
                    step_size_up=step_size,
                    step_size_down=step_size,
                    cycle_momentum=False)


#### The following code is from https://github.com/kohjingyu/fromage/blob/main/fromage/utils.py
def get_params_count(model, max_name_len: int = 60):
    params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
    total_trainable_params = sum([x[1] for x in params if x[-1]])
    total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
    return params, total_trainable_params, total_nontrainable_params


def get_params_count_str(model, max_name_len: int = 60, trainable_only: bool = False):
    padding = 70  # Hardcoded depending on desired amount of padding and separators.
    params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
    param_counts_text = ''
    param_counts_text += '=' * (max_name_len + padding) + '\n'
    param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
    param_counts_text += '-' * (max_name_len + padding) + '\n'
    for name, param_count, shape, trainable in params:
        if trainable_only and not trainable:
            continue
        param_counts_text += f'| {name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
    param_counts_text += '-' * (max_name_len + padding) + '\n'
    param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
    param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
    param_counts_text += '=' * (max_name_len + padding) + '\n'
    return param_counts_text


