import copy
import os
from typing import Sequence, Dict, List

import numpy as np
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .data_constants import IGNORE_INDEX


def print_data_metrics(lens):
    if 'RANK' not in os.environ or int(os.environ['RANK']) == 0:
        print(f"-------------> Data Metrics: <-------------")
        print(f"Max length: {max(lens)}")
        print(f"Min length: {min(lens)}")
        print(f"Mean length: {sum(lens) / len(lens)}")
        print(f"80th percentile: {int(np.percentile(lens, 80))}")
        print(f"85th percentile: {int(np.percentile(lens, 85))}")
        print(f"90th percentile: {int(np.percentile(lens, 90))}")
        print(f"95th percentile: {int(np.percentile(lens, 95))}")
        print(f"99th percentile: {int(np.percentile(lens, 99))}")


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False
        )
        for text in tqdm(strings, desc="Tokenizing samples")
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
        discard_long: bool = True
) -> Dict:
    """Preprocess the data by tokenizing and discarding oversized sources."""
    if discard_long:
        og_max_len = tokenizer.model_max_length
        tokenizer.model_max_length = og_max_len + 128
        filtered_pairs = [(s, t) for s, t in zip(sources, targets) if len(tokenizer.encode(s + t)) <= og_max_len]
        print(f"Discarded {len(sources) - len(filtered_pairs)} out of {len(sources)} samples")
        sources, targets = zip(*filtered_pairs) if filtered_pairs else ([], [])
        tokenizer.model_max_length = og_max_len
    
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized['input_ids']
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    # for label, example_len in zip(labels, examples_tokenized["input_ids_lens"]):
    #     label[example_len+1:] = IGNORE_INDEX
    print_data_metrics(sources_tokenized["input_ids_lens"])
    return dict(input_ids=input_ids, labels=labels, attention_masks=[inp.ne(tokenizer.pad_token_id).long() for inp in input_ids])


def dpo_preprocess(sources: List[str], targets: List[str], negative_targets: List[str], tokenizer: PreTrainedTokenizer):
    # tokenize sources
    tokenized_sources = [tokenizer(
        source,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False,
    ).input_ids[0] for source in sources]

    # tokenize targets
    tokenized_targets = [tokenizer(
        target,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False,
    ).input_ids[0] for target in targets]

    # tokenize negative targets
    tokenized_negative_targets = [tokenizer(
        target,
        return_tensors="pt",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=False,
    ).input_ids[0] for target in negative_targets]

    # concatenate sources and targets
    tokenized_sources_and_targets = [torch.cat((source, target)) for source, target in zip(tokenized_sources, tokenized_targets)]
    labels = copy.deepcopy(tokenized_sources_and_targets)

    # mask sources
    for t, s in zip(labels, tokenized_sources):
        t[:len(s)] = IGNORE_INDEX

    # concatenate sources and negative targets
    tokenized_sources_and_negative_targets = [torch.cat((source, target)) for source, target in zip(tokenized_sources, tokenized_negative_targets)]
    negative_labels = copy.deepcopy(tokenized_sources_and_negative_targets)

    # mask sources
    for t, s in zip(negative_labels, tokenized_sources):
        t[:len(s)] = IGNORE_INDEX

    # prune to max length (right side truncation)
    tokenized_sources_and_targets = [t[-tokenizer.model_max_length:] for t in tokenized_sources_and_targets]
    tokenized_sources_and_negative_targets = [t[-tokenizer.model_max_length:] for t in tokenized_sources_and_negative_targets]
    labels = [t[-tokenizer.model_max_length:] for t in labels]
    negative_labels = [t[-tokenizer.model_max_length:] for t in negative_labels]

    print_data_metrics([len(s) for s in tokenized_sources_and_targets])

    # pad
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenized_sources_and_targets = pad_sequence(tokenized_sources_and_targets, batch_first=True, padding_value=pad_token_id)
    tokenized_sources_and_negative_targets = pad_sequence(tokenized_sources_and_negative_targets, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=pad_token_id)
    negative_labels = pad_sequence(negative_labels, batch_first=True, padding_value=pad_token_id)

    return dict(
        input_ids_and_labels=tokenized_sources_and_targets,
        input_ids_and_negative_labels=tokenized_sources_and_negative_targets,
        labels=labels,
        negative_labels=negative_labels,
        attention_masks=[inp.ne(pad_token_id).long() for inp in tokenized_sources_and_targets],
        negative_attention_masks=[inp.ne(pad_token_id).long() for inp in tokenized_sources_and_negative_targets],
    )
