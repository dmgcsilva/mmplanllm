import os
from typing import Sequence, Dict, Union

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import DataLoader, DistributedSampler

import data_binding as data_binding

from .datasets import TYPE_TO_DATASET_CLASS, DatasetType
from .mmplanllm_dataset import MMPlanLLMDataset, MMPlanLLMModeSampler


def setup():
    # initialize the process group
    print("Initializing process group...")
    rank = int(os.environ['RANK'])
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")


def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    print("world size: ", world_size)
    print("local rank: ", local_rank)
    print("rank: ", rank)
    
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


def get_dataset_type(dataset_type: Union[str, DatasetType]):
    return TYPE_TO_DATASET_CLASS[DatasetType(dataset_type).value]


def find_file_extension(file_path):
    valid_extensions = ['json', 'csv', 'tsv', 'txt']
    for ext in valid_extensions:
        if os.path.exists(f"{file_path}.{ext}"):
            return f"{file_path}.{ext}"

    if not os.path.exists(file_path):
        file_path = file_path.replace("_eval", "_val")
        for ext in valid_extensions:
            if os.path.exists(f"{file_path}.{ext}"):
                return f"{file_path}.{ext}"

    assert os.path.exists(file_path), file_path
    return file_path


def load_train_data(tokenizer: transformers.PreTrainedTokenizer, data_args: data_binding.DataArguments,
                    training_args: data_binding.TrainArgs, init_process_group=True, **kwargs):

    dataset_class = get_dataset_type(data_args.dataset_type)

    print("Loading train data...")
    train_data_path = os.path.join(data_args.data_path, data_args.dataset_name, data_args.dataset_name + "_train")
    train_data_path = find_file_extension(train_data_path)

    train_dataset = dataset_class(tokenizer=tokenizer, data_path=train_data_path, debug=training_args.debug, **kwargs)
    print("Train dataset size: ", len(train_dataset))

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    if isinstance(train_dataset, MMPlanLLMDataset):
        train_sampler = MMPlanLLMModeSampler(train_dataset, batch_size=training_args.per_device_train_batch_size)

    if training_args.evaluation_strategy != "no":
        print("Loading eval data...")
        eval_data_path = os.path.join(data_args.data_path, data_args.dataset_name,
                                      data_args.dataset_name + "_eval")
        eval_data_path = find_file_extension(eval_data_path)

        eval_dataset = dataset_class(tokenizer=tokenizer, data_path=eval_data_path, debug=training_args.debug,
                                     **kwargs)
        print("Eval dataset size: ", len(eval_dataset))

        eval_sampler = DistributedSampler(eval_dataset, rank=rank, num_replicas=world_size)
        if isinstance(train_dataset, MMPlanLLMDataset):
            eval_sampler = MMPlanLLMModeSampler(eval_dataset, batch_size=training_args.per_device_eval_batch_size)
    if init_process_group:
        init_distributed()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        shuffle=False
    )
    eval_dataloader = None
    if training_args.evaluation_strategy != "no":
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            sampler=eval_sampler,
            num_workers=2,
            pin_memory=True,
            shuffle=False
        )

    return train_dataloader, eval_dataloader


def load_test_dataset(tokenizer: transformers.PreTrainedTokenizer, infer_args: data_binding.InferenceArguments, data_args: data_binding.DataArguments,
                      model_args: data_binding.ModelArguments, **kwargs):

    dataset_class = get_dataset_type(data_args.dataset_type)

    print("Loading test data...")
    test_dataset = dataset_class(tokenizer=tokenizer, data_path=infer_args.test_file, debug=False,
                                  model_name=model_args.ckpt_path, **kwargs)
    return test_dataset
