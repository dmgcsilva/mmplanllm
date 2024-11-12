import ast
import json
import os.path
import shutil
from typing import Dict

import torch
import torch.distributed as dist
import transformers
import torch.backends.cudnn as cudnn
from transformers import AutoTokenizer

from constants import *
from data_mod import data_utils
from data_binding import ModelArguments, DataArguments, TrainArgs, ParallelType, PrecisionType, DPOArguments, \
    LoRaArguments
from models.modelling_mmplanllm import FrozenArgs, MMPlanLLM
from trainers.mmplanllm_trainer import MMPlanLLMTrainer
from trainers import trainer_utils
from peft import LoraConfig, TaskType, get_peft_model, LoraModel, PeftModel


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    old_token_count = len(tokenizer)
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    if len(tokenizer) != old_token_count:
        print(f"Tokenizer vocab size changed from {old_token_count} to {len(tokenizer)}")

    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def load_mmplanllm(model_args: FrozenArgs, ckpt_path=None, seq_max_length=DEFAULT_MAX_LEN, dtype=torch.float32):

    tokenizer = AutoTokenizer.from_pretrained(model_args.text_decoder, use_fast=False)
    # Add an image token for loss masking (and visualization) purposes.
    tokenizer.add_special_tokens({"cls_token": "<|image|>"})  # add special image token to tokenizer
    print('Adding [RET] token to vocabulary.')
    print('Before adding new token, tokenizer("[RET]") =', tokenizer('[RET]', add_special_tokens=False))
    num_added_tokens = tokenizer.add_tokens('[RET]')
    print(f'After adding {num_added_tokens} new tokens, tokenizer("[RET]") =',
          tokenizer('[RET]', add_special_tokens=False))
    ret_token_idx = tokenizer.get_vocab().get('[RET]', None)
    assert ret_token_idx is not None and isinstance(ret_token_idx, int), 'Retrieval token not found in tokenizer vocab.'
    model_args.retrieval_token_idx = ret_token_idx

    if tokenizer.pad_token_id is None:
        print("Setting pad token to eos token as default value.")
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'

    model = MMPlanLLM(tokenizer, model_args)

    if ckpt_path is not None:
        if not ckpt_path.endswith("pth.tar"):
            ckpt_path = os.path.join(ckpt_path, "pretrained_model.pth.tar")
        print(f"Loading model from {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        ckpt_path = ckpt_path.replace("pretrained_model.pth.tar", "")
        if any(["base_model" in k for k in checkpoint['state_dict'].keys()]):
            print("DETECTED BASE MODEL IN CHECKPOINT, LOADING LORA MODEL")

            lora_kwargs = {
                'lora_rank': 4,
                'lora_alpha': 8,
                'lora_dropout': 0.1,
            }
            lora_args_path = os.path.join(ckpt_path, 'lora_args.json')
            if not os.path.exists(lora_args_path):
                lora_args_path = os.path.join(ckpt_path, '..', 'lora_args.json')
            if os.path.exists(lora_args_path):
                with open(lora_args_path, 'r') as f:
                    lora_kwargs = json.load(f)
            model.model.make_lm_lora(lora_kwargs['lora_rank'], lora_kwargs['lora_alpha'], lora_kwargs['lora_dropout'])

        model.load_state_dict(checkpoint['state_dict'], strict=False)


    if dtype == torch.float16:
        model = model.half()
    elif dtype == torch.bfloat16:
        model = model.bfloat16()

    model = model.cuda()
    model.model.lm = model.model.lm.cuda()

    # I don't know what this does
    cudnn.benchmark = True

    return model, tokenizer


def add_special_tokens(tokenizer, model):
    tokens_dict = dict()
    if tokenizer.eos_token_id is None:
        print("Setting eos token to default value.")
        if model.config.eos_token_id is not None and model.config.eos_token_id != -1:
            tokens_dict["eos_token"] = tokenizer.convert_ids_to_tokens(model.config.eos_token_id)
        else:
            tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token_id is None:
        print("Setting bos token to default value.")
        if model.config.bos_token_id is not None and model.config.bos_token_id != -1:
            tokens_dict["bos_token"] = tokenizer.convert_ids_to_tokens(model.config.bos_token_id)
        else:
            tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.pad_token_id is None:
        print("Setting pad token to default value.")
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id != -1:
            tokens_dict["pad_token"] = tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 and model.config.pad_token_id is not None else tokenizer.unk_token_id
            )
        else:
            tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.unk_token_id is None:
        if tokenizer.pad_token_id:
            print("Setting unk token to pad token.")
            tokens_dict["unk_token"] = tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 and model.config.pad_token_id is not None else tokenizer.pad_token_id
            )
        elif "pad_token" in tokens_dict:
            print("Setting unk token to pad token.")
            tokens_dict["unk_token"] = tokens_dict["pad_token"]
        else:
            print("Setting unk token to default value.")
            tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    return model, tokenizer


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def main():
    # parse arguments
    parser = transformers.HfArgumentParser(( FrozenArgs, ModelArguments, DataArguments, TrainArgs, DPOArguments, LoRaArguments))
    mmplanllm_args, model_args, data_args, training_args, dpo_args, lora_args = parser.parse_args_into_dataclasses()  # type: (FrozenArgs, ModelArguments, DataArguments, TrainArgs, DPOArguments, LoRaArguments)

    training_args.to_dict()
    model_args.to_dict()
    data_args.to_dict()
    dpo_args.to_dict()
    lora_args.to_dict()

    config = training_args.to_dict() | model_args.to_dict() | data_args.to_dict() | dpo_args.to_dict() | lora_args.to_dict()

    print(training_args)
    data_args.dataset_kwargs = ast.literal_eval(data_args.dataset_kwargs)

    # Build output_dir if not provided
    if training_args.output_dir is not None and training_args.output_dir != "":
        training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name).__str__()
    else:
        training_args.output_dir = os.path.join("/experiments", training_args.project_name,
                                                training_args.run_name).__str__()

    # Get current GPU and GPU_COUNT
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Check if output_dir exists, if so delete or throw warning.
    if rank == 0:
        if not training_args.resume_from_checkpoint:
            if os.path.exists(training_args.output_dir):
                if training_args.overwrite_output_dir:
                    print(f"WARNING: Directory {training_args.output_dir} already exists. OVERWRITING")
                    shutil.rmtree(training_args.output_dir)
                else:
                    print(f"ERROR: Directory {training_args.output_dir} already exists.")
                    return

            # Create output dir
            os.makedirs(training_args.output_dir, exist_ok=True)

            # save training arguments
            json.dump(config, open(f"{training_args.output_dir}/run_arguments.json", "w"), indent=4)

            # Save model args to disk.
            with open(os.path.join(training_args.output_dir, 'model_args.json'), 'w') as f:
                json.dump(vars(mmplanllm_args), f, indent=4)

        else:
            if not os.path.exists(training_args.output_dir):
                print(f"ERROR: Directory {training_args.output_dir} does not exist.")
                return

    # set manual seed
    torch.manual_seed(training_args.seed)
    dtype = torch.float16 if training_args.load_dtype == PrecisionType.FP16 \
        else torch.bfloat16 if training_args.load_dtype == PrecisionType.BF16 \
        else torch.float32

    print(f"Using {dtype} precision")

    # Loading model and tokenizer
    model, tokenizer = load_mmplanllm(mmplanllm_args, ckpt_path=model_args.ckpt_path,
                                      seq_max_length=training_args.seq_max_length, dtype=dtype)

    if lora_args.lora_merge_adapter:
        print("Merging adapter into model")
        model.model.merge_lm_lora()

    original_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # peft
    if dpo_args.use_dpo or lora_args.lora:
        if not isinstance(model, LoraModel) and not isinstance(model, PeftModel):

            if isinstance(model, MMPlanLLM):
                print("LOADING LORA MODEL AS LORA MODEL")
                success = model.model.make_lm_lora(lora_args.lora_rank, lora_args.lora_alpha, lora_args.lora_dropout)

                if success:
                    print("Successfully made Lora model")
                    with open(os.path.join(training_args.output_dir, 'lora_args.json'), 'w') as f:
                        json.dump(vars(lora_args), f, indent=4)
                else:
                    # merge and try again
                    print("FAILED TO MAKE LORA MODEL, MERGING ADAPTER INTO MODEL")
                    model.model.merge_lm_lora()
                    success = model.model.make_lm_lora(lora_args.lora_rank, lora_args.lora_alpha, lora_args.lora_dropout)
                    if success:
                        print("Successfully made Lora model")
                        with open(os.path.join(training_args.output_dir, 'lora_args.json'), 'w') as f:
                            json.dump(vars(lora_args), f, indent=4)
            else:
                print("Loading model into PEFT model")

                modules = find_all_linear_names(model)
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=lora_args.lora_rank,
                    lora_alpha=lora_args.lora_alpha,
                    lora_dropout=lora_args.lora_dropout,
                    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj'],
                )
                model = get_peft_model(model, peft_config)

    if isinstance(model.model.lm, LoraModel) or isinstance(model.model.lm, PeftModel):
        for name, param in model.model.lm.named_parameters():
            # make sure that the input and output embeddings are still trainable
            if "lm_head" in name or "embed" in name:
                param.requires_grad = True
            elif not ('lora' in name or 'Lora' in name):
                param.requires_grad = False
            else:
                param.requires_grad = True

    # print model params and if they require grad
    print()
    print(trainer_utils.get_params_count_str(model, trainable_only=True))
    print()
    # Load dataset
    train_dataloader, eval_dataloader = data_utils.load_train_data(tokenizer, data_args, training_args,
                                                                   init_process_group=world_size > 1,
                                                                   model_name=model_args.base_model if model_args.ckpt_path is None else model_args.ckpt_path,
                                                                   base_model=model_args.base_model,
                                                                   **data_args.dataset_kwargs)

    print("Train dataloader size: ", len(train_dataloader))

    trainer = MMPlanLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=None,
        args=training_args,
        data_args=data_args,
        config=config,
    )

    trainer.train()
    cleanup()


if __name__ == "__main__":
    main()
