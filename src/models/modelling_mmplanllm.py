from dataclasses import dataclass, field, fields
from enum import Enum
from io import BytesIO
from typing import Callable, List, Optional, Tuple, Union, Dict
from collections import namedtuple, OrderedDict
import json
import glob
import math
import numpy as np
import os

import requests
import torch
from info_nce import info_nce
from peft import LoraConfig, TaskType, get_peft_model, LoraModel, PeftModel
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from PIL import Image, UnidentifiedImageError

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoFeatureExtractor, AutoTokenizer
from transformers import OPTForCausalLM, GPT2Tokenizer
from transformers import CLIPVisionModel, CLIPVisionConfig

import torch

# source: https://blog.eleuther.ai/rotary-embeddings/
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _get_cos_sin(self, x, seq_dim=0, offset=0):
        # seq: [seq, batch, heads, hdim]
        seq_len = x.shape[seq_dim] + offset
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached

    def forward(self, x, offset=0):
        cos, sin = self._get_cos_sin(x, offset=offset)
        return (x * cos[offset:, :, :, :]) + (rotate_half(x) * sin[offset:, :, :, :])


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


@dataclass
class FrozenArgs:
    freeze_lm: bool = field(default=True, metadata={"help": "Whether to freeze the language model."})
    freeze_emb: bool = field(default=False, metadata={"help": "Whether to freeze the embeddings, regardless of the freeze_lm flag."})
    freeze_vm: bool = field(default=True, metadata={"help": "Whether to freeze the vision model."})
    text_decoder: str = field(default='facebook/opt-6.7b', metadata={"help": "The name of the text encoder model."})
    visual_encoder: str = field(default='openai/clip-vit-large-patch14', metadata={"help": "The name of the visual encoder model."})
    n_visual_tokens: int = field(default=1, metadata={"help": "The number of visual tokens."})
    image_embed_dropout_prob: float = field(default=0.0, metadata={"help": "The dropout probability for the image embedding."})
    task: str = field(default='captioning', metadata={"help": "The task to perform."})
    shared_emb_dim: Optional[int] = field(default=256, metadata={"help": "The dimension of the shared embedding."})
    text_embed_dropout_prob: float = field(default=0.0, metadata={"help": "The dropout probability for the text embedding."})
    text_emb_layers: List[int] = field(default_factory=lambda: [-1], metadata={"help": "The layers to use for text embeddings."})
    retrieval_token_idx: int = field(default=0, metadata={"help": "The index of the retrieval token."})

    cap_loss_scale: float = field(default=1.0, metadata={"help": "The scaling factor for the captioning loss."})
    ret_loss_scale: float = field(default=1.0, metadata={"help": "The scaling factor for the retrieval loss."})

    use_negatives: bool = field(default=False, metadata={"help": "Whether to use negative sampling."})
    negative_count: int = field(default=512, metadata={"help": "The number of negative samples to use."})

    use_pos_emb: bool = field(default=False, metadata={"help": "Whether to use positional embeddings."})

    @staticmethod
    def from_dict(d: Dict):
        return FrozenArgs(**d)

    def to_dict(self):
        """
            Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
            the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


class MMPlanLLMModel(nn.Module):
    def __init__(self, tokenizer, args: FrozenArgs = FrozenArgs()):
        super().__init__()
        self.tokenizer = tokenizer
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(args.visual_encoder)
        self.image_token = self.tokenizer.cls_token_id
        assert args.text_emb_layers != set(args.text_emb_layers), 'text_emb_layers not unique'
        self.args = args

        text_decoder = args.text_decoder
        visual_encoder = args.visual_encoder
        n_visual_tokens = args.n_visual_tokens
        print(f"Using {text_decoder} for the language model.")
        print(f"Using {visual_encoder} for the visual model with {n_visual_tokens} visual tokens.")

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if 'facebook/opt' in text_decoder:
            self.lm = OPTForCausalLM.from_pretrained(text_decoder)
        else:
            try:
                self.lm = AutoModelForCausalLM.from_pretrained(text_decoder)
            except  Exception as e:
                print(f'Error loading model {text_decoder}: {e}')
                raise e

        self.text_decoder = text_decoder

        if self.args.freeze_lm:
            self.lm.eval()
            print("Freezing the LM.")
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm.train()

        # NOTE: Resizing sets all token embeddings and all lm_head weights (since they are tied in OPT)
        # to be trainable (param.requires_grad = True).
        self.retrieval_token_idx = args.retrieval_token_idx
        print(f'Initializing embedding for the retrieval token [RET] (id = {self.retrieval_token_idx}).')
        self.lm.resize_token_embeddings(len(tokenizer))

        self.input_embeddings = self.lm.get_input_embeddings()

        self.input_embeddings.requires_grad_(True)

        self.output_embeddings = self.lm.get_output_embeddings()

        if not self.lm.config.tie_word_embeddings:
            print("Word embeddings are not tied to the output embeddings, setting output embeddings to be trainable.")
            self.output_embeddings.weight.requires_grad_(True)


        print("Restoring pretrained weights for the visual model.")
        if 'clip' in visual_encoder:
            self.visual_model = CLIPVisionModel.from_pretrained(visual_encoder)
        else:
            self.visual_model = AutoModel.from_pretrained(visual_encoder)

        if 'clip' in visual_encoder:
            hidden_size = self.visual_model.config.hidden_size
        else:
            raise NotImplementedError

        if self.args.freeze_vm:
            print("Freezing the VM.")
            self.visual_model.eval()
            for param in self.visual_model.parameters():
                param.requires_grad = False
        else:
            self.visual_model.train()

        self.visual_model_name = visual_encoder

        embedding_dim = self.input_embeddings.embedding_dim * self.args.n_visual_tokens
        self.text_hidden_fcs = nn.ModuleList([])
        if self.args.shared_emb_dim is None:
            if len(self.args.text_emb_layers) == 1:
                if (self.args.text_emb_layers[0] in [-1, self.lm.config.num_hidden_layers]) and (
                        'bert' not in text_decoder) and "opt" in text_decoder:
                    out_dim = self.lm.config.word_embed_proj_dim
                else:
                    out_dim = self.lm.config.hidden_size
            else:
                if (-1 in self.args.text_emb_layers) or (self.lm.config.num_hidden_layers in self.args.text_emb_layers) \
                        and (self.lm.config.word_embed_proj_dim != self.lm.config.hidden_size):
                    raise ValueError(
                        'No projection dim specified but model uses last output layer and an intermediate one (which have different dims).')
                else:
                    out_dim = self.lm.config.hidden_size
        else:
            out_dim = self.args.shared_emb_dim

            for layer_idx in self.args.text_emb_layers:
                if (layer_idx == -1 or layer_idx == self.lm.config.num_hidden_layers) and ('bert' not in text_decoder):
                    if "opt" in text_decoder:
                        in_dim = self.lm.config.word_embed_proj_dim
                    else:
                        in_dim = self.lm.config.hidden_size

                    text_fc = [nn.Linear(in_dim, out_dim), nn.Dropout(self.args.text_embed_dropout_prob)]
                    self.text_hidden_fcs.append(nn.Sequential(*text_fc))

                elif layer_idx < self.lm.config.num_hidden_layers:
                    text_fc = [nn.Linear(self.lm.config.hidden_size, out_dim),
                               nn.Dropout(self.args.text_embed_dropout_prob)]
                    self.text_hidden_fcs.append(nn.Sequential(*text_fc))
                else:
                    raise ValueError(
                        f'Embedding of layer {layer_idx} was requested but model only has {self.lm.config.num_hidden_layers} layers.')

        self.visual_embeddings = nn.Linear(hidden_size, embedding_dim)
        self.visual_fc = nn.Linear(hidden_size, out_dim)

        self.image_dropout = nn.Dropout(self.args.image_embed_dropout_prob)

        self.negative_queue_text = None
        self.negative_queue_image = None
        self.max_queue_size = self.args.negative_count

        if self.args.use_pos_emb:
            self.pos_embs = Rotary(out_dim)

    def make_lm_lora(self, lora_rank: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1):

        if isinstance(self.lm, LoraModel) or isinstance(self.lm, PeftModel):
            print("Model already has a LoraModel or PeftModel, skipping.")
            return False

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj'],
        )
        self.lm = get_peft_model(self.lm, peft_config)

        # update the input embeddings
        self.input_embeddings = self.lm.get_input_embeddings()
        self.input_embeddings.requires_grad_(True)
        if not self.lm.config.tie_word_embeddings:
            print("Word embeddings are not tied to the output embeddings, setting output embeddings to be trainable.")
            self.output_embeddings = self.lm.get_output_embeddings()
            self.output_embeddings.weight.requires_grad_(True)

        print("---> Converted LM to LoRA")
        return True

    def merge_lm_lora(self):
        if isinstance(self.lm, LoraModel) or isinstance(self.lm, PeftModel):
            print("Merging adapter")
            self.lm = self.lm.merge_and_unload()
            if not self.args.freeze_lm:
                # set all parameters trainable
                for param in self.lm.parameters():
                    param.requires_grad = True

            self.input_embeddings = self.lm.get_input_embeddings()
            self.input_embeddings.requires_grad_(True)
            if not self.lm.config.tie_word_embeddings:
                print("Word embeddings are not tied to the output embeddings, setting output embeddings to be trainable.")
                self.output_embeddings = self.lm.get_output_embeddings()
                self.output_embeddings.weight.requires_grad_(True)

            print("---> Merged LM and LoRA")
        else:
            print(f"WARNING: Can't merge, model is not a LoraModel or PeftModel, but a {type(self.lm)}")

    def get_visual_embs(self, pixel_values: torch.FloatTensor, mode: str = 'captioning', position_offset: torch.LongTensor = 0):
        if mode not in ['captioning', 'retrieval']:
            raise ValueError(f'mode should be one of ["caption", "retrieval"], got {mode} instead.')

        # Extract visual embeddings from the vision encoder.
        if 'clip' in self.visual_model_name:
            outputs = self.visual_model(pixel_values)
            encoder_outputs = outputs.pooler_output
        else:
            raise NotImplementedError

        # Use the correct fc based on function argument.
        if mode == 'captioning':
            visual_embs = self.visual_embeddings(encoder_outputs)  # (2, D * n_visual_tokens)
            visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], self.args.n_visual_tokens, -1))
        elif mode == 'retrieval':
            visual_embs = self.visual_fc(encoder_outputs)  # (2, D * n_visual_tokens)
            if self.args.use_pos_emb:
                visual_embs = self.pos_embs(visual_embs.unsqueeze(1).unsqueeze(1), offset=position_offset)
                visual_embs = visual_embs.squeeze(1).squeeze(1)
            visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 1, -1))
        else:
            raise NotImplementedError

        visual_embs = self.image_dropout(visual_embs)
        return visual_embs

    def remove_img_tokens(self, labels: torch.LongTensor, embs: torch.FloatTensor = None, last_embedding_idx: torch.LongTensor = None):

        was_unbatched = False
        if len(labels.shape) == 1:
            # unbatched case
            labels = labels.unsqueeze(0)
            was_unbatched = True

        new_labels = []
        if embs is not None:
            new_embs = []
        if last_embedding_idx is not None:
            new_last_embedding_idx = []

        for i in range(labels.shape[0]):
            # get the index of all the image tokens in the labels
            label = labels[i]
            img_idxs = (label == self.image_token).nonzero(as_tuple=True)

            if embs is not None:
                emb = embs[i]
            if last_embedding_idx is not None:
                last_idx = last_embedding_idx[i]
            # Check if the tensor is empty
            if len(img_idxs) > 0 and img_idxs[0].shape[0] > 0:
                for j in range(len(img_idxs)):
                    # remove the image tokens from the labels
                    label = torch.cat([label[:img_idxs[j]], label[img_idxs[j] + 1:]])
                    if embs is not None:
                        emb = torch.cat([emb[:img_idxs[j]], emb[img_idxs[j] + 1:]])
                    if last_embedding_idx is not None:
                        last_idx -= 1

            new_labels.append(label)
            if embs is not None:
                new_embs.append(emb)
            if last_embedding_idx is not None:
                new_last_embedding_idx.append(last_idx)

        new_labels = torch.stack(new_labels, dim=0)
        new_embs = torch.stack(new_embs, dim=0) if embs is not None else None
        new_last_embedding_idx = torch.stack(new_last_embedding_idx, dim=0) if last_embedding_idx is not None else None

        if was_unbatched:
            new_labels = new_labels.squeeze(0)

        return new_labels, new_embs, new_last_embedding_idx

    def train(self, mode=True):
        super(MMPlanLLMModel, self).train(mode=mode)
        # Overwrite train() to ensure Frozen models remain frozen.
        if self.args.freeze_lm:
            self.lm.eval()
        if self.args.freeze_vm:
            self.visual_model.eval()


    def _forward_captioning(
            self,
            labels: torch.LongTensor,
            input_embs: torch.FloatTensor,
            visual_embs: torch.FloatTensor,
            concat_captions: bool = False,
    ):

        batch_size, vis_seq_len, _ = visual_embs.shape  # vis_seq_len = n_visual_tokens

        # Concat to text embeddings.
        condition_seq_len = 0
        for i in range(batch_size):
            # find the first occurence of the tokenizer.cls_token_id
            idx = (labels[i] == self.tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                idx = 0
            # replace the input embeddings at idx with the visual embeddings
            input_embs[i, idx, :] = visual_embs[i, 0, :]

        full_labels = torch.zeros(visual_embs.shape[:2], dtype=torch.int64).to(visual_embs.device) - 100

        # Mask out embedding tokens in the labels.
        # full_labels = torch.cat([full_labels, labels], axis=1)
        full_labels = labels.clone()
        for i in range(batch_size):
            idx = (labels[i] == self.tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
            if len(idx) != 1:
                print(f'Warning: {len(idx)} instances of the CLS token found in the label.')
                raise ValueError
            # set the labels to -100 for the visual embeddings
            full_labels[i, idx] = -100

        pad_idx = []

        for label in full_labels:
            for k, token in enumerate(label):
                # Mask out retrieval token if it exists.
                if token in [self.tokenizer.pad_token_id, self.retrieval_token_idx]:
                    label[k:] = -100
                    pad_idx.append(k)
                    break
                if k == len(label) - 1:  # No padding found.
                    pad_idx.append(k + 1)
        assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)

        bs, seq_len, embs_dim = input_embs.shape
        if concat_captions:
            assert len(input_embs.shape) == 3, input_embs
            assert len(full_labels.shape) == 2, full_labels
            assert batch_size % 2 == 0
            all_concat_input_embs = []
            all_concat_labels = []

            # Rearrange embeddings and labels (and their padding) to concatenate captions.
            for i in range(batch_size // 2):
                first_idx = i * 2
                second_idx = first_idx + 1
                first_emb = input_embs[first_idx, :pad_idx[first_idx], :]
                first_labels = full_labels[first_idx, :pad_idx[first_idx]]
                first_padding = input_embs[first_idx, pad_idx[first_idx]:, :]
                first_labels_padding = full_labels[first_idx, pad_idx[first_idx]:]

                second_emb = input_embs[second_idx, :pad_idx[second_idx], :]
                second_labels = full_labels[second_idx, :pad_idx[second_idx]]
                second_padding = input_embs[second_idx, pad_idx[second_idx]:, :]
                second_labels_padding = full_labels[second_idx, pad_idx[second_idx]:]

                assert torch.all(first_labels_padding == -100), first_labels_padding
                assert torch.all(second_labels_padding == -100), second_labels_padding
                concat_input_embs = torch.cat([first_emb, second_emb, first_padding, second_padding],
                                              axis=0)  # (T*2, 768)
                concat_labels = torch.cat(
                    [first_labels, second_labels, first_labels_padding, second_labels_padding],
                    axis=0)  # (T*2, 768)
                all_concat_input_embs.append(concat_input_embs)
                all_concat_labels.append(concat_labels)

            # Pad to max length.
            input_embs = torch.stack(all_concat_input_embs, axis=0)  # (N/2, T*2, 768)
            full_labels = torch.stack(all_concat_labels, axis=0)  # (N/2, T*2, 768)
            assert input_embs.shape == (bs // 2, seq_len * 2, embs_dim), input_embs.shape
            assert full_labels.shape == (bs // 2, seq_len * 2), full_labels.shape

        output = self.lm(inputs_embeds=input_embs,
                         labels=full_labels,
                         output_hidden_states=True)


        ce_loss = output.loss
        ce_loss = ce_loss * self.args.cap_loss_scale
        loss = ce_loss

        return output, full_labels, visual_embs, loss


    def _forward_retrieval(
            self,
            labels: torch.LongTensor,
            input_embs: torch.FloatTensor,
            last_embedding_idx: torch.LongTensor,
            visual_embs: torch.FloatTensor,
            concat_captions: bool = False,
    ):

        ret_token_emb = input_embs[torch.arange(input_embs.shape[0]), last_embedding_idx, :]
        ret_tokens = labels[torch.arange(labels.shape[0]), last_embedding_idx]
        assert torch.all(ret_tokens == self.retrieval_token_idx), (ret_tokens, self.retrieval_token_idx)

        batch_size, vis_seq_len, _ = visual_embs.shape  # vis_seq_len = n_visual_tokens

        # remove the image token from the labels for the retrieval task
        labels, input_embs, last_embedding_idx = self.remove_img_tokens(labels, input_embs, last_embedding_idx)

        full_labels = torch.clone(labels)

        pad_idx = []
        for label in full_labels:
            for k, token in enumerate(label):
                if token == self.tokenizer.pad_token_id:
                    label[k:] = -100
                    pad_idx.append(k)
                    break
                if k == len(label) - 1:  # No padding found.
                    pad_idx.append(k + 1)
        assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)

        output = self.lm(inputs_embeds=input_embs,
                         labels=full_labels,
                         output_hidden_states=True)

        last_embedding = None
        last_output_logit = None
        hidden_states = []

        ce_loss = output.loss

        ce_loss = ce_loss * self.args.ret_loss_scale
        loss = ce_loss

        if self.args.shared_emb_dim is not None:
            for idx, fc_layer in zip(self.args.text_emb_layers, self.text_hidden_fcs):
                hidden_states.append(fc_layer(output.hidden_states[idx]))  # (N, seq_len, 2048)
        else:
            for idx in self.args.text_emb_layers:
                hidden_states.append(output.hidden_states[idx])

        # Add hidden states together.
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        if not concat_captions:
            last_embedding = torch.stack(
                [last_hidden_state[i, last_embedding_idx[i], :] for i in range(batch_size)], axis=0)  # (N, D)
            last_output_logit = torch.stack(
                [output.logits[i, last_embedding_idx[i] - 1, :] for i in range(batch_size)], axis=0)  # (N, D)
        else:
            raise NotImplementedError

        last_embedding = last_embedding / last_embedding.norm(dim=1, keepdim=True)
        # Compute retrieval loss.
        assert visual_embs.shape[1] == vis_seq_len, visual_embs.shape
        ret_loss = []
        for i in range(vis_seq_len):
            visual_embs_i = visual_embs[:, i, :]
            visual_embs_i = visual_embs_i / visual_embs_i.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            visual_embs_i = logit_scale * visual_embs_i

            # Compute InfoNCE loss
            caption_loss = info_nce(last_embedding, visual_embs_i, self.negative_queue_image if self.args.use_negatives else None)
            image_loss = info_nce(visual_embs_i, last_embedding, self.negative_queue_text if self.args.use_negatives else None)

            if self.args.use_negatives:
                self.negative_queue_text = self.update_negative_q(self.negative_queue_text, last_embedding)
                self.negative_queue_image = self.update_negative_q(self.negative_queue_image, visual_embs_i)

            ret_loss.append(self.args.ret_loss_scale * (caption_loss + image_loss) / 2.0)

        loss += torch.stack(ret_loss).sum() / len(ret_loss)

        # assert that the input embs at last_embedding_idx are still the same
        assert torch.all(input_embs[torch.arange(batch_size), last_embedding_idx, :] == ret_token_emb), (input_embs[torch.arange(batch_size), last_embedding_idx, :], ret_token_emb)
        # ditto for the labels
        assert torch.all(labels[torch.arange(batch_size), last_embedding_idx] == ret_tokens), (labels[torch.arange(batch_size), last_embedding_idx], ret_tokens)

        return output, full_labels, last_embedding, last_output_logit, visual_embs, loss

    def update_negative_q(self, queue, new_negatives):
        if queue is None:
            queue = new_negatives
        else:
            queue = torch.cat([queue, new_negatives], dim=0)
            if queue.size(0) > self.max_queue_size:
                queue = queue[-self.max_queue_size:]
        return queue

    def _forward_textgen(
            self,
            labels: torch.LongTensor,
            caption_len: torch.LongTensor,
            attention_mask: Optional[torch.LongTensor] = None,
    ):

        input_ids = labels.clone()

        # caption_len here is treated as the source size to mask the source tokens
        full_labels = labels.clone()
        for i in range(labels.shape[0]):
            full_labels[i, caption_len[i]:] = -100

        output = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=full_labels)

        loss = output.loss

        return output, full_labels, loss

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            labels: torch.LongTensor,
            caption_len: torch.LongTensor,
            mode: str = 'captioning',
            position_ids: Optional[torch.LongTensor] = None,
            concat_captions: bool = False,
            attention_mask: Optional[torch.LongTensor] = None,
            inference: bool = False,
    ):

        if self.negative_queue_text is not None:
            self.negative_queue_text = self.negative_queue_text.to(pixel_values.device).detach()
            self.negative_queue_image = self.negative_queue_image.to(pixel_values.device).detach()

        if mode == 'textgen':
            output, full_labels, loss = self._forward_textgen(labels, caption_len, attention_mask)
            return output, full_labels, None, None, None, loss

        if len(pixel_values.shape) > 4:
            # the shape is bs, n_frames, c, h, w, so we get the visual embeddings for sample in the batch separately and then we stack them
            visual_embs = []
            for i in range(pixel_values.shape[0]):
                visual_embs.append(self.get_visual_embs(pixel_values[i], mode, None if position_ids is None or mode != "retrieval" else position_ids[i]))
            visual_embs = torch.stack(visual_embs, dim=0).squeeze(2)
        else:
            visual_embs = self.get_visual_embs(pixel_values, mode)

        batch_size, vis_seq_len, _ = visual_embs.shape  # vis_seq_len = n_visual_tokens
        if labels is not None:
            assert labels.shape[0] == batch_size, (visual_embs.shape, labels.shape)

        input_embs = self.input_embeddings(labels)  # (N, T, D)

        last_embedding_idx = caption_len - 1  # -1 to retrieve the token before the eos token

        if mode == "retrieval":
            for i in range(labels.shape[0]):
                assert labels[i, last_embedding_idx[i]] == self.retrieval_token_idx, f"Retrieval token ({self.retrieval_token_idx}) not found in label {i} at position {last_embedding_idx[i]}, got {labels[i, last_embedding_idx[i]]} instead."

        loss = 0
        ret_loss = 0
        ce_loss = 0

        last_embedding = None
        last_output_logit = None
        hidden_states = []

        if mode == 'captioning':
            output, full_labels, visual_embs, ce_loss = self._forward_captioning(labels, input_embs, visual_embs, concat_captions)
            loss = ce_loss
        elif mode == 'retrieval':
            output, full_labels, last_embedding, last_output_logit, visual_embs, loss = self._forward_retrieval(labels, input_embs, last_embedding_idx, visual_embs, concat_captions)
            ret_loss = loss
        else:
            raise NotImplementedError

        return output, full_labels, last_embedding, last_output_logit, visual_embs, loss

    def generate(self, embeddings=torch.FloatTensor, max_len: int = 32,
                 temperature: float = 0.0, top_p: float = 1.0, min_word_tokens: int = 0,
                 ret_scale_factor: float = 1.0, filter_value: float = -float('Inf')):
        """Runs greedy decoding and returns generated captions.

        Args:
          embeddings: Input condition that the model uses for autoregressive generation.
          max_len: Maximum number of tokens to generate.
          temperature: Used to modulate logit distribution.
          top_p: If set to < 1, the smallest set of tokens with highest probabilities that add up to top_p or higher are kept for generation.
          min_word_tokens: Minimum number of words to generate before allowing a [RET] output.
          ret_scale_factor: Proportion to scale [RET] token logits by. A higher value may increase the probability of the model generating [RET] outputs.
          filter_value: Value to assign to tokens that should never be generated.
        Outputs:
          out: (N, T) int32 sequence of output tokens.
          output_embeddings: (N, T, 256) sequence of text output embeddings.
        """
        self.lm.eval()

        with torch.no_grad():  # no tracking history
            batch_size, s, _ = embeddings.shape
            # init output with image tokens
            out = None
            past_key_values = None
            output_embeddings = []
            output_logits = []

            for i in range(max_len):
                if 'opt' in self.text_decoder:
                    output = self.lm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
                else:
                    if i == 0:
                        output = self.lm(inputs_embeds=embeddings, use_cache=True, past_key_values=None,
                                         output_hidden_states=True)
                    else:
                        output = self.lm(input_ids=out[:, -1:], use_cache=True, past_key_values=past_key_values,
                                         output_hidden_states=True)

                # Collect and sum the hidden states.
                hidden_states = []
                if self.args.shared_emb_dim is not None:
                    for idx, fc_layer in zip(self.args.text_emb_layers, self.text_hidden_fcs):
                        hidden_states.append(fc_layer(output.hidden_states[idx]))  # (N, seq_len, 2048)
                else:
                    for idx in self.args.text_emb_layers:
                        hidden_states.append(output.hidden_states[idx])
                # Add hidden states together.
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)  # (N, T, 256)
                last_embedding = last_hidden_state / last_hidden_state.norm(dim=-1, keepdim=True)
                output_embeddings.append(last_embedding)

                logits = output.logits[:, -1, :]  # (N, vocab_size)
                if top_p == 1.0:
                    logits = logits.cpu()
                output_logits.append(logits)

                if self.retrieval_token_idx != -1 and self.retrieval_token_idx is not None:
                    if i < min_word_tokens:
                        # Eliminate probability of generating [RET] if this is earlier than min_word_tokens.
                        logits[:, self.retrieval_token_idx] = filter_value
                    else:
                        # Multiply by scaling factor.
                        logits[:, self.retrieval_token_idx] = logits[:, self.retrieval_token_idx] * ret_scale_factor

                past_key_values = output.past_key_values

                if temperature == 0.0:
                    if top_p != 1.0:
                        raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                else:
                    logits = logits / temperature

                    # Apply top-p filtering.
                    if top_p < 1.0:
                        assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # (N, D)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        for j in range(sorted_indices.shape[0]):
                            indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
                            logits[j, indices_to_remove] = filter_value

                    token_weights = logits.exp()  # (N, vocab_size)
                    # remove any nan, inf or < 0 values
                    token_weights[~torch.isfinite(token_weights)] = 0
                    token_weights[token_weights < 0] = 0
                    next_token = torch.multinomial(token_weights, 1)  # (N, 1)

                next_token = next_token.long().to(embeddings.device)
                if out is not None:
                    out = torch.cat([out, next_token], dim=-1)
                else:
                    out = next_token

                # print("Generated token: ", self.tokenizer.decode(next_token[0].tolist()))

                if 'opt' in self.text_decoder:
                    next_embedding = self.input_embeddings(next_token)
                    embeddings = torch.cat([embeddings, next_embedding], dim=1)
                elif (self.tokenizer.eos_token_id and (next_token == self.tokenizer.eos_token_id).all()):
                    # End of generation.
                    break

        return out, output_embeddings, output_logits


class MMPlanLLM(nn.Module):
    def __init__(self, tokenizer, model_args: Optional[FrozenArgs] = None):
        super().__init__()
        self.model = MMPlanLLMModel(tokenizer, model_args)

    def __call__(self, images: Tensor, tgt_tokens: Optional[Tensor] = None, caption_len: Optional[Tensor] = None,
                 generate: bool = False, num_words: int = 32, temperature: float = 1.0, top_p: float = 1.0,
                 ret_scale_factor: float = 1.0, min_word_tokens: int = 0,
                 mode: str = 'captioning', concat_captions: bool = False,
                 inference: bool = False, attention_mask: Optional[Tensor] = None,
                 position_ids: Optional[Tensor] = None) -> Tensor:
        if generate:
            return self.model.generate(images, num_words, temperature=temperature, top_p=top_p,
                                       min_word_tokens=min_word_tokens, ret_scale_factor=ret_scale_factor, filter_value=0.0)
        else:
            output = self.model(
                pixel_values=images,
                labels=tgt_tokens,
                caption_len=caption_len,
                mode=mode,
                concat_captions=concat_captions,
                inference=inference,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            return output


def truncate_caption(caption: str) -> str:
    """Truncate captions at periods and newlines."""
    caption = caption.strip('\n')
    trunc_index = caption.find('\n') + 1
    if trunc_index <= 0:
        trunc_index = caption.find('.') + 1
    if trunc_index > 0:
        caption = caption[:trunc_index]
    return caption


def get_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img = img.convert('RGB')
    return img


def load_mmplanllm(model_dir: str, run_name='pretrained_ckpt.pth.tar') -> MMPlanLLM:
    # adapted from fromage
    model_args_path = os.path.join(model_dir, 'model_args.json')
    model_ckpt_path = os.path.join(model_dir, run_name)

    if not os.path.exists(model_args_path):
        if os.path.exists(os.path.join(os.path.dirname(model_dir), 'model_args.json')):
            model_args_path = os.path.join(os.path.dirname(model_dir), 'model_args.json')
        else:
            raise ValueError(f'model_args.json does not exist in {model_dir} or its parent directory.')
    if not os.path.exists(model_ckpt_path):
        raise ValueError(f'{run_name} does not exist in {model_dir}.')

    with open(model_args_path, 'r') as f:
        model_kwargs = json.load(f)

    # Initialize tokenizer.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    except Exception as e:
        tokenizer = AutoTokenizer.from_pretrained(model_kwargs['text_decoder'], use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Add special tokens to the model to enable [RET].
        tokenizer.add_special_tokens({"cls_token": "<|image|>"})
        tokenizer.add_tokens('[RET]')

        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'

    ret_token_idx = tokenizer.get_vocab().get('[RET]', None)
    assert ret_token_idx is not None and isinstance(ret_token_idx, int), 'Retrieval token not found in tokenizer vocab.'
    model_kwargs['retrieval_token_idx'] = ret_token_idx
    if 'use_pos_emb' not in model_kwargs:
        model_kwargs['use_pos_emb'] = False
    args = namedtuple('args', model_kwargs)(**model_kwargs)

    # Initialize model for inference.
    model = MMPlanLLM(tokenizer, args)

    # Load pretrained linear mappings and [RET] embeddings.
    checkpoint = torch.load(model_ckpt_path, map_location='cpu' if not torch.cuda.is_available() else None)

    is_lora = False
    if any(["base_model" in k for k in checkpoint['state_dict'].keys()]):
        print("DETECTED BASE MODEL IN CHECKPOINT, LOADING LORA MODEL")
        lora_kwargs = {
            'lora_rank': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.1,
        }
        lora_args_path = os.path.join(model_dir, 'lora_args.json')
        if not os.path.exists(lora_args_path):
            lora_args_path = os.path.join(model_dir, '..', 'lora_args.json')
        if os.path.exists(lora_args_path):
            with open(lora_args_path, 'r') as f:
                lora_kwargs = json.load(f)
        model.model.make_lm_lora(lora_kwargs['lora_rank'], lora_kwargs['lora_alpha'], lora_kwargs['lora_dropout'])
        is_lora = True

    model.eval()
    model.bfloat16()

    first_key = next(iter(checkpoint['state_dict'].keys()))
    if 'module' in first_key:
        print('Removing DataParallel module from keys')
        checkpoint['state_dict'] = filter_module_from_keys(checkpoint['state_dict'])

    model.load_state_dict(checkpoint['state_dict'], strict=True)

    if is_lora:
        model.model.merge_lm_lora()

    logit_scale = model.model.logit_scale.exp()

    return model


# util function to remove 'module.' from keys in state dict
def filter_module_from_keys(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

