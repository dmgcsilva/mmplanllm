import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageFont
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from tqdm import tqdm
from transformers import AutoFeatureExtractor
import transformers
import torch


class VMRDataset(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):

        self.kwargs = kwargs
        self.debug = debug

        print(f"Loading data from {data_path}")

        df = pd.read_csv(data_path)

        self.base_image_dir = kwargs.get('base_image_dir', '')
        self.start_frames = df['start_frame'].tolist()
        self.end_frames = df['end_frame'].tolist()
        self.middle_frames = df['middle_frame'].tolist()
        self.instructions = df['instruction'].tolist()
        assert len(self.start_frames) == len(self.end_frames) == len(self.instructions), \
            f"Length mismatch: {len(self.start_frames)}, {len(self.end_frames)}, {len(self.instructions)}"

        self.feature_extractor_model = kwargs.get('feature_extractor_model', 'google/vit-base-patch16-224')
        self.image_size = kwargs.get('image_size', 224)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.feature_extractor_model,
                                                                      do_resize=True,
                                                                      size=self.image_size)
        self.tokenizer = tokenizer
        self.retrieval_token_idx = tokenizer.get_vocab().get('[RET]')
        self.image_token_idx = tokenizer.cls_token_id

        print(f"Retrieval token idx: {self.retrieval_token_idx}")
        print(f"Image token idx: {self.image_token_idx}")
        print(f"Tokenizer padding side: {self.tokenizer.padding_side}")
        print(f"Tokenizer truncation side: {self.tokenizer.truncation_side}")

        assert self.retrieval_token_idx is not None, "Tokenizer must have a [RET] token"
        assert self.image_token_idx is not None, "Tokenizer must have a [CLS] token"

        self.max_len = kwargs.get('max_len', self.tokenizer.model_max_length)
        print(f"Max len: {self.max_len}")

        self.font = None

        self.is_test = kwargs.get('is_test', False) or "_val" in data_path or "_test" in data_path or "_eval" in data_path

        self.recipe_embs_cache = dict()

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        while True:

            image_path = os.path.join(self.base_image_dir, str(self.middle_frames[idx]))
            caption = str(self.instructions[idx])
            try:
                img = Image.open(image_path)
                images = self.feature_extractor(img.convert('RGB'), return_tensors="pt").pixel_values[0, ...]

                caption = f"{self.tokenizer.cls_token}{caption}" if self.is_test else f"{self.tokenizer.cls_token}{caption}[RET]"

                tokenized_data = self.tokenizer(
                    caption,
                    return_tensors="pt",
                    padding='max_length' if not self.is_test else 'do_not_pad',
                    truncation=True,
                    max_length=self.max_len,
                )
                tokens = tokenized_data.input_ids[0]
                attn_mask = tokenized_data.attention_mask[0]

                if self.is_test:
                    # for test mode, we do not need to add the RET or EOS tokens as they should be generated by the model
                    caption_len = tokenized_data.attention_mask[0].sum()
                    assert all([t != self.retrieval_token_idx for t in tokens])
                else:
                    caption_len = tokenized_data.attention_mask[0].sum()  # subtract 1 to account for the eos token

                    if tokens[caption_len - 1] not in [self.retrieval_token_idx]:
                        tokens[caption_len - 1] = self.retrieval_token_idx

                    # assert that the token at caption_len is the [RET] token and that the first token in the [IMG] token
                    assert tokens[
                               caption_len - 1] == self.retrieval_token_idx, f"Token at caption_len is not [RET] token: {tokens[caption_len - 1]}"

                assert tokens[1] == self.image_token_idx or tokens[0] == self.image_token_idx, f"Token at 0 or 1 is not [IMG] token: {tokens} caption: {caption}"

                if self.is_test:
                    # if is test then we want to also return a dict with the target index of the start, middle, and end frame
                    # so that we can evaluate the retrieval performance. To do this we need to extract the file name and
                    # convert it to an integer
                    frame_limits = {"start": 0, "middle":0, "end": 0}
                    frame_limits["start"] = int(os.path.basename(self.start_frames[idx]).split('.')[0])
                    frame_limits["middle"] = int(os.path.basename(self.middle_frames[idx]).split('.')[0])
                    frame_limits["end"] = int(os.path.basename(self.end_frames[idx]).split('.')[0])

                    return dict(images=images, tgt_tokens=tokens, caption_len=caption_len,
                                supported_tasks=['retrieval'], frame_limits=frame_limits)

                return dict(images=images, tgt_tokens=tokens, caption_len=caption_len,
                            supported_tasks=['retrieval', 'captioning'])

            except Exception as e:
                # raise e
                print(f'Error reading {image_path} with caption {caption}: {e}')
                # Pick a new example at random.
                idx = np.random.randint(0, len(self) - 1)

    def get_all_visual_embs(self, model):
        pixel_values = []
        for i in range(len(self.start_frames)):
            pix = []
            img = self.start_frames[i]
            try:
                images = self.feature_extractor(Image.open(os.path.join(self.base_image_dir, str(img))).convert('RGB'), return_tensors="pt").pixel_values[0, ...]
                pix.append(images.unsqueeze(0))
            except Exception as e:
                print(f'Error reading {img}: {e}, continuing...')
                continue

            img = self.end_frames[i]
            try:
                images = self.feature_extractor(Image.open(os.path.join(self.base_image_dir, str(img))).convert('RGB'), return_tensors="pt").pixel_values[0, ...]
                pix.append(images.unsqueeze(0))
            except Exception as e:
                print(f'Error reading {img}: {e}, continuing...')
                continue

            pixel_values.append(torch.cat(pix, dim=0))

        for i in range(len(pixel_values)):
            pixel_values[i] = model.model.get_visual_embs(pixel_values[i], mode='retrieval')

        return torch.stack(pixel_values, dim=0)

    def get_relevant_visual_embs(self, model, idx):
        # Given an idx it return the visual embeddings for every frame that belongs to the same video

        # get video directory (the folder in which the frames are stored)
        video_dir = os.path.dirname(os.path.join(self.base_image_dir, str(self.start_frames[idx])))

        vis_embs = []

        frame_files = [f for f in os.listdir(video_dir) if f.endswith('.jpg')]
        # sort the frames so that the frame at 0 is the first frame in the video (000000.jpg)
        frame_files.sort()

        recipe_name = video_dir.split('/')[-2]

        if recipe_name in self.recipe_embs_cache:
            return self.recipe_embs_cache[recipe_name]

        device = model.model.lm.device

        start_time = time.time()

        for img in frame_files:
            try:
                images = self.feature_extractor(Image.open(os.path.join(video_dir, img)).convert('RGB'), return_tensors="pt").pixel_values[0, ...].to(device)
                emb = model.model.get_visual_embs(images.unsqueeze(0), mode='retrieval')
                vis_embs.append(emb)
            except Exception as e:
                print(f'Error reading {img}: {e}, continuing...')
                continue

        vis_embs = torch.stack(vis_embs, dim=0).squeeze(1).squeeze(1)
        self.recipe_embs_cache[recipe_name] = vis_embs

        return vis_embs
