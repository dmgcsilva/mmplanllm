import json
import os
import time

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from . import tokenization_utils
from .data_constants import IGNORE_INDEX

PLANGPT_PROMPT = ("""<|prompter|> You are a taskbot tasked with helping users cook recipes or DIY projects. I will give you a recipe and I want you to help me do it step by step. You should always be empathetic, honest, and should always help me. If I ask you something that does not relate to the recipe you should politely reject the request and try too get me focused on the recipe. I am unsure how to cook something or do something related to the recipe you should help me to the best of your ability. Please use a {system_tone} tone of voice. Recipe: {title} Steps: {steps} <|endofturn|> <|prompter|> {current_step} <|endofturn|> <|assistant|> ok! <|endofturn|> {dialog_history} <|prompter|> {request} <|endofturn|> <|assistant|> """)

DIALOG_HISTORY_TEMPLATE = ("""<|prompter|> {previous_request} <|endofturn|> <|assistant|> {previous_response} <|endofturn|> """)

HASNT_STARTED = "I haven't started cooking yet."
IS_DOING_STEP = "I am currently on Step {step_number}: {step_text}"

CAPTION_TEMPLATE = "<|uploadedimage|> {caption} <|endofuploadedimage|> "


class MMPlanLLMDataset(Dataset):

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, debug=False, **kwargs):

        self.kwargs = kwargs
        self.debug = debug

        # load files
        with open(data_path, "r") as infile:
            dialogs = json.load(infile)

        if debug:
            # Dialogs is a dict with n dicts inside
            # We only want the first 32 dialogs
            dialogs = {k: v for k, v in list(dialogs.items())[:32]}

        self.raw_sources = []
        self.raw_targets = []
        self.intents = []
        self.dialog_ids = []
        self.turn_numbers = []
        self.images = []
        self.modes = []
        self.target_step_text = []
        self.frame_limits = []
        self.position_ids = []  # for retrieval turns, these hold the position offset of the first frame, to be used with positional embeddings

        self.context_size = kwargs.get("context_size", 1)

        self.tokenizer = tokenizer

        self.feature_extractor_model = kwargs.get('feature_extractor_model', 'google/vit-base-patch16-224')
        self.image_size = kwargs.get('image_size', 224)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.feature_extractor_model,
                                                                      do_resize=True,
                                                                      size=self.image_size)
        self.max_len = kwargs.get("max_len", self.tokenizer.model_max_length)

        self.build_raw_samples(dialogs, tokenizer)

        self.is_test = kwargs.get("is_test", any([t in data_path for t in ['_test', '_val', '_eval']]))

        if kwargs.get('max_samples', None) is not None:
            if kwargs['max_samples'] < len(self.raw_sources):
                self.raw_sources = self.raw_sources[:kwargs['max_samples']]
                self.raw_targets = self.raw_targets[:kwargs['max_samples']]

        self.vis_embs_cache = dict()

    def build_raw_samples(self, dialogs, tokenizer: transformers.PreTrainedTokenizer):
        prompt_template = PLANGPT_PROMPT
        dialog_history_template = DIALOG_HISTORY_TEMPLATE
        hasn_started = HASNT_STARTED
        is_doing_step = IS_DOING_STEP

        only_visual = self.kwargs.get("only_visual", False)
        only_text = self.kwargs.get("only_text", False)

        assert not (only_visual and only_text), "Only one of only_visual or only_text can be True, but got both True."

        target_template = "{system_request} <|endofturn|> {eos_token}"

        context_size = self.kwargs.get("context_size", 1)

        for did, d in tqdm(dialogs.items(), desc="Building samples"):
            # load recipe
            recipe = d['task']["recipe"]
            # get title
            title = recipe['displayName']
            # get steps
            steps = recipe['instructions']  # Type: List[Dict]
            # Steps list to string
            steps_list_text = "\n".join(
                [f"Step {step_number+1}: {step_dict['stepText']}" for step_number, step_dict in enumerate(steps)])
            # get turns
            turns = d['dialog']  # Type: List[Dict]

            current_step_number = 0
            for turn_number, turn_dict in enumerate(turns):
                skip = False
                # current_step_number = turn_dict['current_step']
                user_request = turn_dict['user']
                system_request = turn_dict['system']
                image = turn_dict['relevant_image'] if "visual_request" in turn_dict and turn_dict["visual_request"] else None

                if only_visual and not turn_dict["visual_request"]:
                    current_step_number = turn_dict['current_step']
                    continue

                is_text = turn_dict['intent'] not in ["ARTIFICIAL.VisualStepRetrievalIntent", "ARTIFICIAL.VisualMomentRetrievalIntent"]
                if only_text and not is_text:
                    current_step_number = turn_dict['current_step']
                    continue

                dialog_history_text = ""
                if current_step_number == 0 and turn_number == 0:
                    current_step_text = hasn_started
                else:
                    current_step_text = is_doing_step.format(step_number=current_step_number + 1, step_text=steps[current_step_number]['stepText'])
                    for i in range(min([context_size, turn_number])):
                        intent_text = ""
                        caption_text = ""
                        dialog_history_text = dialog_history_template.format(
                            previous_request= caption_text + turns[turn_number - (i+1)]['user'],
                            previous_response=intent_text + turns[turn_number - (i+1)]['system']
                        ) + dialog_history_text

                        if (only_visual or only_text) and turns[turn_number - (i+1)]["visual_request"]:
                            skip = True
                            break
                        if is_text and "visual_request" in turns[turn_number - (i+1)] and turns[turn_number - (i+1)]['visual_request']:
                            # for text turns, if there is a visual request in the history, skip
                            skip = True
                            break
                if skip:
                    current_step_number = turn_dict['current_step']
                    continue

                mode = ["textgen"]
                if turn_dict["intent"] == "ARTIFICIAL.VisualStepRetrievalIntent":
                    mode = ["captioning"]
                elif turn_dict["intent"] == "ARTIFICIAL.VisualMomentRetrievalIntent":
                    mode = ["retrieval"]
                        
                prompt = prompt_template.format(
                    system_tone=d['system_tone'].replace("_", " "),
                    title=title,
                    steps=steps_list_text,
                    current_step=current_step_text,
                    dialog_history=dialog_history_text,
                    request=user_request
                ).replace("..", ".").replace("  ", " ")
                prompt = prompt.replace("\n", " ")

                if image:  # fix image path typo
                    image = image.replace("frames_reduced_reduced_reduced", "frames_reduced").replace("frames_reduced_reduced", "frames_reduced")

                if is_text:
                    assert mode == ["textgen"], f"Mode is not textgen for text turn: {mode}"

                self.raw_sources.append(prompt.strip())
                target = target_template.format(system_request=system_request, eos_token=tokenizer.eos_token)
                self.raw_targets.append(target.replace("  ", " ").strip())
                self.intents.append(turn_dict['intent'])
                self.dialog_ids.append(did)
                self.turn_numbers.append(turn_number)
                self.images.append(image if image is None else [image] if mode[0] != "retrieval" else self.get_adjacent_frames(image, n=self.kwargs.get("n_adjacent_frames", 0)))
                self.modes.append(mode)
                self.target_step_text.append(steps[turn_dict['current_step']]['stepText'] if mode[0] in ["captioning", "retrieval"] else None)

                turn_frame_limits = None
                frame_offsets = 0
                if mode[0] == "retrieval":
                    # for retrieval frame_limits are not none
                    # get frame limits
                    step_actions = steps[current_step_number]['actions']
                    if len(step_actions) == 1:
                        turn_frame_limits = {
                            "start": step_actions[0]["startFrame"],
                            "middle": step_actions[0]["middleFrame"],
                            "end": step_actions[0]["endFrame"]
                        }
                        assert int(image.split("/")[-1].split(".")[0]) == turn_frame_limits["middle"], f"Image name {image.split('/')[-1].split('.')[0]} does not match middle frame {turn_frame_limits['middle']}"
                    else:
                        # find the action that matches the image
                        image_idx = int(image.split("/")[-1].split(".")[0])
                        for action in step_actions:
                            if action["middleFrame"] == image_idx:
                                turn_frame_limits = {
                                    "start": action["startFrame"],
                                    "middle": action["middleFrame"],
                                    "end": action["endFrame"]
                                }
                                break
                        if turn_frame_limits is None:
                            print(f"Could not find frame limits for image {image} in step {current_step_number+1}")
                            print(f"Step actions: {step_actions}")
                            print(f"Image idx: {image_idx}")
                            print(f"Current step number: {current_step_number}")
                            print(f"Image: {image}")
                            print(f"Turn number: {turn_number}")
                            print(f"Turn dict: {turn_dict}")
                            print(f"Turn frame limits: {turn_frame_limits}")
                            raise ValueError(f"Could not find frame limits for image {image} in step {current_step_number+1}")

                    if turn_frame_limits["start"] is None and turn_frame_limits["end"] is not None and turn_frame_limits["middle"] is not None:
                        turn_frame_limits["start"] = turn_frame_limits["middle"] - (turn_frame_limits["end"] - turn_frame_limits["middle"])
                    elif turn_frame_limits["start"] is not None and turn_frame_limits["end"] is None and turn_frame_limits["middle"] is not None:
                        # find the start frame of the next action
                        recipe_actions = [inst["actions"] for inst in steps]
                        for action in recipe_actions[current_step_number]:
                            if action["startFrame"] > turn_frame_limits["middle"]:
                                turn_frame_limits["end"] = action["startFrame"]
                                break
                        if turn_frame_limits["end"] is None:
                            # find the last frame of the video
                            recipe_folder = "/".join(image.split("/")[:-1])
                            frames = [int(f.split(".")[0]) for f in os.listdir(recipe_folder) if f.endswith(".jpg")]
                            turn_frame_limits["end"] = max(frames)
                    elif turn_frame_limits["start"] is not None and turn_frame_limits["end"] is not None and turn_frame_limits["middle"] is None:
                        turn_frame_limits["middle"] = (turn_frame_limits["start"] + turn_frame_limits["end"]) // 2

                    if any([v is None for v in turn_frame_limits.values()]):
                        print(f"Turn frame limits: {turn_frame_limits}")
                        print(f"Step actions: {step_actions}")
                        raise ValueError(f"Turn frame limits for image {image} in step {current_step_number+1} are not complete")

                    # get the position offset of the middle frame
                    frame_offsets = max(0, turn_frame_limits["middle"] - self.kwargs.get("n_adjacent_frames", 0))

                self.frame_limits.append(turn_frame_limits)
                self.position_ids.append(frame_offsets)
                current_step_number = turn_dict['current_step']

        print(self.raw_sources[10])
        print(self.raw_targets[10])
        print("#####")
        print(self.raw_sources[50])
        print(self.raw_targets[50])
        print("#####")

    def __getitem__(self, item):

        source = self.raw_sources[item]
        target = self.raw_targets[item]
        mode = self.modes[item][0]

        # get source size
        source_size = len(self.tokenizer.encode(source))
        if mode == "captioning":
            assert self.intents[item] == "ARTIFICIAL.VisualStepRetrievalIntent", f"Intent is not VisualStepRetrievalIntent for captioning mode, its {self.intents[item]}"

        if mode == "retrieval":
            seq = f"{source}{' '.join([target.split('<|endofturn|>')[0].strip(), '[RET] <|endofturn|>', target.split('<|endofturn|>')[1]])}"
            if self.is_test:
                seq = f"{source}{target.split('<|endofturn|>')[0].strip()}"
        elif mode == "captioning":
            seq = f"{source}{self.tokenizer.cls_token}{target}"
        else:
            seq = f"{source}{target}"
            if self.is_test:
                seq = source

        self.tokenizer.truncation_side = "left"

        # encode
        tokenized_seq = self.tokenizer(
            seq,
            return_tensors="pt",
            padding='max_length' if not self.is_test else 'do_not_pad',
            truncation=True,
            max_length=self.max_len
        )
        input_ids = tokenized_seq["input_ids"][0]
        attention_mask = tokenized_seq["attention_mask"][0]

        caption_len = attention_mask.sum()

        if mode == "retrieval" and not self.is_test:
            # in this case caption_len is the position of the [RET] token
            caption_len = (input_ids == self.tokenizer.convert_tokens_to_ids("[RET]")).nonzero(as_tuple=True)[0].item() + 1
            assert input_ids[caption_len-1] == self.tokenizer.convert_tokens_to_ids("[RET]"), f"Token at position {caption_len-1} is not the [RET] token"

        # if it was truncated we want to determine the point where the truncation happened
        # so that we know where the target starts for masking

        if not self.is_test:
            if caption_len == self.max_len:
                # update the source size because it was truncated
                source_size = self.max_len - len(self.tokenizer.encode(target))

        # there are no pixel values in this dataset but it has to be passed otherwise it will break
        pixel_values = torch.tensor([])
        if self.images[item]:
            pixel_values = []
            for img in self.images[item]:
                img = Image.open(img)
                pixel_values.append(self.feature_extractor(img.convert('RGB'), return_tensors="pt").pixel_values[0, ...])
            pixel_values = torch.stack(pixel_values, dim=0)

        frame_count = len(self.images[item]) if self.images[item] else 0

        if self.is_test:
            return dict(tgt_tokens=input_ids, images=pixel_values, attention_mask=attention_mask,
                        caption_len=caption_len if mode == "retrieval" else source_size,
                        supported_tasks=[mode], frame_limits=self.frame_limits[item], frame_count=frame_count)

        item_dict =  dict(tgt_tokens=input_ids, images=pixel_values, attention_mask=attention_mask,
                    caption_len=caption_len if mode == "retrieval" else source_size,
                    supported_tasks=[mode], frame_count=frame_count)

        if mode == 'retrieval':
            actual_pos_ids = [self.position_ids[item] + i for i in range(len(self.images[item]))]
            item_dict['position_ids'] = torch.tensor(actual_pos_ids)

        return item_dict

    def __len__(self):
        return len(self.raw_sources)

    def get_adjacent_frames(self, image:str, n:int=0):
        frame_idx = int(image.split("/")[-1].split(".")[0])
        frame_folder = "/".join(image.split("/")[:-1])

        num_of_frames = len([f for f in os.listdir(frame_folder) if f.endswith('.jpg')])

        frames_idxs = [max(min(frame_idx + i, num_of_frames - 1), 0) for i in range(-n, n+1)]

        frames = [f"{frame_folder}/{f:05}.jpg" for f in frames_idxs]

        assert all([os.path.exists(f) for f in frames]), f"Not all frames exist: {frames}"

        return frames

    def get_relevant_visual_embs(self, model, idx):
        # Given an idx it return the visual embeddings for every frame that belongs to the same video

        # get video directory (the folder in which the frames are stored)
        img_path = self.images[idx][0]
        if img_path is None:
            raise ValueError(f"No image path found for item at index {idx}")

        video_dir = os.path.dirname(img_path)

        frame_files = [f for f in os.listdir(video_dir) if f.endswith('.jpg')]
        # sort the frames so that the frame at 0 is the first frame in the video (000000.jpg)
        frame_files.sort()

        recipe_name = video_dir.split('/')[-2]

        if recipe_name in self.vis_embs_cache:
            return self.vis_embs_cache[recipe_name]

        if hasattr(model, "model"):
            device = model.model.lm.device
        else: # is clip model
            device = model.device

        start_time = time.time()

        pixel_values = []

        for img in frame_files:
            try:
                images = self.feature_extractor(Image.open(os.path.join(video_dir, img)).convert('RGB'), return_tensors="pt").pixel_values[0, ...].to(device)
                pixel_values.append(images)
            except Exception as e:
                print(f'Error reading {img}: {e}, continuing...')
                continue

        pixel_values = torch.stack(pixel_values, dim=0)

        with torch.no_grad():
            # check if there is a model inside model
            if hasattr(model, "model"):
                vis_embs = model.model.get_visual_embs(pixel_values, mode="retrieval").squeeze(1)
            else: # is a clip model
                vis_embs = image_features = model.get_image_features(pixel_values).to(device)

        self.vis_embs_cache[recipe_name] = vis_embs


        return vis_embs

    def get_recipe_frames_count(self, idx):
        # Given an idx it return the number of frames that belongs to the same video

        # get video directory (the folder in which the frames are stored)
        img_path = self.images[idx]
        if img_path is None:
            raise ValueError(f"No image path found for item at index {idx}")

        video_dir = os.path.dirname(img_path)

        frame_files = [f for f in os.listdir(video_dir) if f.endswith('.jpg')]
        return len(frame_files)


class MMPlanLLMModeSampler(torch.utils.data.Sampler):

    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.modes_types = list(set([m[0] for m in dataset.modes]))  # List[str]
        self.order = []
        self.batch_size = batch_size

        print(f"Modes: {self.modes_types}")

        self._build_order_list()


    def _build_order_list(self):
        # iterate through the dataset and build the order list so that each batch has only one mode
        mode_lists = {mode: [] for mode in self.modes_types}
        for i in range(len(self.dataset)):
            mode = self.dataset.modes[i]
            mode_lists[mode[0]].append(i)

        print(f"Total valid items b4 prune: {sum([len(mode_lists[mode]) for mode in mode_lists])}")

        # remove the end of the lists so that they are divisible by the batch size
        for mode in mode_lists:
            old_size = len(mode_lists[mode])
            new_size = (old_size // self.batch_size) * self.batch_size
            mode_lists[mode] = mode_lists[mode][:new_size]
            print(f"Mode {mode} pruned from {old_size} to {new_size}")

        print(f"Total valid items: {sum([len(mode_lists[mode]) for mode in mode_lists])}")

        # Calculate total batches for each mode
        total_batches = {mode: len(mode_lists[mode]) // self.batch_size for mode in mode_lists}

        # Determine how often to insert batches of each mode
        min_batches = min(total_batches.values())
        interleave_intervals = {mode: total_batches[mode] // min_batches for mode in mode_lists}

        print(f"Interleave intervals: {interleave_intervals}")

        # Initialize counters and order list
        self.order = []
        mode_batch_counters = {mode: 0 for mode in mode_lists}

        # Interleave batches from different modes
        while any(mode_batch_counters[mode] < total_batches[mode] for mode in mode_lists):
            for mode in sorted(mode_lists, key=lambda x: interleave_intervals[x]):
                if mode_batch_counters[mode] < total_batches[mode]:
                    start_index = mode_batch_counters[mode] * self.batch_size
                    end_index = start_index + (self.batch_size * interleave_intervals[mode])
                    self.order.extend(mode_lists[mode][start_index:min(end_index, len(mode_lists[mode]))])  # Add interleave_interval batches of this mode

                    # Advance the insertion point by the interleave interval times batch size
                    mode_batch_counters[mode] += interleave_intervals[mode]
                    if mode_batch_counters[mode] >= total_batches[mode]:
                        # If we've scheduled all batches for this mode, stop trying to add more
                        mode_batch_counters[mode] = total_batches[mode]

        print(f"Total items in order list: {len(self.order)}")

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)

    def __getitem__(self, item):
        return self.dataset[item]
