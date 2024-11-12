import ast
import json
import os

import numpy as np
import torch
import transformers
from tqdm import tqdm

from data_binding import ModelArguments, InferenceArguments, DataArguments
from data_mod import data_utils, mmplanllm_dataset
from inference import inference_utils
from models.modelling_mmplanllm import load_mmplanllm, MMPlanLLM


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments, DataArguments))
    model_args, inference_args, data_args = parser.parse_args_into_dataclasses()  # type: ModelArguments, InferenceArguments, DataArguments

    if model_args.ckpt_path is None or model_args.ckpt_path == "":
        print(f"ERROR: Please provide a valid ckpt file path. Received: {model_args.ckpt_path}")
        return

    print(f"Loading model and tokenizer from {model_args.ckpt_path}")

    model_args.ckpt_path = model_args.ckpt_path[:-1] if model_args.ckpt_path[-1] == '/' else model_args.ckpt_path

    output_file = model_args.ckpt_path.split('/')[-1]

    output_file = os.path.join(model_args.ckpt_path, f'{output_file}_vmr_outputs.json')

    print(f"Output file: {output_file}")

    # check if parent folder of output file exists
    if not os.path.exists(os.path.dirname(output_file)):
        print(f"Folder {os.path.dirname(output_file)} does not exist. Aborting")
        return

        # load model and tokenizer
    with torch.inference_mode():
        model_dir = model_args.ckpt_path if os.path.isdir(model_args.ckpt_path) else os.path.dirname(
            model_args.ckpt_path)
        run_name = "pretrained_model.pth.tar" if os.path.isdir(model_args.ckpt_path) else os.path.basename(
            model_args.ckpt_path)
        model = load_mmplanllm(model_dir, run_name)  # type: MMPlanLLM
        tokenizer = model.model.tokenizer

        print(f"CLS token id: {tokenizer.cls_token_id}")
        print(f"Model img token id: {model.model.image_token}")

        try:
            model.model.merge_lm_lora()
        except Exception as e:
            print(f"Error: {e}")

        device = torch.device("cuda")
        print(f"Using device: {device}")
        model.to(device)

        model.bfloat16()
        model.eval()

        tokenizer.model_max_length = 2048

        print(f"Tokenizer padding side: {tokenizer.padding_side}")
        print(f"Tokenizer model max length: {tokenizer.model_max_length}")

        dataset_kwargs = ast.literal_eval(data_args.dataset_kwargs)
        dataset_kwargs['max_samples'] = inference_args.max_samples
        dataset_kwargs['is_test'] = True

        print(f"Dataset kwargs: {dataset_kwargs}")

        # load test dataset
        test_dataset = data_utils.load_test_dataset(tokenizer, inference_args, data_args, model_args, **dataset_kwargs)  # type: mmplanllm_dataset.MMPlanLLMDataset


        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                tfile = json.load(f)['test_file']
            if tfile != inference_args.test_file:
                # delete the file
                os.remove(output_file)
                os.remove(output_file.replace('_outputs.json', '_metrics.json'))

        max_samples = inference_args.max_samples if inference_args.max_samples is not None else len(test_dataset)

        all_target_vis_embeds = []

        all_ret_embeds = []
        all_visual_embeds = []

        frame_limits = []

        video_sizes = []

        image_paths = []
        sources = []
        targets = []
        target_step_texts = []
        dev = +1
        print(f"deviation: {dev}")
        # inference
        with torch.no_grad():
            for idx in tqdm(range(len(test_dataset)), desc="Inference"):
                if idx >= max_samples:
                    break

                item = test_dataset[idx]  # type: dict # pixel_values, labels, caption_len, mode
                supported_modes = item['supported_tasks']

                for mode in supported_modes:
                    # build embeddings
                    # get text embeddings
                    if mode == 'retrieval':
                        # remove the image token from the labels
                        labels, _, _ = model.model.remove_img_tokens(item['tgt_tokens'])
                        # get textual embeddings
                        text_embeddings = model.model.input_embeddings(labels.to(device))
                        pixel_values = item['images'][0].to(device=model.model.lm.device,
                                                         dtype=model.model.lm.dtype).unsqueeze(0)
                        visual_embeds = model.model.get_visual_embs(pixel_values, mode='retrieval')
                        # feed to the model
                        outputs, output_embeddings, _ = model(
                            text_embeddings.unsqueeze(0),
                            None,
                            None,
                            generate=True,
                            num_words=10,  # only generate the RET token
                            temperature=0.0,
                            min_word_tokens=1
                        )

                        if model.model.retrieval_token_idx not in outputs[0]:
                            print(f"Warning: RET token not found in the generated tokens for idx: {idx}")
                            print(f"Generated tokens: {tokenizer.convert_ids_to_tokens(outputs[0])}")
                            ret_token_pos = 0
                        else:
                            # get the token embeddings for the RET token after passing through the start and end projection heads
                            # print((outputs[0] == model.model.retrieval_token_idx).nonzero())
                            ret_token_pos = max((outputs[0] == model.model.retrieval_token_idx).nonzero()[0].item()+dev, 0)

                        ret_emb = output_embeddings[ret_token_pos][:, -1, :]

                        all_visual_embeds.append(visual_embeds.cpu().detach().float().numpy())
                        all_ret_embeds.append(ret_emb.cpu().detach().float().numpy())

                        # get the target visual embeddings
                        # print(f"Getting relevant visual embeddings for idx: {idx}")
                        rel_vis_embs = test_dataset.get_relevant_visual_embs(model, idx).cpu().detach()
                        all_target_vis_embeds.append(rel_vis_embs)

                        video_sizes.append(rel_vis_embs.shape[0])

                        frame_limits.append(item['frame_limits'])

                        assert len(test_dataset.images[idx]) == 1, f"Error: More than one image found for idx: {idx}"

                        image_paths.append(test_dataset.images[idx][0])
                        sources.append(test_dataset.raw_sources[idx])
                        targets.append(test_dataset.raw_targets[idx])
                        target_step_texts.append(test_dataset.target_step_text[idx])

                    else:
                        continue

        print(f"Avg video size: {sum(video_sizes) / len(video_sizes)} frames")
        clip_sizes = [m['end'] - m['start'] for m in frame_limits]
        print(f"Avg clip size: {sum(clip_sizes) / len(clip_sizes)} frames")

        # calculate metrics
        metrics = {"ckpt": model_args.ckpt_path, "count": len(all_target_vis_embeds), "test_file": inference_args.test_file, "deviation": dev}
        if len(all_target_vis_embeds) > 0:
            print(len(all_target_vis_embeds))
            print(all_target_vis_embeds[0].shape)
            # all_target_vis_embeds =
            all_ret_embeds = torch.tensor(all_ret_embeds).to(device).squeeze(1)
            # place the all_target_vis_embeds on the device and same dtype as all_ret_embeds
            for i in range(len(all_target_vis_embeds)):
                all_target_vis_embeds[i] = all_target_vis_embeds[i].to(device, dtype=all_ret_embeds.dtype)

            # print(all_target_vis_embeds.shape)
            print(all_ret_embeds.shape)

            assert len(all_target_vis_embeds) == all_ret_embeds.shape[0]

            # build target indices list
            target_indices = []
            tolerance = 0 # 3 frames tolerance
            for i, frame in enumerate(frame_limits):
                start_idx = max(0, frame['middle'] - tolerance)
                end_idx = min(frame['middle'] + tolerance + 1, len(all_target_vis_embeds[i]))
                idxs = list(range(start_idx, end_idx))
                target_indices.append(idxs)
                if idxs == []:
                    print(f"i: {i}, frame: {frame['middle']}, start: {start_idx}, end: {end_idx}, len: {len(all_target_vis_embeds[i])}, indices: {idxs}")

            # calculate recall@k
            recall_at_k = inference_utils.calculate_recall_at_k_vmr(all_target_vis_embeds, all_ret_embeds, target_indices, k=[1, 5, 10])
            metrics.update(recall_at_k)

            # calculate recall@k with IoU = m
            recall_at_k_iou = inference_utils.calculate_recall_at_k_w_iou_version2(all_target_vis_embeds, all_ret_embeds, frame_limits, iou=0.7, k=[1, 5, 10])
            metrics.update(recall_at_k_iou)
            recall_at_k_iou = inference_utils.calculate_recall_at_k_w_iou_version2(all_target_vis_embeds, all_ret_embeds, frame_limits, iou=0.5, k=[1, 5, 10])
            metrics.update(recall_at_k_iou)
            recall_at_k_iou = inference_utils.calculate_recall_at_k_w_iou_version2(all_target_vis_embeds, all_ret_embeds, frame_limits, iou=0.9, k=[1, 5, 10])
            metrics.update(recall_at_k_iou)

            # calculate avg overlap
            metrics['avg_overlap'] = inference_utils.calculate_avg_overlap(all_target_vis_embeds, all_ret_embeds, frame_limits)

            # calculate meanAveragePrecision
            metrics["map"] = inference_utils.calculate_map_vmr(all_target_vis_embeds, all_ret_embeds, [frame['middle'] for frame in frame_limits])

            # Get frame distances
            frame_distances = inference_utils.calculate_frame_distance(all_target_vis_embeds, all_ret_embeds, [frame['middle'] for frame in frame_limits])
            metrics.update(frame_distances)

            # Get step accuracy
            metrics['step_accuracy'] = inference_utils.calculate_step_accuracy(all_target_vis_embeds, all_ret_embeds, frame_limits)

            # save the top k retrievals for each query
            topk_retrievals = inference_utils.get_topk_retrievals(all_target_vis_embeds, all_ret_embeds, k=10)

            # convert the indices to image paths
            assert len(topk_retrievals) == len(image_paths)
            ret_dicts = []
            for i in range(len(topk_retrievals)):
                img_pths = []
                base_path = "/".join(image_paths[i].split("/")[:-1])
                for idx in topk_retrievals[i]['indices']:
                    img_pths.append(f"{base_path}/{idx:05d}.jpg")
                    assert os.path.exists(img_pths[-1]), f"Error: {img_pths[-1]} does not exist"

                ret_dicts.append({
                    "source": sources[i],
                    "target": targets[i].replace('<|endoftext|>', '').replace('<|endofturn|>', ''),
                    "rel_step_text": target_step_texts[i],
                    "target_frame": image_paths[i],
                    "retrievals": img_pths,
                    "scores": [float(i) for i in list(np.array(topk_retrievals[i]['sim_scores']))]
                })

            output_dict = {
                'ckpt_path': model_args.ckpt_path,
                'test_file': inference_args.test_file,
                'dataset_kwargs': dataset_kwargs,
                'count': len(topk_retrievals),
                'predictions': ret_dicts
            }

            with open(output_file.replace('_outputs.json', '_top_10_rets.json'), 'w') as f:
                json.dump(output_dict, f)

            # save the top k retrievals for each query
            all_retrievals_scores = inference_utils.get_topk_retrievals(all_target_vis_embeds, all_ret_embeds, k=10000)

            # convert the indices to image paths
            assert len(all_retrievals_scores) == len(image_paths)
            ret_dicts = []
            for i in range(len(all_retrievals_scores)):
                img_pths = []
                base_path = "/".join(image_paths[i].split("/")[:-1])
                for idx in all_retrievals_scores[i]['indices']:
                    img_pths.append(f"{base_path}/{idx:05d}.jpg")
                    assert os.path.exists(img_pths[-1]), f"Error: {img_pths[-1]} does not exist"

                ret_dicts.append({
                    "source": sources[i],
                    "target": targets[i].replace('<|endoftext|>', '').replace('<|endofturn|>', ''),
                    "rel_step_text": target_step_texts[i],
                    "target_frame": image_paths[i],
                    "frame_limits": frame_limits[i],
                    "retrievals": img_pths,
                    "scores": [float(i) for i in list(np.array(all_retrievals_scores[i]['sim_scores']))]
                })

            output_dict = {
                'ckpt_path': model_args.ckpt_path,
                'test_file': inference_args.test_file,
                'dataset_kwargs': dataset_kwargs,
                'count': len(all_retrievals_scores),
                'predictions': ret_dicts
            }

            with open(output_file.replace('_outputs.json', '_all_rets.json'), 'w') as f:
                json.dump(output_dict, f)


        # save outputs
        print(metrics)
        with open(output_file.replace('_outputs.json', '_metrics.json'), 'w') as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    inference()

