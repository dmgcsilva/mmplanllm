import ast
import json
import os

import torch
import transformers
from torch.utils.data import Sampler, DataLoader
from tqdm import tqdm

from data_binding import ModelArguments, InferenceArguments, DataArguments
from data_mod import data_utils
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

    output_file = os.path.join(model_args.ckpt_path, f'{output_file}_vsg_outputs.json')

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
        test_dataset = data_utils.load_test_dataset(tokenizer, inference_args, data_args, model_args, **dataset_kwargs)  # type: plangpt_dataset.PlanGPTDataset


        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                tfile = json.load(f)['test_file']
            if tfile != inference_args.test_file:
                # delete the file
                os.remove(output_file)
                os.remove(output_file.replace('_outputs.json', '_metrics.json'))


        max_samples = inference_args.max_samples if inference_args.max_samples is not None else len(test_dataset)

        sources = []
        targets = []
        step_results = []
        evald_idxs = []
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
                        continue
                    elif mode == 'captioning':
                        assert test_dataset.intents[idx] == "ARTIFICIAL.VisualStepRetrievalIntent", f"Intent is not VisualStepRetrievalIntent for captioning mode, its {test_dataset.intents[idx]}"
                        # get visual embeddings
                        pixel_values = item['images'][0].to(device=model.model.lm.device, dtype=model.model.lm.dtype).unsqueeze(0)
                        visual_embeds = model.model.get_visual_embs(pixel_values, mode='captioning').squeeze(0)
                        input_embs = model.model.input_embeddings(item['tgt_tokens'].to(device))

                        # print(input_embs.shape)

                        # now find the position of the model.model.image_token_idx and replace the embeddings at that position with the visual embeddings
                        image_token_idx = model.model.image_token
                        img_token_mask = item['tgt_tokens'] == image_token_idx
                        input_embs[img_token_mask] = visual_embeds

                        # discard all tokens past the img token
                        input_embs = input_embs[:img_token_mask.nonzero().squeeze(1).max() + 1]

                        # feed to the model
                        outputs, _, output_logits = model(
                            input_embs.unsqueeze(0),
                            None,
                            None,
                            generate=True,
                            num_words=inference_args.max_new_tokens,
                            temperature=0.0,
                        )

                        # get the caption
                        gen_step = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        sources.append(test_dataset.raw_sources[idx])
                        targets.append(test_dataset.target_step_text[idx])
                        step_results.append(gen_step)
                        evald_idxs.append(idx)
                    else:
                        continue

        clean_predictions = [p.replace(s, "").split("\n###")[0].split("<|endofturn|>")[0].split("<|prompt|>")[0].strip()
                             for p, s in zip(step_results, sources)]

        clean_targets = [t.strip() for t in targets]

        # save predictions
        output_dict = {
            'ckpt_path': model_args.ckpt_path,
            'test_file': inference_args.test_file,
            'dataset_kwargs': dataset_kwargs,
        }
        pred_dicts = []
        for i, data_idx in enumerate(evald_idxs):
            try:
                pred_dicts.append({
                    'id': test_dataset.dialog_ids[data_idx],
                    'turn_number': test_dataset.turn_numbers[data_idx],
                    'intent': test_dataset.intents[data_idx],
                    'prompt': test_dataset.raw_sources[data_idx],
                    'target': clean_targets[i],
                    'prediction': clean_predictions[i],
                })
            except:
                pred_dicts.append({
                    'prompt': test_dataset.raw_sources[data_idx],
                    'target': clean_targets[i],
                    'prediction': clean_predictions[i],
                })

        output_dict['predictions'] = pred_dicts

        # save the targets
        print(f"Saving predictions to {output_file}")
        with open(output_file, "w") as outfile:
            json.dump(output_dict, outfile)

        # calculate metrics
        metrics = {"ckpt": model_args.ckpt_path, "count": len(clean_predictions), "test_file": inference_args.test_file}
        if len(clean_predictions) > 0:
            scores = inference_utils.compute_metrics(clean_predictions, clean_targets[:len(clean_predictions)],
                                                     ['bleu', 'rouge', 'meteor', 'bertscore', 'accuracy'])

            metrics.update(scores)

            # get exact match
            exact_match = inference_utils.calculate_exact_match(clean_predictions, clean_targets[:len(clean_predictions)])
            metrics.update({"exact_match": exact_match})

        print(metrics)
        with open(output_file.replace('_outputs.json', '_metrics.json'), 'w') as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    inference()

