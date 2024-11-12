import json
import os
from typing import List

import pandas as pd
import torch
from PIL.Image import Image
from bert_score import score
from jury import Jury

import matplotlib.pyplot as plt

from sklearn.metrics import recall_score
import numpy as np
import transformers


def compute_metrics(outputs, references, metrics=["bleu", "meteor", "rouge"]):
    jury_metrics = metrics.copy()

    if "accuracy" in metrics:
        jury_metrics.remove("accuracy")
    if "bertscore" in metrics:
        jury_metrics.remove("bertscore")

    scorer = Jury(metrics=jury_metrics)

    outputs = ["gfgfgfgfg" if o == "" else o for o in outputs]
    scores = scorer(predictions=[[o] for o in outputs], references=[[s] for s in references])

    if "accuracy" in metrics:
        acc = [1.0 if r.strip().lower() in o.strip().lower() else 0.0 for o, r in zip(outputs, references)]
        acc_mean = sum(acc) / len(acc)
        scores["accuracy"] = acc_mean

    if "bertscore" in metrics:
        P, R, F1 = score(outputs, references, model_type="microsoft/deberta-xlarge-mnli", lang="en", verbose=True)

        mean_p = sum(P.tolist()) / len(P.tolist())
        mean_r = sum(R.tolist()) / len(R.tolist())
        mean_f1 = sum(F1.tolist()) / len(F1.tolist())

        scores['bertscore'] = {"recall": mean_r, "precision": mean_p, "f1": mean_f1, "model_type": "microsoft/deberta-xlarge-mnli"}

    return scores

def calculate_exact_match(outputs, references):

    # partial exact matche where if reference is contained in output, it is considered a match
    exact_matches = [1.0 if r.strip().lower() in o.strip().lower() else 0.0 for o, r in zip(outputs, references)]
    exact_match = sum(exact_matches) / len(exact_matches)
    exact_match *= 100
    exact_match = round(exact_match, 2)
    return exact_match


def get_similarity_matrix(visual_embeds, ret_embeds, type="default"):
    """
    Calculate the similarity matrix between the visual and retrieval embeddings.

    Args:
        visual_embeds: torch.tensor, the visual embeddings with shape (N, D) where N is the number of samples and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (K, D).

    Returns:
        similarity_matrix: torch.tensor, the similarity matrix between the visual and retrieval embeddings with shape (K, N).
    """
    if len(ret_embeds.shape) == 1:
        ret_embeds = ret_embeds.unsqueeze(0)

    if type == "cosine":
        ret_embeds = ret_embeds.unsqueeze(1).cpu()
        visual_embeds = visual_embeds.unsqueeze(0).cpu()

        similarity_matrix = torch.nn.functional.cosine_similarity(ret_embeds, visual_embeds, dim=2)
    else:
        visual_embeds_normalized = torch.nn.functional.normalize(visual_embeds, p=2, dim=1)
        ret_embeds_normalized = torch.nn.functional.normalize(ret_embeds, p=2, dim=1)

        similarity_matrix = torch.matmul(ret_embeds_normalized, visual_embeds_normalized.T).cpu()

    return similarity_matrix


def calculate_iou(target_start, target_end, pred_start, pred_end):
    """
    Calculate the intersection over union (IoU) between two video moments.

    Args:
        target_start: int, the index of the target start frame.
        target_end: int, the index of the target end frame.
        pred_start: int, the index of the predicted start frame.
        pred_end: int, the index of the predicted end frame.

    Returns:
        iou: float, the intersection over union between the two video moments.
    """
    # check if the predicted video moment is valid
    # print("In calculate_iou")
    # print("target_start: ", target_start)
    # print("target_end: ", target_end)
    # print("pred_start: ", pred_start)
    # print("pred_end: ", pred_end)

    if pred_start >= pred_end:
        return 0.0
    if pred_end < target_start or pred_start > target_end:
        return 0.0

    # Calculate the intersection
    intersection_start = max(target_start, pred_start)
    intersection_end = min(target_end, pred_end)

    # Calculate the union
    union_start = min(target_start, pred_start)
    union_end = max(target_end, pred_end)

    # Calculate the overlap
    overlap = max(0, intersection_end - intersection_start)

    # Calculate the IoU
    iou = overlap / (union_end - union_start)

    return iou


def calculate_recall_at_k_vmr(target_visual_embeds, ret_embeds, target_frames, k=[10]):
    """
    Calculate the recall@k for the video moment retrieval task.
    This function supports multiple target frames

    Args:
        target_visual_embeds: list(torch.tensor), the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_frames: list[list[Dict]], a list of lists where each inner list contains the indices of all acceptable frames
        k: list, the k values for which to calculate the recall@k.

    Returns:
        recall_at_k: list, the recall@k values for each k.
    """
    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    recall = dict()

    for i in range(len(target_frames)):
        assert len(target_frames[i]) > 0, f"Target frames must contain at least one frame for sample {i}"
        r = calculate_recall_at_k_vmr_individual(target_visual_embeds[i], ret_embeds[i], target_frames[i], k)

        for k_val in k:
            if f'recall@{k_val}' not in recall:
                recall[f'recall@{k_val}'] = []
            recall[f'recall@{k_val}'].append(r[f'recall@{k_val}'])

    # Calculate the mean recall@k
    for k_val in k:
        recall[f'recall@{k_val}'] = sum(recall[f'recall@{k_val}']) / len(recall[f'recall@{k_val}'])
        recall[f'recall@{k_val}'] = recall[f'recall@{k_val}'] * 100

    return recall


def calculate_recall_at_k_vmr_individual(target_visual_embeds, ret_embeds, target_indices, k=[10]):
    """
    Calculate the recall@k for the video moment retrieval task for a single sample.

    Args:
        target_visual_embeds: torch.tensor, the visual embeddings for the video frames with shape (M, D) where M is the number of frames in the video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (D).
        target_indices: list[Dict], a list of dictionaries where each dictionary contains the indices of the target start and end frames.
        k: list, the k values for which to calculate the recall@k.

    Returns:
        recall_at_k: list, the recall@k values for each k.
    """

    # Calculate the similarities between retrieval and visual embeddings
    similarities = get_similarity_matrix(target_visual_embeds, ret_embeds).squeeze(0)  # (N,)
    recall = dict()

    for k_val in k:
        # Get the top k indices
        top_k = torch.argsort(similarities, descending=True)[:k_val]

        # Calculate the recall@k
        correct = 0
        for target_index in target_indices:
            if target_index in top_k:
                correct += 1

        # Compute the recall@k knowing that the target is a single video moment
        recall[f"recall@{k_val}"] = correct / min(k_val, len(target_indices))

    return recall


def calculate_recall_at_k_w_iou_version2(target_visual_embeds, ret_embeds, target_frames, iou=0.7, k=[10]):
    """
        Calculate the recall@k for the video moment retrieval task with intersection over union (IoU) constraint.
        Here we first calculate the retrieved video moment middle frame embeddings and then calculate the IoU between
        the retrieved video moment and the target video moment. The retrieved video moment is calculated by assuming a
        clip of the same size as the gt centered on the retrieved frame. If the IoU is greater than the specified
        threshold, we consider the retrieval successful.

        Args:
            target_visual_embeds: torch.tensor, the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
            ret_embeds: torch.tensor, the retrieval embeddings for the start frames with shape (N, D).
            target_frames: list[Dict], a list of dictionaries where each dictionary contains the indices of the target start and end frames.
            iou: float, the intersection over union threshold.
            k: list, the k values for which to calculate the recall@k.

        Returns:
            recall_at_k: list, the recall@k values for each k.
        """

    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    recall = dict()

    for i in range(len(target_frames)):
        assert target_frames[i]["start"] <= target_frames[i]["end"], f"Start frame index must be less than end frame index for target frame {i} but got {target_frames[i]['start']} and {target_frames[i]['end']}"

        r = calculate_recall_at_k_w_iou_version2_individual(target_visual_embeds[i], ret_embeds[i], target_frames[i], iou, k)

        for k_val in k:
            if f'recall@{k_val}_wiou@{iou}' not in recall:
                recall[f'recall@{k_val}_wiou@{iou}'] = []
            recall[f'recall@{k_val}_wiou@{iou}'].append(r[f'recall@{k_val}_wiou@{iou}'])

    # Calculate the mean recall@k
    for k_val in k:
        recall[f'recall@{k_val}_wiou@{iou}'] = sum(recall[f'recall@{k_val}_wiou@{iou}']) / len(recall[f'recall@{k_val}_wiou@{iou}'])

    return recall

def calculate_recall_at_k_w_iou_version2_individual(target_visual_embeds, ret_embeds, target_indices, iou, k=[10]):
    """
        Calculate the recall@k for the video moment retrieval task with intersection over union (IoU) constraint for a single sample.
        Args:
            target_visual_embeds: torch.tensor, the visual embeddings for the video frames with shape (M, D) where M is the number of frames in the video, and D is the dimensionality of the embeddings.
            ret_embeds: torch.tensor, the retrieval embeddings for the end frames with shape (, D).
            target_frames: Dict, a dictionary containing the indices of the target start and end frames.
            iou: float, the intersection over union threshold.
            k: list, the k values for which to calculate the recall@k.

        Returns:
            recall_at_k: list, the recall@k values for each k.
        """

    # Calculate the similarities between retrieval and visual embeddings
    similarities = get_similarity_matrix(target_visual_embeds, ret_embeds).squeeze(0)  # (N,)
    recall = dict()

    for k_val in k:
        # Get the top k indices
        top_k = torch.argsort(similarities, descending=True)[:k_val]

        # Calculate the IoU for each pair of start and end frames
        ious = []
        for i in range(k_val):
            middle_frame = top_k[i]
            gt_size = max(1, target_indices["end"] - target_indices["start"])
            start_frame = max(0, middle_frame - (gt_size // 2))
            end_frame = min(target_visual_embeds.shape[0] - 1, middle_frame + (gt_size // 2))
            iou_val = calculate_iou(target_indices["start"], target_indices["end"], start_frame, end_frame)
            ious.append(iou_val)

        # Calculate the recall@k
        correct = sum([_iou >= iou for _iou in ious])
        # Compute the recall@k knowing that the target is a single video moment
        recall[f"recall@{k_val}_wiou@{iou}"] = 1 if correct > 0 else 0

    return recall

def calculate_avg_overlap(target_visual_embeds, ret_embeds, target_frames):
    """
    Calculate the average overlap between the target and retrieved video moments.

    Args:
        target_visual_embeds: list[torch.tensor], the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_frames: list[Dict], a list of dictionaries where each dictionary contains the indices of the target start and end frames.

    Returns:
        avg_overlap: float, the average overlap between the target and retrieved video moments.
    """

    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    overlaps = []

    for i in range(len(target_frames)):
        assert target_frames[i]["start"] <= target_frames[i]["end"], f"Start frame index must be less than end frame index for target frame {i} but got {target_frames[i]['start']} and {target_frames[i]['end']}"
        overlaps.append(calculate_overlap_individual(target_visual_embeds[i], ret_embeds[i], target_frames[i]))

    avg_overlap = sum(overlaps) / len(overlaps)

    avg_overlap = float(avg_overlap)

    return avg_overlap

def calculate_overlap_individual(target_visual_embeds, ret_embeds, target_indices):
    similarities = get_similarity_matrix(target_visual_embeds, ret_embeds).squeeze(0)  # (N,)

    top_frame = torch.argsort(similarities, descending=True)[0]

    gt_size = target_indices["end"] - target_indices["start"] - 1
    start_frame = max(0, top_frame - (gt_size // 2))
    end_frame = min(target_visual_embeds.shape[0] - 1, top_frame + (gt_size // 2))

    overlap = calculate_iou(target_indices["start"], target_indices["end"], start_frame, end_frame)

    return overlap

def calculate_map_vmr(target_visual_embeds, ret_embeds, target_frames):
    """
    Calculate the mean average precision for the video moment retrieval task.

    Args:
        target_visual_embeds: list[torch.tensor], the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_frames: list[int], a list of integers where each integer is the index of the target frame.

    Returns:
        map: float, the mean average precision for the video moment retrieval task.
    """

    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    aps = []

    for i in range(len(target_frames)):
        assert isinstance(target_frames[i], int), f"Target frames must contain a single frame index for sample {i}"
        aps.append(calculate_map(target_visual_embeds[i], ret_embeds[i].unsqueeze(0), [target_frames[i]]))

    map = sum(aps) / len(aps)

    # map = map * 100

    return map


def calculate_frame_distance(target_visual_embeds, ret_embeds, target_frames):
    """
    Get the average frame distance between the target frame and the retrieved frame.
    To make this resistant to outliers, we will calculate the median of the frame distances.

    Args:
        target_visual_embeds: list[torch.tensor], the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_frames: list[int], a list of integers where each integer is the index of the target frame.

    Returns:
        avg_frame_distance: float, the average frame distance between the target frame and the retrieved frame.
        median_frame_distance: float, the median frame distance between the target frame and the retrieved frame.
    """

    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    frame_distances = []
    video_sizes = [embed.shape[0] for embed in target_visual_embeds]

    for i in range(len(target_frames)):
        assert isinstance(target_frames[i], int), f"Target frames must contain a single frame index for sample {i}"
        frame_distances.append(calculate_avg_frame_distance_individual(target_visual_embeds[i], ret_embeds[i], target_frames[i]))

    # Normalizing frame distances by their respective video sizes
    normd_frame_distances = np.array(frame_distances) / np.array(video_sizes)

    scores = {
        "avg_frame_distance": np.mean(frame_distances),
        "normd_avg_frame_distance": np.mean(normd_frame_distances),
        "median_frame_distance": np.median(frame_distances)
    }

    return scores

def calculate_avg_frame_distance_individual(target_visual_embeds, ret_embeds, target_index):
    similarities = get_similarity_matrix(target_visual_embeds, ret_embeds).squeeze(0)  # (N,)

    top_frame = torch.argsort(similarities, descending=True)[0]

    return abs(top_frame - target_index)

def calculate_step_accuracy(target_visual_embeds, ret_embeds, target_frames):
    """
    Get the video moment accuracy between the target frame and the retrieved frame, meaning if the top retrieved frame is within the target frame.

    Args:
        target_visual_embeds: list[torch.tensor], the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_frames: list[Dict], a list of dictionaries that contain the start and end frame of the relevant video moment.

    Returns:
        step_accuracy: float, the step accuracy between the target frame and the retrieved frame.
    """

    assert ret_embeds.shape[0] == len(target_visual_embeds) == len(target_frames), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]}, {target_visual_embeds.shape[0]}, and {len(target_frames)}"

    step_accuracies = []

    for i in range(len(target_frames)):
        assert len(target_frames[i]) > 0, f"Target frames must contain at least one frame for sample {i}"
        step_accuracies.append(calculate_step_accuracy_individual(target_visual_embeds[i], ret_embeds[i], target_frames[i]))

    step_accuracy = sum(step_accuracies) / len(step_accuracies)

    step_accuracy = step_accuracy * 100

    return step_accuracy

def calculate_step_accuracy_individual(target_visual_embeds, ret_embeds, target_frames):
    similarities = get_similarity_matrix(target_visual_embeds, ret_embeds).squeeze(0)  # (N,)

    top_frame = torch.argsort(similarities, descending=True)[0]

    if target_frames["start"] <= top_frame <= target_frames["end"]:
        return 1
    return 0



def save_similarity_matrix(visual_embeds, ret_embeds, save_path):
    """
    Calculate the similarity matrix between the visual and retrieval embeddings and save it to disk.

    Args:
        visual_embeds: torch.tensor, the visual embeddings with shape (N, D) where N is the number of samples and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        save_path: str, the path where the similarity matrix will be saved.
    """
    # Calculate similarities between retrieval and visual embeddings
    # Assuming cosine similarity for calculation, normalize embeddings first
    similarities = get_similarity_matrix(visual_embeds, ret_embeds)

    # Save the similarity matrix to disk
    torch.save(similarities, save_path)

    print(f"Similarity matrix saved to {save_path}")


def load_similarity_matrix(similarity_matrix_path):
    """
    Load a similarity matrix from disk.

    Args:
        similarity_matrix_path: str, the path to the saved similarity matrix.

    Returns:
        similarity_matrix: torch.tensor, the loaded similarity matrix.
    """
    return torch.load(similarity_matrix_path)


def calculate_map(visual_embeds, ret_embeds, target_indices):
    """
    Calculate the mean average precision for the retrieval task.
    This function assumes a single relevant image per query.

    Args:
        all_vis_embeds: torch.tensor, the visual embeddings with shape (N, D) where N is the number of samples and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        target_indices: list, the indices of the target images in the visual_embeds. (N)

    Returns:
        map: float, the mean average precision.
    """
    similarities = get_similarity_matrix(visual_embeds, ret_embeds)

    # Calculate the average precision for each query
    aps = []
    for i, target_index in enumerate(target_indices):
        # Sort the similarities
        sorted_similarities = torch.argsort(similarities[i], descending=True)
        # Calculate the precision
        aps.append(1 / (torch.where(sorted_similarities == target_index)[0].item() + 1))
    # Calculate the mean average precision
    map = sum(aps) / len(aps)

    # scale the map to 0-100
    map = map * 100

    return map

def get_topk_retrievals(target_visual_embeds, ret_embeds, k=10):
    """
    Get the top k indices retrieved, for each sample.

    Args:
        target_visual_embeds: list[torch.tensor], the visual embeddings for the video frames with shape (N, M, D) where N is the number of samples, M is the number of frames in each video, and D is the dimensionality of the embeddings.
        ret_embeds: torch.tensor, the retrieval embeddings with shape (N, D).
        k: int, the number of top retrievals to get.

    Returns:
        top_k_indices: list, the top k indices for each sample.
    """

    assert ret_embeds.shape[0] == len(target_visual_embeds), f"Number of samples in the embeddings and target frames must match but got {ret_embeds.shape[0]} and {target_visual_embeds.shape[0]}"

    top_k_indices = []

    for i in range(ret_embeds.shape[0]):
        similarities = get_similarity_matrix(target_visual_embeds[i], ret_embeds[i]).squeeze(0)

        top_frames = torch.argsort(similarities, descending=True)
        top_frames = top_frames[:min(k, len(top_frames))].cpu().numpy()

        sims = {
            "indices": top_frames,
            "sim_scores": similarities[top_frames]
        }

        top_k_indices.append(sims)

    return top_k_indices

