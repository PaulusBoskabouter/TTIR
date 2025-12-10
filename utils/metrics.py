#import torch
import numpy as np
import math as log2



def compute_precision_at_k(ground_truth_sets, ranked_item_list, k):
    " compute precision at k per user and average over users"
    precisions = []
    for relevant, recommended in zip(ground_truth_sets, ranked_item_list):
        top_k = recommended[:k]
        correct_hit = len(set(top_k) & relevant)
        precisions.append(correct_hit / k if k > 0 else 0.0)
    return float(np.mean(precisions))


def recall_at_k(ground_truth_sets, ranked_item_list, k):
    " compute recall at k per user and average over users"
    recalls = []
    for relevant, recommended in zip(ground_truth_sets, ranked_item_list):
        # no relevant items, skip
        if len(relevant) == 0:
            recalls.append(0.0)
            continue 
        top_k = recommended[:k]
        correct_hit = len(set(top_k) & relevant)
        recalls.append(correct_hit / len(relevant))
    return float(np.mean(recalls))


def f1_score_at_k(ground_truth_sets, ranked_item_list, k):
    " compute F1 score at k per user and average over users"
    f1_scores= []
    for relevant, recommended in zip(ground_truth_sets, ranked_item_list):
        top_k = recommended[:k]
        correct_hit = len(set(top_k) & relevant)
        # no relevant items, skip
        if len(relevant) == 0 or k == 0:
            f1_scores.append(0.0)
            continue 
        precision = correct_hit / k
        recall = correct_hit / len(relevant) 
        if precision + recall == 0:
            continue
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return float(np.mean(f1_scores))


def average_precision(relevant, recommended, k):
    " compute average precision at k for a single user "
    correct_hit= 0
    scores = []
    if len(recommended) > k:
        recommended = recommended[:k]
    for i, song in enumerate(recommended, start=1):
        if song in relevant:
            correct_hit += 1
            scores.append(correct_hit / i)
    if len(relevant) == 0 or len(scores) == 0:
        return 0.0
    return sum(scores) / len(relevant)


def mean_average_precision(ground_truth_sets, ranked_item_list, k):
    " compute mean average precision at k over all users "
    MAP_scores = []
    for relevant, recommended in zip(ground_truth_sets, ranked_item_list):
        ap_score_per_user = average_precision(relevant, recommended, k)
        MAP_scores.append(ap_score_per_user)
    return float(np.mean(MAP_scores))


def ndcg_at_k(ground_truth_sets, ranked_item_list, k):
    " compute normalized discounted cumulative gain at k per user and average over users"
    ndcgs = []
    for relevant, recommended in zip(ground_truth_sets, ranked_item_list):
        # discounted cumulative gain
        dcg = 0.0
        for i, item in enumerate(recommended[:k], start=1):
            if item in relevant:
                dcg += 1.0 / log2(i + 1)

        ideal_hits = min(len(relevant), k)
        # ideal discounted cumulative gain
        if ideal_hits == 0:
            ndcgs.append(0.0)
            continue
        idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1))
        # normalized discounted cumulative gain
        ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs))
