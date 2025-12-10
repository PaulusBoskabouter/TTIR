# IMPORTS

import numpy as np



# HELPERS

"""
Rescales values to range [0, 1]
    Expects np array
    Mainly relevant for like/dislikes (unbounded negative and positive values)
    For range [-1, 1], will map 0 to 0.5
"""
def rescale(array):
    min_, max_ = min(array), max(array)
    return (array - min_) / (max_ - min_)

"""
Converts values to 0 or 1
    -1 is mapped to 0, 
    1+ is mapped to 1
    0-1 (real numbers) are mapped according to a threshold
"""
def remap(array, t = 0.8):
    return np.array([1 if score > t else 0 for score in array])

"""
Calculates all metrics for a given ranking & k (& t).
Expects np.array for ranking
"""
def metrics(ranking, K, t = 0.8):

    # Calculate metrics for different k
    P       = [precision(ranking, k, t) for k in K]
    R       = [recall(ranking, k, t)    for k in K]
    F1      = [f1_score(ranking, k, t)  for k in K]
    N_K     = [NDCG(ranking, k)         for k in K]
    N       = [NDCG(ranking)]
    metrics = np.array(P + R + F1 + N_K + N)

    return metrics

"""
Aggregate metrics for a set of k. Expects 2D list of metrics
"""
def aggregate(metrics, K):
    
    # Calculate metrics for different k
    P       = [f"Precision @ {k}"   for k in K]
    R       = [f"Recall @ {k}"      for k in K]
    F1      = [f"F1-score @ {k}"    for k in K]
    N_K     = [f"NDCG @ {k}"        for k in K]
    N       = ["NDCG"]
    labels = np.array(P + R + F1 + N_K + N)

    # Aggregate metrics
    aggregate = np.mean(metrics, axis = 0)

    # Save & return in dictionary
    return {str(key): float(value) for key, value in zip(labels, aggregate)}



# METRICS (BINARY)

"""
Calculate precision score. Note: Precision is defined for binary labels
"""
def precision(ranking, k, t = 0.8):

    # Convert non-binary values
    labels = remap(ranking, t)

    # Calculate precision
    return sum(labels[:k]) / k if k > 0 else 0

"""
Calculate recall score. Note: Recall is defined for binary labels
"""
def recall(ranking, k, t = 0.8):

    # Convert non-binary values
    labels = remap(ranking, t)

    # Calculate precision
    return sum(labels[:k]) / sum(labels) if sum(labels) > 0 else 0

"""
Calculate F1 score. Note: F1 is defined for binary labels
"""
def f1_score(ranking, k, t = 0.8):

    # Calculate precision & recall
    p, r = precision(ranking, k, t), recall(ranking, k, t)

    # Calculate F1
    return (2 * p * r) / (p + r) if p != 0 and r != 0 else 0


"""
Calculate AP score. Note: AP is defined for binary labels
"""
def average_precision(ranking, t = 0.8):

    # Calculate average precision
    return sum([ranking[k] * precision(ranking, k + 1, t) for k in range(len(ranking))]) / sum(ranking) if sum(ranking) > 0 else 0



# LABELS (CONTINUOUS/NON-BINARY)

"""
Calculate DCG score. DCG works for non-binary labels
"""
def DCG(ranking, k = None):

    # Rescale labels to [0, 1]
    labels = rescale(ranking)

    if k is not None:

        # Calculate DCG
        return sum([(2 ** labels[i] - 1) / np.log2(2 + i) for i in range(k)])
    
    else:

        # Calculate DCG
        return sum([(2 ** labels[i] - 1) / np.log2(2 + i) for i in range(len(labels))])



"""
Calculate DCG score. DCG works for non-binary labels
"""
def NDCG(ranking, k = None):

    # Sort in descending order
    ideal = np.sort(ranking)[::-1]
    
    # Calculate NDCG
    return DCG(ranking, k) / DCG(ideal, k) if sum(ranking) != 0 else 0