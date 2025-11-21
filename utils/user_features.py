"""
# Author       : Paul Verhoeven
# Date         : 03-11-2025

To keep the notebook as uncluttered as possible, I've thrown a lot of functions down here below.
"""
# from __future__ import annotations
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from utils.dataset_functions import save_processed_data
from tqdm import tqdm_notebook as progress_bar


def get_song_label_and_user_interacton(row: pd.Series, likes:pd.DataFrame, user_data:pd.DataFrame,
                                       unlikes:pd.DataFrame, dislikes:pd.DataFrame, undislikes:pd.DataFrame
                                       ) -> tuple[int, int]:
    """
    (Pardon the horrendous long function name)
    Aggregate a user's interactions with a specific song up to ``timestamp``.

    Args:
    timestamp (int):


    Returns
    -------
    label : int
        1 if the net interaction score is positive, otherwise 0.
    net_interactions : int
        The raw interaction count:
      len(likes) + len(undislikes) - len(dislikes) - len(unlikes)
    """

    user = row['uid']
    song = row['item_id']
    timestamp = row['timestamp']
    pct_played = row['played_ratio_pct']
    song_embed = row['normalized_embed']



    
    # short helper function 
    _filter = lambda df : df[(df["uid"] == user)& (df["item_id"] == song) & (df["timestamp"] <= timestamp)] 

    # Subset our dataframes to contain only user id and timestramp up until the given timestamp parameter.
    likes = _filter(likes)
    unlikes = _filter(unlikes)
    dislikes = _filter(dislikes)
    undislikes = _filter(undislikes)
    listens = _filter(user_data)

    
    netto_interactions = len(likes) + len(undislikes) - len(dislikes) - len(unlikes)
    interactions_label = 1 if netto_interactions > 0 else 0
    pct_80_label = 1 if pct_played >= 80.0 else 0
    pct_100_label = 1 if pct_played >= 100.0 else 0

    multiple_listens_label = 1 if len(listens) > 1 else 0

    return [interactions_label, multiple_listens_label, pct_100_label, pct_80_label], netto_interactions



def calc_centroid(data, label_id):
    mask = data['labels'].apply(lambda lst: lst[label_id] == 1)
    liked_song_subset = data[mask]

    if len(liked_song_subset) == 0:
        return np.zeros(shape=(128,))

    matrix = np.vstack(liked_song_subset['normalized_embed'].values) 
    return matrix.mean(axis=0)



def extract_and_save_features(user_set:list, user_item_data:pd.DataFrame, dataset_type:str, file_loc:Path, 
                              likes:pd.DataFrame, user_data:pd.DataFrame, unlikes:pd.DataFrame, dislikes:pd.DataFrame, undislikes:pd.DataFrame) -> None:
    """
    Extract features per user and save them into seperate files.
    Args:
        
    """
    #assert dataset_type in ["train", "val", 'test']

    # For each user create their user features and determine last listened song (in embedding) + extract label
    for user in progress_bar(user_set, desc=f"{dataset_type}"):
        user_file = file_loc/f"{user}.pt"
        if not user_file.exists(): # Skip if we've already analysed this user.
            # Create subset of our data focussing on only one user at a time
            user_data = user_item_data[user_item_data['uid'] == user].copy()

            


            user_data[["labels", "net_interactions"]] = user_data.apply(
            get_song_label_and_user_interacton,
            axis=1,
            args=(likes, user_data, unlikes, dislikes, undislikes),
            result_type="expand") 
            
            user_feats = []
            label_specific_feats = []
            user_ids = []
            song_embeds = []
            song_labels = []
            interactions = []

            # For all of the user timepoints spanning from 10 timepoints to max timepoints of the user, we analyse their song interacton 
            for t in range(10, len(user_data)-10):
                # Fetch datasubset
                data = user_data.iloc[:t]
                song_embedding = data.iloc[-1]['normalized_embed']
                labels = data.iloc[-1]['labels']
                interaction_count = data.iloc[-1]['net_interactions']
                
                # Calculate the centroid of the liked songs and the dot product with the current song:
                # index cheatsheet: [interactions_label, multiple_listens_label, pct_100_label, pct_80_label]
                in_centroid = calc_centroid(data, 0)
                in_sim = np.dot(in_centroid, song_embedding)
                
                ml_centroid = calc_centroid(data, 1)
                ml_sim = np.dot(ml_centroid, song_embedding)

                pct_100_centroid = calc_centroid(data, 2)
                pct_100_sim = np.dot(pct_100_centroid, song_embedding)

                pct_80_centroid = calc_centroid(data, 3)
                pct_80_sim = np.dot(pct_80_centroid, song_embedding)

                # User interaction and listening statistics:
                played_pct_avg = data['played_ratio_pct'].mean()
                played_pct_std = data['played_ratio_pct'].std()
                track_length_avg = data['track_length_seconds'].mean()
                track_length_std = data['track_length_seconds'].std()
                interact_ratio = data['net_interactions'].sum()/len(data)

                
                # Append the features to list
                specific_feats = [np.append(in_centroid, float(in_sim)), np.append(ml_centroid, float(ml_sim)), np.append(pct_100_centroid, float(pct_100_sim)), np.append(pct_80_centroid, float(pct_80_sim))]
                features = [played_pct_avg, played_pct_std, track_length_avg, track_length_std, interact_ratio]
            
                user_feats.append(features)
                user_ids.append(user)
                label_specific_feats.append(specific_feats)
                song_embeds.append(song_embedding)
                song_labels.append(labels)
                interactions.append(interaction_count)

            # Save this user.
            save_processed_data(user_feats, label_specific_feats, user_ids, song_embeds, song_labels, interactions, file=user_file)