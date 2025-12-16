"""
# Author       : Paul Verhoeven
# Date         : 03-11-2025

To keep the notebook as uncluttered as possible, I've thrown a lot of functions down here below.
"""
# from __future__ import annotations
import pandas as pd
from pathlib import Path
from utils.dataset_functions import save_processed_data
from tqdm import tqdm_notebook as progress_bar
import numpy as np




def get_interactions(df_by_uid, user, song, timestamp):
    if user not in df_by_uid.groups:
        return 0
    df = df_by_uid.get_group(user)   # much smaller subset
    return (df['item_id'].eq(song) & (df['timestamp'] <= timestamp)).sum()


def get_song_label_and_user_interacton(row: pd.Series, likes:pd.DataFrame, dislikes:pd.DataFrame) -> tuple[int, int]:
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

    def count_interactions(df_by_uid):
        if user not in df_by_uid.groups:
            return 0
        df = df_by_uid.get_group(user)
        mask = (df["item_id"].eq(song)) & (df["timestamp"] <= timestamp)
        return mask.sum()

    likes_count = count_interactions(likes)
    dislikes_count = count_interactions(dislikes)

    netto_interactions = likes_count - dislikes_count
    binary_label = int(pct_played >= 80.0)
    continueous_label = min(pct_played / 100.0, 1.0)
    

    return [binary_label, continueous_label], netto_interactions


def extract_and_save_features(user_set:list, file_loc:Path, user_item_data:pd.DataFrame, likes:pd.DataFrame, dislikes:pd.DataFrame) -> None:
    """
    Extract features per user and save them into seperate files.
    Args:
        
    """
    likes = likes.groupby('uid')
    dislikes = dislikes.groupby('uid')

    # For each user create their user features and determine last listened song (in embedding) + extract label
    for user in progress_bar(user_set, desc=""):
        user_file = file_loc/f"{user}.pt"
        if not user_file.exists(): # Skip if we've already analysed this user.
            # Create subset of our data focussing on only one user at a time
            user_data = user_item_data[user_item_data['uid'] == user].copy()
            user_feats = []
            user_ids = []
            song_ids = []
            song_embeds = []
            song_labels = []
            interactions = []
            
            

            interactions_total = 0
            # For all of the user timepoints spanning from 10 timepoints to max timepoints of the user, we analyse their song interacton 
            for t in range(10, len(user_data)-10):
                # Fetch datasubset
                data = user_data.iloc[:t]
                song_embedding = data.iloc[-1]['normalized_embed']
                current_row = data.iloc[-1]
                song_id = current_row['item_id']

                labels, interaction_count = get_song_label_and_user_interacton(current_row, likes, dislikes)
                interactions_total += interaction_count

                # User interaction and listening statistics:
                played_pct_avg = data['played_ratio_pct'].mean()
                played_pct_std = data['played_ratio_pct'].std()
                track_length_avg = data['track_length_seconds'].mean()
                track_length_std = data['track_length_seconds'].std()
                interact_ratio = interactions_total/len(data)

                
                # Append the features to list
                features = [played_pct_avg, played_pct_std, track_length_avg, track_length_std, interact_ratio]
            
                user_feats.append(features)
                user_ids.append(user)
                song_ids.append(song_id)
                song_embeds.append(song_embedding.tolist())
                song_labels.append(labels)
                interactions.append(interaction_count)


            # Save this user.
            save_processed_data(user_feats, user_ids, song_ids, song_embeds, song_labels, interactions, file=user_file)