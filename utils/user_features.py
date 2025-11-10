"""
# Author       : Paul Verhoeven
# Date         : 03-11-2025

To keep the notebook as uncluttered as possible, I've thrown a lot of functions down here below.
"""
# from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np 
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm_notebook as progress_bar


def get_song_label_and_user_interacton(timestamp:int, user:int, song:int, likes:pd.DataFrame, user_data:pd.DataFrame,
                                       unlikes:pd.DataFrame, dislikes:pd.DataFrame, undislikes:pd.DataFrame
                                       ) -> tuple[int, int]:
    """
    (Pardon the horrendous long function name)
    Aggregate a user's interactions with a specific song up to ``timestamp``.

    Returns
    -------
    label : int
        1 if the net interaction score is positive, otherwise 0.
    net_interactions : int
        The raw interaction count:
      len(likes) + len(undislikes) - len(dislikes) - len(unlikes)
    """

    current_row = user_data.iloc[-1]

    pct_80_label = 1 if current_row['played_ratio_pct'] >= 80.0 else 0
    pct_100_label = 1 if current_row['played_ratio_pct'] >= 100.0 else 0
    multiple_listens_label = 1 if (user_data['item_id'] == song).sum() > 1 else 0
    
    # short helper function 
    _filter = lambda df : df[(df["uid"] == user)& (df["item_id"] == song) & (df["timestamp"] <= timestamp)] 

    # Subset our dataframes to contain only user id and timestramp up until the given timestamp parameter.
    likes = _filter(likes)
    unlikes = _filter(unlikes)
    dislikes = _filter(dislikes)
    undislikes = _filter(undislikes)

    
    netto_interactions = len(likes) + len(undislikes) - len(dislikes) - len(unlikes)
    interactions_label = 1 if netto_interactions > 0 else 0

    labels = [interactions_label, multiple_listens_label, pct_100_label, pct_80_label]

    return labels, netto_interactions



def extract_and_save_features(user_set:list, user_item_data:pd.DataFrame, likes:pd.DataFrame, 
                              unlikes:pd.DataFrame, dislikes:pd.DataFrame, undislikes:pd.DataFrame, 
                              dataset_type:str, file_loc:Path = Path("Dataset") / "processed" 
                              ) -> None:
    """
    Function for creating the features. 
    """
    assert dataset_type in ["train", "val", 'test']


    # For each user create their user features and determine last listened song (in embedding) + extract label
    previous_len = None
    for user in progress_bar(user_set, desc=f"{dataset_type}"):
        if not (Path("Dataset")/ "processed" / f"{dataset_type}" / f"{user}.pt").exists(): # Skip if we've already analysed this user.
            subset = user_item_data[user_item_data['uid'] == user]
            user_feats = []
            song_embeds = []
            song_labels = []
            # For all of the user timepoints spanning from 10 timepoints to max timepoints of the user, we analyse their song interacton 
            for t in range(len(subset)-10):
                data = subset.iloc[:t+10]
                song = data.iloc[-1]['item_id']
                song_embedding = data.iloc[-1]['normalized_embed']

                if previous_len is None:
                    previous_len = song_embedding.shape

                else:
                    assert previous_len == song_embedding.shape, f"nee: prev: {previous_len}, nu: {song_embedding.shape}"

                # Extract features:
                labels, interactions = get_song_label_and_user_interacton(timestamp=data.iloc[-1]['timestamp'], user=user, song=song, 
                                                            likes=likes, unlikes=unlikes, dislikes=dislikes, undislikes=undislikes, user_data=data)
                
                features = [t, user, data['played_ratio_pct'].mean(), data['played_ratio_pct'].std(), data['track_length_seconds'].mean(), data['track_length_seconds'].std(), interactions/len(data)]
                

                user_feats.append(features)
                song_embeds.append(song_embedding)
                song_labels.append(labels)

            # Save each user
            file_loc = Path("Dataset") / "processed" / f"{dataset_type}"
            save_tensor_dataset(user, dataset_type, user_feats, song_embeds, song_labels, file_loc)


  
def save_tensor_dataset(file_name:str, user_feats:list, song_embeds:list, labels:list, file_loc:Path) -> None:
    # Convert to tensor and save
    file_loc.mkdir(parents=True, exist_ok=True)

    user_feats   = torch.from_numpy(np.stack(user_feats)).float().clone()
    song_embeds  = torch.from_numpy(np.stack(song_embeds)).float().clone()
    labels       = torch.from_numpy(np.stack(labels)).float().clone()

    data_to_save = {
        "user_feats":   user_feats,
        "song_embeds":  song_embeds,
        "labels":     labels,
    }

    file_loc.mkdir(exist_ok=True)
    torch.save(data_to_save, file_loc / f"{file_name}.pt")


def load_tensor_dataloader(file_name:str, file_loc:Path, batch_size:int=32) -> DataLoader:
    """
    Short function for reloading the afforementioned tensorfiles and store them into torch.utils.data.DataLoader. 
    """

    loaded = torch.load(file_loc / f"{file_name}.pt", map_location="cpu")
    user_feats   = loaded["user_feats"]
    song_embeds  = loaded["song_embeds"]
    labels       = loaded["labels"]

    dataset = TensorDataset(user_feats, song_embeds, labels)
    return DataLoader(dataset, batch_size, shuffle=True)