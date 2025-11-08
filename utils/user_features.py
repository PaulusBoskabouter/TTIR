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
from torch.utils.data import TensorDataset


def get_song_label_and_user_interacton(timestamp:int, user:int, song:int, likes:pd.DataFrame, 
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
    
    # short helper function 
    _filter = lambda df : df[(df["uid"] == user)& (df["item_id"] == song) & (df["timestamp"] <= timestamp)] 

    # Subset our dataframes to contain only user id and timestramp up until the given timestamp parameter.
    likes = _filter(likes)
    unlikes = _filter(unlikes)
    dislikes = _filter(dislikes)
    undislikes = _filter(undislikes)

    
    netto_interactions = len(likes) + len(undislikes) - len(dislikes) - len(unlikes)
    label = 1 if netto_interactions > 0 else 0

    return label, netto_interactions



def extract_and_save_features(user_set:list, user_item_data:pd.DataFrame, likes:pd.DataFrame, 
                              unlikes:pd.DataFrame, dislikes:pd.DataFrame, undislikes:pd.DataFrame, 
                              dataset_type:str, file_loc:Path = Path("Dataset") / "processed" 
                              ) -> None:
  """
    Function for creating the features and saving them into a tensor. 
  """
  assert dataset_type in ["train", "val", 'test']

  user_feats = []
  song_embeds = []
  labels = []
  # For each user create their user features and determine last listened song (in embedding) + extract label
  for user in user_set:
      subset = user_item_data[user_item_data['uid'] == user]
      for t in range(len(subset)-10):
          data = subset.iloc[:t+10]
          song = data.iloc[-1]['item_id']
          song_embedding = data.iloc[-1]['normalized_embed']
          

          # Extract features:
          label, interactions = get_song_label_and_user_interacton(timestamp=data.iloc[-1]['timestamp'], user=user, song=song, 
                                                      likes=likes, unlikes=unlikes, dislikes=dislikes, undislikes=undislikes)
          
          features = [t, user, data['played_ratio_pct'].mean(), data['played_ratio_pct'].std(), data['track_length_seconds'].mean(), data['track_length_seconds'].std(), interactions/len(data)]
          

          user_feats.append(features)
          song_embeds.append(song_embedding)
          labels.append(label)


  # Convert to tensor and save
  user_feats   = torch.from_numpy(np.stack(user_feats)).float().clone()
  song_embeds  = torch.from_numpy(np.stack(song_embeds)).float().clone()
  labels       = torch.tensor(labels, dtype=torch.long) 
  
  data_to_save = {
      "user_feats":   user_feats,
      "song_embeds":  song_embeds,
      "labels":     labels,
  }

  file_loc.mkdir(exist_ok=True)
  torch.save(data_to_save, file_loc/f"{dataset_type}.pt")



def load_tensor_dataset(file_name, file_loc:Path = Path("Dataset") / "processed") -> TensorDataset:
    """
    Short function for reloading the afforementioned tensorfiles. 
    """
    loaded = torch.load(f"{file_name}.pt", map_location="cpu")
    user_feats   = loaded["user_feats"]
    song_embeds  = loaded["song_embeds"]
    labels       = loaded["labels"]

    return TensorDataset(user_feats, song_embeds, labels)