"""
# Author       : Paul Verhoeven
# Date         : 03-11-2025

To keep the notebook as uncluttered as possible, I've thrown a lot of functions down here below.
"""
# from __future__ import annotations
from pathlib import Path
import numpy as np
import numpy as np 
import torch
from torch.utils.data import DataLoader, TensorDataset 
from typing import Literal
from datasets import Dataset, DatasetDict, load_dataset

class YambdaDataset:
    INTERACTIONS = frozenset([
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ])

    def __init__(
        self,
        dataset_type: Literal["flat", "sequential"] = "flat",
        dataset_size: Literal["50m", "500m", "5b"] = "50m"
    ):
        assert dataset_type in {"flat", "sequential"}
        assert dataset_size in {"50m", "500m", "5b"}
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

    def interaction(self, event_type: Literal[
        "likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"
    ]) -> Dataset:
        assert event_type in YambdaDataset.INTERACTIONS
        return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

    def audio_embeddings(self) -> Dataset:
        return self._download("", "embeddings")

    def album_item_mapping(self) -> Dataset:
        return self._download("", "album_item_mapping")

    def artist_item_mapping(self) -> Dataset:
        return self._download("", "artist_item_mapping")


    @staticmethod
    def _download(data_dir: str, file: str) -> Dataset:
        data = load_dataset("yandex/yambda", data_dir=data_dir, data_files=f"{file}.parquet")
        # Returns DatasetDict; extracting the only split
        assert isinstance(data, DatasetDict)
        return data["train"]
    
def download_df(dataset:YambdaDataset, dataset_type, dataset_dir:Path = Path("dataset") / "unprocessed"):
    if not (dataset_dir / f"{dataset_type}.csv").exists():
        df = dataset.interaction(f"{dataset_type}").to_pandas()
        df.to_csv(dataset_dir / f"{dataset_type}.csv", index=False)
        del df

def parse_embedding(s):
    # Remove brackets
    s = s.strip().strip('[]')
    # Split by whitespace
    return np.fromstring(s, sep=' ')


def save_processed_data(user_feats, label_specific_feats, user_ids, song_embeds, song_labels, interactions, file:Path):
    # Convert to tensor and save
    file.parent.mkdir(parents=True, exist_ok=True)
    
    user_feats = torch.from_numpy(np.concatenate(user_feats, axis=0)).float().clone()
    label_specific_feats = torch.from_numpy(np.concatenate(label_specific_feats, axis=0)).float().clone()
    user_ids     = torch.from_numpy(np.concatenate(user_ids, axis=0)).long().clone()
    interactions     = torch.from_numpy(np.concatenate(interactions, axis=0)).long().clone()
    song_embeds  = torch.from_numpy(np.concatenate(song_embeds, axis=0)).float().clone()
    song_labels       = torch.from_numpy(np.concatenate(song_labels, axis=0)).float().clone()

    data_to_save = {
        "user_feats":           user_feats,
        "label_specific_feats": label_specific_feats,
        "user_ids":             user_ids,
        "song_embeds":          song_embeds,
        "labels":               song_labels,
        "interactions":         interactions
    }
    torch.save(data_to_save, file)

def load_tensor_dataloader(file_name:str, file_loc:Path, batch_size:int=32, label_id:int=0) -> DataLoader:
    """
    Short function for reloading the afforementioned tensorfiles and store them into torch.utils.data.DataLoader. 
    """

    loaded = torch.load(file_loc / f"{file_name}.pt", map_location="cpu", weights_only=False)
    
    user_feats              = loaded["user_feats"].squeeze(0)
    label_specific_feats    = loaded["label_specific_feats"].squeeze(0)[:, label_id]
    song_embeds             = loaded["song_embeds"].squeeze(0)
    labels                  = loaded["labels"].squeeze(0)[:, label_id]
    interactions            = loaded["interactions"].squeeze(0)

    # n = user_feats.size(0) // 2
    # user_feats           = user_feats[:n]
    # label_specific_feats = label_specific_feats[:n]
    # song_embeds          = song_embeds[:n]
    # labels               = labels[:n]
    # interactions         = interactions[:n]


    
    dataset = TensorDataset(user_feats, label_specific_feats, song_embeds, labels, interactions)
    dataset = TensorDataset(user_feats, song_embeds, labels, interactions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)