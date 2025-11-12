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
    
def download_df(dataset:YambdaDataset, dataset_type, dataset_dir:Path = Path("Dataset") / "unprocessed"):
    
    if not (dataset_dir / f"{dataset_type}.csv").exists():
        df = dataset.interaction(f"{dataset_type}").to_pandas()
        df.to_csv(dataset_dir / f"{dataset_type}.csv", index=False)
        del df

def parse_embedding(s):
    # Remove brackets
    s = s.strip().strip('[]')
    # Split by whitespace
    return np.fromstring(s, sep=' ')


def save_tensor_dataset(file_name:str, user_feats:list, user_ids:list, song_embeds:list, labels:list, file_loc:Path) -> None:
    # Convert to tensor and save
    file_loc.mkdir(parents=True, exist_ok=True)

    user_feats   = torch.from_numpy(np.stack(user_feats)).float().clone()

    user_feats = user_feats[:, :1]
    user_ids = user_feats[:, 0]
    song_embeds  = torch.from_numpy(np.stack(song_embeds)).float().clone()
    labels       = torch.from_numpy(np.stack(labels)).float().clone()

    data_to_save = {
        "user_feats":   user_feats,
        "user_ids": user_ids,
        "song_embeds":  song_embeds,
        "labels":     labels,
    }

    file_loc.mkdir(exist_ok=True)
    torch.save(data_to_save, file_loc / f"{file_name}.pt")


def load_tensor_dataloader(file_name:str, file_loc:Path, user_to_idx:dict, batch_size:int=32) -> DataLoader:
    """
    Short function for reloading the afforementioned tensorfiles and store them into torch.utils.data.DataLoader. 
    """

    loaded = torch.load(file_loc / f"{file_name}.pt", map_location="cpu")
    user_feats   = loaded["user_feats"]
    user_ids     = loaded["user_ids"]
    song_embeds  = loaded["song_embeds"]
    labels       = loaded["labels"]

    
    dataset = TensorDataset(user_feats, user_ids, song_embeds, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)