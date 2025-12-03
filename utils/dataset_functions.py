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


def save_processed_data(user_feats, user_ids, song_ids, song_embeds, song_labels, interactions, file:Path):#label_specific_feats, user_ids, song_embeds, song_labels, interactions, file:Path):
    # Convert to tensor and save
    file.parent.mkdir(parents=True, exist_ok=True)
    
    user_feats = torch.tensor(user_feats).float()
    user_ids     = torch.tensor(user_ids).long()
    song_ids     = torch.tensor(song_ids).long()
    interactions     = torch.tensor(interactions).long()
    song_embeds  = torch.tensor(song_embeds).float().clone()
    song_labels       = torch.tensor(song_labels).float().clone()


    data_to_save = {
        "user_feats":           user_feats,
        "user_ids":             user_ids,
        "song_ids":             song_ids,
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
    

    user_feats      = loaded["user_feats"].squeeze(0)
    song_embeds     = loaded["song_embeds"].squeeze(0)
    user_ids        = loaded["user_ids"].squeeze(0)
    song_ids        = loaded["song_ids"].squeeze(0)
    labels          = loaded["labels"].squeeze(0)[:, label_id]
    interactions    = loaded["interactions"].squeeze(0)



    
    #dataset = TensorDataset(user_feats, label_specific_feats, song_embeds, labels, interactions)
    dataset = TensorDataset(user_feats, song_embeds, labels, interactions, user_ids, song_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)