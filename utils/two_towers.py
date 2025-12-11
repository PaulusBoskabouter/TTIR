import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim import Adam
from typing import Optional
import faiss


class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.network(x)
        y = F.normalize(y, p=2, dim=1)
        return y



class DualAugmentedTwoTower(nn.Module):
    def __init__(self, name:str, hidden_dim:int, output_dim:int, id_embed_dim:int):
        super().__init__()

        self.name = name

        self.train_loss_history = []
        self.val_loss_history = []

        # user id
        self.user_id_embedder = nn.Embedding(5427, id_embed_dim, padding_idx=0)
        self.song_id_embedder = nn.Embedding(3272, id_embed_dim, padding_idx=0)

        # Tower initialisations
        self.user_tower = Tower(5, hidden_dim, output_dim)
        self.item_tower = Tower(128, hidden_dim, output_dim)

        # ANN Index
        M = 32      # Higher is more accurate but slower
        self.index = faiss.IndexHNSWFlat(output_dim + id_embed_dim, M)


        



    def forward(self, user_features, song_features, user_id, song_id):
        # Embed our ids
        user_vec = self.user_id_embedder(user_id)
        song_vec = self.song_id_embedder(song_id)

        
        y_user = self.user_tower(user_features)
        y_song = self.item_tower(song_features)

        # Concatenate output with lookuptable 
        yu = torch.cat([y_user, user_vec], dim = 1)
        ys = torch.cat([y_song, song_vec], dim = 1)


        # Final dot-product score
        dot = (yu * ys).sum(dim = 1)

        y = F.sigmoid(dot)

        return y


    
    
    def song_pass(self, song_features, song_id):
        """
        Single item tower pass
        """
        # Embed our ids
        with torch.no_grad():
            song_vec = self.song_id_embedder(song_id)

            y = self.item_tower(song_features)

            # Concatenate output with lookuptable 
            y = torch.cat([y, song_vec], dim = 1)

            return y
    


    
    def user_pass(self, user_features, user_id):
        """
        Single user tower pass
        """
        # Embed our ids
        with torch.no_grad():
            user_vec = self.user_id_embedder(user_id)

            y = self.user_tower(user_features)

            # Concatenate output with lookuptable 
            y = torch.cat([y, user_vec], dim = 1)

            return y
    
    
    def create_index(self, song_embeddings):
        """
        Creates index for reccommendation query lookup
        (Expects Numpy array)
        """
        self.index.add(song_embeddings)
    
    
    
    
    def recommendations(self, query, k):
        """
        Get k approximate nearest neighbours
        """
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]

    
    def loss(self, score, labels):
        """
        Calculate & combine all loss terms including:
            - Loss_p: dot-product loss
            - Loss_u: AMM loss of user tower
            - Loss_v: AMM loss of item tower
        """

        
        # Dot-product loss
        loss_p = F.binary_cross_entropy(score, labels)#F.binary_cross_entropy_with_logits(score, labels.float())

       
        return loss_p
    


    def plot(self, epochs:int, last_epoch_saved:int, folder:Optional[Path] = None) -> None: 
        """
        Plot the current progress based on the historic loss data
        Args:
            epochs (int): The number of epochs the model should be trained on, purely used for the xlim.
            folder (Path): If it is not None, we use it to save the plot to this folder.
        """


        x = np.arange(1, len(self.val_loss_history) + 1)
        train_losses = np.array(self.train_loss_history)
        val_losses = np.array(self.val_loss_history)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot both curves
        ax.plot(x, train_losses, label="Training loss")
        ax.plot(x, val_losses, label="Validation loss")

        # Add some polish
        ax.set_xlim(1, epochs+1)
        ax.axvline(x=last_epoch_saved, color="red", linestyle="--", linewidth=1) # Add a horizontal line of our best model (which is saved)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training progress for {self.name} model")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="upper right")


        
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder/ f"{self.name}.png", bbox_inches="tight", dpi=300)
            plt.show()

        else:
            plt.show()

        plt.close(fig)



    def add_losses(self, train_loss:float, val_loss:float) -> None:
        """
        This function is just to add the historic data.
        We save this data as well to make sure we can resume if anything goes wrong mid-training

        Args:
            train_loss (float): The current average epoch loss on the training data
            val_loss (float): The current average epoch loss on the validation data
        """

        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)



    def save(self, folder:Path = Path("models")):
        """
        Save the model state dictionary to a file.
        Args:
            folder (Path): Directory to save the model.
        """

        folder.mkdir(parents=True, exist_ok=True)
        save_data = {
            'weights':        self.state_dict(),
            'train_losses':   self.train_loss_history.copy(),
            'val_losses':     self.val_loss_history.copy()

        }
        torch.save(save_data, folder/f"{self.name}.pt")

    

    def load(self, filepath: Path, device='cuda'):
        """
        Load the model state dictionary from a file.
        Args:
            filepath (Path): Path to the file containing the model state dictionary.
            device: The device to load the model onto
        """
        data = torch.load(filepath, map_location=device)
        self.load_state_dict(data['weights'])
        self.val_loss_history   = data['val_losses']
        self.train_loss_history = data['train_losses']






def train_model(model:DualAugmentedTwoTower, train_dataloader:DataLoader, val_dataloader:DataLoader, optimizer:Adam, patience:int = 10, num_epochs:int = 10, lambda_u:float = 1.0, lambda_v:float = 1.0, device:str = 'cpu'):
    """
        Training function for training the models.
        Args:
            model (DualAugmentedTwoTower): DualAugmentedTwoTower class. What more is there to say.
            train_dataloader (DataLoader): Pretty self explanatory; it's a torch.utils.data.Dataloader consisting of (user_features, user_id, song_embedding, labels).
            val_dataloader   (DataLoader): Let's not be redundant here, it's the same but used for validation.
            optimizer              (Adam): Wow, it's a torch Adam optimizer.
            num_epochs              (int): Number of desired epochs (not counting early stopping)
            lambda_u              (float): ...
            lambda_v              (float): ...
            device:                 (str): The device to load the model onto
    """
    model.to(device)
    
    best_val_loss = np.inf
    patience_counter = 0
    last_save = 0

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        epoch_train_loss = 0.0
        # for user_features, label_features, song_embedding, labels, __ in
        for user_features, song_embeddings, labels, _interactions, user_ids, song_ids in train_dataloader:

            # Move to device
            user_features = user_features.to(device)
            user_ids = user_ids.to(device)
            song_ids = song_ids.to(device)
            song_embeddings = song_embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            score = model(user_features, song_embeddings, user_ids, song_ids)

            # Compute combined loss
            loss = model.loss(score, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * labels.size(0)

        epoch_train_loss /= len(train_dataloader.dataset)


        
        # Validate
        epoch_val_loss = 0.0
        with torch.no_grad():
            # for user_features, label_features, song_embedding, labels, __ in
            for user_features, song_embeddings, labels, _interactions, user_ids, song_ids in val_dataloader:

                # Move to device
                user_features = user_features.to(device)
                user_ids = user_ids.to(device)
                song_ids = song_ids.to(device)
                song_embeddings = song_embeddings.to(device)
                labels = labels.to(device)

                #score, pu, pv = model(user_features, label_features, song_embedding)
                score = model(user_features, song_embeddings, user_ids, song_ids)

                # Compute combined loss
                loss = model.loss(score, labels)

                epoch_val_loss += loss.item() * labels.size(0)
            
        epoch_val_loss /= len(val_dataloader.dataset)
        

        clear_output(wait=True) # make sure our notebook doesn't get cluttered
        print(f"Epoch: [{epoch}/{num_epochs}]")
        
        if epoch_val_loss >= best_val_loss:
            # if not, we wait for 'patience_couter' amount of epochs to improve
            if patience_counter >= patience:
                # if not we early stop
                print('early stopped')
                return
            patience_counter += 1
        
        # if improved, save the model and plot
        else:
            patience_counter = 0
            best_val_loss = epoch_val_loss
            model.save(Path("models"))
            last_save = epoch
        
        model.add_losses(epoch_train_loss, epoch_val_loss)
        model.plot(num_epochs, last_epoch_saved=last_save, folder=Path("models") / "plots")