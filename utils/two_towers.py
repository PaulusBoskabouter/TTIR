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



class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return F.normalize(self.feedforward(x), p = 2, dim = 1) # x-dims = B, F (batch, features)



class DualAugmentedTwoTower(nn.Module):
    def __init__(self, name:str, user_dim:int, item_dim:int, aug_dim:int, hidden_dim:int, embed_dim:int):
        super().__init__()

        self.name = name

        self.train_loss_history = []
        self.val_loss_history = []

        # Categorical  user_id embedding
        # Note: 4457 is hard coded because we know the amount of users in our dataset,
        # which is fine for the experimental setup we're doing. But ideally, this would need improving.
        self.user_id_embedder = nn.Embedding(4457, embed_dim, padding_idx=0)
        
        # Tower initialisations
        self.user_tower = Tower(user_dim + aug_dim + embed_dim, hidden_dim, embed_dim)
        self.item_tower = Tower(item_dim + aug_dim, hidden_dim, embed_dim)

        # Augmentation layers
        self.au = nn.Parameter(torch.randn(aug_dim))  # user augmented vector
        self.av = nn.Parameter(torch.randn(aug_dim))  # item augmented vector

    def forward(self, user_features, user_id, song_features, labels):
        # convert user_ids to the embedded vector and concatinate with user features
        user_vec = self.user_id_embedder(user_id)
        user_features = torch.cat([user_features, user_vec], dim = 1)  # shape (B, sum_embedding_dims)

        # Expand augmented vectors to batch size
        au_batch = self.au.expand(user_features.size(0), -1)  # shape (B, aug_dim)
        av_batch = self.av.expand(song_features.size(0), -1)  # shape (B, aug_dim)
        

        pu = self.user_tower(torch.cat((user_features, au_batch), dim=1))
        pv = self.item_tower(torch.cat((song_features, av_batch), dim=1))

        # Adaptive mimic mechanism (stop gradient for embeddings)
        with torch.no_grad():
            pu_detach = pu.detach()
            pv_detach = pv.detach()

        au = self.au + (pv_detach - self.au) * labels.unsqueeze(1) # au when label = 0, pv when label = 1
        av = self.av + (pu_detach - self.av) * labels.unsqueeze(1) # av when label = 0, pu when label = 1

        # Compute mimic losses (mean squared error)
        loss_u = F.mse_loss(au, pv_detach) # 0 when label = 0
        loss_v = F.mse_loss(av, pu_detach) # 0 when label = 0

        # Final dot-product score
        score = torch.sum(pu * pv, dim = 1)

        return score, loss_u, loss_v

    def loss(self, score, loss_u, loss_v, labels, lambda_u = 1, lambda_v = 1):
        loss_p = F.binary_cross_entropy_with_logits(score, labels.float())
        return loss_p + lambda_u * loss_u + lambda_v * loss_v
    
    def plot(self, epochs:int, folder:Optional[Path] = None) -> None: 
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






def train_model(model:DualAugmentedTwoTower, train_dataloader:DataLoader, val_dataloader:DataLoader, optimizer:Adam, num_epochs:int = 10, lambda_u:float = 1.0, lambda_v:float = 1.0, device:str = 'cpu'):
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
    patience = 10
    patience_counter = 0
    

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        epoch_train_loss = 0.0
        
        for user_features, user_id, song_embedding, labels in train_dataloader:
     
            # Move to device
            user_features = user_features.to(device)
            user_id = user_id.to(device)
            song_embedding = song_embedding.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            score, loss_u, loss_v = model(user_features, user_id, song_embedding, labels)

            # Compute combined loss
            loss = model.loss(score, loss_u, loss_v, labels, lambda_u, lambda_v)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * labels.size(0)

        epoch_train_loss /= len(train_dataloader.dataset)


        
        # Validate
        epoch_val_loss = 0.0
        with torch.no_grad():
            for user_features, user_id, song_embedding, labels in val_dataloader:
                model.eval()

                # Move to device
                user_features = user_features.to(device)
                user_id = user_id.to(device)
                song_embedding = song_embedding.to(device)
                labels = labels.to(device)

                score, loss_u, loss_v = model(user_features, user_id, song_embedding, labels)
                loss = model.loss(score, loss_u, loss_v, labels, lambda_u, lambda_v)

                epoch_val_loss += loss.item() * labels.size(0)
            
        epoch_val_loss /= len(val_dataloader.dataset)
        

        clear_output(wait=True) # make sure our notebook doesn't get cluttered
        print(f"Epoch: [{epoch}/{num_epochs}]")
        
        if epoch_val_loss > best_val_loss:
            # if not, we wait for 'patience_couter' amount of epochs to improve
            if patience_counter > patience:
                # if not we early stop
                print('early stopped')
                return
            patience_counter += 1
        
        # if improved, save the model and plot
        else:
            patience_counter = 0
            best_val_loss = epoch_val_loss
            model.save(Path("models"))
        
        model.add_losses(epoch_train_loss, epoch_val_loss)
        model.plot(num_epochs, Path("models") / "plots")
