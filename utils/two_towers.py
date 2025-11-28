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
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return F.normalize(self.feedforward(x), p = 2, dim = 1) # x-dims = B, F (batch, features)



class DualAugmentedTwoTower(nn.Module):
    def __init__(self, name:str, user_dim:int, item_dim:int, hidden_dim:int, aug_dim:int, user_embed_dim):
        super().__init__()

        self.name = name

        self.train_loss_history = []
        self.val_loss_history = []

        # user id
        self.user_id_embedder = nn.Embedding(4457, user_embed_dim, padding_idx=0)

        # Tower initialisations
        self.user_tower = Tower(user_dim + aug_dim+ user_embed_dim, hidden_dim, aug_dim)
        self.item_tower = Tower(item_dim + aug_dim, hidden_dim, aug_dim)

        # Augmentation layers
        self.au = nn.Parameter(torch.randn(aug_dim))  # user augmented vector
        self.av = nn.Parameter(torch.randn(aug_dim))  # item augmented vector

        



    def forward(self, user_features, song_features, user_id):
        # convert user_ids to the embedded vector and concatinate with user features
        user_vec = self.user_id_embedder(user_id)
        user_features = torch.cat([user_features, user_vec], dim = 1)

        # Expand augmented vectors to batch size
        au_batch = self.au.expand(user_features.size(0), -1)  # shape (B, aug_dim)
        av_batch = self.av.expand(song_features.size(0), -1)  # shape (B, aug_dim)
        
        pu = self.user_tower(torch.cat((user_features, au_batch), dim=1))
        pv = self.item_tower(torch.cat((song_features, av_batch), dim=1))

        # Final dot-product score
        score = (pu * pv).sum(dim = 1)

        # Return pu_detach & pv_detach for loss computation
        return score, pu, pv


    
    def loss(self, score, pu, pv, labels, lambda_u = 1, lambda_v = 1, tau:float = 0.07):
        """
        Calculate & combine all loss terms including:
            - Loss_p: dot-product loss
            - Loss_u: AMM loss of user tower
            - Loss_v: AMM loss of item tower
        """

        
        # Dot-product loss
        loss_p = F.binary_cross_entropy(score, labels)#F.binary_cross_entropy_with_logits(score, labels.float())

        # Stop gradient
        pu_detach = pu.detach()
        pv_detach = pv.detach()

        # Expand learnable aug vectors to batch
        au_exp = self.au.unsqueeze(0).expand_as(pv_detach)  # (B, D)
        av_exp = self.av.unsqueeze(0).expand_as(pu_detach)  # (B, D)

    
        labels = labels.unsqueeze(1).float()    # [B, 1]
        

        diff_u = labels * (au_exp - pv_detach)    # [B, D]
        diff_v = labels * (av_exp - pu_detach)

        loss_u = diff_u.pow(2).mean()
        loss_v = diff_v.pow(2).mean()
        
        
        # per-element mse -> per-sample mse (mean over dim)


        # mse_u_per_sample = F.mse_loss(au_exp, pv_detach, reduction='none').mean(dim = 1)  # (B,)
        # mse_v_per_sample = F.mse_loss(av_exp, pu_detach, reduction='none').mean(dim = 1)  # (B,)

        # # mask positives and average only over positives
        # pos_mask = (labels.view(-1) == 1)
        # loss_u = mse_u_per_sample[pos_mask].mean()
        # loss_v = mse_v_per_sample[pos_mask].mean()
        # print(loss_u)
        # return 0
        return loss_p + lambda_u * loss_u + lambda_v * loss_v
    


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
        for user_features, song_embedding, labels, __, uidx in train_dataloader:
            # Move to device
            user_features = user_features.to(device)
            uidx = uidx.to(device)
            #label_features = label_features.to(device)
            song_embedding = song_embedding.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            # score, pu, pv = model(user_features, label_features, song_embedding)
            score, pu, pv = model(user_features, song_embedding, uidx)

            # Compute combined loss
            loss = model.loss(score, pu, pv, labels, lambda_u, lambda_v)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * labels.size(0)

        epoch_train_loss /= len(train_dataloader.dataset)


        
        # Validate
        epoch_val_loss = 0.0
        with torch.no_grad():
            # for user_features, label_features, song_embedding, labels, __ in
            for user_features, song_embedding, labels, __, uidx in val_dataloader:

                # Move to device
                user_features = user_features.to(device)
                uidx = uidx.to(device)
                # label_features = label_features.to(device)
                song_embedding = song_embedding.to(device)
                labels = labels.to(device)

                #score, pu, pv = model(user_features, label_features, song_embedding)
                score, pu, pv = model(user_features, song_embedding, uidx)
                loss = model.loss(score, pu, pv, labels, lambda_u, lambda_v)

                epoch_val_loss += loss.item() * labels.size(0)
            
        epoch_val_loss /= len(val_dataloader.dataset)
        

        clear_output(wait=True) # make sure our notebook doesn't get cluttered
        print(f"Epoch: [{epoch}/{num_epochs}]")
        
        if epoch_val_loss > best_val_loss:
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