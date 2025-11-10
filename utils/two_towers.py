import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np


class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return F.normalize(self.feedforward(x), p = 2, dim = 1) # x-dims = B, F (batch, features)



class DualAugmentedTwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, hidden_dim, embed_dim):
        super().__init__()

        # User feature embedding
        self.user_embeddings = nn.ModuleList([
            nn.Embedding(feat_cardinality, feat_embed_dim)
            for feat_cardinality, feat_embed_dim in list(user_feat.values())
        ])

        # Tower initialisations
        self.user_tower = Tower(user_dim + aug_dim, hidden_dim, embed_dim)
        self.item_tower = Tower(item_dim + aug_dim, hidden_dim, embed_dim)
        self.au = nn.Parameter(torch.randn(embed_dim))  # user augmented vector
        self.av = nn.Parameter(torch.randn(embed_dim))  # item augmented vector

    def forward(self, user_x, item_x, labels):

        # user_x shape: (B, F) with indices for each categorical feature
        user_embedding = [
            embed(user_x[:, f]) for f, embed in enumerate(self.user_embeddings)
        ]
        user_x = torch.cat(user_embedding, dim = 1)  # shape (B, sum_embedding_dims)

        # Expand augmented vectors to batch size
        au_batch = self.au.expand(user_x.size(0), -1)  # shape (B, aug_dim)
        av_batch = self.av.expand(item_x.size(0), -1)  # shape (B, aug_dim)

        pu = self.user_tower(torch.cat((user_x, au_batch), dim=1))
        pv = self.item_tower(torch.cat((item_x, av_batch), dim=1))

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


def train_model(model, dataloader, optimizer, num_epochs = 10, lambda_u = 1.0, lambda_v = 1.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            user_x, item_x, labels = batch

            # Move to device
            user_x = user_x.to(device)
            item_x = item_x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            score, loss_u, loss_v = model(user_x, item_x, labels)

            # Compute combined loss
            loss = model.loss(score, loss_u, loss_v, labels, lambda_u, lambda_v)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)

        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}: Avg Loss = {avg_loss:.4f}")