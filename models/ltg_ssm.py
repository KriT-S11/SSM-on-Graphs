import torch
import torch.nn as nn
from typing import List, Tuple

from ltg_ssm.layers.gnn import SpatialDiffusion
from ltg_ssm.layers.mixing import TemporalFeatureMixing
from ltg_ssm.layers.ssm import AdaptiveSSM

class LTGSSMBlock(nn.Module):
    """
    A single block of the LTG-SSM architecture.
    Comprises Spatial Diffusion -> Temporal Feature Mixing -> Adaptive SSM -> Residual Projection.
    """
    def __init__(self, in_channels: int, hidden_dim: int, state_dim: int = 16, gnn_type: str = 'sage', dropout: float = 0.1):
        super(LTGSSMBlock, self).__init__()
        
        # 1. Spatial Diffusion
        self.diffusion = SpatialDiffusion(in_channels=in_channels, out_channels=hidden_dim, gnn_type=gnn_type, dropout=dropout)
        
        # 2. Temporal Feature Mixing
        self.mixing = TemporalFeatureMixing(hidden_dim=hidden_dim)
        
        # 3. Adaptive State Space Model
        self.ssm = AdaptiveSSM(hidden_dim=hidden_dim, state_dim=state_dim)
        
        # 4. Residual Projection
        self.residual_proj = nn.Linear(in_channels, hidden_dim)
        
        # Output activation
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_seq: List[torch.Tensor], edge_index_seq: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for a sequence of temporal graph snapshots.
        
        Args:
            x_seq (List[torch.Tensor]): List of length L, where each element is node features at time l of shape (N, in_channels)
            edge_index_seq (List[torch.Tensor]): List of length L, where each element is edge_index at time l
            
        Returns:
            List[torch.Tensor]: Output representations at each time step, each of shape (N, hidden_dim)
        """
        L = len(x_seq)
        out_seq = []
        
        z_prev = None
        u_prev = None
        
        for l in range(L):
            x_l = x_seq[l]
            edge_index_l = edge_index_seq[l]
            
            # 1. Spatial Diffusion: Z_l = GNN(X_l, G_l)
            z_l = self.diffusion(x_l, edge_index_l)
            
            # 2. Temporal Feature Mixing: H_l = Mix(Z_l, Z_{l-1})
            h_l = self.mixing(z_l, z_prev)
            
            # 3. Adaptive State Space Model: Y_l, U_l = SSM(H_l, U_{l-1})
            y_l, u_l = self.ssm(h_l, u_prev)
            
            # 4. Residual Projection: Y_hat_l = Y_l + Linear(X_l)
            y_hat_l = y_l + self.residual_proj(x_l)
            
            # 5. Output activation
            out_l = self.dropout(self.activation(y_hat_l))
            out_seq.append(out_l)
            
            # Update history states
            z_prev = z_l
            u_prev = u_l
            
        return out_seq


class LTGSSM(nn.Module):
    """
    Full LTG-SSM network for Node Classification.
    """
    def __init__(self, num_features: int, hidden_dim: int, num_classes: int, num_layers: int = 2, state_dim: int = 16, gnn_type: str = 'sage', dropout: float = 0.1):
        super(LTGSSM, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First block maps from input features to hidden dim
        self.layers.append(LTGSSMBlock(in_channels=num_features, hidden_dim=hidden_dim, state_dim=state_dim, gnn_type=gnn_type, dropout=dropout))
        
        # Subsequent blocks map hidden dim to hidden dim
        for _ in range(num_layers - 1):
            self.layers.append(LTGSSMBlock(in_channels=hidden_dim, hidden_dim=hidden_dim, state_dim=state_dim, gnn_type=gnn_type, dropout=dropout))
            
        # Final classifier node
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x_seq: List[torch.Tensor], edge_index_seq: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process the entire sequence of temporal graphs and output class logits.
        """
        current_x_seq = x_seq
        
        for layer in self.layers:
            current_x_seq = layer(current_x_seq, edge_index_seq)
            
        # Pass the final layer's output through the classifier for each timestamp
        logits_seq = [self.classifier(x) for x in current_x_seq]
        
        return logits_seq
