import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GCNConv

class SpatialDiffusion(nn.Module):
    """
    Spatial Diffusion (GNN) module for LTG-SSM.
    Propagates information across the observed graph structure using SAGEConv or GCNConv.
    """
    def __init__(self, in_channels: int, out_channels: int, gnn_type: str = 'sage', dropout: float = 0.0):
        super(SpatialDiffusion, self).__init__()
        
        self.gnn_type = gnn_type.lower()
        if self.gnn_type == 'sage':
            self.conv = SAGEConv(in_channels, out_channels)
        elif self.gnn_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
            
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single graph snapshot.
        
        Args:
            x (torch.Tensor): Node features of shape (N, in_channels)
            edge_index (torch.Tensor): Edge indices of shape (2, E)
            
        Returns:
            torch.Tensor: Diffused node representations Z_l of shape (N, out_channels)
        """
        z = self.conv(x, edge_index)
        z = self.act(z)
        z = self.dropout(z)
        return z
