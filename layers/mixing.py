import torch
import torch.nn as nn

class TemporalFeatureMixing(nn.Module):
    """
    Temporal Feature Mixing via Interpolation Gating.
    Mixes diffused representations Z_l and Z_{l-1} from consecutive timestamps.
    """
    def __init__(self, hidden_dim: int):
        super(TemporalFeatureMixing, self).__init__()
        # Linear layer computes the interpolation gate based on concatenated representations
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, z_t: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for mixing.
        
        Args:
            z_t (torch.Tensor): Diffused node features at current time step t. Shape (N, hidden_dim)
            z_prev (torch.Tensor): Diffused node features at previous time step t-1. Shape (N, hidden_dim)
            
        Returns:
            torch.Tensor: Mixed representations H_t of shape (N, hidden_dim)
        """
        # If this is the first timestamp, z_prev might be None or zeros.
        if z_prev is None:
            return z_t
            
        # Compute interpolation gate: sigma(W [Z_t, Z_{t-1}])
        combined = torch.cat([z_t, z_prev], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))
        
        # Interpolate
        # If gate is close to 1, we rely on the current representation Z_t
        # If gate is close to 0, we rely on the historical representation Z_{t-1}
        h_t = gate * z_t + (1.0 - gate) * z_prev
        
        return h_t
