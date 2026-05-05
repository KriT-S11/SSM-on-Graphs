import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveSSM(nn.Module):
    """
    Adaptive State Space Model with Diagonal Approximation for Temporal Graphs.
    Runs an independent Single-Input Single-Output (SISO) SSM for each feature channel.
    """
    def __init__(self, hidden_dim: int, state_dim: int = 16):
        super(AdaptiveSSM, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Adaptive step size projection
        # Computes Delta for each feature channel based on the full feature vector
        self.delta_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # SSM parameters (independent for each channel)
        # A: Diagonal state matrix (parameterized in log space to enforce negativity/stability)
        self.log_A = nn.Parameter(torch.empty(hidden_dim, state_dim))
        
        # B: Input projection matrix
        self.B = nn.Parameter(torch.empty(hidden_dim, state_dim))
        
        # C: Output projection matrix
        self.C = nn.Parameter(torch.empty(hidden_dim, state_dim))
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize A to represent stable continuous-time dynamics (e.g., negative eigenvalues)
        # We store log(A) so that A = -exp(log_A) is strictly negative.
        nn.init.normal_(self.log_A, mean=0.0, std=0.1)
        nn.init.normal_(self.B, mean=0.0, std=0.1)
        nn.init.normal_(self.C, mean=0.0, std=0.1)
        
    def forward(self, h_t: torch.Tensor, u_prev: torch.Tensor = None):
        """
        Forward pass for the SSM layer.
        
        Args:
            h_t (torch.Tensor): Mixed features at time t. Shape (N, hidden_dim)
            u_prev (torch.Tensor): SSM state from previous time step. Shape (N, hidden_dim, state_dim)
                                   If None, assumes start of sequence.
                                   
        Returns:
            y_t (torch.Tensor): Output features at time t. Shape (N, hidden_dim)
            u_t (torch.Tensor): Updated SSM state at time t. Shape (N, hidden_dim, state_dim)
        """
        N = h_t.size(0)
        
        if u_prev is None:
            # Initialize hidden state if not provided
            u_prev = torch.zeros(N, self.hidden_dim, self.state_dim, device=h_t.device, dtype=h_t.dtype)
            
        # 1. Compute adaptive step size (Delta_t)
        # Shape: (N, hidden_dim)
        delta_t = F.softplus(self.delta_proj(h_t))
        
        # 2. Get continuous-time A matrix
        # Enforce A to be strictly negative for stable dynamics
        A = -torch.exp(self.log_A) # Shape: (hidden_dim, state_dim)
        
        # 3. Discretize dynamics
        # Using diagonal approximation: exp(Delta * A)
        # delta_t shape: (N, hidden_dim) -> (N, hidden_dim, 1) to broadcast with A
        delta_t_expanded = delta_t.unsqueeze(-1)
        
        # exp_Delta_A shape: (N, hidden_dim, state_dim)
        exp_Delta_A = torch.exp(delta_t_expanded * A)
        
        # 4. State Update: U_t = U_{t-1} * exp(Delta_t * A) + Delta_t * H_t * B
        # B shape: (hidden_dim, state_dim)
        # H_t shape: (N, hidden_dim) -> (N, hidden_dim, 1)
        h_t_expanded = h_t.unsqueeze(-1)
        
        # The input term: Delta_t * H_t * B
        # Shape: (N, hidden_dim, 1) * (N, hidden_dim, 1) * (1, hidden_dim, state_dim) -> (N, hidden_dim, state_dim)
        input_term = delta_t_expanded * h_t_expanded * self.B.unsqueeze(0)
        
        u_t = u_prev * exp_Delta_A + input_term
        
        # 5. Output Projection: Y_t = U_t * C
        # Shape: (N, hidden_dim, state_dim) * (1, hidden_dim, state_dim) -> sum over state_dim -> (N, hidden_dim)
        y_t = torch.sum(u_t * self.C.unsqueeze(0), dim=-1)
        
        return y_t, u_t
