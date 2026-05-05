import torch
import torch.nn.functional as F

from ltg_ssm.models.ltg_ssm import LTGSSM

def test_ltg_ssm():
    print("Testing LTG-SSM architecture...")
    
    # Synthetic Graph Parameters
    num_nodes = 100
    num_features = 32
    num_classes = 5
    hidden_dim = 64
    seq_length = 5  # Number of temporal snapshots
    
    # 1. Generate Synthetic Temporal Graph Sequence
    x_seq = []
    edge_index_seq = []
    y_true_seq = []
    
    for t in range(seq_length):
        # Node features
        x_seq.append(torch.randn(num_nodes, num_features))
        
        # Random edge indices (approx 300 edges)
        src = torch.randint(0, num_nodes, (300,))
        dst = torch.randint(0, num_nodes, (300,))
        edge_index_seq.append(torch.stack([src, dst], dim=0))
        
        # Ground truth labels (for node classification)
        y_true_seq.append(torch.randint(0, num_classes, (num_nodes,)))
        
    print(f"Generated temporal graph with {seq_length} snapshots, {num_nodes} nodes each.")
    
    # 2. Instantiate Model
    model = LTGSSM(
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=2,
        state_dim=16,
        gnn_type='sage',
        dropout=0.1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 3. Forward Pass
    model.train()
    logits_seq = model(x_seq, edge_index_seq)
    
    print(f"Forward pass successful. Output length: {len(logits_seq)}, Output shape per step: {logits_seq[0].shape}")
    
    # 4. Loss Computation & Backward Pass
    loss = 0
    for t in range(seq_length):
        loss += F.cross_entropy(logits_seq[t], y_true_seq[t])
        
    loss = loss / seq_length
    print(f"Computed Cross-Entropy Loss: {loss.item():.4f}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Verify gradients
    has_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
    if has_grad:
        print("Backward pass successful. Gradients successfully computed and backpropagated.")
    else:
        print("WARNING: Backward pass failed or zero gradients.")
        
    print("LTG-SSM architecture test completed successfully!")

if __name__ == "__main__":
    test_ltg_ssm()
