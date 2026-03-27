import torch
import torch.nn as nn
import numpy as np
from scipy.special import legendre

class LegendrePositionEmbedding(nn.Module):
    """
    Legendre Polynomial Position Embedding
    Uses Legendre polynomials to encode positional information
    """
    def __init__(self, d_model, max_len=5000, degree=10):
        super(LegendrePositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.degree = degree
        
        # Pre-compute Legendre polynomial embeddings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Normalize positions to [-1, 1] for Legendre polynomials
        normalized_pos = 2 * (position / max_len) - 1
        
        # Compute Legendre polynomials for each position
        for i in range(d_model):
            poly_degree = i % degree
            legendre_poly = legendre(poly_degree)
            pe[:, i] = torch.from_numpy(legendre_poly(normalized_pos.numpy())).squeeze()
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Positional embeddings of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        return self.pe[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)

# Made with Bob
