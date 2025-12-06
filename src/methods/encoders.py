"""
Encoder architectures for trajectory → feature extraction.

All encoders share the same interface:
    Input:  trajectory (batch, T, 2)
    Output: features (batch, output_dim)
"""

import torch
import torch.nn as nn


class LinearEncoder(nn.Module):
    """
    Simplest possible encoder: flatten + linear projection.
    
    Serves as a baseline to see if complex architectures are needed.
    """
    
    def __init__(self, seq_len: int = 200, output_dim: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # Flatten trajectory and project
        self.fc = nn.Linear(seq_len * 2, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 2)
        Returns:
            (batch, output_dim)
        """
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # (batch, seq_len * 2)
        return self.fc(x_flat)


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder.
    
    Flattens trajectory and passes through several dense layers.
    """
    
    def __init__(
        self,
        seq_len: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 3,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        layers = []
        in_dim = seq_len * 2
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        
        # Remove last dropout
        self.net = nn.Sequential(*layers[:-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        return self.net(x_flat)


class LSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for sequential trajectory processing.
    
    Captures temporal correlations and long-range dependencies.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        self.output_dim = hidden_dim * self.num_directions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        
        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            out = torch.cat([h_forward, h_backward], dim=-1)
        else:
            out = h_n[-1]
        
        return out


class CNNEncoder(nn.Module):
    """
    Multi-scale 1D CNN encoder.
    
    Processes trajectory at multiple time scales using different kernel sizes.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Fine scale: small kernels
        self.fine_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        
        # Medium scale
        self.medium_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        
        # Coarse scale: large kernels
        self.coarse_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = channels * 3
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, 2, seq_len)
        
        fine = self.global_pool(self.fine_conv(x)).squeeze(-1)
        medium = self.global_pool(self.medium_conv(x)).squeeze(-1)
        coarse = self.global_pool(self.coarse_conv(x)).squeeze(-1)
        
        out = torch.cat([fine, medium, coarse], dim=-1)
        return self.dropout(out)


class HybridEncoder(nn.Module):
    """
    Hybrid CNN + LSTM encoder.
    
    Combines local feature extraction (CNN) with sequential modeling (LSTM).
    This was previously called "PINN encoder" but is really just a hybrid architecture.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        cnn_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # CNN branch for local pattern extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        
        # LSTM branch for sequential dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim + cnn_channels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        
        self.output_dim = hidden_dim * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN features
        x_cnn = x.transpose(1, 2)  # (batch, 2, seq_len)
        cnn_features = self.cnn(x_cnn).transpose(1, 2)  # (batch, seq_len, cnn_channels)
        
        # Concatenate with original trajectory
        x_combined = torch.cat([x, cnn_features], dim=-1)
        
        # LSTM encoding
        _, (h_n, _) = self.lstm(x_combined)
        
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        
        return torch.cat([h_forward, h_backward], dim=-1)


# Registry for easy access
ENCODERS = {
    "linear": LinearEncoder,
    "mlp": MLPEncoder,
    "lstm": LSTMEncoder,
    "cnn": CNNEncoder,
    "hybrid": HybridEncoder,
}


def create_encoder(name: str, **kwargs) -> nn.Module:
    """Factory function to create an encoder by name."""
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODERS.keys())}")
    return ENCODERS[name](**kwargs)



