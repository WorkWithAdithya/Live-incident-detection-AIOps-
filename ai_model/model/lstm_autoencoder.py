"""
lstm_autoencoder.py
-------------------
LSTM Autoencoder for unsupervised anomaly detection.

Architecture:
    Input  (batch, seq_len=60, 3)
      │
      ▼
    LSTMEncoder  ──►  hidden state  ──►  Linear  ──►  latent (batch, 32)
      │
      ▼
    LSTMDecoder  ◄──  repeat latent across seq_len
      │
      ▼
    Linear  ──►  reconstructed (batch, seq_len, 3)

Anomaly score = MSE(input, reconstructed)
High score  →  pattern deviates from what the model learned as "normal"
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]          # (batch, hidden_size)


class LSTMDecoder(nn.Module):
    def __init__(self, latent_size: int, hidden_size: int, output_size: int,
                 num_layers: int, dropout: float, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.lstm    = nn.LSTM(
            input_size  = latent_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # z: (batch, latent_size)
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)   # (batch, seq_len, latent)
        out, _ = self.lstm(z_rep)                            # (batch, seq_len, hidden)
        return self.out(out)                                 # (batch, seq_len, output)


class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Autoencoder.

    Args:
        input_size  : Number of features (3: cpu, memory, disk)
        hidden_size : LSTM hidden units
        latent_size : Bottleneck dimension
        num_layers  : Stacked LSTM depth
        dropout     : Dropout between LSTM layers
        seq_len     : Sequence window length
    """
    def __init__(
        self,
        input_size  : int   = 3,
        hidden_size : int   = 64,
        latent_size : int   = 32,
        num_layers  : int   = 2,
        dropout     : float = 0.2,
        seq_len     : int   = 60,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.input_size = input_size

        self.encoder          = LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.encoder_to_latent = nn.Linear(hidden_size, latent_size)
        self.decoder          = LSTMDecoder(latent_size, hidden_size, input_size,
                                             num_layers, dropout, seq_len)

    def forward(self, x):
        hidden = self.encoder(x)
        latent = self.encoder_to_latent(hidden)
        return self.decoder(latent)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample MSE between input and reconstruction.
        Use this as the anomaly score at inference time.

        Args:
            x : (batch, seq_len, input_size)
        Returns:
            errors : (batch,)
        """
        with torch.no_grad():
            recon  = self.forward(x)
            errors = torch.mean((x - recon) ** 2, dim=(1, 2))
        return errors


if __name__ == "__main__":
    model = LSTMAutoencoder()
    x     = torch.randn(8, 60, 3)
    out   = model(x)
    errs  = model.reconstruction_error(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Errors : {errs.shape}  →  {errs.numpy().round(4)}")
    print("✅ Model OK")