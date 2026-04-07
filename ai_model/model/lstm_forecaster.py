"""
lstm_forecaster.py
------------------
Sequence-to-sequence LSTM that predicts future system metric values.

Architecture:
    Input  : (batch, lookback=60, 3)   — last 60 readings (cpu, mem, disk)
    Encoder: Stacked LSTM → final hidden + cell state
    Decoder: LSTM unrolled for horizon=12 steps → 12 future readings
    Output : (batch, horizon=12, 3)    — next 12 predicted readings

At 5s logging interval:
    60 input steps  = 5-minute history window
    12 output steps = 1-minute forecast horizon

Usage:
    model = LSTMForecaster()
    x     = torch.randn(batch, 60, 3)   # normalised input
    y_hat = model(x)                     # (batch, 12, 3) normalised predictions
    # scaler.inverse_transform to get back to % values
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    Encoder-Decoder LSTM for multi-step multivariate time series forecasting.

    The encoder reads the full input sequence and compresses it into a
    hidden state. The decoder uses that hidden state to generate predictions
    one step at a time (autoregressive), feeding each prediction back as
    input to the next decoder step.

    Args:
        input_size      : Number of input features (3: cpu, memory, disk)
        hidden_size     : LSTM hidden units
        num_layers      : Stacked LSTM depth
        dropout         : Dropout between LSTM layers
        lookback        : Input sequence length (must match training)
        horizon         : How many steps ahead to forecast
    """

    def __init__(
        self,
        input_size  : int   = 3,
        hidden_size : int   = 128,
        num_layers  : int   = 2,
        dropout     : float = 0.2,
        lookback    : int   = 60,
        horizon     : int   = 12,
    ):
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lookback    = lookback
        self.horizon     = horizon

        # ── Encoder ───────────────────────────────────────────────────────────
        # Reads the full 60-step input window
        self.encoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        # Generates predictions one step at a time using encoder's hidden state
        self.decoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Projects hidden state → metric prediction (3 values)
        self.output_projection = nn.Linear(hidden_size, input_size)

    def forward(
        self,
        x             : torch.Tensor,
        teacher_forcing: float = 0.0,
        target        : torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x               : (batch, lookback, input_size) — normalised input
            teacher_forcing : Probability of using true target as next decoder input
                              (only used during training, set to 0.0 at inference)
            target          : (batch, horizon, input_size) — true future values
                              (required only when teacher_forcing > 0)

        Returns:
            predictions : (batch, horizon, input_size) — normalised forecasts
        """
        batch_size = x.size(0)

        # ── Encode full input sequence ────────────────────────────────────────
        _, (hidden, cell) = self.encoder(x)
        # hidden, cell: (num_layers, batch, hidden_size)

        # ── Decode step by step ───────────────────────────────────────────────
        # First decoder input = last observed value in the input sequence
        decoder_input = x[:, -1:, :]   # (batch, 1, input_size)

        predictions = []

        for t in range(self.horizon):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            # out: (batch, 1, hidden_size)

            pred = self.output_projection(out)  # (batch, 1, input_size)
            predictions.append(pred)

            # Next decoder input: use prediction OR true value (teacher forcing)
            use_teacher = (
                teacher_forcing > 0.0
                and target is not None
                and torch.rand(1).item() < teacher_forcing
            )
            decoder_input = target[:, t:t+1, :] if use_teacher else pred

        predictions = torch.cat(predictions, dim=1)  # (batch, horizon, input_size)

        # Clamp to [0, 1] since inputs are normalised to this range
        return torch.clamp(predictions, 0.0, 1.0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-only forward pass (no teacher forcing, no gradients).

        Args:
            x : (batch, lookback, 3) normalised tensor
        Returns:
            (batch, horizon, 3) normalised forecasts
        """
        self.eval()
        return self.forward(x, teacher_forcing=0.0)


if __name__ == "__main__":
    model  = LSTMForecaster()
    dummy  = torch.randn(4, 60, 3)
    output = model.predict(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {output.shape}")
    print(f"Output range : [{output.min():.4f}, {output.max():.4f}]")

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
    print("✅ LSTMForecaster OK")