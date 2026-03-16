from pathlib import Path

import torch
import torch.nn as nn

MODEL_PATH = Path(__file__).parent / "model.pth"


class QNetwork(nn.Module):
    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 3) -> None:
        """Build a feedforward Q-network with two hidden layers.

        Args:
            input_size:  Number of input features (state vector length).
            hidden_size: Number of neurons in each hidden layer.
            output_size: Number of output Q-values (one per action).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Q-value tensor of shape (batch_size, output_size).
        """
        return self.net(x)

    def save(self, path: Path = MODEL_PATH) -> None:
        """Save the model weights to disk.

        Args:
            path: File path to save the weights to.
        """
        torch.save(self.state_dict(), path)
        print(f"[model] saved → {path}")

    def load(self, path: Path = MODEL_PATH) -> bool:
        """Load model weights from disk if the file exists.

        Args:
            path: File path to load the weights from.

        Returns:
            True if weights were loaded successfully, False otherwise.
        """
        if path.exists():
            try:
                self.load_state_dict(torch.load(path, weights_only=True))
                print(f"[model] loaded ← {path}")
                return True
            except RuntimeError:
                print(f"[model] {path} is incompatible (architecture changed) — starting fresh")
        return False
