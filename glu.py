import torch
import torch.nn as nn

class GLU(nn.Module):
    
    def __init__(self, input_dim: int) -> None:
        """Initializes GLU module.
        
        Args:
            input_dim: Feature dimension of input tensor (int)
        """
        super(GLU, self).__init__()
        # Linear transformation layer: maps input to same dimension
        self.linear = nn.Linear(input_dim, input_dim)
        # Gating layer: generates sigmoid gate signal
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computation.
        
        Args:
            x: Input tensor with shape [batch_size, input_dim]
        
        Returns:
            Output tensor with same shape as input [batch_size, input_dim]
        
        Computation Flow:
        1. Linear transform: x_linear = self.linear(x)
        2. Gate activation: x_gate = torch.sigmoid(self.gate(x))
        3. Gated output: x_linear * x_gate
        """
        linear_out = self.linear(x)  # [batch, input_dim]
        gate_out = torch.sigmoid(self.gate(x))  # [batch, input_dim]
        return linear_out * gate_out  # [batch, input_dim]

if __name__ == "__main__":
    # Create random input: batch_size=32, input_dim=64
    x = torch.randn(32, 64)
    glu = GLU(input_dim=64)
    output = glu(x)
    assert output.shape == x.shape, f"Dimension mismatch! Expected: {x.shape}, Got: {output.shape}"
    print("GLU input-output dimension validation successful:", output.shape)
