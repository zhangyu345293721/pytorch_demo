import torch
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self, input_dim : int):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
            
        """
        linear_out = self.linear(x)
        gate_out = torch.sigmoid(self.gate(x))
        return linear_out * gate_out

# for example demo
x = torch.randn(32, 64)  # batch size 32, dim = 64
glu = GLU(input_dim=64)
output = glu(x)
print(output.shape)  # output shape
