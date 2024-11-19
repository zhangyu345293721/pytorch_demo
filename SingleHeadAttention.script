import torch
import torch.nn
import math


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int):
        super(SingleHeadAttention, self).__init__()
        self.d_model = d_model
        self.query_linear = nn.linear(d_model, d_model)
        self.key_linear = nn.linear(d_model, d_model)
        self.value_linear = nn.linear(d_model, d_model)
        self.output_linear = nn.linear(d_model, d_model)

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            forward tensor
        Args:
            input: input tensor
            mask: masked matrix

        Returns:
            tensor
        """
        query = self.query_linear(input)
        key = self.key_linear(input)
        value = self.value_linear(input)
        # attention value
        attention = self.attention(query, key, value, mask)
        # add a linear
        return self.output_linear(attention)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.tensor,
                  mask: torch.Tensor) -> torch.tensor:
        """
            get attention score
        Args:
            query: tensor
            key: tensor
            value: tensor
            mask: masked tensor

        Returns:
            attention score
        """
        # d_k size
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        # softmax
        soft_score = scores.softmax(dim=-1)
        return torch.matmul(soft_score, value)
