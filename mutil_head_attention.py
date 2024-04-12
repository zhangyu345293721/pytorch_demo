import math
import torch.nn as nn
import torch
import torch.nn.functional as F


"""
    mutil head attention code 
"""


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int) -> None:
        """
            mutil head attention init
        Args:
            h: head number
            d_model: vector dim
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # define W^q, W^k, W^v and W^o matrix, w_list
        self.linear_list = [
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            forward function,
        Args:
            x: input matrix
        Returns:
            output tensor matrix
        """
        n_batches = x.size(0)
        # compute query,key and value
        query, key, value = [
            linear(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2) for linear, x in
            zip(self.linear_list, (x, x, x))
        ]
        x = self.attention(query, key, value)
        # concat attention to mutil-head-attention
        x = (x.transpose(1, 2).contiguous().view(n_batches, -1, self.h * self.d_k))
        # w * o
        return self.linear_list[-1](x)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
              compute Attention result
              input Q,K,V  MultiHeadedAttention class。
        Args:
            query: query matrix
            key: key matrix
            value: value matrix
        Returns:
            attention value
        """
        d_k = query.size(-1)
        # execute QK^T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # equal dim == 2, row softmax
        p_attn = scores.softmax(dim=-1)
        return torch.matmul(p_attn, value)
