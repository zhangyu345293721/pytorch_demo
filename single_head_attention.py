import math
import torch.nn as nn
import torch

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        """
        单头注意力机制初始化

        Args:
            d_model (int): 模型的维度
        """
        super(SingleHeadAttention, self).__init__()
        self.d_model = d_model
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: 输出张量
        """
        n_batches = x.size(0)
        
        # 计算 query, key, value
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        
        # 计算注意力
        x = self.attention(query, key, value)
        
        # 通过最终的线性层
        return self.out_linear(x)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        计算注意力值

        Args:
            query (torch.Tensor): 查询矩阵
            key (torch.Tensor): 键矩阵
            value (torch.Tensor): 值矩阵

        Returns:
            torch.Tensor: 注意力输出
        """
        d_k = query.size(-1)
        
        # 计算分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 对分数进行softmax操作
        p_attn = scores.softmax(dim=-1)
        
        # 乘以值矩阵
        return torch.matmul(p_attn, value)

# 示例使用
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 512
    input_seq = torch.randn(batch_size, seq_len, d_model)

    single_head_attention = SingleHeadAttention(d_model)
    output = single_head_attention(input_seq)

    print(output.shape)  # 输出的形状
