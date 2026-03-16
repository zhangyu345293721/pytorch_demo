import torch
import torch.nn as nn
import torch.nn.functional as F

class MOE(nn.Module):
    def __init__(self, input_dim, expert_dim, output_dim, num_experts, top_k):
        super(MOE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gating = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # 门控网络计算专家权重
        gating_weights = F.softmax(self.gating(x), dim=-1)
        
        # 选择Top-K专家
        top_k_weights, top_k_indices = torch.topk(gating_weights, self.top_k, dim=-1)
        
        # 聚合Top-K专家的输出
        expert_outputs = []
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_output = self.experts[expert_idx](x)
            expert_outputs.append(expert_output * top_k_weights[:, i].unsqueeze(1))
        
        return torch.sum(torch.stack(expert_outputs), dim=0)

# 使用示例
model = MOE(input_dim=100, expert_dim=50, output_dim=10, num_experts=4, top_k=2)
x = torch.randn(10, 100)
output = model(x)
