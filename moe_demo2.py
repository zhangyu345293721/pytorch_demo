import torch
import torch.nn as nn
import torch.nn.functional as F

class MOE(nn.Module):
    def __init__(self, input_dim, expert_dim, output_dim, num_experts, top_k):
        """
        初始化 MoE 模型
        Args:
            input_dim:    输入特征维度
            expert_dim:   每个专家网络中间隐藏层维度
            output_dim:   模型输出维度
            num_experts:  总专家数量
            top_k:        每次前向传播选择 top-k 个专家
        """
        super(MOE, self).__init__()  # 调用父类初始化（固定写法）
        
        # 保存模型超参数
        self.num_experts = num_experts  # 总专家数
        self.top_k = top_k              # 每次选用的专家数量

        # ==================== 1. 定义门控网络（负责分配专家）====================
        # 线性层：输入 -> 每个专家的权重分数
        self.gating = nn.Linear(input_dim, num_experts)

        # ==================== 2. 定义所有专家网络 ====================
        # ModuleList：专门存储多个网络层/子模型，支持参数训练
        self.experts = nn.ModuleList([
            # 每个专家 = 两层线性层 + 激活函数（标准MLP结构）
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),  # 第一层线性变换
                nn.ReLU(),                         # 激活函数（引入非线性）
                nn.Linear(expert_dim, output_dim)  # 第二层线性变换，输出结果
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        前向传播：数据输入 -> 选专家 -> 计算 -> 输出
        Args:
            x: 输入张量 shape [batch_size, input_dim]
        Returns:
            加权融合后的输出 [batch_size, output_dim]
        """
        # ==================== 步骤1：门控网络计算专家权重 ====================
        # 1) 输入通过线性层得到原始分数
        gate_logits = self.gating(x)
        # 2) softmax 归一化，所有权重和为1，表示每个专家被选择的概率
        gating_weights = F.softmax(gate_logits, dim=-1)

        # ==================== 步骤2：选出权重最高的 top-k 个专家 ====================
        # topk：返回 最大的k个权重值 + 对应专家的编号
        top_k_weights, top_k_indices = torch.topk(gating_weights, self.top_k, dim=-1)

        # ==================== 步骤3：逐个调用选中的专家并加权计算 ====================
        expert_outputs = []  # 存储每个专家的加权输出
        for i in range(self.top_k):
            # 获取当前选中的专家编号
            expert_idx = top_k_indices[:, i]
            # 获取当前专家的权重（并扩展维度方便相乘）
            weight = top_k_weights[:, i].unsqueeze(1)
            
            # 用选中的专家对输入x进行计算
            current_output = self.experts[expert_idx](x)
            # 结果 × 权重（加权）
            weighted_output = current_output * weight
            
            expert_outputs.append(weighted_output)

        # ==================== 步骤4：所有专家结果求和，得到最终输出 ====================
        final_output = torch.sum(torch.stack(expert_outputs), dim=0)

        return final_output


# ==================== 模型使用示例 ====================
if __name__ == '__main__':
    # 初始化 MoE 模型
    model = MOE(
        input_dim=100,    # 输入维度100
        expert_dim=50,    # 专家中间层维度50
        output_dim=10,    # 输出维度10
        num_experts=4,    # 一共4个专家
        top_k=2           # 每次选2个专家
    )

    # 构造一批测试数据：10个样本，每个样本100维
    test_input = torch.randn(10, 100)

    # 模型前向计算
    output = model(test_input)

    # 打印输出形状，验证是否正确
    print("模型输入形状:", test_input.shape)
    print("模型输出形状:", output.shape)
