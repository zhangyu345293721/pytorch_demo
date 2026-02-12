import math
import torch
from torch import nn
from transformers.activations import ACT2FN
import torch.nn.functional as F
from transformers import PretrainedConfig


class Config(PretrainedConfig):
    """MoE模型的配置类，继承自Hugging Face的PretrainedConfig"""
    
    def __init__(
            self,
            dropout: float = 0.0,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        """
        初始化MoE配置
        
        Args:
            dropout: Dropout率
            hidden_act: 激活函数类型
            hidden_size: 隐藏层维度
            intermediate_size: 中间层维度，如为None则自动计算
            num_experts_per_tok: 每个token选择的专家数量
            n_routed_experts: 总的专家数量
            n_shared_experts: 共享专家数量
            scoring_func: 评分函数类型，默认为'softmax'
            aux_loss_alpha: 辅助损失的alpha参数
            seq_aux: 是否在序列级别计算辅助损失
            norm_topk_prob: 是否标准化top-k概率
        """
        super().__init__(**kwargs)
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts_per_tok = num_experts_per_tok    
        self.n_routed_experts = n_routed_experts          
        self.n_shared_experts = n_shared_experts          
        self.scoring_func = scoring_func                  
        self.aux_loss_alpha = aux_loss_alpha              
        self.seq_aux = seq_aux                            
        self.norm_topk_prob = norm_topk_prob              


class FeedForward(nn.Module):
    """前馈神经网络模块"""
    
    def __init__(self, config: Config):
        super().__init__()
        # 如果未指定中间层维度，则自动计算（基于隐藏层维度的8/3倍）
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 将中间层维度对齐到64的倍数，优化内存访问
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        # 定义网络层
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数

    def forward(self, x):
        # 使用门控激活机制：gate_proj(x) * up_proj(x)
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """MoE门控机制，负责专家选择和权重分配"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  
        self.n_routed_experts = config.n_routed_experts  

        self.scoring_func = config.scoring_func  
        self.alpha = config.aux_loss_alpha       
        self.seq_aux = config.seq_aux           

        self.norm_topk_prob = config.norm_topk_prob  
        self.gating_dim = config.hidden_size         
        
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用Kaiming均匀分布初始化权重参数"""
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: 输入隐藏状态  
        Returns:
            topk_idx: 选择的专家索引
            topk_weight: 专家权重
            aux_loss: 辅助损失
        """
        # 获取输入形状
        bsz, seq_len, h = hidden_states.shape
        # 展平输入：(batch_size * seq_len, hidden_size)
        hidden_states = hidden_states.view(-1, h)
        
        # 计算专家得分：每个token对每个专家的偏好分数
        logits = F.linear(hidden_states, self.weight, None)
        
        # 使用softmax将得分转换为概率分布
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选择top-k专家和对应的权重
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果选择多个专家且需要标准化概率，则进行归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20  # 避免除零
            topk_weight = topk_weight / denominator

        # 计算辅助损失（仅在训练时且alpha>0时计算）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # 序列级别的辅助损失计算
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # token级别的辅助损失计算
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
            
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # 创建专家网络列表
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 创建门控模块
        self.gate = MoEGate(config)
        # 如果配置了共享专家，则创建共享专家网络
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x  # 保存原始输入用于残差连接
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        # 训练和推理使用不同的处理策略
        if self.training:
            # 训练模式：为每个token复制输入，分别处理
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            
            # 对每个专家处理其负责的token
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            
            # 加权合并专家输出
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式：使用优化的推理方法
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 如果配置了共享专家，则添加共享专家的输出
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        # 保存辅助损失用于反向传播
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        
        # 对专家索引进行排序，便于批量处理
        idxs = flat_expert_indices.argsort()
        # 计算每个专家处理的token数量
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算对应的token索引
        token_idxs = idxs // self.config.num_experts_per_tok
        
        # 对每个专家批量处理其负责的token
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
                
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            
            # 专家处理并加权
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # 将结果累加到缓存中
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache
