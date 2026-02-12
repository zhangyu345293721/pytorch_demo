import math
import torch
from torch import nn
from transformers.activations import ACT2FN
import torch.nn.functional as F
from transformers import PretrainedConfig


class Config(PretrainedConfig):
    """Configuration class for MoE models, inheriting from Hugging Face's PretrainedConfig.
    
    This class defines all hyperparameters for the Mixture of Experts (MoE) architecture.
    Key features:
    - Handles both routed experts and shared experts
    - Supports dimension-preserving intermediate layers
    - Configures expert selection and loss calculation
    
    Note: If intermediate_size is None, it's automatically calculated as 8/3 * hidden_size
    and aligned to the nearest multiple of 64 for memory optimization.
    """
    
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
    ) -> None:
        """Initialize MoE configuration.
        
        Args:
            dropout: Dropout rate for regularization (float, default=0.0)
            hidden_act: Activation function for hidden layers (str, default='silu')
            hidden_size: Dimension of hidden layers (int, default=512)
            intermediate_size: Dimension of intermediate layers (int, default=None)
            num_experts_per_tok: Number of experts to select per token (int, default=2)
            n_routed_experts: Total number of routed experts (int, default=4)
            n_shared_experts: Number of shared experts (int, default=1)
            scoring_func: Scoring function for expert selection (str, default='softmax')
            aux_loss_alpha: Weight for auxiliary loss (float, default=0.1)
            seq_aux: Whether to compute auxiliary loss at sequence level (bool, default=True)
            norm_topk_prob: Whether to normalize top-k probabilities (bool, default=True)
            **kwargs: Additional keyword arguments for PretrainedConfig
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
    """Feed-forward neural network module with gate mechanism.
    
    Implements a GLU-like structure where:
    output = down_proj(activation(gate_proj(x) * up_proj(x)))
    
    Key features:
    - Automatically calculates intermediate_size if not provided
    - Uses activation function specified in config
    - Includes dropout for regularization
    
    Note: intermediate_size is rounded to nearest multiple of 64 for memory optimization.
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize FeedForward module.
        
        Args:
            config: MoE configuration object (Config)
        """
        super().__init__()
        # Calculate intermediate_size if not provided
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # Align to 64 for memory optimization
            config.intermediate_size = 64 * ((intermediate_size + 63) // 64)
        
        # Define layers
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # Activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computation.
        
        Args:
            x: Input tensor with shape [batch_size, seq_len, hidden_size]
        
        Returns:
            Output tensor with same shape as input [batch_size, seq_len, hidden_size]
        
        Computation:
        1. Compute gate and up projections: gate = gate_proj(x), up = up_proj(x)
        2. Apply activation: activation(gate * up)
        3. Apply down projection: down_proj(activation(gate * up))
        4. Apply dropout
        """
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """MoE gating mechanism for expert selection and weighting.
    
    Responsible for:
    - Computing expert scores for each token
    - Selecting top-k experts per token
    - Generating expert weights
    - Calculating auxiliary loss for training
    
    Key features:
    - Supports both softmax and custom scoring functions
    - Handles top-k probability normalization
    - Computes auxiliary loss for expert balance
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize MoE gating mechanism.
        
        Args:
            config: MoE configuration object (Config)
        """
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        
        # Weight matrix for expert scoring
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weight parameters using Kaiming uniform distribution.
        
        Note: Uses math.sqrt(5) for fan-in initialization as per PyTorch's default.
        """
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute expert selection and weights.
        
        Args:
            hidden_states: Input tensor with shape [batch_size, seq_len, hidden_size]
        
        Returns:
            topk_idx: Indices of selected experts (shape [batch_size * seq_len, top_k])
            topk_weight: Corresponding weights for selected experts (shape same as topk_idx)
            aux_loss: Auxiliary loss for expert balance (scalar tensor)
        
        Computation flow:
        1. Reshape input: [batch_size * seq_len, hidden_size]
        2. Compute logits: scores = hidden_states * weight^T
        3. Apply scoring function (softmax by default)
        4. Select top-k experts and weights
        5. Normalize top-k weights if needed
        6. Compute auxiliary loss (during training only)
        """
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)  # [batch_size*seq_len, hidden_size]
        
        # Compute expert scores
        logits = F.linear(hidden_states, self.weight)
        
        # Apply scoring function
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"Unsupported scoring function for MoE gating: {self.scoring_func}")
        
        # Get top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        # Normalize top-k weights
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        
        # Compute auxiliary loss (only during training)
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # Sequence-level auxiliary loss
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, 
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                               ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # Token-level auxiliary loss
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
        
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """Mixture of Experts feed-forward network.
    
    Combines multiple expert networks with gating mechanism. Handles:
    - Routed experts (selective activation)
    - Shared experts (always active)
    - Training/inference optimization
    - Auxiliary loss for expert balance
    
    Key features:
    - Dimension-preserving output
    - Efficient training with token replication
    - Optimized inference using batched expert processing
    """
    
    def __init__(self, config: Config) -> None:
        """Initialize MOE feed-forward network.
        
        Args:
            config: MoE configuration object (Config)
        """
        super().__init__()
        self.config = config
        
        # Initialize routed experts
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        
        # Initialize gating mechanism
        self.gate = MoEGate(config)
        
        # Initialize shared experts (if configured)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computation.
        
        Args:
            x: Input tensor with shape [batch_size, seq_len, hidden_size]
        
        Returns:
            Output tensor with same shape as input [batch_size, seq_len, hidden_size]
        
        Training vs Inference:
        - Training: Replicates input tokens for each expert, processes separately
        - Inference: Uses optimized batched processing (moe_infer)
        """
        identity = x  # For residual connection
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        
        # Get expert selection and weights
        topk_idx, topk_weight, aux_loss = self.gate(x)
        
        # Prepare for expert processing
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:
            # Training mode: Token replication
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            
            # Process each expert's tokens
            for i, expert in enumerate(self.experts):
                # Get tokens assigned to expert i
                mask = flat_topk_idx == i
                y[mask] = expert(x[mask]).to(y.dtype)  # Maintain dtype consistency
            
            # Combine expert outputs with weights
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # Inference mode: Optimized batched processing
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # Add shared experts (if configured)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        
        # Save auxiliary loss for training
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self,x: torch.Tensor,flat_expert_indices: torch.Tensor,flat_expert_weights: torch.Tensor) -> torch.Tensor:
        """Optimized inference for MoE layer.
        
        Args:
            x: Input tensor (flattened) [batch_size*seq_len, hidden_size]
            flat_expert_indices: Expert indices for each token [batch_size*seq_len]
            flat_expert_weights: Expert weights for each token [batch_size*seq_len, 1]
        
        Returns:
            Combined output tensor with same shape as input [batch_size*seq_len, hidden_size]
        
        Optimization strategy:
        - Sorts tokens by expert index
        - Groups tokens by expert
        - Processes each expert's tokens in batch
        - Uses scatter_add for efficient accumulation
        """
        expert_cache = torch.zeros_like(x)
        
        # Sort tokens by expert index
        idxs = flat_expert_indices.argsort()
        # Count tokens per expert
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        
        # Compute token indices for each expert
        token_idxs = idxs 
        
        # Process each expert in batches
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx >= end_idx:
                continue
                
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            
            # Process expert tokens
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out = expert_out * flat_expert_weights[idxs[start_idx:end_idx]]
            
            # Accumulate results
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        
        return expert_cache
