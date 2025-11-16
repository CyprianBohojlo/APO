# Direct Preference Optimisation policy head + loss,  

import torch
import torch.nn as nn
import torch.nn.functional as F

def preference_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
    reference_free: bool = True,
):
    
    # Eq. 2 from the DPO paper.
    # If reference_free, ref_chosen/ref_rejected are zeroed → IPO variant.
    
    if reference_free:
        ref_chosen = torch.zeros_like(chosen_logp)
        ref_rejected = torch.zeros_like(rejected_logp)

    # Δ = β [ (logπθ(y⁺)-logπref(y⁺)) − (logπθ(y⁻)-logπref(y⁻)) ]
    logits = beta * ((chosen_logp - ref_chosen) - (rejected_logp - ref_rejected))
    targets = torch.ones_like(logits)
    return F.binary_cross_entropy_with_logits(logits, targets), logits


class PromptPolicy(nn.Module):
    """
    Stateless MLP: dummy zero‐state → logits over 'pool_size' prompts.
    Mirrors PPO.ActorCritic but only the actor head.
    """

    def __init__(self, pool_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pool_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, pool_size),
        )
        # buffer for the single zero-state
        self.register_buffer("state_zero", torch.zeros(1, pool_size))

    def forward(self) -> torch.Tensor:
        # raw logits shape = [pool_size]
        return self.net(self.state_zero).squeeze(0)

    def log_softmax(self) -> torch.Tensor:
        return self.forward().log_softmax(-1)

    def softmax(self) -> torch.Tensor:
        return self.forward().softmax(-1)
