"""
Direct Preference Optimisation (DPO) Alignment Layer
=====================================================
Implements:
  - DPO loss (Rafailov et al. 2023) for the symbolic planning model
  - Reference model management (frozen copy)
  - Reward computation for evaluation
"""

from __future__ import annotations
import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DPOTrainer(nn.Module):
    """
    Wraps a policy model and its frozen reference copy.
    Implements the DPO objective:

        L_DPO = -E[log sigmoid(beta * (log pi(y_w|x) - log pi_ref(y_w|x))
                              - beta * (log pi(y_l|x) - log pi_ref(y_l|x)))]

    where y_w = chosen, y_l = rejected continuations.
    beta controls the KL penalty strength.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        beta:          float = 0.1,
    ):
        super().__init__()
        self.policy    = policy_model
        self.reference = copy.deepcopy(policy_model)
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.beta = beta

    def _log_probs(
        self,
        model: nn.Module,
        input_ids: Tensor,
        labels:    Tensor,
        mask:      Optional[Tensor] = None,
    ) -> Tensor:
        logits, _ = model(input_ids, attention_mask=mask)
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            token_lp = token_lp * mask.float()
        return token_lp.sum(dim=-1)

    def dpo_loss(
        self,
        prompt_ids:    Tensor,
        prompt_mask:   Tensor,
        chosen_ids:    Tensor,
        chosen_mask:   Tensor,
        rejected_ids:  Tensor,
        rejected_mask: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        def concat_and_score(cont_ids, cont_mask):
            full_ids  = torch.cat([prompt_ids,  cont_ids],  dim=1)
            full_mask = torch.cat([prompt_mask, cont_mask], dim=1)
            label_mask = torch.cat([
                torch.zeros_like(prompt_mask),
                cont_mask
            ], dim=1)
            inp    = full_ids[:, :-1]
            labels = full_ids[:, 1:]
            lmask  = label_mask[:, 1:]
            amask  = full_mask[:, :-1]
            lp_policy = self._log_probs(self.policy, inp, labels * lmask, amask)
            with torch.no_grad():
                lp_ref = self._log_probs(self.reference, inp, labels * lmask, amask)
            return lp_policy, lp_ref

        lp_chosen_pol,  lp_chosen_ref  = concat_and_score(chosen_ids,   chosen_mask)
        lp_rejected_pol, lp_rejected_ref = concat_and_score(rejected_ids, rejected_mask)

        chosen_diff   = lp_chosen_pol   - lp_chosen_ref
        rejected_diff = lp_rejected_pol - lp_rejected_ref
        logits_dpo    = self.beta * (chosen_diff - rejected_diff)
        loss          = -F.logsigmoid(logits_dpo).mean()

        metrics = {
            "dpo_loss":         loss.item(),
            "chosen_reward":    chosen_diff.mean().item(),
            "rejected_reward":  rejected_diff.mean().item(),
            "reward_margin":    (chosen_diff - rejected_diff).mean().item(),
            "accuracy":         (logits_dpo > 0).float().mean().item(),
        }
        return loss, metrics

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        return self.dpo_loss(
            batch["prompt_ids"],   batch["prompt_mask"],
            batch["chosen_ids"],   batch["chosen_mask"],
            batch["rejected_ids"], batch["rejected_mask"],
        )

    def sync_reference(self):
        """Hard-update reference model from current policy weights."""
        self.reference.load_state_dict(
            copy.deepcopy(self.policy.state_dict()))
        for p in self.reference.parameters():
            p.requires_grad_(False)
