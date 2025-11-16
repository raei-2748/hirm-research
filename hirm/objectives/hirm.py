"""Head-invariant risk minimization objective."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from hirm.objectives.common import compute_env_risks, flatten_head_gradients


class HIRMObjective:
    """Head-Invariant Risk Minimization objective operating on head gradients."""

    def __init__(self, cfg: Any) -> None:
        self.lambda_invariance = float(getattr(cfg, "lambda_invariance", 1.0))
        self.grad_normalization = getattr(cfg, "grad_normalization", "l2").lower()

    def __call__(
        self,
        policy,
        batch: Dict[str, Tensor],
        env_ids: Tensor,
        risk_fn,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        env_risks, pnl, _, _ = compute_env_risks(policy, batch, env_ids, risk_fn)
        risks = torch.stack(list(env_risks.values()))
        mean_risk = risks.mean()
        penalty, grad_cos = self._head_gradient_penalty(policy, env_risks)
        loss = mean_risk + self.lambda_invariance * penalty
        logs: Dict[str, Tensor] = {
            "loss": loss.detach(),
            "risk/mean": mean_risk.detach(),
            "hirm/penalty": penalty.detach(),
            "hirm/lambda": torch.tensor(self.lambda_invariance),
            "pnl/mean": pnl.mean().detach(),
        }
        if grad_cos is not None:
            logs["hirm/grad_cosine"] = grad_cos.detach()
        for env, risk in env_risks.items():
            logs[f"risk/env_{env}"] = risk.detach()
        return loss, logs

    def _head_gradient_penalty(
        self,
        policy,
        env_risks: Dict[int, Tensor],
    ) -> Tuple[Tensor, Tensor | None]:
        head_params = list(policy.head_parameters())
        if not head_params:
            raise ValueError("Policy must expose head parameters for HIRM")
        grad_vectors: list[Tensor] = []
        for risk in env_risks.values():
            grads = torch.autograd.grad(
                risk,
                head_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            grad_vec = flatten_head_gradients(grads)
            grad_vec = self._normalize_grad(grad_vec)
            grad_vectors.append(grad_vec)
        if len(grad_vectors) < 2:
            penalty = torch.zeros(1, device=grad_vectors[0].device)
            return penalty, None
        stacked = torch.stack(grad_vectors)
        mean = stacked.mean(dim=0, keepdim=True)
        penalty = ((stacked - mean) ** 2).mean()
        cosine = self._pairwise_cosine(grad_vectors)
        return penalty, cosine

    def _normalize_grad(self, grad_vec: Tensor) -> Tensor:
        if self.grad_normalization in {"l2", "unit"}:
            norm = grad_vec.norm(p=2) + 1e-12
            return grad_vec / norm
        return grad_vec

    @staticmethod
    def _pairwise_cosine(grad_vectors: list[Tensor]) -> Tensor:
        device = grad_vectors[0].device
        total = torch.zeros(1, device=device)
        count = 0
        for idx in range(len(grad_vectors)):
            for jdx in range(idx + 1, len(grad_vectors)):
                total += F.cosine_similarity(grad_vectors[idx], grad_vectors[jdx], dim=0)
                count += 1
        if count == 0:
            return torch.ones(1, device=device)
        return total / count


__all__ = ["HIRMObjective"]
