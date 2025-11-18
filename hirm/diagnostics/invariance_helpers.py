"""Helpers for collecting invariance diagnostics signals."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
from torch import nn

from hirm.objectives.common import compute_env_risks, concat_state, flatten_head_gradients
from hirm.objectives.hirm import _get_parameters


def _select_layers(model) -> Dict[str, nn.Module]:
    layers: Dict[str, nn.Module] = {}
    if getattr(model, "representation", None) is not None:
        layers["representation"] = model.representation  # type: ignore[assignment]
    if getattr(model, "phi_encoder", None) is not None:
        layers["phi_encoder"] = model.phi_encoder  # type: ignore[assignment]
    if getattr(model, "r_encoder", None) is not None:
        layers["r_encoder"] = model.r_encoder  # type: ignore[assignment]
    if getattr(model, "head", None) is not None:
        layers["head"] = model.head  # type: ignore[assignment]
    return layers


def _sample_batch(batch: Mapping[str, torch.Tensor], max_samples: int) -> Dict[str, torch.Tensor]:
    first = next(iter(batch.values()))
    count = min(max_samples, first.shape[0]) if hasattr(first, "shape") else max_samples
    if count <= 0:
        return {k: v[:0] for k, v in batch.items()}
    idx = torch.arange(count, device=next(iter(batch.values())).device)
    return {k: v[idx] for k, v in batch.items()}


def collect_invariance_signals(
    model,
    train_batch_by_env: Mapping[str, Mapping[str, torch.Tensor]],
    invariance_mode: str,
    device: torch.device,
    risk_fn,
    max_samples_per_env: int = 256,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    """Collect gradients and activations needed for ISI diagnostics."""

    model.eval()
    head_gradients: Dict[str, torch.Tensor] = {}
    layer_activations: Dict[str, Dict[str, torch.Tensor]] = {}

    layers = _select_layers(model)
    for name in layers:
        layer_activations[name] = {}

    params: Iterable[nn.Parameter] = _get_parameters(model, invariance_mode)
    params = list(params)

    for env_name, raw_batch in train_batch_by_env.items():
        batch = {k: v.to(device) for k, v in raw_batch.items()}
        batch = _sample_batch(batch, max_samples_per_env)
        if batch["env_ids"].numel() == 0:
            continue
        env_id = int(batch["env_ids"][0].item())
        env_key = f"env_{env_id}"

        # Collect activations with lightweight forward hooks.
        captured: Dict[str, torch.Tensor] = {}

        def _make_hook(layer_key: str):  # type: ignore[no-untyped-def]
            def hook(_module, _inp, output):
                if isinstance(output, torch.Tensor):
                    captured[layer_key] = output.detach().cpu()

            return hook

        hooks = [layers[name].register_forward_hook(_make_hook(name)) for name in layers]
        inputs = concat_state(batch)
        with torch.no_grad():
            model(inputs, env_ids=batch.get("env_ids"))
        for hook in hooks:
            hook.remove()
        for name, tensor in captured.items():
            layer_activations[name][env_key] = tensor

        if not params:
            continue

        env_risks, _, _, _ = compute_env_risks(model, batch, batch["env_ids"], risk_fn)
        env_risk = env_risks.get(env_id)
        if env_risk is None:
            continue
        grads = torch.autograd.grad(env_risk, params, retain_graph=False, allow_unused=True)
        head_gradients[env_key] = flatten_head_gradients(grads).detach().cpu()

    return head_gradients, layer_activations


__all__ = ["collect_invariance_signals"]
