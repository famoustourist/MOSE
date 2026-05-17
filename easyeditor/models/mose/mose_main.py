from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook
from ...util.generate import generate_fast

from ..alphaedit.compute_ks import compute_ks
from ..alphaedit.compute_z import (
    compute_z,
    get_module_input_output_at_words,
)
from ..rome.layer_stats import layer_stats

from .mose_hparams import MOSEHyperParams


CONTEXT_TEMPLATES_CACHE: Optional[List[List[str]]] = None
COV_CACHE: Dict[Tuple[str, str], torch.Tensor] = {}


def apply_mose_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MOSEHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)

    weights_copy: Dict[str, torch.Tensor] = {}

    layers = list(hparams.layers)
    if hparams.auto_layer_selection:
        layers = _select_layers(model, tok, requests, hparams)

    new_weights = _execute_mose(model, tok, requests, hparams, layers)

    with torch.no_grad():
        for w_name, new_w in new_weights.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] = new_w.to(w.device, dtype=w.dtype)

    print(f"[MOSE] New weights successfully inserted into {list(new_weights.keys())}")

    return model, weights_copy


def execute_mose(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MOSEHyperParams,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    layers = list(hparams.layers)
    if hparams.auto_layer_selection:
        layers = _select_layers(model, tok, requests, hparams)
    return _execute_mose(model, tok, requests, hparams, layers)


def _execute_mose(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MOSEHyperParams,
    layers: List[int],
) -> Dict[str, torch.Tensor]:
    device = torch.device(f"cuda:{hparams.device}" if torch.cuda.is_available() else "cpu")

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            requests[i]["target_new"] = " " + request["target_new"]
        if "subject" not in request:
            requests[i]["subject"] = request["prompt"]
        if "{}" not in request["prompt"]:
            assert request["subject"] in request["prompt"], (
                f"Subject:{request['subject']} does not exist in prompt: {request['prompt']}"
            )
            requests[i]["prompt"] = requests[i]["prompt"].replace(
                requests[i]["subject"], "{}"
            )
        print(
            f"[MOSE] editing: [{requests[i]['prompt']}] -> [{requests[i]['target_new']}]"
        )

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates = _get_context_templates(model, tok)
    z_layer = layers[-1]
    z_list: List[torch.Tensor] = []
    for request in requests:
        cur_z = compute_z(
            model,
            tok,
            request,
            hparams,
            z_layer,
            context_templates,
        )
        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=1)

    new_weights: Dict[str, torch.Tensor] = {}

    for i, layer in enumerate(layers):
        print(f"\n[MOSE] === LAYER {layer} ===")

        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        layer_ks = layer_ks.to(device=device, dtype=torch.float32)
        print(f"[MOSE] writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        cur_zs = cur_zs.to(device=device, dtype=torch.float32)
        z_targets = (zs.to(device=device, dtype=torch.float32) - cur_zs)
        print(f"[MOSE] z error {torch.linalg.norm(z_targets, dim=0).mean():.4f}")

        repeat_factor = layer_ks.size(1) // z_targets.size(1)
        z_targets = z_targets.repeat_interleave(repeat_factor, dim=1)
        resid = z_targets / max(1, len(layers) - i)

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        W_orig = weights[weight_name].detach().to(device=device, dtype=torch.float32)
        transpose_back = False
        if W_orig.shape[1] == layer_ks.shape[0]:
            W0 = W_orig
        elif W_orig.shape[0] == layer_ks.shape[0]:
            W0 = W_orig.T
            transpose_back = True
        else:
            raise RuntimeError(
                f"Cannot align weight {weight_name} shape {tuple(W_orig.shape)} "
                f"with key shape {tuple(layer_ks.shape)}"
            )

        d, p = W0.shape
        K_E = layer_ks
        V_E = (W0 @ K_E) + resid.to(device=device, dtype=torch.float32)

        cov = _get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams,
        ).to(device=device, dtype=torch.float32)

        lam = float(hparams.preserve_lambda) * float(hparams.mom2_update_weight)
        M = lam * (W0 @ cov @ W0.T) + V_E @ K_E.T @ W0.T

        if hparams.svd_reg and hparams.svd_reg > 0:
            M = M + hparams.svd_reg * torch.eye(d, device=device, dtype=M.dtype)

        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh

        W_new = R @ W0
        W_write = W_new.T if transpose_back else W_new
        new_weights[weight_name] = W_write.to(weights[weight_name].dtype)

        with torch.no_grad():
            weights[weight_name][...] = new_weights[weight_name].to(weights[weight_name].device)

        for x in (layer_ks, cur_zs, z_targets):
            del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"[MOSE] orthogonal updates computed for {list(new_weights.keys())}")
    return new_weights


def _select_layers(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MOSEHyperParams,
) -> List[int]:
    candidate_layers = list(hparams.layers)
    if len(candidate_layers) <= 1:
        return candidate_layers

    device = torch.device(f"cuda:{hparams.device}" if torch.cuda.is_available() else "cpu")
    context_templates = _get_context_templates(model, tok)

    proc_requests = deepcopy(requests)
    for i, r in enumerate(proc_requests):
        if r["target_new"][0] != " ":
            proc_requests[i]["target_new"] = " " + r["target_new"]
        if "subject" not in r:
            proc_requests[i]["subject"] = r["prompt"]
        if "{}" not in r["prompt"]:
            proc_requests[i]["prompt"] = r["prompt"].replace(r["subject"], "{}")

    z_layer = candidate_layers[-1]
    zs = torch.stack(
        [compute_z(model, tok, req, hparams, z_layer, context_templates) for req in proc_requests],
        dim=1,
    )

    best_layer, best_score = candidate_layers[0], float("inf")
    for layer in candidate_layers:
        K_E = compute_ks(model, tok, proc_requests, hparams, layer, context_templates).T
        K_E = K_E.to(device=device, dtype=torch.float32)
        W_orig = nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        ).detach().to(device=device, dtype=torch.float32)
        if W_orig.shape[1] == K_E.shape[0]:
            W0 = W_orig
        elif W_orig.shape[0] == K_E.shape[0]:
            W0 = W_orig.T
        else:
            raise RuntimeError(
                f"Cannot align layer {layer} weight shape {tuple(W_orig.shape)} "
                f"with key shape {tuple(K_E.shape)}"
            )

        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[r["prompt"] for r in proc_requests],
            words=[r["subject"] for r in proc_requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T.to(device=device, dtype=torch.float32)
        V_E = zs.to(device=device, dtype=torch.float32) - cur_zs + (W0 @ K_E)

        diff = V_E - (W0 @ K_E)
        w_norm = torch.linalg.matrix_norm(W0, ord=2)
        sigma = torch.relu(K_E)
        score = torch.linalg.norm(diff / (w_norm + 1e-8) * sigma)
        if score.item() < best_score:
            best_score = score.item()
            best_layer = layer

    layer_set = sorted({best_layer - 1, best_layer, best_layer + 1})
    layer_set = [l for l in layer_set if l in candidate_layers]
    if not layer_set:
        layer_set = [best_layer]
    print(f"[MOSE] auto-selected layers: {layer_set} (anchor={best_layer})")
    return layer_set


def _get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    hparams: MOSEHyperParams,
    force_recompute: bool = False,
) -> torch.Tensor:
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    if key in COV_CACHE and not force_recompute:
        return COV_CACHE[key]

    print(f"[MOSE] retrieving covariance statistics for {model_name} @ {layer_name}.")
    stat = layer_stats(
        model,
        tok,
        layer_name,
        hparams.stats_dir,
        hparams.mom2_dataset,
        to_collect=["mom2"],
        sample_size=(
            hparams.mom2_n_samples
            if not force_recompute
            else max(1, hparams.mom2_n_samples // 10)
        ),
        precision=hparams.mom2_dtype,
        hparams=hparams,
        force_recompute=force_recompute,
    )
    COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
    return COV_CACHE[key]


def _get_context_templates(model, tok) -> List[List[str]]:
    global CONTEXT_TEMPLATES_CACHE
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]
        ]
        print(f"[MOSE] cached context templates {CONTEXT_TEMPLATES_CACHE}")
    return CONTEXT_TEMPLATES_CACHE
