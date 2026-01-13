from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from ...util import nethook

from .lora_hparams import LoRAHyperParams
from functools import partial



class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
        
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
        

def apply_lora_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: LoRAHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)
    '''
    for param_tensor in model.state_dict(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        if "lora" in param_tensor:
            break
        else:
    '''

    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)


    temp=0
    for param_tensor in model.state_dict(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        if "lora" in param_tensor:
            temp=1
            break
    
    hparams.layers=[20, 21, 22, 23, 24]
    if temp==0:
        '''
        for layer in hparams.layers:
            model.model.layers[layer].mlp.down_proj = assign_lora(model.model.layers[layer].mlp.down_proj)
            #model.model.layers[layer].mlp.up_proj = assign_lora(model.model.layers[layer].mlp.up_proj)
            #model.model.layers[layer].mlp.gate_proj = assign_lora(model.model.layers[layer].mlp.gate_proj)
            #model.model.layers[layer].self_attn.q_proj = assign_lora(model.model.layers[layer].self_attn.q_proj)
            #model.model.layers[layer].self_attn.v_proj = assign_lora(model.model.layers[layer].self_attn.v_proj)
            #layer.mlp.up_proj = assign_lora(layer.mlp.up_proj)

        '''
        for layer in model.model.layers:
            layer.self_attn.q_proj = assign_lora(layer.self_attn.q_proj)
            layer.self_attn.v_proj = assign_lora(layer.self_attn.v_proj)
            #layer.mlp.down_proj = assign_lora(layer.mlp.down_proj)
            #layer.mlp.up_proj = assign_lora(layer.mlp.up_proj) 


    '''
    if not keep_original_weight and hasattr(model,'peft_config'):
        peft_model = model    
    '''
    
    
    deltas = execute_lora(model, tok, requests, hparams)
    
    upd_matrixs={i:[] for i in list(deltas.keys())}

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
                
            #print(upd_matrix,"rankis:", np.linalg.matrix_rank(upd_matrix.cpu().numpy()))
            
            w[...] += upd_matrix
            
            
            #print(upd_matrix,"rankis:", np.linalg.matrix_rank(upd_matrix.cpu().numpy()))

            #rank = np.linalg.matrix_rank(w.cpu().numpy())
            rank=0
            upd_matrixs[w_name]=[w.cpu().numpy(),rank]
            #print(w.device)
            #print("Rank of the matrix:", rank)



    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_lora(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: LoRAHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        for rewrite_module in hparams.rewrite_module_tmp
        if rewrite_module.format(layer) in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    
    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            mask_token = -100
            
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                device
            )
            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            loss_mask = target_ids != tok.unk_token_id
            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            if 't5' in hparams.model_name.lower():
                inputs['labels'] = target_ids
                logits = model(**inputs).logits
                unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['labels'].unsqueeze(-1)).squeeze(-1)

                mask = inputs['labels'] != -100
                n_tokens = mask.float().sum()
                avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                nll = -avg_log_prob
                loss = nll
            else:

                full_prompt = [f"{p} {l}" for p, l in zip(txt, tgt)]
                prompt_ids = tok(list(txt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
                num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
                tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
                bs = tokens["input_ids"].shape[0]
                tokens["labels"] = tokens["input_ids"].clone()
                num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
                for i in range(len(txt)):
                    tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i]+num_prompt_toks[i]] = mask_token
                tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
                tokens = tokens.to(device)
                pred = model(**tokens)
                loss = pred.loss

                '''
                probs = torch.nn.functional.log_softmax(
                    model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
                )
                loss = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                    1
                ) / loss_mask.sum(1)
                loss = loss.mean()
                '''
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward()
                opt.step()
                
                

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break
            

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}
    
    #print("before editing:", weights_copy,'\n\n', "after editing", weights)
        

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
