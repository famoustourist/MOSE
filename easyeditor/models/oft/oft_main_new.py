from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from ...util import nethook

from .oft_hparams import OFTHyperParams
import torch.nn as nn

from functools import partial



class OFTInjectedLinear(nn.Module):
    def __init__(
        self, linear, bias=False, r=4, eps=1e-5, is_coft=True, block_share=False,
    ):
        super().__init__()

        assert linear.in_features % r == 0, "in_features must be divisible by r"

        # Get the number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        # Set the device IDs for distributed training
        self.device_ids = list(range(self.num_gpus))

        self.in_features=linear.in_features
        self.out_features=linear.out_features

        # Define the fixed Linear layer: v
        self.OFT = linear

        # Define the reduction rate:
        self.r = r
        self.is_coft = is_coft

        self.fix_filt_shape = [self.in_features, self.out_features]
        
        self.R_shape = [self.in_features // self.r, self.in_features // self.r]
        R = 1e-2*torch.ones(self.in_features, 1)
        
        R = torch.stack([R] * self.r)
        
        self.R = nn.Parameter(R, requires_grad=True)
        self.eps = eps * self.R_shape[0] * self.R_shape[0]
        
        self.block_share = block_share
        '''
        # Define the trainable matrix parameter: R
        self.block_share = block_share
        if self.block_share:
            # Initialized as an identity matrix
            self.R_shape = [self.in_features // self.r, self.in_features // self.r]
            self.R = nn.Parameter(torch.zeros(self.R_shape[0], self.R_shape[0]), requires_grad=True)
  
            self.eps = eps * self.R_shape[0] * self.R_shape[0]
        else:
            # Initialized as an identity matrix
            self.R_shape = [self.r, self.in_features // self.r, self.in_features // self.r]
            R = torch.zeros(self.R_shape[1], self.R_shape[1])
            R = torch.stack([R] * self.r)
            self.R = nn.Parameter(R, requires_grad=True)
            self.eps = eps * self.R_shape[1] * self.R_shape[1]
        '''
        #addtion
        self.mode="train"

    def forward(self, x):
        
        #print(self.r)
        orig_dtype = x.dtype
        dtype = self.R.dtype
        
        if self.mode=="test":
            bias_term = self.OFT.bias.data if self.OFT.bias is not None else None
            out = nn.functional.linear(input=x, weight=self.weight, bias=bias_term)
            return out #.to(orig_dtype)
        #print(self.R)
        
        #print(self.R, self.R.norm())
        with torch.no_grad():
            self.R.copy_(nn.functional.normalize(self.R, dim=1))
        
        #print(self.R)
        #Rt = self.R.transpose(1,2)
        #I = torch.zeros((self.R.size(1), self.R.size(1)), device=self.R.device, dtype=self.R.dtype).unsqueeze(0).expand_as(torch.bmm(self.R, Rt))
        
        #print(I.shape)
        
        I=torch.eye(self.in_features, device=self.R.device)
        #orth_rotate= I - 2*torch.bmm(self.R, Rt)
        orth_rotate=I
        for j in range(self.r):
            orth_rotate = torch.mm(orth_rotate, (I-2*self.R[j] @ self.R[j].t()))
        
        #print(orth_rotate.shape)
        '''
        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    self.R.copy_(project(self.R, eps=self.eps))
            orth_rotate = self.cayley(self.R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    self.R.copy_(project_batch(self.R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.R)
        #print(self.R)
        '''
        # Block-diagonal parametrization
        block_diagonal_matrix = orth_rotate
        #self.block_diagonal(orth_rotate)
        #self.block_diagonal(orth_rotate)
        
        #print(block_diagonal_matrix.shape)
        # fix filter
        fix_filt = self.OFT.weight.data
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
        filt = torch.transpose(filt, 0, 1)
        
        #print(x.shape, filt.shape)

        # Apply the trainable identity matrix
        bias_term = self.OFT.bias.data if self.OFT.bias is not None else None
        out = nn.functional.linear(input=x, weight=filt, bias=bias_term)
        
        #print(filt,"\n\n")
        
        self.weight=filt

        return out #.to(orig_dtype)

    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        
        # Perform the Cayley parametrization
        Q = torch.mm(I + skew, torch.inverse(I - skew))
        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    def block_diagonal(self, R):
        if self.block_share:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))


def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out


def apply_oft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: OFTHyperParams,
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


    assign_oft = partial(OFTInjectedLinear)

    for name, w in model.named_parameters():
        if name=="model.layers.19.mlp.down_proj.weight":
            print(w, w.shape)

    temp=0
    for param_tensor in model.state_dict(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        if "OFT" in param_tensor:
            temp=1
            break
    
    #hparams.layers=[19, 20 ,21]
    print(hparams.layers)
    if temp==0:

        for layer in hparams.layers:
            model.model.layers[layer].mlp.down_proj = assign_oft(model.model.layers[layer].mlp.down_proj, r=hparams.block)
            #model.model.layers[layer].mlp.gate_proj = assign_oft(model.model.layers[layer].mlp.up_proj)
            #model.model.layers[layer].mlp.gate_proj = assign_oft(model.model.layers[layer].mlp.gate_proj)
            #model.model.layers[layer].self_attn.q_proj = assign_oft(model.model.layers[layer].self_attn.q_proj)
            #model.model.layers[layer].self_attn.v_proj = assign_oft(model.model.layers[layer].self_attn.v_proj)
    '''
    for name, w in model.named_parameters():
        print(name)
    #print(model)
    '''


    deltas = execute_oft(model, tok, requests, hparams)
    
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


def execute_oft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: OFTHyperParams,
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
        '''
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        '''
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
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
            opt.zero_grad()
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                device
            )
            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            loss_mask = target_ids != tok.unk_token_id
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

            #if loss.item() >= 1e-2:
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

        #if loss_meter.avg < 1e-2:
        #    break
            

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
