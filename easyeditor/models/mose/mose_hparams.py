from dataclasses import dataclass
from typing import List, Literal, Optional
import yaml

from ...util.hparams import HyperParams


@dataclass
class MOSEHyperParams(HyperParams):
    layers: List[int]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ] = "subject_last"

    preserve_lambda: float = 1.0
    svd_reg: float = 0.0

    v_num_grad_steps: int = 25
    v_lr: float = 5e-1
    v_loss_layer: int = 31
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 0.75
    kl_factor: float = 0.0625

    rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj"
    layer_module_tmp: str = "model.layers.{}"
    mlp_module_tmp: str = "model.layers.{}.mlp"
    attn_module_tmp: str = "model.layers.{}.self_attn"
    ln_f_module: str = "model.norm"
    lm_head_module: str = "lm_head"

    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 100000
    mom2_dtype: str = "float32"
    stats_dir: str = "./data/stats"
    mom2_update_weight: float = 15000.0

    auto_layer_selection: bool = False

    device: int = 0
    alg_name: str = "MOSE"
    model_name: str = ""
    batch_size: int = 1
    max_length: int = 40
    model_parallel: bool = False

    num_steps: int = 0
    lr: float = 0.0
    weight_decay: float = 0.0
    norm_constraint: Optional[float] = None
    block: int = 4

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'MOSE') or print(
            f'MOSEHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} '
        )

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in valid_fields}
        return cls(**filtered)
