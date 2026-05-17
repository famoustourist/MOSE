from ..models.mose import MOSEHyperParams, apply_mose_to_model
from ..dataset import ZsreDataset, CounterFactDataset

ALG_DICT = {
    "MOSE": apply_mose_to_model,
}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset,
}

ALG_MULTIMODAL_DICT = {}
MULTIMODAL_DS_DICT = {}
PER_ALG_DICT = {}
PER_DS_DICT = {}
Safety_DS_DICT = {}
