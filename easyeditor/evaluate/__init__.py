from .evaluate import compute_edit_quality, compute_icl_edit_quality

def compute_sent_metric(*args, **kwargs):
    raise NotImplementedError(
        "compute_sent_metric is disabled in the MOSE-only import patch. "
        "It is not needed for normal MOSE editing."
    )
