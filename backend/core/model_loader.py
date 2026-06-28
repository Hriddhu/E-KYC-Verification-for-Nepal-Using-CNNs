import torch
from pathlib import Path


def load_weights(source: str, device: str = "cpu"):
    """
    Loads raw weights/checkpoint from a given source.
    For now: supports local disk only.
    Later: extend with elif branches for s3://, hf:// etc.
    without ever touching services/tampering.py or services/liveness.py.
    """
    # 1. check if source starts with "s3://" or "hf://" → raise NotImplementedError for now
    # 2. otherwise treat as local path
    # 3. torch.load(source, map_location=device)
    # 4. return the raw checkpoint object (caller decides how to interpret it)
    pass