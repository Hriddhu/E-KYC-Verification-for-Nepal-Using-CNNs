import torch

def load_weights(source: str, device: str = "cpu"):
    if source.startswith("s3://"):
        raise NotImplementedError("S3 model loading not implemented yet.")
    if source.startswith("hf://"):
        raise NotImplementedError("HuggingFace Hub model loading not implemented yet.")

    # local disk path
    try:
        return torch.load(source, map_location=device, weights_only=False)
    except TypeError:
        # older torch versions don't support weights_only kwarg
        return torch.load(source, map_location=device)