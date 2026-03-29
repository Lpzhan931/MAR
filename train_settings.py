import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16
    BASE_MODEL_PATH = "/home/share/models/Qwen3-8B/"
    SMALL_MODEL_PATH = "/home/pzli/Project/Spec/SpS/models/qwen3-tiny-ep3/"
