import torch

def get_device(pref_cuda: str = "cuda:0"):

    if torch.cuda.is_available():
        device, device_name = torch.device(pref_cuda), pref_cuda
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device, device_name = torch.device("mps"), "mps"
    else:
        device, device_name = torch.device("cpu"), "cpu"
    
    print(f"Using device: {device}")
    return device, device_name
