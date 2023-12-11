import torch 
import torch.nn as nn

def weights_init(param) -> None:
    """Initializes weights of Conv and fully connected."""

    if isinstance(param, nn.Conv2d):
        torch.nn.init.xavier_uniform_(param.weight.data)
        if param.bias is not None:
            torch.nn.init.constant_(param.bias.data, 0.2)
    elif isinstance(param, nn.Linear):
        torch.nn.init.normal_(param.weight.data, mean=0.0, std=0.01)
        torch.nn.init.constant_(param.bias.data, 0.0)