from torch import nn
import torch
from transformers import Owlv2VisionModel
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DetectorModelOwl(nn.Module):
    owl: Owlv2VisionModel

    def __init__(self, model_path: str, dropout: float, n_hidden: int = 768):
        super().__init__()

        owl = Owlv2VisionModel.from_pretrained(model_path)
        assert isinstance(owl, Owlv2VisionModel)
        self.owl = owl
        self.owl.requires_grad_(False)
        self.transforms = None

        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(n_hidden, eps=1e-5)
        self.linear1 = nn.Linear(n_hidden, n_hidden * 2)
        self.act1 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(n_hidden * 2, eps=1e-5)
        self.linear2 = nn.Linear(n_hidden * 2, 2)
    
    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None):
        with torch.autocast(device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            outputs = self.owl(pixel_values=pixel_values, output_hidden_states=True)
            x = outputs.last_hidden_state

            x = self.dropout1(x)
            x = self.ln1(x)
            x = self.linear1(x)
            x = self.act1(x)

            x = self.dropout2(x)
            x, _ = x.max(dim=1)
            x = self.ln2(x)

            x = self.linear2(x)
        
        if labels is not None:
            loss = F.cross_entropy(x, labels)
            return (x, loss)

        return (x,)

