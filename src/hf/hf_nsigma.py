from transformers import LogitsProcessor
import torch

class TopNSigma(LogitsProcessor):
    def __init__(self, nsigma: float, device: str):
        self.nsigma = torch.tensor(nsigma, device=device)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        maximum = scores.max(dim=-1, keepdim=True).values
        std = scores.std(dim=-1, keepdim=True)
        threshold = maximum - self.nsigma * std
        mask = scores >= threshold
        return torch.where(mask, scores, float('-inf'))
