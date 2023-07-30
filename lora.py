import torch
import torch.nn as nn


class LoraModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lora_rank: int,
        alpha: int,
    ):
        super().__init__()
        self._lora_rank = lora_rank
        self._alpha = alpha

        self.A = nn.Linear(input_dim, lora_rank)
        self.B = nn.Linear(lora_rank, output_dim)

    @property
    def lora_rank(self):
        return self._lora_rank

    @property
    def alpha(self):
        return self._alpha

    def foward(self, x: torch.Tensor) -> torch.Tensor:
        return self._alpha * (
            self.B.transpose(1, 0) @ self.A.transpose(1, 0) @ x
        )



class DynamicLoraModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lora_maximum_rank: int,
        alpha: int,
    ):
        super().__init__()
        
        self._lora_maximum_rank = lora_maximum_rank
        self._alpha = alpha

        self.A = nn.Linear(input_dim, lora_maximum_rank)
        self.B = nn.Linear(lora_maximum_rank, output_dim)

        # Setting default current rank
        self._current_rank = 0
    
    def get_current_rank(self):
        return self._current_rank
    
    def set_current_rank(self, n: int):
        self._current_rank = n

    def forward(self):
        pass