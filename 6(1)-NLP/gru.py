import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # 구현하세요!
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate parameter
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(input_size, hidden_size, bias=False)

        # Reset gate parameter
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(input_size, hidden_size, bias=False)

        # Candidate hidden state
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        z = torch.sigmoid(self.W_z(x) + self.U_z(h)) # Update gate
        r = torch.sigmoid(self.W_r(x) + self.U_r(h)) # Reset gate
        h_tidle = torch.tanh(self.W_h(x) + self.U_h(r * h)) # Candidate hidden state
        h_new = (1 - z) * h + z * h_tidle # New hidden state
        return h_new # Return


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        """
        inputs: (batch_size, seq_len, input_size)
        return: last hidden state (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = inputs.size()
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        
        for t in range(seq_len):
            x_t = inputs[:, t, :] # (batch_size, input_size)
            h = self.cell(x_t, h)
        
        return h # return last hidden state