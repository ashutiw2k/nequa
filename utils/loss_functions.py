import torch
from torch import nn


class QuantumFidelityLoss(nn.Module):
    def __init__(self, is_statevector=True, num_shots=1024, eps: float = 1e-8):
        """
        @param is_statevector: If True, expects ideal and measured to be complex statevectors.
                               If False, treats them as shot counts or probability distributions.
        @param num_shots: Used only when input is shot counts.
        """
        super(QuantumFidelityLoss, self).__init__()
        self.is_statevector = is_statevector
        self.num_counts = num_shots
        self.eps = eps

    def forward(self, ideal: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
        if self.is_statevector:
            # Ensure inputs are complex tensors
            if not torch.is_complex(ideal):
                raise ValueError("Ideal input must be a complex-valued statevector")
            if not torch.is_complex(measured):
                raise ValueError("Measured input must be a complex-valued statevector")

            # Normalize just in case
            psi = ideal / torch.linalg.norm(ideal)
            phi = measured / torch.linalg.norm(measured)

            overlap = torch.dot(torch.conj(psi), phi)
            fidelity = torch.abs(overlap) ** 2
            return 1.0 - torch.real(fidelity)

        else:
            # Classical fidelity loss using counts or probs
            p = ideal / self.num_counts + self.eps
            q = measured / self.num_counts + self.eps
            p = p / p.sum()
            q = q / q.sum()

            fidelity = torch.square(torch.sum(torch.sqrt(p * q)))
            return 1.0 - fidelity