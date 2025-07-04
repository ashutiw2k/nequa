import torch
from torch.utils.data import Dataset
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from tqdm import tqdm

from circuits.modify_circuits import get_param_input_circuit


class FidelityDataset(Dataset):
    """Yields (param_tensor, ideal_state_tensor, noisy_unitary_matrix) per sample."""
    def __init__(self, ideal_data, base_circuit:QuantumCircuit, perm, noise_model):
        """
        ideal_data: list of (params_np, ideal_state_np)
        base_circuit: Qiskit QuantumCircuit (static)
        perm: precomputed bit-reversal index list
        noise_model: BitPhaseFlipNoise instance
        """
        self.params, self.ideal   = zip(*ideal_data)
        self.base_circuit = base_circuit.copy()
        self.perm         = perm
        self.noise_model  = noise_model
        self.unitaries = [self._generate_unitary(p, self.base_circuit) for p in tqdm(self.params, f'Building Unitaries')]

    def __len__(self):
        return len(self.ideal)
    
    def _generate_unitary(self, params, circuit:QuantumCircuit):
        circ = circuit.copy()
        circ = self.base_circuit.copy()
        # 2) build the noisy Qiskit circuit & extract its unitary
        noisy_qc = get_param_input_circuit(
            params,
            self.noise_model.get_noisy_circuit_for_model(circ),
        )
        U = Operator(noisy_qc).data                                  # numpy array
        # 3) re-index for little→big endian
        return U[np.ix_(self.perm, self.perm)]



    def __getitem__(self, idx):

        return self.ideal[idx].to(torch.complex128), self.unitaries[idx]
