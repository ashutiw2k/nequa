import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
import pennylane as qml

from simulator.simulate import run_circuit_sim
from circuits.pqc_circuits import simple_PQC_pennylane


class SimplePennyLaneQuantumModel(nn.Module):
    def __init__(self, num_qubits, num_params, pqc_arch_func, 
                 qdevice="default.qubit", diff='backprop', device=torch.device('cpu')):
        
        super(SimplePennyLaneQuantumModel, self).__init__()

        self.num_qubits = num_qubits
        self.raw_params = nn.Parameter(torch.randn(num_params) * 2 * torch.pi)
        self.pqc_arch = pqc_arch_func
        
        self.qml_device = qml.device(qdevice, wires=num_qubits)

        # Get indexing based on flipped endianness - 1 -> 001 becomes 100 -> 4, if n=3, etc. 
        self.perm = [int(f"{i:0{num_qubits}b}"[::-1], 2) for i in range(2**num_qubits)] 

        self._gate_map = {
            "h":        qml.Hadamard,
            "x":        qml.PauliX,    "pauli-x": qml.PauliX,
            "y":        qml.PauliY,    "pauli-y": qml.PauliY,
            "z":        qml.PauliZ,    "pauli-z": qml.PauliZ,
            "cx":       qml.CNOT,      "cnot":    qml.CNOT,
            "cz":       qml.CZ,
            "rx":       qml.RX,
            "rz":       qml.RZ,
        }


        @qml.qnode(self.qml_device, interface='torch', diff_method=diff)
        def circuit_sim(param_tensor, circuit:QuantumCircuit):
            # # ⬇ Base circuit (e.g., noisy GHZ, etc.)
            # qml_unitary = circuit_unitary[np.ix_(self.perm, self.perm)]  # reorder rows and columns
            # qml.QubitUnitary(qml_unitary, wires=range(self.num_qubits))  # fixed input/inverse
            
            for ins, qreg, _ in circuit:
                name = ins.name.lower()
                op   = self._gate_map.get(name)
                if op is None:
                    raise ValueError(f"Unsupported gate: {ins.name}")

                wires = [q._index for q in qreg]
                params = ins.params

                # one‐line apply
                if params:
                    op(float(params[0]), wires=wires)
                else:
                    op(wires=wires)



            # ⬇ Append parameterized PQC (θ = π·sin(x))
            # bounded_params = torch.pi * torch.sin(param_tensor)
            pqc_arch_func(num_qubits, param_tensor)

            # ⬇ Measurement: return state (or expectation values)
            return qml.state()

        self.qnode = circuit_sim


    def forward(self, circuit=QuantumCircuit):
        # circuit_op = Operator(circuit).data
        return self.qnode(self.raw_params, circuit)
    


class SimplePennylaneQuantumStateModel(nn.Module):
    def __init__(self, num_qubits, num_params, pqc_arch_func, 
                 qdevice="default.qubit", diff='backprop', device=torch.device('cpu')):
        
        super(SimplePennylaneQuantumStateModel, self).__init__()

        self.num_qubits = num_qubits
        self.raw_params = nn.Parameter(torch.randn(num_params) * 2 * torch.pi)
        self.pqc_arch = pqc_arch_func
        
        self.qml_device = qml.device(qdevice, wires=num_qubits)

        # Get indexing based on flipped endianness - 1 -> 001 becomes 100 -> 4, if n=3, etc. 
        self.perm = [int(f"{i:0{num_qubits}b}"[::-1], 2) for i in range(2**num_qubits)] 

        @qml.qnode(self.qml_device, interface='torch', diff_method=diff)
        def circuit_sim(param_tensor, circuit_unitary:Operator):
            # ⬇ Base circuit (e.g., noisy GHZ, etc.)
            qml_unitary = circuit_unitary[np.ix_(self.perm, self.perm)]  # reorder rows and columns
            qml.QubitUnitary(qml_unitary, wires=range(self.num_qubits))  # fixed input/inverse

            # ⬇ Append parameterized PQC (θ = π·sin(x))
            # bounded_params = torch.pi * torch.sin(param_tensor)
            pqc_arch_func(num_qubits, param_tensor)

            # ⬇ Measurement: return probabilities (or expectation values)
            return qml.state()

        self.qnode = circuit_sim
        
    def _qiskit_to_pl_matrix(self, U: np.ndarray, n: int) -> np.ndarray:
        """
        ChatGPT code snippet
        Re-index little-endian unitary U to big-endian ordering."""
        
        return         # reorder rows and columns

    def forward(self, circuit=QuantumCircuit):
        circuit_op = Operator(circuit).data

        return self.qnode(self.raw_params, circuit_op)



class SimplePennylaneUnitaryStateModel(nn.Module):
    def __init__(self, num_qubits, num_params, pqc_arch_func, 
                 qdevice="default.qubit", diff='backprop', device=torch.device('cpu')):
        
        super(SimplePennylaneUnitaryStateModel, self).__init__()

        self.num_qubits = num_qubits
        self.raw_params = nn.Parameter(torch.randn(num_params) * 2 * torch.pi)
        self.pqc_arch = pqc_arch_func
        
        self.qml_device = qml.device(qdevice, wires=num_qubits)

        # Get indexing based on flipped endianness - 1 -> 001 becomes 100 -> 4, if n=3, etc. 
        # self.perm = [int(f"{i:0{num_qubits}b}"[::-1], 2) for i in range(2**num_qubits)] 

        @qml.qnode(self.qml_device, interface='torch', diff_method=diff)
        def circuit_sim(param_tensor, circuit_unitary:Operator):
            # ⬇ Base circuit (e.g., noisy GHZ, etc.)
            # qml_unitary = circuit_unitary[np.ix_(self.perm, self.perm)]  # reorder rows and columns
            qml.QubitUnitary(circuit_unitary, wires=range(self.num_qubits))  # fixed input/inverse

            # ⬇ Append parameterized PQC (θ = π·sin(x))
            # bounded_params = torch.pi * torch.sin(param_tensor)
            pqc_arch_func(num_qubits, param_tensor)

            # ⬇ Measurement: return probabilities (or expectation values)
            return qml.state()

        self.qnode = circuit_sim
        

    def forward(self, circuit_op:torch.Tensor):

        if circuit_op.dim() > 1:
            return torch.stack([self.qnode(self.raw_params, U) for U in circuit_op]).to(torch.complex128)
        else:
            return self.qnode(self.raw_params, circuit_op).to(torch.complex128)
