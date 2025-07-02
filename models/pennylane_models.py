import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
import pennylane as qml

from simulator.simulate import run_circuit_sim
from circuits.pqc_circuits import simple_PQC_pennylane

class SimpleCustomQuantumModel(nn.Module):
    def __init__(self, num_params:int, num_qubits: int, 
                 simulator:AerSimulator, num_shots:int, 
                 circuit_runner = run_circuit_sim, param_init_noise=torch.pi/1000):
        super().__init__()

        self.pqc_params = nn.Parameter(param_init_noise * torch.rand(num_params))
        self.simulator = simulator
        self.transpile = transpile
        self.num_shots = num_shots
        self.runner = circuit_runner
        self.num_qubits = num_qubits
        self.pennylane_dev = qml.device("default.qubit", wires=self.base_circuit.num_qubits)

        pass

    
    def forward(self, unitary:Operator):
        """
        @param circuit: The quantum circuit with the input and it's inverse appended. 
        """    
        # circuit = circuit.remove_final_measurements(inplace=False)
        # circuit_pqc = append_pqc_to_quantum_circuit(circuit, self.pqc_params)
        # measured_tensor = self.runner(circuit_pqc, self.simulator, self.num_shots)

        # return measured_tensor

        qnode = self.make_pennylane_qnode()
        probs = qnode(self.pqc_params, unitary, self.num_qubits)
        # shots_scaled = probs * self.num_shots

        return probs

    def make_pennylane_qnode(self):
        dev = qml.device("default.qubit", wires=self.base_circuit.num_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def pennylane_pqc_circ(params, unitary, num_qubits):
            qml.QubitUnitary(unitary, wires=range(num_qubits))  # fixed input/inverse
            simple_PQC_pennylane(num_qubits=num_qubits, params=params)
            
            return qml.probs(wires=range(num_qubits))

        return pennylane_pqc_circ


class SimplePennyLaneQuantumModel(nn.Module):
    def __init__(self, num_qubits: int, num_params: int,
                 pqc_arch_func, prob_dist = False,
                 device="default.qubit", num_shots=1024):
        super().__init__()

        # Raw trainable parameters (unconstrained)
        self.raw_params = nn.Parameter(torch.randn(num_params) * 2 * torch.pi)


        self.num_shots = num_shots
        self.prob_dist = prob_dist


        # PennyLane device (can be finite shots or exact)
        self.q_device = qml.device(device, wires=num_qubits, shots=(num_shots if prob_dist else None))

        # PennyLane QNode using torch interface
        @qml.qnode(self.q_device, interface="torch", diff_method=('parameter-shift' if prob_dist else 'backprop'))
        def circuit(param_tensor, circuit_unitary:Operator):
            # ⬇ Base circuit (e.g., noisy GHZ, etc.)
            qml.QubitUnitary(circuit_unitary, wires=range(num_qubits))  # fixed input/inverse

            # ⬇ Append parameterized PQC (θ = π·sin(x))
            # bounded_params = torch.pi * torch.sin(param_tensor)
            pqc_arch_func(num_qubits, param_tensor)

            # ⬇ Measurement: return probabilities (or expectation values)
            return qml.probs(wires=range(num_qubits))

        self.qnode = circuit

    def forward(self, circuit=QuantumCircuit):
        circuit_op = Operator(circuit)
        val = self.num_shots if self.prob_dist else 1
        return self.qnode(self.raw_params, circuit_op.data) * val
    


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
