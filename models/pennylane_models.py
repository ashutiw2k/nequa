import torch
from torch import nn
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