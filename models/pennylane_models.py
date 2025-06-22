import torch
from torch import nn
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
import pennylane as qml

from simulator.simulate import run_circuit_sim
from circuits.pqc_circuits import simple_PQC_pennylane

class SimpleCustomQuantumModel(nn.Module):
    def __init__(self, num_params:int, base_circuit: QuantumCircuit, 
                 simulator:AerSimulator, num_shots:int, 
                 circuit_runner = run_circuit_sim, param_init_noise=torch.pi/1000):
        super().__init__()

        self.pqc_params = nn.Parameter(param_init_noise * torch.rand(num_params))
        self.simulator = simulator
        self.transpile = transpile
        self.num_shots = num_shots
        self.runner = circuit_runner
        self.base_circuit = base_circuit
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
        probs = qnode(self.pqc_params, unitary, self.base_circuit.num_qubits)
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

