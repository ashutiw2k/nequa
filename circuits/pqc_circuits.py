import numpy as np
from qiskit import QuantumCircuit
import torch

import pennylane as qml

def get_simple_PQC_qiskit(num_qubits:int, params:torch.Tensor):
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    # print(params)
    for i in range(num_qubits):
        pqc.rz(params[0], i)
        pqc.rx(params[1], i)
        pqc.rz(params[2], i)

    return pqc

def simple_PQC_pennylane(num_qubits:int, params:torch.Tensor):

    for i in range(num_qubits):
        qml.RY(params[0], i)
        qml.RX(params[1], i)
        qml.RZ(params[2], i)

        
