import numpy as np
from qiskit import QuantumCircuit
import torch

import pennylane as qml

def qiskit_PQC_RZRX(num_qubits:int, params:torch.Tensor):
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    assert len(params) == 2, f"Length of parameter tensor is {len(params)}, expected 2."
    # print(params)
    for i in range(num_qubits):
        pqc.rz(params[0], i)
        pqc.rx(params[1], i)


    return pqc

def qiskit_PQC_RZRX_unique(num_qubits:int, params:torch.Tensor):
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    assert len(params) == 2*num_qubits, f"Length of parameter tensor is {len(params)}, expected 2."
    # print(params)
    for i in range(num_qubits):
        pqc.rz(params[i*2], i)
        pqc.rx(params[i*2 + 1], i)

    return pqc

def qiskit_PQC_RXRZ_unique(num_qubits:int, params:torch.Tensor):
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    assert len(params) == 2*num_qubits, f"Length of parameter tensor is {len(params)}, expected 2."
    # print(params)
    for i in range(num_qubits):
        pqc.rx(params[i*2], i)
        pqc.rz(params[i*2 + 1], i)

    return pqc


def qiskit_PQC_RXRZ(num_qubits:int, params:torch.Tensor):
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    assert len(params) == 2, f"Length of parameter tensor is {len(params)}, expected 2."

    for i in range(num_qubits):
        pqc.rx(params[0], i)
        pqc.rz(params[1], i)


    return pqc

def qiskit_PQC_RXRZRY(num_qubits:int, params:torch.Tensor):
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    assert len(params) == 3, f"Length of parameter tensor is {len(params)}, expected 3."
    # print(params)
    for i in range(num_qubits):
        pqc.rx(params[0], i)
        pqc.rz(params[1], i)
        pqc.ry(params[2], i)


    return pqc


def simple_PQC_pennylane(num_qubits:int, params:torch.Tensor):

    for i in range(num_qubits):
        qml.RZ(params[0], i)
        qml.RX(params[1], i)
        # qml.RX(params[2], i)

        
