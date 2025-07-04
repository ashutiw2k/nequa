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

def qiskit_PQC_RZRXRZ_unique(num_qubits:int, params:torch.Tensor):
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    assert len(params) == 3*num_qubits, f"Length of parameter tensor is {len(params)}, expected 2."
    # print(params)
    for i in range(num_qubits):
        pqc.rz(params[i*3], i)
        pqc.rx(params[i*3 + 1], i)
        pqc.rz(params[i*3 + 2], i)

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

        
def pennylane_PQC_RZRX_unique(num_qubits:int, params:torch.Tensor):
    
    assert len(params) == 2*num_qubits, f"Length of parameter tensor is {len(params)}, expected 2."
    for i in range(num_qubits):
        qml.RZ(params[2*i], i)
        qml.RX(params[2*i + 1], i)

        
def pennylane_PQC_RZRXRY_unique(num_qubits:int, params:torch.Tensor):
    
    assert len(params) == 3*num_qubits, f"Length of parameter tensor is {len(params)}, expected {3*num_qubits}."
    for i in range(num_qubits):
        qml.RZ(params[3*i], i)
        qml.RX(params[3*i + 1], i)
        qml.RY(params[3*i + 2], i)


        
def pennylane_PQC_RZRXRZ_unique(num_qubits:int, params:torch.Tensor):
    
    assert len(params) == 3*num_qubits, f"Length of parameter tensor is {len(params)}, expected {3*num_qubits}."
    for i in range(num_qubits):
        qml.RZ(params[3*i], i)
        qml.RX(params[3*i + 1], i)
        qml.RZ(params[3*i + 2], i)
        

def qiskit_PQC_U3_unique(num_qubits:int, params:torch.Tensor):
    
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    assert len(params) == 3*num_qubits, f"Length of parameter tensor is {len(params)}, expected {3*num_qubits}."
    # print(params)
    for i in range(num_qubits):
        pqc.u(params[3*i], params[3*i + 1], params[3*i + 2], i)


    return pqc

def pennylane_PQC_U3_unique(num_qubits:int, params:torch.Tensor):
    
    assert len(params) == 3*num_qubits, f"Length of parameter tensor is {len(params)}, expected {3*num_qubits}."
    for i in range(num_qubits):
        # qml.RZ(params[3*i], i)
        # qml.RX(params[3*i + 1], i)
        # qml.RZ(params[3*i + 2], i)
        qml.U3(params[3*i], params[3*i + 1], params[3*i + 2], i)


def qiskit_PQC_U3U3_unique(num_qubits:int, params:torch.Tensor):
    
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    assert len(params) == 6*num_qubits, f"Length of parameter tensor is {len(params)}, expected {6*num_qubits}."
    # print(params)
    for i in range(num_qubits):
        pqc.u(params[6*i], params[6*i + 1], params[6*i + 2], i)
        pqc.u(params[6*i + 3], params[6*i + 4], params[6*i + 5], i)


    return pqc

def pennylane_PQC_U3U3_unique(num_qubits:int, params:torch.Tensor):
    
    assert len(params) == 6*num_qubits, f"Length of parameter tensor is {len(params)}, expected {6*num_qubits}."
    for i in range(num_qubits):
        # qml.RZ(params[3*i], i)
        # qml.RX(params[3*i + 1], i)
        # qml.RZ(params[3*i + 2], i)
        qml.U3(params[6*i], params[6*i + 1], params[6*i + 2], i)
        qml.U3(params[6*i + 3], params[6*i + 4], params[6*i + 5], i)



def pennylane_PQC_RZRXRZ_CX_unique(num_qubits:int, params:torch.Tensor):
    
    assert len(params) == 3*num_qubits, f"Length of parameter tensor is {len(params)}, expected {3*num_qubits}."
    
    for i in range(num_qubits):
        qml.RZ(params[3*i], i)
        qml.RX(params[3*i + 1], i)
        qml.RZ(params[3*i + 2], i)

    for i in range(num_qubits-1):
        qml.CNOT([i, i+1])
    

        
def qiskit_PQC_RZRXRZ_CX_unique(num_qubits:int, params:torch.Tensor):
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    assert len(params) == 3*num_qubits, f"Length of parameter tensor is {len(params)}, expected 2."
    # print(params)
    for i in range(num_qubits):
        pqc.rz(params[i*3], i)
        pqc.rx(params[i*3 + 1], i)
        pqc.rz(params[i*3 + 2], i)

    for i in range(num_qubits - 1):
        pqc.cx(i, i+1)

    return pqc

def pennylane_PQC_RZRXRZ_CZ_unique(num_qubits:int, params:torch.Tensor):
    
    assert len(params) == 3*num_qubits, f"Length of parameter tensor is {len(params)}, expected {3*num_qubits}."
    
    for i in range(num_qubits):
        qml.RZ(params[3*i], i)
    
    for i in range(num_qubits-1):
        qml.CZ([i, i+1])
    
    for i in range(num_qubits):
        qml.RX(params[3*i + 1], i)
    
    for i in range(num_qubits-1):
        qml.CZ([i, i+1])
    
    for i in range(num_qubits):
        qml.RZ(params[3*i + 2], i)

        
def qiskit_PQC_RZRXRZ_CZ_unique(num_qubits:int, params:torch.Tensor):
    pqc = QuantumCircuit(num_qubits)
    params = params.detach().numpy()
    assert len(params) == 3*num_qubits, f"Length of parameter tensor is {len(params)}, expected 2."
    # print(params)
    for i in range(num_qubits):
        pqc.rz(params[i*3], i)
    
    for i in range(num_qubits - 1):
        pqc.cz(i, i+1)
    
    for i in range(num_qubits):
        pqc.rx(params[i*3 + 1], i)
    
    for i in range(num_qubits - 1):
        pqc.cz(i, i+1)
    
    for i in range(num_qubits):
        pqc.rz(params[i*3 + 2], i)

    return pqc

