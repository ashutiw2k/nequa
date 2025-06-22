import numpy as np
from qiskit import QuantumCircuit, quantum_info
import torch

import pennylane as qml

from .pqc_circuits import qiskit_PQC_RZRX

def append_pqc_to_quantum_circuit(circuit:QuantumCircuit, params: torch.Tensor, pqc_func=qiskit_PQC_RZRX):
    pqc_circ = pqc_func(circuit.num_qubits, params)
    # print(pqc_circ.draw())
    
    circuit_with_pqc = circuit.remove_final_measurements(inplace=False)
    # print(circuit_with_pqc.draw())
    
    # circuit_with_pqc.append(pqc_circ)
    circuit_with_pqc = circuit_with_pqc.compose(pqc_circ)
    circuit_with_pqc.measure_all()

    return circuit_with_pqc


def append_inverse_to_quantum_circuit(circuit:QuantumCircuit, add_measure=False):
    circ = circuit.remove_final_measurements(inplace=False)
    circ = circ.compose(circ.inverse())
    
    if add_measure:
        circ.measure_all()

    return circ


def get_circuit_for_model(input:str, circuit:QuantumCircuit):
    """
    Returns a circuit with the input bitstring prefixed as a sequence of X gates, and the circuit's inverse appended. 
    """
    assert len(input) == circuit.num_qubits, f"The number of bits in the input {input} do not equal the number of qubits {circuit.num_qubits}"
    input_circ = QuantumCircuit(len(input))
    for i,b in enumerate(input):
        if b == '1':
            input_circ.x(i)

    model_circ = input_circ.compose(append_inverse_to_quantum_circuit(circuit))

    return model_circ


def get_unitary_for_model_pennylane(input:str, circuit:QuantumCircuit):
    qc = get_circuit_for_model(input, circuit)
    qc = qc.remove_final_measurements(inplace=False)

    model_circuit_unitary = quantum_info.Operator(qc).data

    return model_circuit_unitary


