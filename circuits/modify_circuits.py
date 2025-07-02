import numpy as np
from qiskit import QuantumCircuit, quantum_info
from qiskit.circuit.library import RZGate, RXGate
import torch
import copy

import pennylane as qml

from .pqc_circuits import qiskit_PQC_RZRX
from models.noise_models import BitPhaseFlipNoise

def append_pqc_to_quantum_circuit(circuit:QuantumCircuit, params: torch.Tensor, pqc_func=qiskit_PQC_RZRX):
    # # First, bound the parameters between -pi and pi
    # bounded_params = torch.pi * torch.tanh(params)

    pqc_circ = pqc_func(circuit.num_qubits, params)
    # print(pqc_circ.draw())
    
    circuit_with_pqc = circuit.remove_final_measurements(inplace=False)
    # print(circuit_with_pqc.draw())
    
    # circuit_with_pqc.append(pqc_circ)
    circuit_with_pqc = circuit_with_pqc.compose(pqc_circ)
    circuit_with_pqc.measure_all()

    return circuit_with_pqc


def append_inverse(circuit:QuantumCircuit, add_measure=False):
    circ = circuit.remove_final_measurements(inplace=False)
    circ = circ.compose(circ.inverse())
    
    if add_measure:
        circ.measure_all()

    return circ


def get_str_circuit_for_model(input:str, circuit:QuantumCircuit):
    """
    Returns a circuit with the input bitstring prefixed as a sequence of X gates, and the circuit's inverse appended. 
    """
    assert len(input) == circuit.num_qubits, f"The number of bits in the input {input} do not equal the number of qubits {circuit.num_qubits}"
    input_circ = QuantumCircuit(len(input))
    for i,b in enumerate(input):
        if b == '1':
            input_circ.x(i)

    model_circ = input_circ.compose(append_inverse(circuit))

    return model_circ


def get_unitary_for_model_pennylane(input:str, circuit:QuantumCircuit):
    qc = get_str_circuit_for_model(input, circuit)
    qc = qc.remove_final_measurements(inplace=False)

    model_circuit_unitary = quantum_info.Operator(qc).data

    return model_circuit_unitary


def append_custom_noisy_inverse(circuit:QuantumCircuit, noise:BitPhaseFlipNoise=None):
    inverse_circuit = QuantumCircuit(circuit.num_qubits)
    circuit_ins = copy.deepcopy(circuit.data)
    circuit_ins.reverse()

    if noise is None:
        noise = BitPhaseFlipNoise()

    noise_list_x = iter(np.random.uniform(low=(noise.x_noise - noise.delta_x), 
                                    high= (noise.x_noise + noise.delta_x),
                                    size= 2 * len(circuit_ins)))
    
    noise_list_z = iter(np.random.uniform(low=(noise.z_noise - noise.delta_z), 
                                     high= (noise.z_noise + noise.delta_z),
                                     size= 2 * len(circuit_ins)))

    for ins in circuit_ins:
        if ins.label is None or 'noise' not in ins.label:
            inverse_circuit.append(ins)
            for q in ins.qubits:
                inverse_circuit.append(RZGate(phi=next(noise_list_z), label='z_noise'), [q])
                inverse_circuit.append(RXGate(theta=next(noise_list_x), label='x_noise'), [q])


    return circuit.compose(inverse_circuit)
    
def get_param_circuit_for_model(input:torch.Tensor, circuit:QuantumCircuit):
    """
    Returns a circuit with the input bitstring prefixed as a sequence of X gates, and the circuit's inverse appended. 
    """
    assert len(input) == circuit.num_qubits, f"The number of parameter rows in the input {input} do not equal the number of qubits {circuit.num_qubits}"
    input_circ = QuantumCircuit(len(input))
    for i,p in enumerate(input):
        input_circ.rx(p[0], i)
        input_circ.rz(p[1], i)
        

    model_circ = input_circ.compose(circuit)

    return model_circ

