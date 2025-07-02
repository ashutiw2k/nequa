from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RZGate, XGate, ZGate, HGate, CXGate, CZGate
import numpy as np
import random

from models.noise_models import BitPhaseFlipNoise
from .modify_circuits import append_inverse, append_custom_noisy_inverse

GATES = {'x':XGate, 'h':HGate, 'z':ZGate, 'cx': CXGate, 'cz': CZGate}
QUBITS_FOR_GATES = {'x':1, 'h':1, 'z':1, 'cx': 2, 'cz':2}


def custom_ghz(num_q: int, hammard_on: int, single_qubit_cnot=True, add_measure=False):
    
    assert hammard_on < num_q, f"Qubit chosen to have hammard gate {hammard_on} greater than number of available qubits {num_q} "
    circ = QuantumCircuit(num_q)
    circ.h(hammard_on)
    if single_qubit_cnot:
        for i in range(num_q):
            if not i == hammard_on:
                circ.cx(hammard_on, i)

    else:
        for i in range(num_q-1):
            c = (hammard_on+i) % num_q
            t = (hammard_on+i+1) % num_q
            circ.cx(c,t)

    if add_measure:
        circ.measure_all()

    return circ



def custom_noisy_ghz(num_q: int, hammard_on: int, single_qubit_cnot=True, add_measure=False, noise:BitPhaseFlipNoise=None):
    
    assert hammard_on < num_q, f"Qubit chosen to have hammard gate {hammard_on} greater than number of available qubits {num_q} "
    if noise is None:
        noise = BitPhaseFlipNoise()

    noise_list_x = iter(np.random.uniform(low=(noise.x_noise - noise.delta_x), 
                                     high= (noise.x_noise + noise.delta_x),
                                     size= 2 * num_q + 1))
    
    noise_list_z = iter(np.random.uniform(low=(noise.z_noise - noise.delta_z), 
                                     high= (noise.z_noise + noise.delta_z),
                                     size= 2 * num_q + 1))
    
    circ = QuantumCircuit(num_q)
    circ.h(hammard_on)
    # circ.rz(next(noise_list_z), hammard_on)
    # circ.rx(next(noise_list_x), hammard_on)
    circ.append(RZGate(phi=next(noise_list_z), label='z_noise'), [hammard_on])
    circ.append(RXGate(theta=next(noise_list_x), label='x_noise'), [hammard_on])

    if single_qubit_cnot:
        for i in range(num_q):
            if not i == hammard_on:
                circ.cx(hammard_on, i)
                
                circ.append(RZGate(phi=next(noise_list_z), label='z_noise'), [hammard_on])
                circ.append(RZGate(phi=next(noise_list_z), label='z_noise'), [i])
                
                circ.append(RXGate(theta=next(noise_list_x), label='x_noise'), [hammard_on])
                circ.append(RXGate(theta=next(noise_list_x), label='x_noise'), [i])
                

    else:
        for i in range(num_q-1):
            c = (hammard_on+i) % num_q
            t = (hammard_on+i+1) % num_q
            circ.cx(c,t)
            circ.append(RZGate(phi=next(noise_list_z), label='z_noise'), [c])
            circ.append(RZGate(phi=next(noise_list_z), label='z_noise'), [t])

            
            circ.append(RXGate(theta=next(noise_list_x), label='x_noise'), [c])
            circ.append(RXGate(theta=next(noise_list_x), label='x_noise'), [t])


    if add_measure:
        circ.measure_all()

    return circ


class GetGHZCircuitsForModel():
    def __init__(self, num_qubits, hammard_on, noise:BitPhaseFlipNoise=None,
                    single_qubit_cnot=True):
        self.num_qubits = num_qubits
        self.hammard = hammard_on
        self.noise = noise
        self.single_qubit_cnot = single_qubit_cnot

    def get_ideal_model_circuit(self, input:str):
        """
        Get a quantum circuit without "noise" and it's inverse appended. 
        @param input: Input is a bitstring for now
        """
        assert len(input) == self.num_qubits, f"The number of bits in the input {input} do not equal the number of qubits {self.num_qubits}"
        input_circ = QuantumCircuit(len(input))
        for i,b in enumerate(input):
            if b == '1':
                input_circ.x(i)

        model_circ = input_circ.compose(
            append_inverse(
            custom_ghz(
            num_q=self.num_qubits,
            hammard_on=self.hammard,
            single_qubit_cnot=self.single_qubit_cnot,
            add_measure=False
        )))

        return model_circ

    def get_noisy_model_circuit_bitstring(self, input:str):

        assert len(input) == self.num_qubits, f"The number of bits in the input {input} do not equal the number of qubits {self.num_qubits}"
        input_circ = QuantumCircuit(len(input))
        for i,b in enumerate(input):
            if b == '1':
                input_circ.x(i)

        model_circ = input_circ.compose(
            append_custom_noisy_inverse(
                custom_noisy_ghz(
                    num_q=self.num_qubits, 
                    hammard_on=self.hammard,
                    single_qubit_cnot=self.single_qubit_cnot,
                    add_measure=False,
                    noise=self.noise
                ), 
                self.noise
            )
        )

        return model_circ
 
    
    def get_noisy_model_circuit_params(self, params:np.ndarray):

        assert len(params) == self.num_qubits, f"The number of params in the input {params} do not equal the number of qubits {self.num_qubits}"
        input_circ = QuantumCircuit(self.num_qubits)
        
        for i in range(self.num_qubits):
            input_circ.rx(params[i][0], i)
            input_circ.rz(params[i][1], i)
        
        model_circ = input_circ.compose(
            append_custom_noisy_inverse(
                custom_noisy_ghz(
                    num_q=self.num_qubits, 
                    hammard_on=self.hammard,
                    single_qubit_cnot=self.single_qubit_cnot,
                    add_measure=False,
                    noise=self.noise
                ), 
                self.noise
            )
        )

        return model_circ


def generate_random_circuit(num_qubits: int, num_gates: int, gate_dist:dict=None, seed=None):
    circuit = QuantumCircuit(num_qubits)

    if seed is not None:
        random.seed(seed)

    if gate_dist is None:
        gate_dist = {gate:1/len(GATES) for gate in GATES}

    # print(gate_dist)
    gates = random.choices(
        population=list(gate_dist.keys()), 
        weights=list(gate_dist.values()), 
        k=num_gates)
    
    for gate in gates:
        gate_q = QUBITS_FOR_GATES.get(gate)
        gate_f = GATES.get(gate)
        q = random.sample(population=range(num_qubits), k=gate_q)

        circuit.append(gate_f(), q)

    return circuit
