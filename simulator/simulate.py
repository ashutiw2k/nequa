import random
import torch
from qiskit import QuantumCircuit, transpile
import pennylane as qml
from qiskit_aer import AerSimulator


def _fill_missing_bitstrings(data_dict):
    n_bits = len(next(iter(data_dict)))  # infer bitstring length
    all_bitstrings = [format(i, f'0{n_bits}b') for i in range(2 ** n_bits)]
    return {bit: data_dict.get(bit, 0) for bit in all_bitstrings}


def get_ideal_shots(input:str, shots:int, assert_check=True):
    if assert_check:
        assert all(c in '01' for c in input), f'Input value {input} not a bitstring'
    ideal_measurement = torch.zeros(2 ** len(input))
    ideal_measurement[int(input, 2)] = shots

    return ideal_measurement

def get_ideal_prob(input:str, assert_check=True):
    if assert_check:
        assert all(c in '01' for c in input), f'Input value {input} not a bitstring'
    
    ideal_measurement = torch.zeros(2 ** len(input))
    ideal_measurement[int(input, 2)] = 1

    return ideal_measurement

def get_soft_ideal_prob(input: str, smoothing: float = 1e-3, assert_check=True):
    if assert_check:
        assert all(c in '01' for c in input), f'Input value {input} not a bitstring'

    n = len(input)
    ideal = torch.full((2**n,), smoothing)
    target_index = int(input, 2)
    ideal[target_index] = 1.0 - (2**n - 1) * smoothing
    return ideal


def get_ideal_data(num_qubits:int, measure_counts:int, num_values:int=100, prob_dist=False):
    valid_bitstrings = [''.join(random.choice('01') for _ in range(num_qubits)) for _ in range(num_values)]
    if prob_dist:
        ideal_data = [(bstring, get_ideal_prob(bstring, assert_check=False)) for bstring in valid_bitstrings]
    else:
        ideal_data = [(bstring, get_ideal_shots(bstring, measure_counts, assert_check=False)) for bstring in valid_bitstrings]

    return ideal_data


def run_circuit_sim(circuit:QuantumCircuit, simulator:AerSimulator, num_shots=2**10):
    transpiled_circ = transpile(circuit, simulator)
    result = simulator.run(transpiled_circ, shots=num_shots).result()
    counts = _fill_missing_bitstrings(result.get_counts(transpiled_circ))
    return torch.tensor([x[1] for x in sorted(counts.items(), key=lambda x:x[0])])

