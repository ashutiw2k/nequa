import random
import torch
from qiskit import QuantumCircuit, transpile
import pennylane as qml
from qiskit_aer import AerSimulator


def _fill_missing_bitstrings(data_dict, n_bits=None):
    """Return a dictionary containing all bitstrings of a given length.

    Qiskit only returns bitstrings that appear at least once. If ``data_dict``
    is empty or missing some bitstrings, those entries are filled with zeros.

    Parameters
    ----------
    data_dict : dict
        Dictionary returned by ``qiskit`` with bitstrings as keys.
    n_bits : int, optional
        Number of qubits used in the circuit. If ``None`` it will be inferred
        from ``data_dict``. When ``data_dict`` is empty this parameter is
        required.
    """

    if n_bits is None:
        try:
            n_bits = len(next(iter(data_dict)))
        except StopIteration:
            raise ValueError(
                "Unable to infer bitstring length from empty data_dict."
            )

    all_bitstrings = [format(i, f"0{n_bits}b") for i in range(2 ** n_bits)]
    return {bit: data_dict.get(bit, 0) for bit in all_bitstrings}


def get_ideal_shots(input:str, shots:int, assert_check=True):
    if assert_check:
        assert all(c in '01' for c in input), f'Input value {input} not a bitstring'
    ideal_measurement = torch.zeros(2 ** len(input))
    ideal_measurement[int(input, 2)] = shots

    return ideal_measurement


def get_soft_ideal_shots(input:str, shots:int, assert_check=True):
    if assert_check:
        assert all(c in '01' for c in input), f'Input value {input} not a bitstring'
    num_vals = 2 ** len(input)
    ideal_measurement = torch.ones(num_vals)
    ideal_measurement[int(input, 2)] = shots - ideal_measurement.sum() + 1

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


def get_ideal_data(num_qubits:int, measure_counts:int, num_values:int=100, prob_dist=False, get_soft=False):
    valid_bitstrings = [''.join(random.choice('01') for _ in range(num_qubits)) for _ in range(num_values)]
    if prob_dist:
        if get_soft:
            ideal_data = [(bstring, get_ideal_prob(bstring, assert_check=False)) for bstring in valid_bitstrings]
        else:
            ideal_data = [(bstring, get_ideal_prob(bstring, assert_check=False)) for bstring in valid_bitstrings]
    else:
        if get_soft:
            ideal_data = [(bstring, get_soft_ideal_shots(bstring, measure_counts, assert_check=False)) for bstring in valid_bitstrings]
        else:
            ideal_data = [(bstring, get_ideal_shots(bstring, measure_counts, assert_check=False)) for bstring in valid_bitstrings]

    return ideal_data


def run_circuit_sim(circuit: QuantumCircuit, simulator: AerSimulator, num_shots=2**10):
    """Run ``circuit`` on ``simulator`` and return the counts as a tensor."""

    transpiled_circ = transpile(circuit, simulator)
    result = simulator.run(transpiled_circ, shots=num_shots).result()
    counts = _fill_missing_bitstrings(
        result.get_counts(transpiled_circ), n_bits=circuit.num_qubits
    )
    return torch.tensor(
        [x[1] for x in sorted(counts.items(), key=lambda x: x[0])]
    )

