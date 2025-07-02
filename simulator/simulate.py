import random
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import BaseEstimatorV1      # Terra ≥ 0.46
from qiskit.quantum_info import Statevector, Operator

from qiskit_aer import AerSimulator
import pennylane as qml
from tqdm import tqdm
import numpy as np

from collections import Counter

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
    return {bit: data_dict.get(bit, 1e-8) for bit in all_bitstrings}


def get_ideal_shots(input:str, shots:int, assert_check=True):
    if assert_check:
        assert all(c in '01' for c in input), f'Input value {input} not a bitstring'
    ideal_measurement = torch.full((2 ** len(input),), 1e-10)
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


def get_ideal_data_state(num_qubits:int, num_vals:int=1000):
    ideal_data_list = []

    # Analytic, shot-free device
    dev = qml.device("default.qubit", wires=num_qubits, shots=None)
    @qml.qnode(dev, diff_method=None)        # ← no grads tracked
    def rx_rz_layer(param_matrix):
        """|ψ⟩ produced by RX–RZ layer with per-qubit params (shape: [n,2])."""
        for w in range(num_qubits):
            qml.RX(param_matrix[w, 0], wires=w)
            qml.RZ(param_matrix[w, 1], wires=w)
        return qml.state()                   # big-endian ordering

    for _ in tqdm(range(num_vals), desc="Generating Ideal Data"):
        # Random angles in [0, 2π)
        params = (torch.rand((num_qubits, 2)) * torch.pi * 2).detach().cpu().numpy()

        # Evaluate circuit → NumPy array, then convert to Torch tensor
        state_tensor = torch.tensor(rx_rz_layer(params))

        ideal_data_list.append((params, state_tensor))

    return ideal_data_list



def get_ideal_data_superpos(num_qubits:int, num_shots:int=1024, num_vals:int=1000, prob_dist=False, statevector=False):
    ideal_data_list = []
    # num_qubits = circuit.num_qubits
    
    for i in tqdm(range(num_vals), f'Generating Ideal Data'):
        params = (torch.rand((num_qubits, 2)) * torch.pi * 2).detach().cpu().numpy() # Get random values between 0 and 2pi
        circ = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            circ.rx(params[i][0], i)
            circ.rz(params[i][1], i)
        
        if statevector:
            output_state = Statevector.from_instruction(circ)
            state_tensor = torch.tensor(output_state.data)
            ideal_data_list.append((params, state_tensor))
        
        else:
            circ.measure_all()

        # transpiled_circ = transpile(circ, ideal_sim)
        # result = ideal_sim.run(transpiled_circ, shots=num_shots).result()
        # counts = result.get_counts(transpiled_circ)
        # full_counts = _fill_missing_bitstrings(counts, num_qubits)
            counts = run_circuit_sampler(circ, num_shots, prob_dist)
            ideal_data_list.append((params, counts))

    return ideal_data_list

def run_circuit_sampler(circuit:QuantumCircuit, shots=2**10, prob_dist=False):
    """
    Run ``circuit`` with Terra’s default Sampler backend (falls back
    to pure-Python BasicAer if nothing faster is installed) and return
    a length-2**n tensor of shot counts.
    """
    job      = AerSimulator().run(circuit, shots=shots)   # no transpiler needed
    qdist    = job.result().get_counts()             # {bitstring:prob}
    num_qubits = circuit.num_qubits

    counts   = torch.tensor(
        [qdist.get(i, 0) for i in range(2**num_qubits)],
    )

    if not prob_dist:
        counts = torch.round(counts).int()
    return counts


def run_sampler_pennylane(circuit:QuantumCircuit, shots=2**10):
    
    num_qubits = circuit.num_qubits

    dev = qml.device("default.qubit", wires=num_qubits, shots=shots)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def sample_qnode(U_big):
        # inject your Qiskit-built circuit (now re-indexed)
        qml.QubitUnitary(U_big, wires=range(num_qubits))
        # full-register sampling
        return qml.sample(wires=range(num_qubits))
        
    circuit_op = Operator(circuit.remove_final_measurements(inplace=False)).data
    perm = [int(f"{i:0{num_qubits}b}"[::-1], 2) for i in range(2**num_qubits)] 
    circuit_op_pennylane =  circuit_op[np.ix_(perm, perm)] 
    samples = sample_qnode(circuit_op_pennylane)

    bitstrings = ["".join(str(bit.item()) for bit in samp) for samp in samples]

    return Counter(bitstrings)


def run_state_pennylane(circuit:QuantumCircuit):
    
    num_qubits = circuit.num_qubits

    dev = qml.device("default.qubit", wires=num_qubits, shots=None)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def state_qnode(U_big):
        # inject your Qiskit-built circuit (now re-indexed)
        qml.QubitUnitary(U_big, wires=range(num_qubits))
        # full-register sampling
        return qml.state()
        
    circuit_op = Operator(circuit.remove_final_measurements(inplace=False)).data
    perm = [int(f"{i:0{num_qubits}b}"[::-1], 2) for i in range(2**num_qubits)] 
    circuit_op_pennylane =  circuit_op[np.ix_(perm, perm)] 
    return state_qnode(circuit_op_pennylane)

