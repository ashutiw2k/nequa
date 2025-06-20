import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector


def test_parameterized_gates_equivalence():
    num_qubits = 3
    params = [Parameter(f"theta{i}") for i in range(num_qubits)]
    circ_param = QuantumCircuit(num_qubits)
    for idx, p in enumerate(params):
        circ_param.rx(p, idx)

    # Generate deterministic parameter values for reproducibility
    values = [0.1 * (i + 1) for i in range(num_qubits)]
    bound = circ_param.assign_parameters(
        {p: v for p, v in zip(params, values)}, inplace=False
    )
    sv_param = Statevector.from_instruction(bound)

    circ_const = QuantumCircuit(num_qubits)
    for idx, v in enumerate(values):
        circ_const.rx(v, idx)
    sv_const = Statevector.from_instruction(circ_const)

    assert np.allclose(sv_param.data, sv_const.data)
