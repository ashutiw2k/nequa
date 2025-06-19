from qiskit import QuantumCircuit

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

