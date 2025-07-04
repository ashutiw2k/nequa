from circuits.pqc_circuits import *

PQC_MAPPINGS = {
    'unique_rzrxrz' : {'qiskit': qiskit_PQC_RZRXRZ_unique, 'pennylane':pennylane_PQC_RZRXRZ_unique, 'mult':3},
    'unique_rzrx' : {'qiskit': qiskit_PQC_RZRX_unique, 'pennylane':pennylane_PQC_RZRX_unique, 'mult':2},
    'unique_u3' : {'qiskit': qiskit_PQC_U3_unique, 'pennylane':pennylane_PQC_U3_unique, 'mult':3},
    'unique_rzrxrz_cz' : {'qiskit': qiskit_PQC_RZRXRZ_CZ_unique, 'pennylane':pennylane_PQC_RZRXRZ_CZ_unique, 'mult':3},
}