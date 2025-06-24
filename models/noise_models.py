from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import coherent_unitary_error
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import numpy as np

def create_coherent_noise_error(dx=np.pi, dz=np.pi):
    """Create a small unitary rotation along X and Z."""
    # Convert degrees to radians


    # Combine: RX after RZ (note: matrix multiplication order matters)
    noise_circuit = QuantumCircuit(1)
    noise_circuit.rz(dz, 0)
    noise_circuit.rx(dx, 0)

    # Return as a QuantumError (even though it's unitary, this lets us insert it)
    return coherent_unitary_error(Operator(noise_circuit).to_matrix())


class CoherentXZNoiseModel(NoiseModel):
    def __init__(self, delta_x_deg=5, delta_z_deg=5, noisy_gates=None):
        super().__init__()
        self.delta_x = np.deg2rad(delta_x_deg)
        self.delta_z = np.deg2rad(delta_z_deg)
        if noisy_gates:
            self.noisy_gates = noisy_gates
        else:
            self.noisy_gates = noisy_gates or ['x', 'h', 'u1', 'u2', 'u3', 'rx', 'rz']
        # self._inject_noise()
        self.custom_quantum_error = create_coherent_noise_error(dx=self.delta_x, dz=self.delta_z)
        
        self.add_all_qubit_quantum_error(self.custom_quantum_error, self.noisy_gates)



