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



class BitPhaseFlipNoise:
    """
    Simple container for parametrising a combined bit-flip (X) and
    phase-flip (Z) coherent over-rotation noise model.

    Parameters
    ----------
    x_rad : float, optional
        Nominal rotation angle (in **radians**) used to model *bit-flip* noise,
        i.e. an unwanted RX-type over-rotation.  Default: π/30.

    z_rad : float, optional
        Nominal rotation angle (in **radians**) used to model *phase-flip*
        noise, i.e. an unwanted RZ-type over-rotation.  Default: π/30.

    delta_x : float, optional
        The percentage by which `x_rad` should randomly deviate.  Default: 5 %.

    delta_z : float, optional
        The percentage by which `z_rad` should randomly deviate.  Default: 5 %.

    noisy_gates : list[str] | None, optional
        Sequence of gate names (lower-case Qiskit mnemonics) that should be
        tagged as noisy when constructing a Qiskit‐Aer ``NoiseModel``.
        If *None*, a sensible default covering single-qubit rotations is used.
    """

    def __init__(
        self,
        x_rad: float = np.pi / 30,
        z_rad: float = np.pi / 30,
        delta_x: float = 5,
        delta_z: float = 5,
        noisy_gates: list[str] | None = None,
    ):
        # Store the *nominal* over-rotation angles
        self.x_noise = x_rad
        self.z_noise = z_rad

        # Convert the percentage deltas into absolute radian offsets
        self.delta_x = delta_x * self.x_noise / 100.0
        self.delta_z = delta_z * self.z_noise / 100.0

        # Use caller-supplied gate list or fall back to a default set
        self.noisy_gates = (
            noisy_gates
            if noisy_gates is not None
            else ["x", "h", "u1", "u2", "u3", "rx", "rz"]
        )
