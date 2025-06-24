import torch
from torch import nn
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from circuits.modify_circuits import append_pqc_to_quantum_circuit
from circuits.pqc_circuits import qiskit_PQC_RZRX
from simulator.simulate import run_circuit_sim

# Autograd function, should not be needing it 
# import torch
# from torch.autograd import Function

# class CircuitRunnerFunction(Function):
#     @staticmethod
#     def forward(ctx, params, base_circuit, simulator, num_shots, runner):
#         # 1) stash everything we’ll need for backward
#         ctx.base_circuit = base_circuit
#         ctx.simulator    = simulator
#         ctx.num_shots    = num_shots
#         ctx.runner       = runner
#         ctx.shift        = torch.pi / 2

#         # 2) save the “current” params
#         ctx.save_for_backward(params)

#         # 3) run your sim exactly as before
#         circ_pqc = append_pqc_to_quantum_circuit(base_circuit, params)
#         out = runner(circ_pqc, simulator, num_shots)      # → torch.Tensor, no grad yet
#         out = out.to(params.device)                       # move to same device
#         out = out + params.sum() * 0.0

#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         # Retrieve saved values
#         params,           = ctx.saved_tensors
#         base_circ         = ctx.base_circuit
#         simulator, shots  = ctx.simulator, ctx.num_shots
#         runner            = ctx.runner
#         shift             = ctx.shift
#         device            = params.device

#         # Prepare gradient vector
#         grad_params = torch.zeros_like(params)

#         # For each parameter θᵢ, do two shifted evaluations
#         for i in range(params.numel()):
#             # +shift
#             p_plus = params.clone()
#             p_plus[i] += shift
#             circ_p = append_pqc_to_quantum_circuit(base_circ, p_plus)
#             out_plus = runner(circ_p, simulator, shots).to(device)

#             # –shift
#             p_minus = params.clone()
#             p_minus[i] -= shift
#             circ_m = append_pqc_to_quantum_circuit(base_circ, p_minus)
#             out_minus = runner(circ_m, simulator, shots).to(device)

#             # parameter-shift derivative of probabilities
#             dP_dθ = 0.5 * (out_plus - out_minus)

#             # chain rule: sum over output dims
#             grad_params[i] = torch.dot(grad_output, dP_dθ)

#         # gradients only for params; other inputs (circuit, simulator…) get None
#         return grad_params, None, None, None, None




class SimpleQiskitQuantumModel(nn.Module):
    def __init__(self, num_params:int, base_circuit: QuantumCircuit, 
                 simulator:AerSimulator, num_shots:int, 
                 circuit_runner = run_circuit_sim, pqc_arch_func=qiskit_PQC_RZRX):
        super().__init__()

        self.pqc_params = nn.Parameter(2 * torch.pi * torch.rand(num_params)) # Convert parameters from [0,1) to [0,2\pi)
        self.simulator = simulator
        self.transpile = transpile
        self.num_shots = num_shots
        self.runner = circuit_runner
        self.base_circuit = base_circuit
        self.pqc_arch_func = pqc_arch_func
        pass

    
    def forward(self, circuit:QuantumCircuit=None):
        """
        @param circuit: The quantum circuit with the input and it's inverse appended. 
        """    
        circ = circuit.remove_final_measurements(inplace=False)
        circuit_pqc = append_pqc_to_quantum_circuit(circuit=circ, 
                                                    params=self.pqc_params,
                                                    pqc_func=self.pqc_arch_func)
        
        measured_tensor = self.runner(circuit_pqc, self.simulator, self.num_shots)

        return measured_tensor + (self.pqc_params.sum() * 0.0) # The params.sum is for the required_grad=True error fix. 




        
