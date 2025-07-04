import argparse
import yaml
import sys, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))  # Makes other scripts and functions discoverable

from utils.mappings import PQC_MAPPINGS
from utils.loss_functions import QuantumFidelityLoss
from circuits.generate_circuits import generate_random_circuit
from circuits.modify_circuits import get_param_input_circuit
from simulator.simulate import run_operator_pennylane, get_ideal_data_state
from models.noise_models import BitPhaseFlipNoise
from models.pennylane_models import SimplePennylaneUnitaryStateModel
from models.data_models import FidelityDataset


def load_config(path: str) -> dict:
    """
    Load a YAML configuration file and return its contents as a dict.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


# --------ARG PARSE LOGIC----------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Parse arguments for PQC experiment runner.'
    )
    # Config file argument
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML config file'
    )
    # Range of number of qubits: one or two integers (shorthand -q)
    parser.add_argument(
        '--qubit_range', '-q',
        type=int,
        nargs='+',
        # metavar=('MIN_QUBITS', 'MAX_QUBITS', 'STEP'),
        help='One or three ints: single qubit count or min, max qubits with step'
    )
    # Range of number of gates: one or two integers (shorthand -g)
    parser.add_argument(
        '--gate_range', '-g',
        type=int,
        nargs='+',
        # metavar=('MIN_GATES', 'MAX_GATES', 'STEP'),
        help='One or three ints: single gate count or min, max gates with step'
    )
    # PQC function: string identifier or module path (shorthand -p)
    parser.add_argument(
        '--pqc_function', '-p',
        type=str,
        help='Name of the PQC function to use'
    )
    # Figure output path (shorthand -o)
    parser.add_argument(
        '--figure_output', '-o',
        type=str,
        nargs='?',
        default='plots/',
        help='Output path for generated figures'
    )
        # Optional training parameters
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        nargs='?',
        default=5,
        help='Number of training epochs (optional)'
    )
    parser.add_argument(
        '--num_data', '-n',
        type=int,
        nargs='?',
        default=5000,
        help='Number of training data samples (optional)',
    )

    parser.add_argument(
        '--num_test', '-t',
        type=int,
        nargs='?',
        default=50,
        help='Number of values to test against (optional)',
    )

    parser.add_argument(
        '--learning_rate', '-l',
        type=float,
        nargs='?',
        default=0.005,
        help='Learning Rate for optimizer (optional)',
    )

    parser.add_argument(
        '--seed',
        nargs='?',
        default=None,
        help='Seed to initialize randomizer with (optional)',
    )

    parser.add_argument(
        '--gate_dist', '-d',
        type=dict,
        nargs='?',
        default=None,
        help='Gate Distribution in Random Circuit (optional)',
    )

    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Run on GPU if available',
    )
    
    parser.add_argument(
        '--save_circuit',
        action='store_true',
        help='Save the circuit to the figure output folder',
    )

    parser.add_argument(
        '--batch', '-b',
        type=int,
        nargs='?',
        default=4,
        help='Number of batches value to pass to the dataloader (optional)',
    )

    return parser.parse_args()


def normalize_range(val, name: str):
    """
    Normalize a range specification to a (min, max) tuple.
    Accepts:
      - int -> (int, int)
      - list/tuple of length 1 -> (x, x)
      - list/tuple of length 2 -> (min, max)
    Otherwise, exits with error.
    """
    if isinstance(val, int):
        return [val, val]
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            return [val[0], val[0]]
        if len(val) == 3:
            return [val[0], val[1], val[2]]
    print(f"Invalid {name}: must be one or three integers", file=sys.stderr)
    sys.exit(1)

# ---------------------------------


def train_model(pqc_model:torch.nn.Module, data:DataLoader, epochs, optimizer, scheduler, loss_fn, 
                device=torch.device('cpu'),):
    
    param_list = []  # Stores parameter evolution across epochs
    loss_list_epoch = []
    enable_debug_logs = True  # Set False to suppress grad/param printouts
    lambda_reg = 0.01         # Regularization weight
    epsil = 0.1


    for epoch in range(epochs):
        print(f'\n Starting Epoch {epoch+1}')
        pqc_model.train()
        epoch_loss_list = []
        grad_norms = []        # per‐batch gradient norms

        data_iterator = tqdm(data)

        for step, (ideal, unitary) in enumerate(data_iterator):
            
            ideal = ideal.to(device)

            optimizer.zero_grad()

            measured = pqc_model(unitary)

            loss = loss_fn(ideal, measured)

            # Add optional regularization to keep angles from zeroing out

            loss.backward()
            optimizer.step()


            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(loss)
            new_lr = optimizer.param_groups[0]["lr"]

            # 4) Check if a plateau was detected
            if new_lr < prev_lr:
                with torch.no_grad():
                    noise = torch.randn_like(pqc_model.raw_params) * epsil * -1
                    pqc_model.raw_params.add_(noise)
                    epsil *= epsil

            raw_param_values = pqc_model.raw_params.detach().cpu().numpy()
            param_list.append(raw_param_values)
            epoch_loss_list.append(loss.item())
            gn = pqc_model.raw_params.grad.norm().item()
            grad_norms.append(gn)

            data_iterator.set_postfix_str(f"Loss: {loss.item():.4e}")

            # Optional: print debug stats
            if enable_debug_logs and step == 0:
                grad_norm = pqc_model.raw_params.grad.norm().item()
                # bounded = pqc_model.raw_params.detach().cpu().numpy()
                print(f"\tStep {step}, grad norm: {grad_norm:.4e}")
                print(f"\tRaw param range: [{raw_param_values.min():.3f}, {raw_param_values.max():.3f}]")
                # print(f"\tBounded param range: [{bounded.min():.3f}, {bounded.max():.3f}]")


        # Log epoch summary
        avg_loss = np.mean(epoch_loss_list)
        loss_list_epoch.append(avg_loss)


        print(f"Epoch {epoch+1} complete: avg loss = {avg_loss:.4e}")

    return pqc_model

def test_model(pqc_model, data:DataLoader):

    pqc_model.eval()
    loss_fn = QuantumFidelityLoss()

    fid_noisy = []
    fid_noisy_pqc = []

    for ideal, U in tqdm(data):

        noi_out_st = run_operator_pennylane(U, pqc_model.num_qubits)
        npqc_out_st = pqc_model(U)

        fid_i_n = [loss_fn.state_fidelity(i, noi) for i,noi in zip(ideal, noi_out_st)]
        fid_i_ne = [loss_fn.state_fidelity(i, npqc.detach()) for i,npqc in zip(ideal, npqc_out_st)]

        fid_noisy.extend(fid_i_n)
        fid_noisy_pqc.extend(fid_i_ne)

    return fid_noisy, fid_noisy_pqc

    pass

def experiment_runner(num_qubits, num_gates, pqc_arch_dict, gate_dist, num_test, batch,
                      epochs=3, num_vals=1000, lr = 0.005, device=torch.device('cpu'), seed=None, circuit_path=None) :
    """
    Returns a tuple of floats, where the first value is the fidelity of the noisy circuit without PQC and the
    second value is the fidelity of the pqc-inserted circuit, against the ideal (identity) values. 
    """
    # print(num_qubits, num_gates, pqc_arch_dict)

    pennylane_pqc = pqc_arch_dict['pennylane']
    qiskit_pqc = pqc_arch_dict['qiskit']
    param_mult = pqc_arch_dict['mult']

    pqc_model = SimplePennylaneUnitaryStateModel(
        num_qubits=num_qubits, num_params=param_mult*num_qubits, 
        pqc_arch_func=pennylane_pqc, 
    )
    optimizer = torch.optim.AdamW(params=pqc_model.parameters(), 
                                  lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',        # look for loss decreasing
                factor=0.5,        # lr ← lr * 0.5
                patience=num_vals / 10,        # wait 1 loop with no improvement
                min_lr=1e-2
            )

    loss_fn = QuantumFidelityLoss()

    random_circuit = generate_random_circuit(num_qubits, num_gates, gate_dist, seed=seed)
    # ideal_data = get_ideal_data_state(num_qubits, num_vals)

    if circuit_path:
        random_circuit.draw('mpl', filename=circuit_path)
        plt.close()

    perm = [int(f"{i:0{num_qubits}b}"[::-1], 2) for i in range(2**num_qubits)]
    noise_model = BitPhaseFlipNoise()

    # 2) build Dataset + DataLoader
    ideal_data = get_ideal_data_state(num_qubits, num_vals)
    train_dataset    = FidelityDataset(ideal_data, random_circuit, perm, noise_model)
    train_loader     = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
    )    
    train_model(
        pqc_model=pqc_model, 
        data=train_loader,
        epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn
    )


    test_data = get_ideal_data_state(num_qubits, int(num_test))
    test_dataset    = FidelityDataset(test_data, random_circuit, perm, noise_model)
    test_loader     = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    ) 
    fid_list_noisy, fid_list_pqc = test_model(
        pqc_model=pqc_model,
        data=test_loader,
    )
    
    return fid_list_noisy, fid_list_pqc

def main():
    args = parse_args()
    config = {}

    # Load YAML config if provided
    if args.config:
        config = load_config(args.config)

    # For each parameter, prefer config value; fall back to CLI
    raw_qubit = config.get('qubit_range', args.qubit_range)
    raw_gate = config.get('gate_range', args.gate_range)
    pqc_function = config.get('pqc_function', args.pqc_function)
    figure_output = config.get('figure_output', args.figure_output)
    epochs = config.get('epochs', args.epochs)
    num_data = config.get('num_data', args.num_data)
    gate_dist = config.get('gate_dist', args.gate_dist)
    learining_rate = config.get('learning_rate', args.learning_rate)
    gpu = config.get('gpu', args.gpu)
    seed = config.get('seed', args.seed)
    save_circuit = config.get('save_circuit', args.save_circuit)
    num_test = config.get('num_test', args.num_test)
    batch = config.get('batch', args.batch)

    # Determine device
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    # Validate required arguments
    missing = []
    if raw_qubit is None:
        missing.append('qubit_range')
    if raw_gate is None:
        missing.append('gate_range')
    if pqc_function is None:
        missing.append('pqc_function')
    # if figure_output is None:
    #     missing.append('figure_output')

    if missing:
        print(f"Missing required parameters: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    if (isinstance(raw_qubit, int) or (isinstance(raw_qubit, (list, tuple)) and len(raw_qubit) == 1)):
        single_qubit = True
    else:
        single_qubit = False
    
    if (isinstance(raw_gate, int) or (isinstance(raw_gate, (list, tuple)) and len(raw_gate) == 1)):
        single_gate = True
    else:
        single_gate = False

    if single_gate and single_qubit:
        single_exp = True
    else:
        single_exp=False

    # Normalize ranges
    qubit_range = normalize_range(raw_qubit, 'qubit_range')
    gate_range = normalize_range(raw_gate, 'gate_range')

    if figure_output:
        os.makedirs(figure_output, exist_ok=True)

    # Example usage printout (replace with real logic)
    print(f"Running PQC experiment with:")
    print(f"  Qubits: {qubit_range[0]} to {qubit_range[1]}")
    print(f"  Gates: {gate_range[0]} to {gate_range[1]}")
    print(f"  PQC Function: {pqc_function}")
    print(f"  Plot Output Dir: {figure_output}")
    print(f"  Num Epochs per Experiment: {epochs}")
    print(f"  Training Data Samples: {num_data}")
    print(f"  Training Data Batch Size: {batch}")
    print(f"  Testing Data Samples: {num_test}")
    print(f"  Gate Distribution: {gate_dist}")
    print(f"  Learning Rate: {learining_rate}")

    fig_qubit_start = qubit_range[0]
    fig_qubit_end = qubit_range[1]
    fig_gate_start = gate_range[0]
    fig_gate_end = gate_range[1]


    gate_nam = f"{fig_gate_start}" if fig_gate_start == fig_gate_end else f"{fig_gate_start}-{fig_gate_end}"
    qubit_nam = f"{fig_qubit_start}" if fig_qubit_start == fig_qubit_end else f"{fig_qubit_start}-{fig_qubit_end}"
    seed_nam = f"_{seed}-seed" if seed else ""
    

    qubit_range[1] += 1
    gate_range[1] += 1
    
    # TODO: Insert experiment logic here

    avg_qubit_fid_noisy = []
    avg_qubit_fid_pqc = []
    for num_q in range(*qubit_range):
        avg_gate_fid_noisy = []
        avg_gate_fid_pqc = []
        for num_g in range(*gate_range):
            print(f"\n{'-'*25}\nRunning Experiment and training random circuit with {num_q} qubits and {num_g} gates.")
            if save_circuit:
                circuit_path = figure_output + f"Circuit_{num_q}q_{num_g}g_{seed}-seed.png"
            else:
                circuit_path = None

            fid_val_noisy, fid_val_pqc = experiment_runner(num_q, num_g, PQC_MAPPINGS[pqc_function],
                                        epochs=epochs, num_vals=num_data, num_test=num_test, batch=batch,
                                        lr=learining_rate, gate_dist=gate_dist, device=device, seed=seed, 
                                        circuit_path=circuit_path)
            if not single_exp:
                avg_gate_fid_noisy.append(np.mean(fid_val_noisy))
                avg_gate_fid_pqc.append(np.mean(fid_val_pqc))
            else:
                print(f"Plotting Fidelity results for {num_q} qubits, {num_g} gates.")
                plt.plot(range(int(num_test)), fid_val_noisy,      label='Fidelity: Ideal, Noisy')
                plt.plot(range(int(num_test)), fid_val_pqc,  label='Fidelity: Ideal, PQC-corrected')
                plt.xlabel('Experiment Number')          # optional cosmetics
                plt.ylabel('Fidelity')
                plt.title(f'Fidelity for {num_q} qubits, {num_g} gates.')
                plt.legend()                      # shows the two labels
                plt.savefig(figure_output+f"Fidelity_Exp_{num_q}q_{num_g}g{seed_nam}_{batch}b.png")
                plt.close()


        if not single_exp and not single_gate:
            print("Plotting average fidelity across gates.")
            plt.plot(range(*gate_range), avg_gate_fid_noisy,      label='Fidelity: Ideal, Noisy')
            plt.plot(range(*gate_range), avg_gate_fid_pqc,  label='Fidelity: Ideal, PQC-corrected')
            plt.xlabel('Number of Gates')          # optional cosmetics
            plt.ylabel('Fidelity')
            plt.title('Average Fidelity across varying number of gates')
            plt.legend()                      # shows the two labels
            plt.savefig(figure_output+f"Fidelity_Avg_{num_q}q_{gate_nam}g{seed_nam}_{batch}b.png")
            plt.close()
        
        if not single_qubit:
            avg_qubit_fid_noisy.append(np.mean(avg_gate_fid_noisy))
            avg_qubit_fid_pqc.append(np.mean(avg_gate_fid_pqc))
    
    if not single_exp and not single_qubit:
        print("Plotting average fidelity across qubits.")
        plt.plot(range(*qubit_range), avg_qubit_fid_noisy,      label='Fidelity: Ideal, Noisy')
        plt.plot(range(*qubit_range), avg_qubit_fid_pqc,  label='Fidelity: Ideal, PQC-corrected')
        plt.xlabel('Number of Qubits')          # optional cosmetics
        plt.ylabel('Fidelity')        
        plt.title('Average Fidelity across varying number of qubits')
        plt.legend() 
        plt.savefig(figure_output+f"Fidelity_Avg_{qubit_nam}q_{gate_nam}g{seed_nam}_{batch}b.png")
        plt.close()



if __name__ == '__main__':
    main()
