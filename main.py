import argparse
import yaml
import torch

from circuits.generate_circuits import custom_ghz
from circuits.modify_circuits import get_circuit_for_model, get_unitary_for_model_pennylane
from simulator.simulate import get_ideal_data

from qiskit_aer import AerSimulator

from models.qiskit_models import SimpleQiskitQuantumModel
from models.pennylane_models import SimpleCustomQuantumModel

MODEL_MAP = {
    "SimpleQiskitQuantumModel": SimpleQiskitQuantumModel,
    "SimpleCustomQuantumModel": SimpleCustomQuantumModel,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train quantum model")
    parser.add_argument("--config", type=str, help="Path to yaml config file")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--num_shots", type=int, help="Number of measurement shots")
    parser.add_argument("--num_qubits", type=int, help="Number of qubits")
    parser.add_argument("--dataset_size", type=int, help="Dataset size")
    parser.add_argument("--num_params", type=int, help="Number of parameters for PQC")
    parser.add_argument("--model", type=str, help="Model class name")
    return parser.parse_args()

def load_config(args):
    cfg = {}
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
    for k, v in vars(args).items():
        if k == "config":
            continue
        if k not in cfg or cfg[k] is None:
            cfg[k] = v
    return cfg

def instantiate_model(cfg, device):
    model_name = cfg.get("model", "SimpleQiskitQuantumModel")
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model {model_name}")

    base_circuit = custom_ghz(cfg["num_qubits"], 2)
    simulator = AerSimulator()
    if model_name == "SimpleQiskitQuantumModel":
        model = MODEL_MAP[model_name](
            num_params=cfg["num_params"],
            base_circuit=base_circuit,
            simulator=simulator,
            num_shots=cfg["num_shots"],
        )
    else:
        model = MODEL_MAP[model_name](
            num_params=cfg["num_params"],
            base_circuit=base_circuit,
            simulator=simulator,
            num_shots=cfg["num_shots"],
        )
    model.to(device)
    return model, base_circuit

def train(cfg):
    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    model, base_circuit = instantiate_model(cfg, device)
    dataset = get_ideal_data(
        cfg["num_qubits"],
        cfg["num_shots"],
        num_values=cfg["dataset_size"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.MSELoss()

    for epoch in range(cfg["epochs"]):
        print(f"Epoch {epoch+1}/{cfg['epochs']}")
        for bitstring, ideal in dataset:
            optimizer.zero_grad()
            ideal = ideal.float().to(device)
            circ = get_circuit_for_model(bitstring, base_circuit)
            measured = model(circuit=circ).float().to(device)
            loss = loss_fn(measured, ideal)
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item():.4f}")

    print("Training complete")
    return model

def main():
    args = parse_args()
    cfg = load_config(args)
    defaults = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 1,
        "lr": 0.01,
        "num_shots": 1024,
        "num_qubits": 3,
        "dataset_size": 10,
        "num_params": 2,
        "model": "SimpleQiskitQuantumModel",
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    train(cfg)

if __name__ == "__main__":
    main()
