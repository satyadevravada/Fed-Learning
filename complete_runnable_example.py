"""
Complete Federated Learning Example
Run this script directly to start training federated learning models!

Usage:
    python main.py --algorithm fedavg --num_rounds 30
    python main.py --compare-all --num_rounds 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import numpy as np
from tqdm import tqdm
import argparse
import json
import os
from datetime import datetime

# ============================================
# Import all components (assuming they're in the same directory)
# ============================================

try:
    from fed_learning_framework import (
        FederatedOptimizer, FederatedClient, FederatedServer,
        FedAvg, FedAdam, FedAdagrad, create_non_iid_split,
        federated_learning_round
    )
    from additional_optimizers import (
        FedRMSprop, FedMomentum, FedNAdam, FedAdamW, FedYogi
    )
except ImportError:
    print("Note: Import classes from separate files, or include them in this file")


# ============================================
# Complete Standalone Implementation
# ============================================

class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def load_mnist_for_federated(test_ratio=0.1):
    """Load MNIST dataset (fallback if FEMNIST not available)"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download MNIST
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    print(f"Loaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_dataset, test_dataset


def run_single_experiment(
    algorithm_name,
    train_dataset,
    test_dataset,
    num_clients=10,
    num_rounds=50,
    local_epochs=5,
    learning_rate=0.01,
    alpha=0.5,
    batch_size=32,
    device='cuda',
    verbose=True
):
    """
    Run a single federated learning experiment
    
    Args:
        algorithm_name: Name of algorithm ('fedavg', 'fedadam', etc.)
        train_dataset: Training dataset
        test_dataset: Test dataset
        num_clients: Number of federated clients
        num_rounds: Number of communication rounds
        local_epochs: Local training epochs per round
        learning_rate: Learning rate
        alpha: Dirichlet concentration parameter
        batch_size: Training batch size
        device: Device to use
        verbose: Print progress
    
    Returns:
        Dictionary with training history
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {algorithm_name}")
        print(f"Device: {device}")
        print(f"Clients: {num_clients}, Rounds: {num_rounds}, Local Epochs: {local_epochs}")
        print(f"Learning Rate: {learning_rate}, Alpha: {alpha}")
        print(f"{'='*60}\n")
    
    # Create non-IID data split
    client_indices = create_non_iid_split(train_dataset, num_clients, alpha)
    
    # Create data loaders
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Parse algorithm configuration
    config = parse_algorithm_config(algorithm_name, learning_rate)
    
    # Initialize clients
    clients = []
    for i, loader in enumerate(client_loaders):
        client = FederatedClient(
            client_id=i,
            train_loader=loader,
            device=device,
            optimizer_name=config['client_optimizer'],
            lr=config['client_lr']
        )
        clients.append(client)
    
    # Initialize server
    num_classes = 10  # MNIST/FEMNIST classes
    if 'femnist' in str(train_dataset.__class__).lower():
        num_classes = 62
    
    model = SimpleCNN(num_classes=num_classes)
    fed_optimizer = config['server_optimizer']
    
    server = FederatedServer(
        model=model,
        test_loader=test_loader,
        device=device,
        fed_optimizer=fed_optimizer
    )
    
    # Training loop
    history = {
        'algorithm': algorithm_name,
        'rounds': [],
        'test_loss': [],
        'test_accuracy': [],
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'local_epochs': local_epochs,
            'learning_rate': learning_rate,
            'alpha': alpha,
            'batch_size': batch_size
        }
    }
    
    pbar = tqdm(range(num_rounds), desc=algorithm_name) if verbose else range(num_rounds)
    
    for round_num in pbar:
        test_loss, test_accuracy = federated_learning_round(
            server, clients, local_epochs=local_epochs
        )
        
        history['rounds'].append(round_num)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)
        
        if verbose and isinstance(pbar, tqdm):
            pbar.set_postfix({
                'Loss': f'{test_loss:.4f}',
                'Acc': f'{test_accuracy:.2f}%'
            })
        elif verbose and round_num % 5 == 0:
            print(f"Round {round_num:3d} | Loss: {test_loss:.4f} | Acc: {test_accuracy:.2f}%")
    
    if verbose:
        print(f"\nFinal Results for {algorithm_name}:")
        print(f"  Test Loss: {history['test_loss'][-1]:.4f}")
        print(f"  Test Accuracy: {history['test_accuracy'][-1]:.2f}%")
        print(f"  Best Accuracy: {max(history['test_accuracy']):.2f}%")
    
    return history


def parse_algorithm_config(algorithm_name, learning_rate):
    """Parse algorithm name and return configuration"""
    name = algorithm_name.lower()
    
    # Default configuration
    config = {
        'client_optimizer': 'sgd',
        'client_lr': learning_rate,
        'server_optimizer': None
    }
    
    if name == 'fedavg':
        config['server_optimizer'] = FedAvg(learning_rate)
    
    elif name == 'fedadam' or name == 'fedadam-server':
        config['server_optimizer'] = FedAdam(learning_rate)
    
    elif name == 'fedadam-client':
        config['client_optimizer'] = 'adam'
        config['client_lr'] = learning_rate * 0.1  # Adam needs smaller LR
        config['server_optimizer'] = FedAvg(learning_rate)
    
    elif name == 'fedadagrad' or name == 'fedadagrad-server':
        config['server_optimizer'] = FedAdagrad(learning_rate)
    
    elif name == 'fedadagrad-client':
        config['client_optimizer'] = 'adagrad'
        config['server_optimizer'] = FedAvg(learning_rate)
    
    elif name == 'fedmomentum' or name == 'fedmomentum-server':
        config['server_optimizer'] = FedMomentum(learning_rate)
    
    elif name == 'fedmomentum-client':
        config['client_optimizer'] = 'momentum'
        config['server_optimizer'] = FedAvg(learning_rate)
    
    elif name == 'fedrmsprop' or name == 'fedrmsprop-server':
        config['server_optimizer'] = FedRMSprop(learning_rate)
    
    elif name == 'fedrmsprop-client':
        config['client_optimizer'] = 'rmsprop'
        config['client_lr'] = learning_rate * 0.1
        config['server_optimizer'] = FedAvg(learning_rate)
    
    elif name == 'fednadam' or name == 'fednadam-server':
        config['server_optimizer'] = FedNAdam(learning_rate)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    return config


def compare_algorithms(
    train_dataset,
    test_dataset,
    algorithms=['fedavg', 'fedadam', 'fedadam-client', 'fedmomentum', 'fednadam'],
    num_rounds=50,
    **kwargs
):
    """
    Compare multiple algorithms
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        algorithms: List of algorithm names to compare
        num_rounds: Number of rounds
        **kwargs: Additional arguments for run_single_experiment
    
    Returns:
        Dictionary mapping algorithm names to histories
    """
    results = {}
    
    for algo in algorithms:
        print(f"\n{'#'*70}")
        print(f"# Running {algo}")
        print(f"{'#'*70}")
        
        history = run_single_experiment(
            algorithm_name=algo,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_rounds=num_rounds,
            **kwargs
        )
        
        results[algo] = history
    
    return results


def save_results(results, output_dir='results'):
    """Save experiment results to JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/results_{timestamp}.json"
    
    # Convert to serializable format
    serializable_results = {}
    for name, history in results.items():
        serializable_results[name] = {
            'algorithm': history['algorithm'],
            'rounds': history['rounds'],
            'test_loss': history['test_loss'],
            'test_accuracy': history['test_accuracy'],
            'config': history['config']
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to {filename}")
    return filename


def plot_results(results, output_dir='results/plots'):
    """Plot comparison of results"""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, history in results.items():
        plt.plot(history['rounds'], history['test_accuracy'], 
                label=name, linewidth=2, marker='o', markersize=3, markevery=5)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Federated Learning: Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    for name, history in results.items():
        plt.plot(history['rounds'], history['test_loss'], 
                label=name, linewidth=2, marker='s', markersize=3, markevery=5)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.title('Federated Learning: Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {filename}")
    
    plt.show()


def print_summary_table(results):
    """Print summary table of results"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"{'Algorithm':<25} {'Final Acc':<12} {'Best Acc':<12} {'Final Loss':<12}")
    print("-"*80)
    
    for name, history in results.items():
        final_acc = history['test_accuracy'][-1]
        best_acc = max(history['test_accuracy'])
        final_loss = history['test_loss'][-1]
        
        print(f"{name:<25} {final_acc:>10.2f}%  {best_acc:>10.2f}%  {final_loss:>10.4f}")
    
    print("="*80 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Federated Learning with SGD Variants')
    
    # Experiment configuration
    parser.add_argument('--algorithm', type=str, default='fedavg',
                       help='Algorithm to run (fedavg, fedadam, etc.)')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all algorithms')
    
    # Training parameters
    parser.add_argument('--num_clients', type=int, default=10,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=30,
                       help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Local training epochs per round')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet alpha (lower = more non-IID)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset
    print("Loading dataset...")
    try:
        from datasets import load_dataset
        print("Attempting to load FEMNIST...")
        # Try FEMNIST first
        # train_dataset, test_dataset = load_femnist_data()
        # For now, use MNIST as fallback
        train_dataset, test_dataset = load_mnist_for_federated()
    except:
        print("Using MNIST dataset...")
        train_dataset, test_dataset = load_mnist_for_federated()
    
    # Run experiment(s)
    if args.compare_all:
        algorithms = ['fedavg', 'fedadam', 'fedadam-client', 
                     'fedmomentum', 'fednadam']
        results = compare_algorithms(
            train_dataset,
            test_dataset,
            algorithms=algorithms,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            alpha=args.alpha,
            batch_size=args.batch_size,
            device=args.device
        )
    else:
        history = run_single_experiment(
            algorithm_name=args.algorithm,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_clients=args.num_clients,
            num_rounds=args.num_rounds,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            alpha=args.alpha,
            batch_size=args.batch_size,
            device=args.device
        )
        results = {args.algorithm: history}
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    save_results(results)
    
    # Plot results
    if not args.no_plot and len(results) > 1:
        try:
            plot_results(results)
        except Exception as e:
            print(f"Warning: Could not plot results: {e}")
    
    print("\n✨ Experiment completed successfully! ✨\n")


if __name__ == "__main__":
    main()
