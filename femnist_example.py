import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Import the framework classes
from fed_learning_framework import (
    FederatedClient, FederatedServer, FedAvg, FedAdam, FedAdagrad,
    FedNAdam, FedAdamW, FedYogi, create_non_iid_split,
    federated_learning_round_sequential
)


class SimpleCNN(nn.Module):
    """
    Simple CNN for FEMNIST based on PyTorch MNIST example
    https://github.com/pytorch/examples/blob/main/mnist/main.py
    """
    
    def __init__(self, num_classes=62):
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


class FEMNISTDataset(torch.utils.data.Dataset):
    """Wrapper for FEMNIST dataset from HuggingFace"""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
        # Pre-extract labels for efficient access
        self.labels = []
        for item in hf_dataset:
            label = item.get('label', item.get('character', item.get('digit')))
            if label is None:
                raise KeyError(f"No label found in dataset item: {list(item.keys())}")
            self.labels.append(label)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0
        
        return image, label


def load_femnist_data(test_split_ratio=0.15, subset_size=None, seed=42, verbose=True):
    """
    Load FEMNIST dataset with train/test split
    
    Args:
        test_split_ratio: Fraction of data for testing (server-side)
        subset_size: Optional limit on total dataset size
        seed: Random seed
        verbose: Print statistics
    
    Returns:
        train_dataset, test_dataset
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if verbose:
        print("=" * 70)
        print("Loading FEMNIST dataset from HuggingFace...")
        print("=" * 70)
    
    # Load full dataset
    dataset = load_dataset("flwrlabs/femnist", split="train", trust_remote_code=True)
    total_size = len(dataset)
    
    if verbose:
        print(f"Original dataset size: {total_size:,} samples")
    
    # Shuffle indices
    indices = np.random.permutation(total_size)
    
    # Apply subset if requested
    if subset_size is not None and subset_size < total_size:
        indices = indices[:subset_size]
        if verbose:
            print(f"Using subset: {subset_size:,} samples")
    
    # Train/test split
    test_size = int(len(indices) * test_split_ratio)
    train_size = len(indices) - test_size
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create datasets
    train_dataset = FEMNISTDataset(dataset.select(train_indices.tolist()))
    test_dataset = FEMNISTDataset(dataset.select(test_indices.tolist()))
    
    if verbose:
        print(f"\n✓ Train size: {len(train_dataset):,} samples")
        print(f"✓ Test size: {len(test_dataset):,} samples")
        print(f"✓ Split ratio: {(1-test_split_ratio)*100:.0f}/{test_split_ratio*100:.0f}")
        print(f"✓ Number of classes: 62 (digits + upper + lower case)")
        print("=" * 70 + "\n")
    
    return train_dataset, test_dataset


def run_federated_experiment(
    model_class,
    train_dataset,
    test_dataset,
    num_clients=10,
    num_rounds=100,
    local_epochs=5,
    client_optimizer='sgd',
    server_aggregator='fedavg',
    client_lr=0.01,
    server_lr=0.01,
    alpha=0.5,
    batch_size=32,
    device='cuda',
    seed=42,
    save_dir='results'
):
    """
    Run a complete federated learning experiment with detailed tracking
    
    Args:
        model_class: Model class to instantiate
        train_dataset: Training dataset
        test_dataset: Test dataset
        num_clients: Number of federated clients
        num_rounds: Number of federated learning rounds
        local_epochs: Number of local training epochs per round
        client_optimizer: Client-side optimizer ('sgd', 'adam', etc.)
        server_aggregator: Server aggregation method ('fedavg', 'fedadam', etc.)
        client_lr: Client learning rate
        server_lr: Server learning rate
        alpha: Dirichlet alpha for non-IID split (lower = more heterogeneous)
        batch_size: Batch size for training
        device: Device to use
        seed: Random seed
        save_dir: Directory to save results
    
    Returns:
        Dictionary with experiment results
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING EXPERIMENT")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  • Clients: {num_clients}")
    print(f"  • Rounds: {num_rounds}")
    print(f"  • Local epochs: {local_epochs}")
    print(f"  • Client optimizer: {client_optimizer} (lr={client_lr})")
    print(f"  • Server aggregator: {server_aggregator} (lr={server_lr})")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Non-IID alpha: {alpha}")
    print(f"  • Device: {device}")
    print("=" * 70 + "\n")
    
    # Create non-IID split
    print(f"Creating non-IID data split (alpha={alpha})...")
    client_indices = create_non_iid_split(train_dataset, num_clients, alpha=alpha)
    
    # Analyze data distribution
    print("\nClient data distribution:")
    for i in range(min(num_clients, 10)):  # Show first 10 clients
        indices = client_indices[i]
        labels = [train_dataset.labels[idx] for idx in indices[:500]]  # Sample
        unique_labels = len(set(labels))
        print(f"  Client {i:2d}: {len(indices):5d} samples, "
              f"{unique_labels:2d} unique classes")
    
    if num_clients > 10:
        print(f"  ... ({num_clients - 10} more clients)")
    
    # Create client data loaders
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    
    # Create test loader (server-side data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize clients
    clients = []
    for i, loader in enumerate(client_loaders):
        client = FederatedClient(
            client_id=i,
            train_loader=loader,
            device=device,
            optimizer_name=client_optimizer,
            lr=client_lr
        )
        clients.append(client)
    
    # Initialize server with appropriate aggregator
    model = model_class()
    
    # Create server optimizer
    if server_aggregator.lower() == 'fedavg':
        fed_optimizer = FedAvg(learning_rate=server_lr)
    elif server_aggregator.lower() == 'fedadam':
        fed_optimizer = FedAdam(learning_rate=server_lr)
    elif server_aggregator.lower() == 'fedadagrad':
        fed_optimizer = FedAdagrad(learning_rate=server_lr)
    elif server_aggregator.lower() == 'fednadam':
        fed_optimizer = FedNAdam(learning_rate=server_lr)
    elif server_aggregator.lower() == 'fedadamw':
        fed_optimizer = FedAdamW(learning_rate=server_lr)
    elif server_aggregator.lower() == 'fedyogi':
        fed_optimizer = FedYogi(learning_rate=server_lr)
    else:
        raise ValueError(f"Unknown server aggregator: {server_aggregator}")
    
    server = FederatedServer(
        model=model,
        test_loader=test_loader,
        device=device,
        fed_optimizer=fed_optimizer
    )
    
    # Training loop with detailed tracking
    print("\n" + "=" * 70)
    print("Starting federated training...")
    print("=" * 70 + "\n")
    
    results = {
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'local_epochs': local_epochs,
            'client_optimizer': client_optimizer,
            'server_aggregator': server_aggregator,
            'client_lr': client_lr,
            'server_lr': server_lr,
            'alpha': alpha,
            'batch_size': batch_size,
            'seed': seed
        },
        'rounds': [],
        'train_loss': [],  # Track training loss if possible
        'test_loss': [],
        'test_accuracy': []
    }
    
    best_accuracy = 0.0
    
    for round_num in range(num_rounds):
        # Run federated round
        test_loss, test_accuracy = federated_learning_round_sequential(
            server, clients, local_epochs=local_epochs, show_progress=False
        )
        
        # Store results
        results['rounds'].append(round_num + 1)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_accuracy)
        
        # Track best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
        
        # Print progress
        if (round_num + 1) % 5 == 0 or round_num == 0:
            print(f"Round {round_num+1:3d}/{num_rounds} | "
                  f"Loss: {test_loss:.4f} | "
                  f"Accuracy: {test_accuracy:.2f}% | "
                  f"Best: {best_accuracy:.2f}%")
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Final Test Loss: {results['test_loss'][-1]:.4f}")
    print(f"Final Test Accuracy: {results['test_accuracy'][-1]:.2f}%")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("=" * 70 + "\n")
    
    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        exp_name = f"{client_optimizer}_{server_aggregator}_alpha{alpha}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = os.path.join(save_dir, f"{exp_name}_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {json_path}")
    
    return results


def compare_algorithms(train_dataset, test_dataset, configs, save_dir='results'):
    """
    Compare multiple federated learning configurations
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        configs: List of configuration dictionaries
        save_dir: Directory to save results
    
    Returns:
        Dictionary mapping config names to results
    """
    all_results = {}
    
    print("\n" + "=" * 70)
    print(f"COMPARING {len(configs)} CONFIGURATIONS")
    print("=" * 70 + "\n")
    
    for i, config in enumerate(configs, 1):
        name = config.pop('name')
        print(f"\n[{i}/{len(configs)}] Running: {name}")
        print("-" * 70)
        
        try:
            results = run_federated_experiment(
                SimpleCNN,
                train_dataset,
                test_dataset,
                save_dir=save_dir,
                **config
            )
            all_results[name] = results
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def plot_comparison(results_dict, save_dir='results'):
    """
    Create comprehensive comparison plots
    
    Args:
        results_dict: Dictionary mapping experiment names to results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # Plot 1: Test Accuracy
    for (name, results), color in zip(results_dict.items(), colors):
        ax1.plot(results['rounds'], results['test_accuracy'], 
                label=name, linewidth=2.5, color=color, alpha=0.8)
    
    ax1.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Test Accuracy vs Communication Rounds', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=1)
    
    # Plot 2: Test Loss
    for (name, results), color in zip(results_dict.items(), colors):
        ax2.plot(results['rounds'], results['test_loss'], 
                label=name, linewidth=2.5, color=color, alpha=0.8)
    
    ax2.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Test Loss', fontsize=13, fontweight='bold')
    ax2.set_title('Test Loss vs Communication Rounds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=1)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    # Create summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Algorithm':<40} {'Final Acc':<12} {'Best Acc':<12} {'Final Loss':<12}")
    print("-" * 70)
    
    for name, results in results_dict.items():
        final_acc = results['test_accuracy'][-1]
        best_acc = max(results['test_accuracy'])
        final_loss = results['test_loss'][-1]
        print(f"{name:<40} {final_acc:>10.2f}% {best_acc:>10.2f}% {final_loss:>11.4f}")
    
    print("=" * 70 + "\n")


def plot_alpha_comparison(results_dict, save_dir='results'):
    """
    Plot comparison across different alpha values (non-IID levels)
    
    Args:
        results_dict: Dictionary with alpha values as keys
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by alpha value
    sorted_results = sorted(results_dict.items(), key=lambda x: float(x[0].split('=')[1]))
    
    for name, results in sorted_results:
        ax.plot(results['rounds'], results['test_accuracy'], 
               label=name, linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Non-IID Level (α) on Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'alpha_comparison_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Alpha comparison plot saved to: {save_path}")
    
    plt.show()


# Main experiment script
if __name__ == "__main__":
    # Set random seeds
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load FEMNIST data
    print("Loading FEMNIST dataset...")
    train_dataset, test_dataset = load_femnist_data(
        test_split_ratio=0.15,
        subset_size=50000,  # Use 100k samples for faster experiments
        seed=SEED,
        verbose=True
    )
    
    # =========================================================================
    # EXPERIMENT 1: Baseline Comparison
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: BASELINE ALGORITHM COMPARISON")
    print("="*70)
    
    baseline_configs = [
        {
            'name': 'FedAvg (Baseline)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedavg',
            'client_lr': 0.01,
            'server_lr': 1.0,  # Not used in FedAvg
            'num_clients': 10,
            'num_rounds': 100,
            'local_epochs': 5,
            'alpha': 0.5,
            'batch_size': 32,
            'seed': SEED
        },
        {
            'name': 'FedAdam',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedadam',
            'client_lr': 0.01,
            'server_lr': 0.01,
            'num_clients': 10,
            'num_rounds': 100,
            'local_epochs': 5,
            'alpha': 0.5,
            'batch_size': 32,
            'seed': SEED
        },
        {
            'name': 'FedYogi',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedyogi',
            'client_lr': 0.01,
            'server_lr': 0.01,
            'num_clients': 10,
            'num_rounds': 100,
            'local_epochs': 5,
            'alpha': 0.5,
            'batch_size': 32,
            'seed': SEED
        },
        {
            'name': 'FedAdagrad',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedadagrad',
            'client_lr': 0.01,
            'server_lr': 0.1,
            'num_clients': 10,
            'num_rounds': 100,
            'local_epochs': 5,
            'alpha': 0.5,
            'batch_size': 32,
            'seed': SEED
        },
        {
            'name': 'Client-Adam + FedAvg',
            'client_optimizer': 'adam',
            'server_aggregator': 'fedavg',
            'client_lr': 0.001,
            'server_lr': 1.0,
            'num_clients': 10,
            'num_rounds': 100,
            'local_epochs': 5,
            'alpha': 0.5,
            'batch_size': 32,
            'seed': SEED
        }
    ]
    
    baseline_results = compare_algorithms(train_dataset, test_dataset, baseline_configs)
    plot_comparison(baseline_results, save_dir='results')
    
    # =========================================================================
    # EXPERIMENT 2: Non-IID Level Analysis (Different Alpha Values)
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: NON-IID LEVEL ANALYSIS")
    print("="*70)
    
    alpha_configs = [
        {
            'name': 'α=0.1 (Highly Non-IID)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedavg',
            'client_lr': 0.01,
            'server_lr': 1.0,
            'num_clients': 10,
            'num_rounds': 100,
            'local_epochs': 5,
            'alpha': 0.1,
            'batch_size': 32,
            'seed': SEED
        },
        {
            'name': 'α=0.5 (Moderate Non-IID)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedavg',
            'client_lr': 0.01,
            'server_lr': 1.0,
            'num_clients': 10,
            'num_rounds': 100,
            'local_epochs': 5,
            'alpha': 0.5,
            'batch_size': 32,
            'seed': SEED
        },
        {
            'name': 'α=1.0 (Mild Non-IID)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedavg',
            'client_lr': 0.01,
            'server_lr': 1.0,
            'num_clients': 10,
            'num_rounds': 100,
            'local_epochs': 5,
            'alpha': 1.0,
            'batch_size': 32,
            'seed': SEED
        },
        {
            'name': 'α=10.0 (Nearly IID)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedavg',
            'client_lr': 0.01,
            'server_lr': 1.0,
            'num_clients': 10,
            'num_rounds': 100,
            'local_epochs': 5,
            'alpha': 10.0,
            'batch_size': 32,
            'seed': SEED
        }
    ]
    
    alpha_results = compare_algorithms(train_dataset, test_dataset, alpha_configs)
    plot_alpha_comparison(alpha_results, save_dir='results')
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print("Results and plots saved in 'results/' directory")