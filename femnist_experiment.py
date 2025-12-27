import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# framework imports 
from fed_learning_framework import (
    FederatedClient, FederatedServer, FedAvg, FedAdam,
    FedYogi, FedAMSGrad, FedAdamW, ClientAdamServerAvg,
    create_non_iid_split, federated_learning_round_sequential
)


class SimpleCNN(nn.Module):
    """Simple CNN for FEMNIST."""
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))           # conv + relu
        x = F.relu(self.conv2(x))           # conv + relu
        x = F.max_pool2d(x, 2)              # pooling
        x = self.dropout1(x)                # dropout
        x = torch.flatten(x, 1)             # flatten for FC
        x = F.relu(self.fc1(x))             # fc + relu
        x = self.dropout2(x)                # dropout
        x = self.fc2(x)                     # final logits
        return F.log_softmax(x, dim=1)      # log-probabilities


class FEMNISTDataset(torch.utils.data.Dataset):
    """Wrapper for HF FEMNIST dataset (returns image, label)."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        # pre-extract labels for speed
        self.labels = []
        for item in hf_dataset:
            label = item.get('label', item.get('character', item.get('digit')))
            if label is None:
                raise KeyError(f"No label in item keys: {list(item.keys())}")
            self.labels.append(label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        # ensure grayscale
        if image.mode != 'L':
            image = image.convert('L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0
        return image, label


def load_femnist_data(test_split_ratio=0.15, subset_size=25000, seed=42, verbose=True):
    """Load FEMNIST from HF, take a shuffled subset and split train/test."""
    np.random.seed(seed); torch.manual_seed(seed)
    if verbose:
        print("Loading FEMNIST from HuggingFace...")
    dataset = load_dataset("flwrlabs/femnist", split="train", trust_remote_code=True)
    total = len(dataset)
    if verbose:
        print(f"Original size: {total:,}")
    # sample subset
    indices = np.random.permutation(total)[:subset_size]
    test_size = int(len(indices) * test_split_ratio)
    train_idx = indices[:-test_size]
    test_idx = indices[-test_size:]
    train_dataset = FEMNISTDataset(dataset.select(train_idx.tolist()))
    test_dataset = FEMNISTDataset(dataset.select(test_idx.tolist()))
    if verbose:
        print(f"Using subset {subset_size:,}: train={len(train_dataset):,}, test={len(test_dataset):,}")
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
    save_dir='./results'
):
    """Run one federated experiment and save results."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Experiment: agg={server_aggregator}, clients={num_clients}, rounds={num_rounds}, local_epochs={local_epochs}")
    # create non-iid splits
    client_indices = create_non_iid_split(train_dataset, num_clients, alpha=alpha, seed=seed)
    # build client loaders
    client_loaders = []
    for idx in client_indices:
        subset = Subset(train_dataset, idx)
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # small subset for train loss eval
    train_eval_size = min(2000, len(train_dataset) // 5)
    train_eval_indices = np.random.choice(len(train_dataset), train_eval_size, replace=False)
    train_eval_loader = DataLoader(Subset(train_dataset, train_eval_indices), batch_size=batch_size, shuffle=False)

    # instantiate clients
    clients = []
    for i, loader in enumerate(client_loaders):
        clients.append(FederatedClient(client_id=i, train_loader=loader, device=device, optimizer_name=client_optimizer, lr=client_lr))

    # choose server aggregator
    agg = server_aggregator.lower()
    if agg == 'fedavg':
        fed_optimizer = FedAvg(learning_rate=server_lr)
    elif agg == 'fedadam':
        fed_optimizer = FedAdam(learning_rate=server_lr)
    elif agg == 'fedyogi':
        fed_optimizer = FedYogi(learning_rate=server_lr)
    elif agg == 'fedamsgrad':
        fed_optimizer = FedAMSGrad(learning_rate=server_lr)
    elif agg == 'fedadamw':
        fed_optimizer = FedAdamW(learning_rate=server_lr, weight_decay=0.01)
    elif agg == 'clientadam':
        fed_optimizer = ClientAdamServerAvg(learning_rate=server_lr)
    else:
        raise ValueError(f"Unknown server aggregator: {server_aggregator}")

    # server and model
    model = model_class()
    server = FederatedServer(model=model, test_loader=test_loader, device=device, fed_optimizer=fed_optimizer)

    # results bookkeeping
    results = {'config': dict(num_clients=num_clients, num_rounds=num_rounds, local_epochs=local_epochs,
                              client_optimizer=client_optimizer, server_aggregator=server_aggregator,
                              client_lr=client_lr, server_lr=server_lr, alpha=alpha, batch_size=batch_size, seed=seed),
               'rounds': [], 'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    best_acc = 0.0

    # training loop
    for rnd in range(num_rounds):
        test_loss, test_acc, train_loss = federated_learning_round_sequential(
            server, clients, local_epochs=local_epochs, train_eval_loader=train_eval_loader, show_progress=False
        )
        results['rounds'].append(rnd + 1)
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['test_accuracy'].append(test_acc)
        best_acc = max(best_acc, test_acc)
        if (rnd + 1) % 10 == 0 or rnd == 0:
            print(f"Round {rnd+1}/{num_rounds} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    # save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        exp_name = f"{client_optimizer}_{server_aggregator}_alpha{alpha}"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"{exp_name}_{ts}.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {path}")

    return results


def compare_algorithms(train_dataset, test_dataset, configs, save_dir='./results'):
    """Run multiple experiment configs and collect their results."""
    all_results = {}
    for i, cfg in enumerate(configs, 1):
        cfg = cfg.copy()
        name = cfg.pop('name')
        print(f"[{i}/{len(configs)}] Running {name}")
        try:
            res = run_federated_experiment(SimpleCNN, train_dataset, test_dataset, save_dir=save_dir, **cfg)
            all_results[name] = res
        except Exception as e:
            print(f"ERROR {name}: {e}")
            import traceback; traceback.print_exc()
    return all_results


def plot_comparison(results_dict, save_dir='./results'):
    """Plot accuracy, train loss, and test loss for each algorithm."""
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    # Test Accuracy
    for (name, res), color in zip(results_dict.items(), colors):
        ax1.plot(res['rounds'], res['test_accuracy'], label=name, linewidth=2.0)
    ax1.set_xlabel('Round'); ax1.set_ylabel('Test Accuracy (%)'); ax1.set_title('Test Accuracy')
    ax1.legend(fontsize=9); ax1.grid(True); ax1.set_xlim(left=1)

    # Training Loss
    for (name, res), color in zip(results_dict.items(), colors):
        ax2.plot(res['rounds'], res['train_loss'], label=name, linewidth=2.0)
    ax2.set_xlabel('Round'); ax2.set_ylabel('Train Loss'); ax2.set_title('Train Loss')
    ax2.legend(fontsize=9); ax2.grid(True); ax2.set_xlim(left=1)

    # Test Loss
    for (name, res), color in zip(results_dict.items(), colors):
        ax3.plot(res['rounds'], res['test_loss'], label=name, linewidth=2.0)
    ax3.set_xlabel('Round'); ax3.set_ylabel('Test Loss'); ax3.set_title('Test Loss')
    ax3.legend(fontsize=9); ax3.grid(True); ax3.set_xlim(left=1)

    plt.tight_layout()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'comparison_{ts}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.show()

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Algorithm':<30} {'Final Acc':<12} {'Best Acc':<12} {'Final Train':<12} {'Final Test':<12} {'Gap':<8}")
    print("-" * 100)
    for name, res in results_dict.items():
        final_acc = res['test_accuracy'][-1]
        best_acc = max(res['test_accuracy'])
        final_train = res['train_loss'][-1]
        final_test = res['test_loss'][-1]
        gap = final_test - final_train
        print(f"{name:<30} {final_acc:>10.2f}% {best_acc:>10.2f}% {final_train:>12.4f} {final_test:>12.4f} {gap:>8.4f}")
    print("=" * 100)


if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED); np.random.seed(SEED)
    os.makedirs('results', exist_ok=True)

    # load data (25k subset)
    train_dataset, test_dataset = load_femnist_data(test_split_ratio=0.15, subset_size=25000, seed=SEED, verbose=True)

    # experiment configurations
    NUM_ROUNDS = 100
    NUM_CLIENTS = 10
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32
    ALPHA = 0.5

    configs = [
        {'name': '1_FedAvg_Baseline', 'client_optimizer': 'sgd', 'server_aggregator': 'fedavg',
         'client_lr': 0.01, 'server_lr': 1.0, 'num_clients': NUM_CLIENTS, 'num_rounds': NUM_ROUNDS,
         'local_epochs': LOCAL_EPOCHS, 'alpha': ALPHA, 'batch_size': BATCH_SIZE, 'seed': SEED},
        {'name': '2_FedAdam', 'client_optimizer': 'sgd', 'server_aggregator': 'fedadam',
         'client_lr': 0.01, 'server_lr': 0.01, 'num_clients': NUM_CLIENTS, 'num_rounds': NUM_ROUNDS,
         'local_epochs': LOCAL_EPOCHS, 'alpha': ALPHA, 'batch_size': BATCH_SIZE, 'seed': SEED},
        {'name': '3_FedYogi', 'client_optimizer': 'sgd', 'server_aggregator': 'fedyogi',
         'client_lr': 0.01, 'server_lr': 0.01, 'num_clients': NUM_CLIENTS, 'num_rounds': NUM_ROUNDS,
         'local_epochs': LOCAL_EPOCHS, 'alpha': ALPHA, 'batch_size': BATCH_SIZE, 'seed': SEED},
        {'name': '5_FedAMSGrad', 'client_optimizer': 'sgd', 'server_aggregator': 'fedamsgrad',
         'client_lr': 0.01, 'server_lr': 0.01, 'num_clients': NUM_CLIENTS, 'num_rounds': NUM_ROUNDS,
         'local_epochs': LOCAL_EPOCHS, 'alpha': ALPHA, 'batch_size': BATCH_SIZE, 'seed': SEED},
        {'name': '6_FedAdamW', 'client_optimizer': 'sgd', 'server_aggregator': 'fedadamw',
         'client_lr': 0.01, 'server_lr': 0.01, 'num_clients': NUM_CLIENTS, 'num_rounds': NUM_ROUNDS,
         'local_epochs': LOCAL_EPOCHS, 'alpha': ALPHA, 'batch_size': BATCH_SIZE, 'seed': SEED},
        {'name': '7_ClientAdam_ServerAvg', 'client_optimizer': 'adam', 'server_aggregator': 'clientadam',
         'client_lr': 0.001, 'server_lr': 1.0, 'num_clients': NUM_CLIENTS, 'num_rounds': NUM_ROUNDS,
         'local_epochs': LOCAL_EPOCHS, 'alpha': ALPHA, 'batch_size': BATCH_SIZE, 'seed': SEED}
    ]

    # run experiments and plot
    results = compare_algorithms(train_dataset, test_dataset, configs)
    plot_comparison(results, save_dir='./results')
    print("All experiments finished; results in ./results")
