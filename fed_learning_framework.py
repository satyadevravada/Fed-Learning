import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import copy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedOptimizer:
    """Base class for server-side federated optimizers"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.state = {}
    
    def aggregate(self, global_params: OrderedDict, 
                  client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        """
        Aggregate client updates
        
        Args:
            global_params: Current global model parameters
            client_updates: List of (client_params, num_samples) tuples
        
        Returns:
            Updated global parameters
        """
        raise NotImplementedError


class FedAvg(FederatedOptimizer):
    """Standard FedAvg: Weighted averaging (baseline)"""
    
    def aggregate(self, global_params: OrderedDict, 
                  client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        total_samples = sum(num_samples for _, num_samples in client_updates)
        new_params = OrderedDict()
        
        for key in global_params.keys():
            new_params[key] = torch.zeros_like(global_params[key])
            for client_params, num_samples in client_updates:
                weight = num_samples / total_samples
                new_params[key] += client_params[key] * weight
        
        return new_params


class FedAdam(FederatedOptimizer):
    """Server-side Adam optimizer for federated learning"""
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
    
    def aggregate(self, global_params: OrderedDict, 
                  client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        self.t += 1
        
        # Weighted average of client updates (pseudo-gradient)
        total_samples = sum(num_samples for _, num_samples in client_updates)
        pseudo_grad = OrderedDict()
        
        for key in global_params.keys():
            pseudo_grad[key] = torch.zeros_like(global_params[key])
            for client_params, num_samples in client_updates:
                weight = num_samples / total_samples
                # Calculate delta (pseudo-gradient)
                delta = global_params[key] - client_params[key]
                pseudo_grad[key] += delta * weight
            
            # Initialize momentum if needed
            if key not in self.state:
                self.state[key] = {
                    'm': torch.zeros_like(global_params[key]),
                    'v': torch.zeros_like(global_params[key])
                }
            
            # Adam update
            m = self.state[key]['m']
            v = self.state[key]['v']
            
            m = self.beta1 * m + (1 - self.beta1) * pseudo_grad[key]
            v = self.beta2 * v + (1 - self.beta2) * (pseudo_grad[key] ** 2)
            
            self.state[key]['m'] = m
            self.state[key]['v'] = v
        
        # Apply update with bias correction
        new_params = OrderedDict()
        for key in global_params.keys():
            m = self.state[key]['m']
            v = self.state[key]['v']
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Update parameters
            new_params[key] = global_params[key] - self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        
        return new_params


class FedAdagrad(FederatedOptimizer):
    """Server-side Adagrad optimizer"""
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
    
    def aggregate(self, global_params: OrderedDict, 
                  client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        total_samples = sum(num_samples for _, num_samples in client_updates)
        pseudo_grad = OrderedDict()
        
        for key in global_params.keys():
            pseudo_grad[key] = torch.zeros_like(global_params[key])
            for client_params, num_samples in client_updates:
                weight = num_samples / total_samples
                delta = global_params[key] - client_params[key]
                pseudo_grad[key] += delta * weight
            
            # Initialize accumulated gradient if needed
            if key not in self.state:
                self.state[key] = torch.zeros_like(global_params[key])
            
            # Accumulate squared gradients
            self.state[key] += pseudo_grad[key] ** 2
        
        # Apply update
        new_params = OrderedDict()
        for key in global_params.keys():
            adapted_lr = self.lr / (torch.sqrt(self.state[key]) + self.epsilon)
            new_params[key] = global_params[key] - adapted_lr * pseudo_grad[key]
        
        return new_params


class FedNAdam(FederatedOptimizer):
    """Server-side NAdam (Nesterov-accelerated Adam) optimizer"""
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
    
    def aggregate(self, global_params: OrderedDict, 
                  client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        self.t += 1
        
        # Compute weighted average of client updates (pseudo-gradient)
        total_samples = sum(num_samples for _, num_samples in client_updates)
        pseudo_grad = OrderedDict()
        
        for key in global_params.keys():
            pseudo_grad[key] = torch.zeros_like(global_params[key])
            for client_params, num_samples in client_updates:
                weight = num_samples / total_samples
                delta = global_params[key] - client_params[key]
                pseudo_grad[key] += delta * weight
            
            # Initialize momentum if needed
            if key not in self.state:
                self.state[key] = {
                    'm': torch.zeros_like(global_params[key]),
                    'v': torch.zeros_like(global_params[key])
                }
            
            # NAdam update
            m = self.state[key]['m']
            v = self.state[key]['v']
            
            # Update biased first and second moment estimates
            m = self.beta1 * m + (1 - self.beta1) * pseudo_grad[key]
            v = self.beta2 * v + (1 - self.beta2) * (pseudo_grad[key] ** 2)
            
            self.state[key]['m'] = m
            self.state[key]['v'] = v
        
        # Apply Nesterov momentum and update
        new_params = OrderedDict()
        for key in global_params.keys():
            m = self.state[key]['m']
            v = self.state[key]['v']
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # Nesterov step: use current momentum + next momentum term
            m_bar = (self.beta1 * m_hat + 
                    (1 - self.beta1) * pseudo_grad[key] / (1 - self.beta1 ** self.t))
            
            # Update parameters
            new_params[key] = (global_params[key] - 
                             self.lr * m_bar / (torch.sqrt(v_hat) + self.epsilon))
        
        return new_params


class FedAdamW(FederatedOptimizer):
    """Server-side AdamW optimizer with decoupled weight decay"""
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, 
                 weight_decay: float = 0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
    
    def aggregate(self, global_params: OrderedDict, 
                  client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        self.t += 1
        
        # Compute weighted average of client updates (pseudo-gradient)
        total_samples = sum(num_samples for _, num_samples in client_updates)
        pseudo_grad = OrderedDict()
        
        for key in global_params.keys():
            pseudo_grad[key] = torch.zeros_like(global_params[key])
            for client_params, num_samples in client_updates:
                weight = num_samples / total_samples
                delta = global_params[key] - client_params[key]
                pseudo_grad[key] += delta * weight
            
            # Initialize momentum if needed
            if key not in self.state:
                self.state[key] = {
                    'm': torch.zeros_like(global_params[key]),
                    'v': torch.zeros_like(global_params[key])
                }
            
            # Adam update (same as FedAdam)
            m = self.state[key]['m']
            v = self.state[key]['v']
            
            m = self.beta1 * m + (1 - self.beta1) * pseudo_grad[key]
            v = self.beta2 * v + (1 - self.beta2) * (pseudo_grad[key] ** 2)
            
            self.state[key]['m'] = m
            self.state[key]['v'] = v
        
        # Apply update with decoupled weight decay
        new_params = OrderedDict()
        for key in global_params.keys():
            m = self.state[key]['m']
            v = self.state[key]['v']
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # AdamW: apply weight decay directly to parameters
            new_params[key] = (global_params[key] * (1 - self.lr * self.weight_decay) -
                             self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon))
        
        return new_params


class FedYogi(FederatedOptimizer):
    """Server-side Yogi optimizer (adaptive learning rate with better convergence)"""
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-3):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
    
    def aggregate(self, global_params: OrderedDict, 
                  client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        self.t += 1
        
        # Compute weighted average of client updates (pseudo-gradient)
        total_samples = sum(num_samples for _, num_samples in client_updates)
        pseudo_grad = OrderedDict()
        
        for key in global_params.keys():
            pseudo_grad[key] = torch.zeros_like(global_params[key])
            for client_params, num_samples in client_updates:
                weight = num_samples / total_samples
                delta = global_params[key] - client_params[key]
                pseudo_grad[key] += delta * weight
            
            # Initialize momentum if needed
            if key not in self.state:
                self.state[key] = {
                    'm': torch.zeros_like(global_params[key]),
                    'v': torch.zeros_like(global_params[key])
                }
            
            # Yogi update
            m = self.state[key]['m']
            v = self.state[key]['v']
            
            # First moment (same as Adam)
            m = self.beta1 * m + (1 - self.beta1) * pseudo_grad[key]
            
            # Second moment (Yogi-specific: sign-based adaptive update)
            # CORRECTED: proper sign and order
            grad_squared = pseudo_grad[key] ** 2
            v = v + (1 - self.beta2) * torch.sign(grad_squared - v) * grad_squared
            
            self.state[key]['m'] = m
            self.state[key]['v'] = v
        
        # Apply update with bias correction (added for better performance)
        new_params = OrderedDict()
        for key in global_params.keys():
            m = self.state[key]['m']
            v = self.state[key]['v']
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            new_params[key] = (global_params[key] - 
                             self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon))
        
        return new_params


class FederatedClient:
    """Client for federated learning"""
    
    def __init__(self, client_id: int, train_loader: DataLoader, 
                 device: torch.device, optimizer_name: str = 'sgd',
                 lr: float = 0.01):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.optimizer_name = optimizer_name
        self.lr = lr
    
    def train(self, model: nn.Module, epochs: int = 1) -> Tuple[OrderedDict, int]:
        """
        Train model locally
        
        Returns:
            Tuple of (updated_params, num_samples)
        """
        try:
            model = model.to(self.device)
            model.train()
            
            # Create optimizer based on specified variant
            if self.optimizer_name.lower() == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=self.lr)
            elif self.optimizer_name.lower() == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=self.lr)
            elif self.optimizer_name.lower() == 'adagrad':
                optimizer = optim.Adagrad(model.parameters(), lr=self.lr)
            elif self.optimizer_name.lower() == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(), lr=self.lr)
            elif self.optimizer_name.lower() == 'momentum':
                optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
            
            criterion = nn.CrossEntropyLoss()
            
            # FIXED: Count samples correctly (don't multiply by epochs)
            num_samples = len(self.train_loader.dataset)
            
            for epoch in range(epochs):
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            return model.state_dict(), num_samples
            
        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            raise


class FederatedServer:
    """Server for federated learning"""
    
    def __init__(self, model: nn.Module, test_loader: DataLoader,
                 device: torch.device, fed_optimizer: FederatedOptimizer):
        self.global_model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.fed_optimizer = fed_optimizer
    
    def aggregate_and_update(self, client_updates: List[Tuple[OrderedDict, int]]) -> None:
        """Aggregate client updates and update global model"""
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return
            
        try:
            global_params = self.global_model.state_dict()
            new_params = self.fed_optimizer.aggregate(global_params, client_updates)
            self.global_model.load_state_dict(new_params)
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate global model on test set"""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy
    
    def get_global_model(self) -> nn.Module:
        """Return a copy of the global model"""
        return copy.deepcopy(self.global_model)


def create_non_iid_split(dataset, num_clients: int, alpha: float = 0.5) -> List[List[int]]:
    """
    Split dataset using Dirichlet distribution for non-IID data
    
    Args:
        dataset: PyTorch dataset with targets
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
    
    Returns:
        List of client indices
    """
    # IMPROVED: Better label extraction with error handling
    labels = []
    
    try:
        # Try direct attribute access first
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        elif hasattr(dataset, 'character'):
            labels = np.array(dataset.character)
        else:
            # Fallback: iterate through dataset
            logger.info("Extracting labels by iteration...")
            for i in range(len(dataset)):
                item = dataset[i]
                
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    # Standard (data, label) format
                    label = item[1]
                elif isinstance(item, dict):
                    # Dictionary format - try common keys
                    label = None
                    for key in ['label', 'y', 'target', 'digit', 'character', 'class']:
                        if key in item:
                            label = item[key]
                            break
                    if label is None:
                        raise ValueError(f"Could not find label in dict with keys: {item.keys()}")
                else:
                    raise ValueError(f"Unsupported dataset item format: {type(item)}")
                
                labels.append(label)
            
            labels = np.array(labels)
        
        if len(labels) == 0:
            raise ValueError("No labels extracted from dataset")
            
    except Exception as e:
        logger.error(f"Failed to extract labels: {e}")
        raise
    
    num_classes = len(np.unique(labels))
    logger.info(f"Dataset: {len(labels)} samples, {num_classes} classes")
    
    # Get indices for each class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx in class_indices:
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        
        # Split indices according to proportions
        class_splits = np.split(class_idx, proportions)
        
        for client_id, indices in enumerate(class_splits):
            client_indices[client_id].extend(indices.tolist())
    
    # Shuffle each client's data
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    # Log statistics
    for i in range(min(5, num_clients)):  # Show first 5 clients
        logger.info(f"Client {i}: {len(client_indices[i])} samples")
    
    return client_indices


def federated_learning_round_sequential(
    server: FederatedServer, 
    clients: List[FederatedClient], 
    local_epochs: int = 1,
    show_progress: bool = True
) -> Tuple[float, float]:
    """
    Sequential client training with clear progress tracking
    Most reliable and debuggable approach
    
    For GPU training, this is often FASTER than threading due to:
    - No device contention
    - No serialization overhead
    - Better GPU utilization
    
    Args:
        server: FederatedServer instance
        clients: List of FederatedClient instances
        local_epochs: Number of local training epochs
        show_progress: Show progress bar
    
    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        if show_progress:
            logger.warning("tqdm not available, progress bar disabled")
    
    global_model = server.get_global_model()
    client_updates = []
    
    iterator = tqdm(clients, desc="Training clients") if (show_progress and has_tqdm) else clients
    
    for client in iterator:
        try:
            # Create fresh model copy with proper device handling
            client_model = copy.deepcopy(global_model)
            client_model = client_model.to(client.device)
            
            # Train locally
            updated_params, num_samples = client.train(client_model, epochs=local_epochs)
            client_updates.append((updated_params, num_samples))
            
            # Free memory
            del client_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            logger.error(f"Client {client.client_id} failed: {e}")
            # Continue with other clients
            continue
    
    if not client_updates:
        raise RuntimeError("All clients failed to train")
    
    # Aggregate on server
    server.aggregate_and_update(client_updates)
    
    # Evaluate
    test_loss, test_accuracy = server.evaluate()
    return test_loss, test_accuracy


# Example usage template
if __name__ == "__main__":
    print("Federated Learning Framework")
    print("=============================")
    print("\nSupported configurations:")
    print("1. Client optimizers: SGD, Adam, AdaGrad, RMSprop, Momentum")
    print("2. Server aggregators: FedAvg, FedAdam, FedAdagrad, FedNAdam, FedAdamW, FedYogi")
    print("\nKey fixes applied:")
    print("✓ Corrected FedYogi second moment update")
    print("✓ Added bias correction to FedYogi")
    print("✓ Fixed sample counting in client training")
    print("✓ Improved label extraction with error handling")
    print("✓ Added comprehensive error handling")
    print("✓ Added logging throughout")
    print("✓ Better memory management")
    print("\nTo use this framework:")
    print("- Define your model architecture")
    print("- Load FEMNIST or custom dataset")
    print("- Create non-IID splits using create_non_iid_split()")
    print("- Initialize clients and server")
    print("- Run federated_learning_round_sequential() for multiple rounds")