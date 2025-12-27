import torch
from collections import OrderedDict
from typing import List, Tuple
from fed_learning_framework import FederatedOptimizer
# Add these to your federated_framework.py file


class FedRMSprop(FederatedOptimizer):
    """Server-side RMSprop optimizer for federated learning"""
    
    def __init__(self, learning_rate: float = 0.01, alpha: float = 0.99, 
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.alpha = alpha
        self.epsilon = epsilon
    
    def aggregate(self, global_params: OrderedDict, 
                  client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        # Compute weighted average of client updates (pseudo-gradient)
        total_samples = sum(num_samples for _, num_samples in client_updates)
        pseudo_grad = OrderedDict()
        
        for key in global_params.keys():
            pseudo_grad[key] = torch.zeros_like(global_params[key])
            for client_params, num_samples in client_updates:
                weight = num_samples / total_samples
                delta = global_params[key] - client_params[key]
                pseudo_grad[key] += delta * weight
            
            # Initialize velocity if needed
            if key not in self.state:
                self.state[key] = torch.zeros_like(global_params[key])
            
            # Momentum update: accumulate velocity
            self.state[key] = self.momentum * self.state[key] + pseudo_grad[key]
        
        # Apply update with momentum
        new_params = OrderedDict()
        for key in global_params.keys():
            new_params[key] = global_params[key] - self.lr * self.state[key]
        
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
            v = v - (1 - self.beta2) * torch.sign(v - pseudo_grad[key] ** 2) * (pseudo_grad[key] ** 2)
            
            self.state[key]['m'] = m
            self.state[key]['v'] = v
        
        # Apply update
        new_params = OrderedDict()
        for key in global_params.keys():
            m = self.state[key]['m']
            v = self.state[key]['v']
            
            # Simple update without bias correction (as in original Yogi paper)
            new_params[key] = (global_params[key] - 
                             self.lr * m / (torch.sqrt(v) + self.epsilon))
        
        return new_params


# Extended example configurations for comprehensive comparison
def get_all_configurations():
    """
    Returns a comprehensive list of federated learning configurations
    for comparison against baseline
    """
    
    base_config = {
        'num_clients': 10,
        'local_epochs': 5,
        'alpha': 0.5,
        'batch_size': 32
    }
    
    configurations = [
        # Baseline
        {
            **base_config,
            'name': 'FedAvg (Baseline - SGD + Averaging)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedavg',
            'learning_rate': 0.01
        },
        
        # Client-side optimizer variants
        {
            **base_config,
            'name': 'FedAdam-Client (Adam on clients)',
            'client_optimizer': 'adam',
            'server_aggregator': 'fedavg',
            'learning_rate': 0.001
        },
        {
            **base_config,
            'name': 'FedAdagrad-Client (Adagrad on clients)',
            'client_optimizer': 'adagrad',
            'server_aggregator': 'fedavg',
            'learning_rate': 0.01
        },
        {
            **base_config,
            'name': 'FedMomentum-Client (Momentum on clients)',
            'client_optimizer': 'momentum',
            'server_aggregator': 'fedavg',
            'learning_rate': 0.01
        },
        {
            **base_config,
            'name': 'FedRMSprop-Client (RMSprop on clients)',
            'client_optimizer': 'rmsprop',
            'server_aggregator': 'fedavg',
            'learning_rate': 0.001
        },
        
        # Server-side optimizer variants
        {
            **base_config,
            'name': 'FedAdam-Server (Adam on server)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedadam',
            'learning_rate': 0.01
        },
        {
            **base_config,
            'name': 'FedAdagrad-Server (Adagrad on server)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedadagrad',
            'learning_rate': 0.01
        },
        {
            **base_config,
            'name': 'FedMomentum-Server (Momentum on server)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedmomentum',
            'learning_rate': 0.01
        },
        {
            **base_config,
            'name': 'FedRMSprop-Server (RMSprop on server)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fedrmsprop',
            'learning_rate': 0.01
        },
        {
            **base_config,
            'name': 'FedNAdam-Server (NAdam on server)',
            'client_optimizer': 'sgd',
            'server_aggregator': 'fednadam',
            'learning_rate': 0.01
        },
    ]
    
    return configurations


# Performance comparison utilities
def create_comparison_table(results_dict):
    """Create a comparison table of final results"""
    import pandas as pd
    
    data = []
    for name, results in results_dict.items():
        data.append({
            'Algorithm': name,
            'Final Accuracy': f"{results['test_accuracy'][-1]:.2f}%",
            'Final Loss': f"{results['test_loss'][-1]:.4f}",
            'Best Accuracy': f"{max(results['test_accuracy']):.2f}%",
            'Best Round': results['test_accuracy'].index(max(results['test_accuracy']))
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('Final Accuracy', ascending=False)
    
    return df


def save_results_to_csv(results_dict, filename='federated_results.csv'):
    """Save detailed results to CSV"""
    import pandas as pd
    
    all_data = []
    for name, results in results_dict.items():
        for i in range(len(results['rounds'])):
            all_data.append({
                'Algorithm': name,
                'Round': results['rounds'][i],
                'Test Loss': results['test_loss'][i],
                'Test Accuracy': results['test_accuracy'][i]
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# Example usage
if __name__ == "__main__":
    print("Additional Federated Learning Optimizers")
    print("=" * 50)
    print("\nAvailable server-side optimizers:")
    print("1. FedRMSprop - RMSprop with exponential moving average")
    print("2. FedMomentum - Classical momentum SGD")
    print("3. FedNAdam - Nesterov-accelerated Adam")
    print("4. FedAdamW - Adam with decoupled weight decay")
    print("5. FedYogi - Adaptive learning rate with sign-based updates")
    print("\nThese can be combined with any client-side optimizer!")
    print("\nFor complete experiments, use get_all_configurations()")
    print("to get a full comparison against the FedAvg baseline.")



class FedMomentum(FederatedOptimizer):
    """Server-side Momentum SGD for federated learning."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum

    def aggregate(self, global_params: OrderedDict, 
                  client_updates: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        """
        Aggregate client updates using momentum-based server optimization.
        Args:
            global_params (OrderedDict): Current global model parameters
            client_updates (List[Tuple[OrderedDict, int]]): 
                List of (client_params, num_samples) tuples
        Returns:
            OrderedDict: Updated global parameters
        """
        total_samples = sum(num_samples for _, num_samples in client_updates)
        pseudo_grad = OrderedDict()

        # Compute weighted pseudo-gradient (difference between global and client models)
        for key in global_params.keys():
            pseudo_grad[key] = torch.zeros_like(global_params[key])
            for client_params, num_samples in client_updates:
                weight = num_samples / total_samples
                delta = global_params[key] - client_params[key]
                pseudo_grad[key] += delta * weight

            # Initialize velocity (momentum buffer) if needed
            if key not in self.state:
                self.state[key] = torch.zeros_like(global_params[key])

            # Momentum update: v_t = μ*v_{t-1} + g_t
            self.state[key] = self.momentum * self.state[key] + pseudo_grad[key]

        # Apply parameter update: w = w - η * v_t
        new_params = OrderedDict()
        for key in global_params.keys():
            new_params[key] = global_params[key] - self.lr * self.state[key]

        return new_params
