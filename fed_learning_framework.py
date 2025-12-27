import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedOptimizer:
    """Base server-side optimizer."""
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.state = {}

    def aggregate(self, global_params, client_updates):
        """Aggregate client updates into new server params (override)."""
        raise NotImplementedError


class FedAvg(FederatedOptimizer):
    """Simple weighted average of client params."""
    def aggregate(self, global_params, client_updates):
        total = sum(n for _, n in client_updates)
        new = OrderedDict()
        for k in global_params:
            new[k] = torch.zeros_like(global_params[k])
            for p, n in client_updates:
                # weight client param by its sample count
                new[k] += p[k] * (n / total)
        return new


class FedAdam(FederatedOptimizer):
    """Server-side Adam on pseudo-gradients."""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-3):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def aggregate(self, global_params, client_updates):
        self.t += 1
        total = sum(n for _, n in client_updates)
        pseudo = OrderedDict()

        for k in global_params:
            # compute weighted pseudo-gradient (client delta)
            pseudo[k] = torch.zeros_like(global_params[k])
            for p, n in client_updates:
                pseudo[k] += (p[k] - global_params[k]) * (n / total)

            # init moments if missing
            if k not in self.state:
                self.state[k] = {'m': torch.zeros_like(global_params[k]), 'v': torch.zeros_like(global_params[k])}

            # update biased first and second moments
            m = self.state[k]['m']
            v = self.state[k]['v']
            m = self.beta1 * m + (1 - self.beta1) * pseudo[k]
            v = self.beta2 * v + (1 - self.beta2) * (pseudo[k] ** 2)
            self.state[k]['m'] = m
            self.state[k]['v'] = v

        new = OrderedDict()
        for k in global_params:
            # bias-correct and apply adaptive update
            m = self.state[k]['m']
            v = self.state[k]['v']
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            new[k] = global_params[k] + self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        return new


class FedYogi(FederatedOptimizer):
    """Server-side Yogi adaptive aggregation."""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-3):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def aggregate(self, global_params, client_updates):
        self.t += 1
        total = sum(n for _, n in client_updates)
        pseudo = OrderedDict()

        for k in global_params:
            # weighted pseudo-gradient
            pseudo[k] = torch.zeros_like(global_params[k])
            for p, n in client_updates:
                pseudo[k] += (p[k] - global_params[k]) * (n / total)

            if k not in self.state:
                self.state[k] = {'m': torch.zeros_like(global_params[k]), 'v': torch.zeros_like(global_params[k])}

            # update first moment
            m = self.state[k]['m']
            v = self.state[k]['v']
            m = self.beta1 * m + (1 - self.beta1) * pseudo[k]

            # Yogi second-moment update (sign-based)
            gsq = pseudo[k] ** 2
            v = v - (1 - self.beta2) * torch.sign(v - gsq) * gsq

            self.state[k]['m'] = m
            self.state[k]['v'] = v

        new = OrderedDict()
        for k in global_params:
            # bias-correct and update
            m = self.state[k]['m']
            v = self.state[k]['v']
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            new[k] = global_params[k] + self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        return new


class FedAMSGrad(FederatedOptimizer):
    """Server-side AMSGrad aggregation."""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-3):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def aggregate(self, global_params, client_updates):
        self.t += 1
        total = sum(n for _, n in client_updates)
        pseudo = OrderedDict()

        for k in global_params:
            # compute pseudo-gradient
            pseudo[k] = torch.zeros_like(global_params[k])
            for p, n in client_updates:
                pseudo[k] += (p[k] - global_params[k]) * (n / total)

            if k not in self.state:
                self.state[k] = {'m': torch.zeros_like(global_params[k]),
                                 'v': torch.zeros_like(global_params[k]),
                                 'v_hat_max': torch.zeros_like(global_params[k])}

            # update moments
            m = self.state[k]['m']
            v = self.state[k]['v']
            m = self.beta1 * m + (1 - self.beta1) * pseudo[k]
            v = self.beta2 * v + (1 - self.beta2) * (pseudo[k] ** 2)
            self.state[k]['m'] = m
            self.state[k]['v'] = v

        new = OrderedDict()
        for k in global_params:
            # bias-correct and use max of v-hats
            m = self.state[k]['m']
            v = self.state[k]['v']
            v_hat_max = self.state[k]['v_hat_max']
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            v_hat_max = torch.maximum(v_hat_max, v_hat)
            self.state[k]['v_hat_max'] = v_hat_max
            new[k] = global_params[k] + self.lr * m_hat / (torch.sqrt(v_hat_max) + self.epsilon)
        return new


class FedAdamW(FederatedOptimizer):
    """Server-side AdamW (decoupled weight decay)."""
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-3, weight_decay=0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0

    def aggregate(self, global_params, client_updates):
        self.t += 1
        total = sum(n for _, n in client_updates)
        pseudo = OrderedDict()

        for k in global_params:
            # compute weighted client delta
            pseudo[k] = torch.zeros_like(global_params[k])
            for p, n in client_updates:
                pseudo[k] += (p[k] - global_params[k]) * (n / total)

            if k not in self.state:
                self.state[k] = {'m': torch.zeros_like(global_params[k]), 'v': torch.zeros_like(global_params[k])}

            # Adam-style moment updates
            m = self.state[k]['m']
            v = self.state[k]['v']
            m = self.beta1 * m + (1 - self.beta1) * pseudo[k]
            v = self.beta2 * v + (1 - self.beta2) * (pseudo[k] ** 2)
            self.state[k]['m'] = m
            self.state[k]['v'] = v

        new = OrderedDict()
        for k in global_params:
            # bias-correct and apply decoupled weight decay
            m = self.state[k]['m']
            v = self.state[k]['v']
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            new[k] = global_params[k] * (1 - self.lr * self.weight_decay) + self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        return new


class ClientAdamServerAvg(FederatedOptimizer):
    """Clients use Adam; server does simple averaging."""
    def aggregate(self, global_params, client_updates):
        total = sum(n for _, n in client_updates)
        new = OrderedDict()
        for k in global_params:
            new[k] = torch.zeros_like(global_params[k])
            for p, n in client_updates:
                new[k] += p[k] * (n / total)
        return new


class FederatedClient:
    """Client wrapper for local training."""
    def __init__(self, client_id, train_loader, device, optimizer_name='sgd', lr=0.01):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.optimizer_name = optimizer_name
        self.lr = lr

    def train(self, model, epochs=1):
        # move model to client device and set train mode
        model = model.to(self.device)
        model.train()

        # pick optimizer
        optn = self.optimizer_name.lower()
        if optn == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.lr)
        elif optn == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        elif optn == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=self.lr)
        elif optn == 'momentum':
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        criterion = nn.CrossEntropyLoss()
        num_samples = len(self.train_loader.dataset)

        # local training loop
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # return local state and sample count
        return model.state_dict(), num_samples


class FederatedServer:
    """Server that holds global model and aggregates updates."""
    def __init__(self, model, test_loader, device, fed_optimizer):
        self.global_model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.fed_optimizer = fed_optimizer

    def aggregate_and_update(self, client_updates):
        # skip if no updates
        if not client_updates:
            logger.warning("No client updates")
            return

        # get current params and aggregate via chosen optimizer
        global_params = self.global_model.state_dict()
        new_params = self.fed_optimizer.aggregate(global_params, client_updates)

        # ensure tensors are on server device and sanity-check shapes
        new_params = OrderedDict((k, v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in new_params.items())
        self._sanity_check_params(global_params, new_params)
        self.global_model.load_state_dict(new_params)

    def _sanity_check_params(self, global_params, new_params):
        # verify matching keys and shapes
        if set(global_params.keys()) != set(new_params.keys()):
            raise ValueError("Parameter key mismatch")
        for k in global_params:
            if new_params[k].shape != global_params[k].shape:
                raise ValueError(f"Shape mismatch for {k}")

    def evaluate(self):
        # evaluate model on test set, return loss and accuracy
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        test_loss /= len(self.test_loader)
        accuracy = 100.0 * correct / total
        return test_loss, accuracy

    def compute_train_loss(self, train_loader):
        # compute average loss on provided train loader
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        train_loss = 0.0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                train_loss += criterion(output, target).item()
        train_loss /= len(train_loader)
        return train_loss

    def get_global_model(self):
        # return a deep copy of the global model
        return copy.deepcopy(self.global_model)


def create_non_iid_split(dataset, num_clients, alpha=0.5, seed=None):
    """Create Dirichlet non-iid splits; returns list of index lists per client."""
    if seed is not None:
        np.random.seed(seed)

    # extract labels robustly
    if hasattr(dataset, 'targets'):
        raw = dataset.targets
    elif hasattr(dataset, 'labels'):
        raw = dataset.labels
    else:
        raw = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                raw.append(item[1])
            elif isinstance(item, dict):
                for key in ['label', 'y', 'target', 'class']:
                    if key in item:
                        raw.append(item[key])
                        break
                else:
                    raise ValueError("Could not find label in dataset dict")
            else:
                raise ValueError("Unsupported dataset item format")

    # convert to numpy int array
    if isinstance(raw, torch.Tensor):
        labels = raw.cpu().numpy().astype(np.int64)
    elif isinstance(raw, np.ndarray):
        labels = raw.astype(np.int64)
    else:
        labels = np.array([int(x.item()) if isinstance(x, torch.Tensor) else int(x) for x in raw], dtype=np.int64)

    if len(labels) == 0:
        raise ValueError("No labels found")

    # build per-class index lists
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    client_indices = [[] for _ in range(num_clients)]

    # split each class via Dirichlet proportions
    for idx in class_indices:
        if len(idx) == 0:
            continue
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        parts = np.split(idx, splits)
        for cid, p in enumerate(parts):
            client_indices[cid].extend(p.tolist())

    # shuffle and validate
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        if len(client_indices[i]) == 0:
            raise ValueError("Some client received 0 samples")

    return client_indices


def federated_learning_round_sequential(server, clients, local_epochs=1, train_eval_loader=None, show_progress=True):
    """Run one round: sequential client training, then server aggregation and evaluation."""
    try:
        from tqdm import tqdm
        has_tqdm = True
    except Exception:
        has_tqdm = False
        if show_progress:
            logger.warning("tqdm not available")

    global_model = server.get_global_model()  # copy of global model for local client updates
    client_updates = []
    iterator = tqdm(clients, desc="Training clients") if (show_progress and has_tqdm) else clients

    for client in iterator:
        try:
            client_model = copy.deepcopy(global_model)
            client_model = client_model.to(client.device)
            updated, ns = client.train(client_model, epochs=local_epochs)  # local training
            client_updates.append((updated, ns))
            del client_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Client {client.client_id} failed: {e}")
            continue

    if not client_updates:
        raise RuntimeError("All clients failed")

    server.aggregate_and_update(client_updates)  # aggregate and update global model
    test_loss, test_accuracy = server.evaluate()  # evaluate global model
    train_loss = server.compute_train_loss(train_eval_loader) if train_eval_loader is not None else None
    return test_loss, test_accuracy, train_loss


if __name__ == "__main__":
    print("Federated Learning Framework")
    print("Variants: FedAvg, FedAdam, FedYogi, FedAMSGrad, FedAdamW, ClientAdamServerAvg")
