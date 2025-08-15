# =======================================================================
# PART 1: IMPORTS
# All necessary libraries like torch, numpy, matplotlib, etc.
# =======================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# =======================================================================
# PART 2: MODEL DEFINITIONS
# All model classes are here, clearly separated by comments.
# =======================================================================

# --- Positional Encoding (Shared) ---
class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        # pe shape needs to be broadcastable to x.
        # pe from (max_len, 1, embed_dim) -> (seq_len, 1, embed_dim)
        # x from (batch, seq_len, embed_dim) will be added to pe (1, seq_len, embed_dim) after unsqueeze
        x = x.permute(1, 0, 2) # to (seq_len, batch, embed_dim)
        x = x + self.pe[:x.size(0)]
        return x.permute(1, 0, 2) # back to (batch, seq_len, embed_dim)

# --- Zarvan Model Architecture ---
class _HolisticExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.s_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, S, E = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        s = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weights = F.softmax(s, dim=2)
        head_outputs = torch.sum(weights * v, dim=2, keepdim=True)
        return self.combine(head_outputs.reshape(B, 1, E))

class _AssociativeExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.s_proj = nn.Linear(embed_dim, 1)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        s, v = self.s_proj(x), self.v_proj(x)
        weights = F.softmax(s, dim=1)
        return torch.sum(weights * v, dim=1, keepdim=True)

class _SequentialExtractor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.s_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.angle_calculator = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        B, S, E = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        weights = torch.cumsum(s * v, dim=1)
        alpha = self.norm(self.angle_calculator(weights / S))
        omega = alpha * math.pi
        phases = torch.cat([torch.cos(omega), torch.sin(omega)], dim=-1)
        return self.out_proj(phases)

class _ZarvanBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads):
        super().__init__()
        self.input_adapter = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.holistic_ctx = _HolisticExtractor(embed_dim, num_heads)
        self.associative_ctx = _AssociativeExtractor(embed_dim)
        self.sequential_ctx = _SequentialExtractor(embed_dim)
        self.expert_gate = nn.Sequential(nn.Linear(embed_dim, 3), nn.SiLU())
        self.ffn = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, S, E = x.shape
        x_residual = x
        x_adapted = self.input_adapter(x)
        q_holistic = self.holistic_ctx(x_adapted)
        q_associative = self.associative_ctx(x_adapted)
        q_sequential = self.sequential_ctx(x_adapted)
        gates = self.expert_gate(x_adapted)
        g_h, g_a, g_s = gates.chunk(3, dim=-1)
        h_candidate = (
            g_h * q_holistic.expand(-1, S, -1) +
            g_a * q_associative.expand(-1, S, -1) +
            g_s * q_sequential
        )
        out = x_residual + self.ffn(self.norm(h_candidate))
        return out

class ZarvanModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, num_layers, num_classes, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([_ZarvanBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)])
        self.output_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        h = self.embedding(x)
        h = self.pos_encoder(h)
        for layer in self.layers:
            h = layer(h)
        return self.output_head(h)


# --- Baseline Model Architectures ---
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.embedding(x)
        h, _ = self.lstm(h)
        return self.output_head(h)

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.embedding(x)
        h, _ = self.gru(h)
        return self.output_head(h)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        h = self.embedding(x)
        h = self.pos_encoder(h)
        h = self.transformer_encoder(h)
        return self.output_head(h)


# =======================================================================
# PART 3: TASK & DATASET DEFINITIONS (CORRECTED)
# All Dataset classes for our reasoning tests are here.
# =======================================================================
class BaseTaskDataset(Dataset):
    """Base class for tasks to handle common initialization."""
    def __init__(self, num_samples, seq_len):
        self.num_samples = num_samples
        self.seq_len = seq_len
        # The _generate_data call is now moved to child classes
        # after they have defined their specific vocab and num_classes.

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# --- Task 1: Binary State Flip ---
class BinaryFlipDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len, flip_token_prob=0.1):
        super().__init__(num_samples, seq_len)
        self.flip_token_prob = flip_token_prob
        self.vocab = {'PAD': 0, 'DATA': 1, 'TOKEN_FLIP': 2}
        self.num_classes = 2 # State A, State B
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        for _ in range(self.num_samples):
            seq, lab, current_state = [], [], 0
            for _ in range(self.seq_len):
                lab.append(current_state)
                if torch.rand(1).item() < self.flip_token_prob:
                    token = self.vocab['TOKEN_FLIP']
                    current_state = 1 - current_state
                else:
                    token = self.vocab['DATA']
                seq.append(token)
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# --- Task 2: Ternary State Cycle ---
class TernaryCycleDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len, cycle_token_prob=0.1):
        super().__init__(num_samples, seq_len)
        self.cycle_token_prob = cycle_token_prob
        self.vocab = {'PAD': 0, 'DATA': 1, 'TOKEN_CYCLE': 2}
        self.num_classes = 3 # State A, B, C
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        for _ in range(self.num_samples):
            seq, lab, current_state = [], [], 0
            for _ in range(self.seq_len):
                lab.append(current_state)
                if torch.rand(1).item() < self.cycle_token_prob:
                    token = self.vocab['TOKEN_CYCLE']
                    current_state = (current_state + 1) % 3
                else:
                    token = self.vocab['DATA']
                seq.append(token)
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# --- Task 3: Parallel State Tracking ---
class ParallelStatesDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len, flip_prob=0.15):
        super().__init__(num_samples, seq_len)
        self.flip_prob = flip_prob
        self.vocab = {'PAD': 0, 'DATA': 1, 'FLIP_A': 2, 'FLIP_B': 3}
        self.num_classes = 4 # A0B0, A0B1, A1B0, A1B1
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        for _ in range(self.num_samples):
            seq, lab, state_a, state_b = [], [], 0, 0
            for _ in range(self.seq_len):
                lab.append(state_a * 2 + state_b)
                if torch.rand(1).item() < self.flip_prob:
                    if torch.rand(1).item() < 0.5:
                        token, state_a = self.vocab['FLIP_A'], 1 - state_a
                    else:
                        token, state_b = self.vocab['FLIP_B'], 1 - state_b
                else:
                    token = self.vocab['DATA']
                seq.append(token)
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# --- Task 4: Long-Term Dependency ---
class LongDependencyDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len):
        super().__init__(num_samples, seq_len)
        self.vocab = {'PAD': 0, 'DATA': 1, 'TOKEN_FLIP': 2}
        self.num_classes = 2
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        for _ in range(self.num_samples):
            seq = [self.vocab['DATA']] * self.seq_len
            flip_pos_1 = torch.randint(0, self.seq_len // 4, (1,)).item()
            flip_pos_2 = torch.randint(self.seq_len * 3 // 4, self.seq_len, (1,)).item()
            seq[flip_pos_1] = self.vocab['TOKEN_FLIP']
            seq[flip_pos_2] = self.vocab['TOKEN_FLIP']
            lab, current_state = [], 0
            for token in seq:
                lab.append(current_state)
                if token == self.vocab['TOKEN_FLIP']:
                    current_state = 1 - current_state
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# --- Task 5: Reasoning in Noise ---
class ReasoningInNoiseDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len, cycle_prob=0.1, noise_prob=0.5):
        super().__init__(num_samples, seq_len)
        self.cycle_prob = cycle_prob
        self.noise_prob = noise_prob
        self.vocab = {'PAD': 0, 'DATA': 1, 'NOISE': 2, 'TOKEN_CYCLE': 3}
        self.num_classes = 3
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        for _ in range(self.num_samples):
            seq, lab, current_state = [], [], 0
            for _ in range(self.seq_len):
                lab.append(current_state)
                if torch.rand(1).item() < self.cycle_prob:
                    token = self.vocab['TOKEN_CYCLE']
                    current_state = (current_state + 1) % 3
                elif torch.rand(1).item() < self.noise_prob:
                    token = self.vocab['NOISE']
                else:
                    token = self.vocab['DATA']
                seq.append(token)
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# --- Task 6: Conditional Logic ---
class ConditionalLogicDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len, event_prob=0.2):
        super().__init__(num_samples, seq_len)
        self.event_prob = event_prob
        self.vocab = {'PAD': 0, 'DATA': 1, 'TOKEN_ACTION': 2, 'TOKEN_RESET': 3}
        self.num_classes = 2 # State A (0), State B (1)
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        for _ in range(self.num_samples):
            seq, lab, current_state = [], [], 0
            for _ in range(self.seq_len):
                lab.append(current_state)
                if torch.rand(1).item() < self.event_prob:
                    if torch.rand(1).item() < 0.5:
                        token = self.vocab['TOKEN_ACTION']
                        if current_state == 0: # Action only works in state A
                           current_state = 1
                    else:
                        token = self.vocab['TOKEN_RESET']
                        current_state = 0
                else:
                    token = self.vocab['DATA']
                seq.append(token)
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# --- Task 7: Hierarchical Reasoning ---
class HierarchicalReasoningDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len, event_prob=0.2):
        super().__init__(num_samples, seq_len)
        self.event_prob = event_prob
        self.vocab = {'PAD': 0, 'DATA': 1, 'MASTER_FLIP': 2, 'SUB_CYCLE': 3}
        self.num_classes = 6 # (Master OFF, Sub A/B/C), (Master ON, Sub A/B/C)
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        for _ in range(self.num_samples):
            seq, lab = [], []
            master_state, sub_state = 1, 0 # Master ON, Sub A
            for _ in range(self.seq_len):
                lab.append(master_state * 3 + sub_state)
                if torch.rand(1).item() < self.event_prob:
                    if torch.rand(1).item() < 0.4:
                        token = self.vocab['MASTER_FLIP']
                        master_state = 1 - master_state
                    else:
                        token = self.vocab['SUB_CYCLE']
                        if master_state == 1: # Sub-state only cycles if master is ON
                            sub_state = (sub_state + 1) % 3
                else:
                    token = self.vocab['DATA']
                seq.append(token)
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# --- Task 8: Counting ---
class CountingDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len, count_prob=0.25, threshold=3):
        super().__init__(num_samples, seq_len)
        self.count_prob = count_prob
        self.threshold = threshold
        self.vocab = {'PAD': 0, 'DATA': 1, 'COUNT_TOKEN': 2}
        self.num_classes = 2 # State A, State B
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        for _ in range(self.num_samples):
            seq, lab, current_state, counter = [], [], 0, 0
            for _ in range(self.seq_len):
                lab.append(current_state)
                if torch.rand(1).item() < self.count_prob:
                    token = self.vocab['COUNT_TOKEN']
                    counter += 1
                    if counter >= self.threshold:
                        current_state = 1 - current_state
                        counter = 0
                else:
                    token = self.vocab['DATA']
                seq.append(token)
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# =======================================================================
# PART 4: TRAINING & EVALUATION FRAMEWORK
# The generic functions for running the experiments.
# =======================================================================

def train_and_evaluate(model, model_name, train_loader, test_loader, epochs, lr, device):
    """A generic training and evaluation loop."""
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore PAD token
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_samples = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train] {model_name}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)
            
            mask = labels != 0 # Do not calculate loss on PAD tokens
            loss = criterion(outputs[mask], labels[mask])
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs[mask], 1)
            train_correct += (predicted == labels[mask]).sum().item()
            train_samples += labels[mask].size(0)
            
            if train_samples > 0:
                pbar.set_postfix({'loss': f"{train_loss / len(pbar):.3f}", 'acc': f"{train_correct / train_samples:.3f}"})

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_samples if train_samples > 0 else 0)
        
        # --- Evaluation ---
        model.eval()
        test_loss, test_correct, test_samples = 0, 0, 0
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Eval] {model_name}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)
                
                mask = labels != 0
                loss = criterion(outputs[mask], labels[mask])
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs[mask], 1)
                test_correct += (predicted == labels[mask]).sum().item()
                test_samples += labels[mask].size(0)

                if test_samples > 0:
                   pbar.set_postfix({'loss': f"{test_loss / len(pbar):.3f}", 'acc': f"{test_correct / test_samples:.3f}"})

        history['test_loss'].append(test_loss / len(test_loader))
        history['test_acc'].append(test_correct / test_samples if test_samples > 0 else 0)
        
    return history

def plot_results(results, task_name):
    """Plots accuracy and loss for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Experiment Results for: {task_name}', fontsize=16)
    
    # Accuracy Plot
    axes[0].set_title("Test Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    for model_name, history in results.items():
        axes[0].plot(history['test_acc'], label=model_name, marker='o')
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    
    # Loss Plot
    axes[1].set_title("Test Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    for model_name, history in results.items():
        axes[1].plot(history['test_loss'], label=model_name, marker='o')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{task_name}.png')
    plt.show()

def run_experiment(task_name, dataset_class, config):
    """Runs a full experiment for a given task."""
    print(f"\n{'='*30}\nRunning Experiment: {task_name}\n{'='*30}")
    
    # 1. Create Dataset and Dataloaders
    torch.manual_seed(config['seed'])
    
    # Use task-specific params if provided, else use default
    task_params = config.get(task_name, {})
    train_dataset = dataset_class(num_samples=config['num_samples_train'], seq_len=config['seq_len'], **task_params)
    test_dataset = dataset_class(num_samples=config['num_samples_test'], seq_len=config['seq_len'], **task_params)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    vocab_size = train_dataset.vocab_size
    num_classes = train_dataset.num_classes
    
    # 2. Initialize Models
    model_configs = {
        "embed_dim": config['embed_dim'],
        "hidden_dim": config['hidden_dim'],
        "num_heads": config['num_heads'],
        "num_layers": config['num_layers'],
        "max_len": config['seq_len'],
        "vocab_size": vocab_size,
        "num_classes": num_classes
    }
    
    models = {
        "Zarvan": ZarvanModel(**model_configs),
        "LSTM": LSTMModel(**model_configs),
        "GRU": GRUModel(**model_configs),
        "Transformer": TransformerModel(**model_configs),
    }
    
    # 3. Train and Evaluate each model
    results = {}
    for name, model in models.items():
        # Ensure each model starts with the same weights for a fair comparison
        torch.manual_seed(config['seed'])
        model.apply(lambda m: [m.reset_parameters() for m in m.children() if hasattr(m, 'reset_parameters')])

        history = train_and_evaluate(model, name, train_loader, test_loader, config['epochs'], config['lr'], config['device'])
        results[name] = history
        print(f"Final Test Accuracy for {name}: {history['test_acc'][-1]:.4f}")

    # 4. Plot Results
    plot_results(results, task_name)


# =======================================================================
# PART 5: EXPERIMENT EXECUTION
# The main block to configure and run the desired experiment.
# =======================================================================
if __name__ == '__main__':
    # --- General configuration for all experiments ---
    config = {
        'seed': 42,
        'seq_len': 48,
        'batch_size': 128,
        'epochs': 10,  # Increase for better convergence, 5 is for a quick demo
        'lr': 1e-3,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'num_samples_train': 8000,
        'num_samples_test': 1000,
        
        # Model Hyperparameters (kept similar for fairness)
        'embed_dim': 64,
        'hidden_dim': 128,  # For RNNs and Transformer FFN
        'num_heads': 4,
        'num_layers': 2,

        # Task-specific parameters (optional)
        'long_dependency': {'seq_len': 128}, # Override seq_len for this specific task
        'counting': {'threshold': 4},
        'reasoning_in_noise': {'noise_prob': 0.6},
    }

    # --- Dictionary to map task names to their dataset classes ---
    all_tasks = {
        "1_binary_flip": BinaryFlipDataset,
        "2_ternary_cycle": TernaryCycleDataset,
        "3_parallel_states": ParallelStatesDataset,
        "4_long_dependency": LongDependencyDataset,
        "5_reasoning_in_noise": ReasoningInNoiseDataset,
        "6_conditional_logic": ConditionalLogicDataset,
        "7_hierarchical_reasoning": HierarchicalReasoningDataset,
        "8_counting": CountingDataset,
    }

    # #################################################
    # ### CHOOSE YOUR EXPERIMENT TO RUN HERE ###
    # #################################################
    #
    # Instructions:
    # 1. Choose one of the keys from the `all_tasks` dictionary above.
    # 2. Assign it to the `task_to_run` variable.
    
    task_to_run = "8_counting"
    
    # #################################################

    print(f"Using device: {config['device']}")
    if task_to_run in config:
        print(f"Applying specific config for {task_to_run}: {config[task_to_run]}")
        # Update general config with task-specific one
        config.update(config[task_to_run])

    if task_to_run in all_tasks:
        dataset_class = all_tasks[task_to_run]
        run_experiment(task_to_run, dataset_class, config)
    else:
        print(f"Error: Task '{task_to_run}' not found.")
        print("Available tasks are:", list(all_tasks.keys()))