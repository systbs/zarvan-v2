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
import random

# =======================================================================
# PART 2: MODEL DEFINITIONS
# This section contains all model classes: Zarvan and the baselines.
# It is identical to the previous version.
# =======================================================================

# --- Positional Encoding (Shared) ---
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        return x.permute(1, 0, 2)

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
# PART 3: SSM-LIKE STATE-MAINTAINING TASKS
# =======================================================================
class BaseTaskDataset(Dataset):
    def __init__(self, num_samples, seq_len):
        self.num_samples = num_samples
        self.seq_len = seq_len

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# --- Test 16: Selective Copying ---
class SelectiveCopyingDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len, num_copy=5):
        self.num_copy = num_copy
        super().__init__(num_samples, seq_len)
        self.vocab = {'PAD': 0, 'DATA': 1, 'SEP': 2, **{f'D{i}': i+3 for i in range(10)}} # Digits 0-9
        self.num_classes = len(self.vocab) # Predict the copied digits
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        digit_tokens = list(range(3, 13))

        for _ in range(self.num_samples):
            # Content to be copied
            copy_content = random.choices(digit_tokens, k=self.num_copy)
            
            # Distractor content
            distractor_len = self.seq_len - (2 * self.num_copy) - 1
            distractors = [self.vocab['DATA']] * distractor_len
            
            # Input sequence
            seq = copy_content + distractors + [self.vocab['SEP']] + ([self.vocab['PAD']] * self.num_copy)
            
            # Label sequence: ignore everything except the final part
            lab = [0] * (self.seq_len - self.num_copy) + copy_content
            
            sequences.append(torch.tensor(seq[:self.seq_len], dtype=torch.long))
            labels.append(torch.tensor(lab[:self.seq_len], dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# --- Test 17: Signal Accumulation & Thresholding ---
class SignalAccumulationDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len, threshold=5):
        self.threshold = threshold
        super().__init__(num_samples, seq_len)
        self.vocab = {'PAD': 0, 'INC': 1, 'DEC': 2}
        self.num_classes = 2 # State A (below threshold), State B (above threshold)
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        for _ in range(self.num_samples):
            seq, lab = [], []
            accumulator = 0
            current_state = 0
            for _ in range(self.seq_len):
                lab.append(current_state)
                
                op = random.choice(['INC', 'DEC'])
                seq.append(self.vocab[op])
                
                if op == 'INC':
                    accumulator += 1
                else:
                    accumulator -= 1
                
                if accumulator >= self.threshold:
                    current_state = 1
            
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# --- Test 18: Dominant Frequency Detection ---
class DominantFrequencyDataset(BaseTaskDataset):
    def __init__(self, num_samples, seq_len):
        super().__init__(num_samples, seq_len)
        self.vocab = {'PAD': 0, 'A': 1, 'B': 2, 'C': 3}
        self.num_classes = len(self.vocab) # Predict the dominant token
        self.sequences, self.labels = self._generate_data()

    def _generate_data(self):
        sequences, labels = [], []
        tokens_to_count = list(self.vocab.keys())[1:]

        for _ in range(self.num_samples):
            seq, lab = [], []
            counts = {token: 0 for token in tokens_to_count}
            dominant_token = tokens_to_count[0]
            
            for _ in range(self.seq_len):
                # Label is the dominant token from the *previous* step
                lab.append(self.vocab[dominant_token])
                
                # Input is the current random token
                current_token = random.choice(tokens_to_count)
                seq.append(self.vocab[current_token])
                
                # Update counts and find new dominant token
                counts[current_token] += 1
                dominant_token = max(counts, key=counts.get)
            
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(lab, dtype=torch.long))
        return torch.stack(sequences), torch.stack(labels)

# =======================================================================
# PART 4: TRAINING & EVALUATION FRAMEWORK
# =======================================================================

def train_and_evaluate(model, model_name, train_loader, test_loader, epochs, lr, device, num_classes):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # PAD token is 0
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
            
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
            
            mask = labels != 0 
            if mask.sum() == 0: continue
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs[mask], 1)
            train_correct += (predicted == labels[mask]).sum().item()
            train_samples += mask.sum().item()
            
            if train_samples > 0:
                pbar.set_postfix({'loss': f"{train_loss / len(pbar):.3f}", 'acc': f"{train_correct / train_samples:.3f}"})

        history['train_loss'].append(train_loss / len(pbar) if len(pbar) > 0 else 0)
        history['train_acc'].append(train_correct / train_samples if train_samples > 0 else 0)
        
        model.eval()
        test_loss, test_correct, test_samples = 0, 0, 0
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Eval] {model_name}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                outputs = outputs.view(-1, num_classes)
                labels = labels.view(-1)
                
                mask = labels != 0
                if mask.sum() == 0: continue
                
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs[mask], 1)
                test_correct += (predicted == labels[mask]).sum().item()
                test_samples += mask.sum().item()

                if test_samples > 0:
                   pbar.set_postfix({'loss': f"{test_loss / len(pbar):.3f}", 'acc': f"{test_correct / test_samples:.3f}"})

        history['test_loss'].append(test_loss / len(test_loader) if len(test_loader) > 0 else 0)
        history['test_acc'].append(test_correct / test_samples if test_samples > 0 else 0)
        
    return history

def plot_results(results, task_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Experiment Results for: {task_name}', fontsize=16)
    
    axes[0].set_title("Test Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    for model_name, history in results.items():
        axes[0].plot(history['test_acc'], label=model_name, marker='o')
    axes[0].grid(True); axes[0].legend(); axes[0].set_ylim(0, 1.05)
    
    axes[1].set_title("Test Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    for model_name, history in results.items():
        axes[1].plot(history['test_loss'], label=model_name, marker='o')
    axes[1].grid(True); axes[1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{task_name}.png')
    plt.show()

# =======================================================================
# PART 5: EXPERIMENT EXECUTION
# =======================================================================
def run_experiment(task_name, dataset_class, config):
    print(f"\n{'='*30}\nRunning Experiment: {task_name}\n{'='*30}")
    
    torch.manual_seed(config['seed'])
    task_params = config.get(task_name, {})
    
    train_dataset = dataset_class(num_samples=config['num_samples_train'], seq_len=config['seq_len'], **task_params)
    test_dataset = dataset_class(num_samples=config['num_samples_test'], seq_len=config['seq_len'], **task_params)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    vocab_size = train_dataset.vocab_size
    num_classes = train_dataset.num_classes
    
    model_configs = {
        "embed_dim": config['embed_dim'], "hidden_dim": config['hidden_dim'],
        "num_heads": config['num_heads'], "num_layers": config['num_layers'],
        "max_len": config['seq_len'], "vocab_size": vocab_size, "num_classes": num_classes
    }
    
    models = {
        "Zarvan": ZarvanModel(**model_configs), "LSTM": LSTMModel(**model_configs),
        "GRU": GRUModel(**model_configs), "Transformer": TransformerModel(**model_configs),
    }
    
    results = {}
    for name, model in models.items():
        torch.manual_seed(config['seed'])
        model.apply(lambda m: [m.reset_parameters() for m in m.children() if hasattr(m, 'reset_parameters')])
        history = train_and_evaluate(model, name, train_loader, test_loader, config['epochs'], config['lr'], config['device'], num_classes)
        results[name] = history
        print(f"Final Test Accuracy for {name}: {history['test_acc'][-1]:.4f}")

    plot_results(results, task_name)


if __name__ == '__main__':
    config = {
        'seed': 42, 
        'seq_len': 128, # Longer sequence length to test long-range dependencies
        'batch_size': 128, 
        'epochs': 15, 
        'lr': 0.001, 
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'num_samples_train': 10000, 
        'num_samples_test': 2000,
        'embed_dim': 128, 
        'hidden_dim': 512, 
        'num_heads': 4, 
        'num_layers': 4,
    }

    all_tasks = {
        "18_selective_copying": SelectiveCopyingDataset,
        "19_signal_accumulation": SignalAccumulationDataset,
        "20_dominant_frequency": DominantFrequencyDataset,
    }

    # #################################################
    # ### CHOOSE YOUR SSM-LIKE EXPERIMENT HERE ###
    # #################################################
    
    task_to_run = "20_dominant_frequency"
    
    # #################################################

    print(f"Using device: {config['device']}")
    if task_to_run in all_tasks:
        dataset_class = all_tasks[task_to_run]
        run_experiment(task_to_run, dataset_class, config)
    else:
        print(f"Error: Task '{task_to_run}' not found. Available tasks:", list(all_tasks.keys()))