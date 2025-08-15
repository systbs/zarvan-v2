# =======================================================================
# PART 1: IMPORTS
# All necessary libraries, including torchaudio
# =======================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import os
import random

# =======================================================================
# PART 2: MODEL DEFINITIONS
# Includes Zarvan for Audio, AST, 2D CNN, and a 1D CNN (M5).
# =======================================================================

# --- Zarvan Sub-modules (Unchanged) ---
class _HolisticExtractor(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
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
        alpha = self.norm(self.angle_calculator(weights / (S or 1.0)))
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
        h_candidate = (g_h * q_holistic.expand(-1, S, -1) +
                       g_a * q_associative.expand(-1, S, -1) +
                       g_s * q_sequential)
        out = x_residual + self.ffn(self.norm(h_candidate))
        return out

# --- Model 1: Zarvan for Audio (Spectrogram input) ---
class ZarvanForAudio(nn.Module):
    def __init__(self, n_mels, seq_len, embed_dim, hidden_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.input_projection = nn.Linear(n_mels, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.layers = nn.ModuleList([_ZarvanBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        # x shape: (B, 1, n_mels, seq_len) -> Spectrogram
        x = x.squeeze(1).transpose(1, 2) # (B, seq_len, n_mels)
        x = self.input_projection(x)
        x = x + self.pos_encoder
        for layer in self.layers:
            x = layer(x)
        cls_token = x.mean(dim=1)
        return self.mlp_head(cls_token)

# --- Model 2: Audio Spectrogram Transformer (AST) ---
class AST(nn.Module):
    def __init__(self, n_mels, seq_len, patch_size, embed_dim, hidden_dim, num_heads, num_layers, num_classes):
        super().__init__()
        num_patches_h = n_mels // patch_size
        num_patches_w = seq_len // patch_size
        num_patches = num_patches_h * num_patches_w
        
        self.patch_embedding = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        cls_token = x[:, 0]
        return self.mlp_head(cls_token)

# --- Model 3: 2D CNN (Spectrogram input) ---
class CNN2D(nn.Module):
    def __init__(self, n_mels, seq_len, num_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        
        # Calculate flattened size dynamically
        dummy_input = torch.randn(1, 1, n_mels, seq_len)
        dummy_output = self.conv_block3(self.conv_block2(self.conv_block1(dummy_input)))
        self.flattened_size = dummy_output.numel()

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.flattened_size, num_classes))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x)

# --- Model 4: 1D CNN - M5 (Raw waveform input) ---
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2).squeeze()


# =======================================================================
# PART 3: DATA LOADING & PREPARATION
# =======================================================================
class SpeechCommandsDataset(torchaudio.datasets.SPEECHCOMMANDS):
    def __init__(self, subset: str):
        super().__init__("./", download=True)
        
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

def get_data_loaders(batch_size, n_mels, sample_rate):
    labels = sorted(list(set(p.split(os.path.sep)[-2] for p in torchaudio.datasets.SPEECHCOMMANDS("./", download=True)._walker)))
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # Transformation for spectrogram-based models
    mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=n_mels)
    
    def pad_sequence(batch):
        # Make all tensors in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)

    def collate_fn_spectrogram(batch):
        tensors, targets = [], []
        for waveform, _, label, _, _ in batch:
            tensors += [waveform]
            targets += [label_to_index[label]]
        tensors = pad_sequence(tensors)
        targets = torch.tensor(targets)
        # Apply spectrogram transformation
        tensors = mel_spectrogram(tensors)
        return tensors, targets

    def collate_fn_raw(batch):
        tensors, targets = [], []
        for waveform, _, label, _, _ in batch:
            tensors += [waveform]
            targets += [label_to_index[label]]
        tensors = pad_sequence(tensors)
        targets = torch.tensor(targets)
        return tensors, targets

    train_set = SpeechCommandsDataset("training")
    test_set = SpeechCommandsDataset("testing")

    # Dataloader for models that use spectrogram
    train_loader_spec = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_spectrogram, num_workers=2)
    test_loader_spec = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_spectrogram, num_workers=2)

    # Dataloader for models that use raw waveform (1D CNN)
    train_loader_raw = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_raw, num_workers=2)
    test_loader_raw = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_raw, num_workers=2)

    return train_loader_spec, test_loader_spec, train_loader_raw, test_loader_raw, len(labels)


# =======================================================================
# PART 4: TRAINING & EVALUATION FRAMEWORK
# =======================================================================
def train_and_evaluate(model, model_name, train_loader, test_loader, epochs, lr, device):
    model.to(device)
    # Using Negative Log Likelihood Loss for M5 model's log_softmax output
    criterion = nn.NLLLoss() if isinstance(model, M5) else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_samples = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train] {model_name}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Clip gradients to prevent exploding gradients, common in Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_samples += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f"{(train_loss / len(pbar)):.3f}", 'acc': f"{(train_correct / train_samples):.3f}"})

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_samples)
        
        model.eval()
        test_loss, test_correct, test_samples = 0, 0, 0
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Eval] {model_name}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_samples += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                pbar.set_postfix({'loss': f"{(test_loss / len(pbar)):.3f}", 'acc': f"{(test_correct / test_samples):.3f}"})

        history['test_loss'].append(test_loss / len(test_loader))
        history['test_acc'].append(test_correct / test_samples)
        scheduler.step()
        
    return history

def plot_results(results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Speech Commands Classification Performance', fontsize=16)
    
    axes[0].set_title("Test Accuracy")
    axes[0].set_xlabel("Epochs"); axes[0].set_ylabel("Accuracy")
    for model_name, history in results.items():
        axes[0].plot(history['test_acc'], label=model_name, marker='o')
    axes[0].grid(True); axes[0].legend()
    
    axes[1].set_title("Test Loss")
    axes[1].set_xlabel("Epochs"); axes[1].set_ylabel("Loss")
    for model_name, history in results.items():
        axes[1].plot(history['test_loss'], label=model_name, marker='o')
    axes[1].grid(True); axes[1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('speech-audio.png')
    plt.show()

# =======================================================================
# PART 5: EXPERIMENT EXECUTION (MODIFIED WITH MODEL-SPECIFIC LEARNING RATE)
# =======================================================================
if __name__ == '__main__':
    config = {
        'seed': 42,
        'batch_size': 128, # Smaller batch size can also help with stability
        'epochs': 10, 
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        
        # Audio Hyperparameters
        'sample_rate': 16000,
        'n_mels': 128,
        'seq_len': 32, # Will be updated dynamically
        
        # Model Hyperparameters
        'patch_size': 16, 
        'embed_dim': 192,
        'hidden_dim': 384,
        'num_heads': 4,
        'num_layers': 4,
    }

    print(f"Using device: {config['device']}")
    
    # --- Load Data ---
    train_spec, test_spec, train_raw, test_raw, num_classes = get_data_loaders(
        config['batch_size'], config['n_mels'], config['sample_rate'])
    
    # --- Adjust seq_len based on actual data ---
    dummy_spec, _ = next(iter(train_spec))
    config['seq_len'] = dummy_spec.shape[3]
    print(f"Spectrogram sequence length set to: {config['seq_len']}")

    # --- Initialize Models with specific configurations ---
    # We can now assign a different learning rate to each model
    models_config = {
        "AST": {
            "model": AST(n_mels=config['n_mels'], seq_len=config['seq_len'], patch_size=config['patch_size'],
                         embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'],
                         num_heads=config['num_heads'], num_layers=config['num_layers'],
                         num_classes=num_classes),
            "lr": 0.0001, # CRITICAL: Much lower LR for Transformer stability
            "train_loader": train_spec,
            "test_loader": test_spec
        },
        
        "ZarvanForAudio": {
            "model": ZarvanForAudio(n_mels=config['n_mels'], seq_len=config['seq_len'], embed_dim=config['embed_dim'],
                                    hidden_dim=config['hidden_dim'], num_heads=config['num_heads'],
                                    num_layers=config['num_layers'], num_classes=num_classes),
            "lr": 0.001, # Standard LR
            "train_loader": train_spec,
            "test_loader": test_spec
        },
        
        "CNN2D": {
            "model": CNN2D(n_mels=config['n_mels'], seq_len=config['seq_len'], num_classes=num_classes),
            "lr": 0.001, # Standard LR
            "train_loader": train_spec,
            "test_loader": test_spec
        },
        "M5_1D_CNN": {
            "model": M5(n_output=num_classes),
            "lr": 0.001, # Standard LR
            "train_loader": train_raw,
            "test_loader": test_raw
        }
    }
    
    # --- Run Experiments ---
    results = {}
    for name, params in models_config.items():
        print(f"\n{'='*30}\nTraining {name}\n{'='*30}")
        torch.manual_seed(config['seed'])
        
        model = params["model"]
        def weight_reset(m):
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                m.reset_parameters()
        model.apply(weight_reset)
        
        # Use the specific learning rate for this model
        lr = params["lr"]
        train_loader = params["train_loader"]
        test_loader = params["test_loader"]

        print(f"Using learning rate: {lr} for {name}")
        
        history = train_and_evaluate(model, name, train_loader, test_loader, config['epochs'], lr, config['device'])
        results[name] = history
        print(f"Final Test Accuracy for {name}: {history['test_acc'][-1]:.4f}")

    # --- Plot Final Results ---
    plot_results(results)