# =======================================================================
# PART 1: IMPORTS
# All necessary libraries
# =======================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# =======================================================================
# PART 2: MODEL DEFINITIONS
# Models are adapted for the 32x32x3 CIFAR-10 dataset.
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

# --- Model 1: Zarvan for Vision (Adapted for CIFAR-10) ---
class ZarvanForVision(nn.Module):
    def __init__(self, img_size, in_channels, embed_dim, hidden_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        
        # Each row of pixels has (img_size * in_channels) features
        self.input_projection = nn.Linear(img_size * in_channels, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, img_size, embed_dim)) # Learnable pos encoding for H rows
        
        self.layers = nn.ModuleList([_ZarvanBlock(embed_dim, hidden_dim, num_heads) for _ in range(num_layers)])
        
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (B, C, H, W) -> (B, 3, 32, 32)
        # Permute to (B, H, W, C) and flatten W and C for each row
        x = x.permute(0, 2, 3, 1).flatten(2) # (B, H, W*C) -> (B, 32, 32*3)
        
        # Project input rows to embed_dim
        x = self.input_projection(x) # (B, 32, embed_dim)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        for layer in self.layers:
            x = layer(x)
        
        # Take the mean of the sequence output for classification
        cls_token = x.mean(dim=1)
        return self.mlp_head(cls_token)

# --- Model 2: Simple Vision Transformer (ViT for CIFAR-10) ---
class ViT(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, embed_dim, hidden_dim, num_heads, num_layers, num_classes):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
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

# --- Model 3: Standard CNN Baseline (for CIFAR-10) ---
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)

# =======================================================================
# PART 3: DATA LOADING & PREPARATION
# =======================================================================
def get_data_loaders(batch_size):
    # Mean and std for CIFAR-10 normalization
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    
    # Data augmentation for the training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# =======================================================================
# PART 4: TRAINING & EVALUATION FRAMEWORK
# =======================================================================

def train_and_evaluate(model, model_name, train_loader, test_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
    fig.suptitle('CIFAR-10 Classification Performance Comparison', fontsize=16)
    
    axes[0].set_title("Test Accuracy")
    axes[0].set_xlabel("Epochs"); axes[0].set_ylabel("Accuracy")
    for model_name, history in results.items():
        axes[0].plot(history['test_acc'], label=model_name, marker='o')
    axes[0].grid(True); axes[0].legend(); axes[0].set_ylim(bottom=0.5)
    
    axes[1].set_title("Test Loss")
    axes[1].set_xlabel("Epochs"); axes[1].set_ylabel("Loss")
    for model_name, history in results.items():
        axes[1].plot(history['test_loss'], label=model_name, marker='o')
    axes[1].grid(True); axes[1].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('cifar-10.png')
    plt.show()

# =======================================================================
# PART 5: EXPERIMENT EXECUTION
# =======================================================================
if __name__ == '__main__':
    config = {
        'seed': 42,
        'batch_size': 256,
        'epochs': 15, # CIFAR-10 requires more epochs to train well
        'lr': 0.001,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        
        # Model Hyperparameters
        'img_size': 32,
        'in_channels': 3,
        'patch_size': 4, # For ViT (32x32 image -> 8x8=64 patches)
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'num_classes': 10
    }

    print(f"Using device: {config['device']}")
    
    # --- Load Data ---
    train_loader, test_loader = get_data_loaders(config['batch_size'])
    
    # --- Initialize Models ---
    models = {
        "ZarvanForVision": ZarvanForVision(img_size=config['img_size'], in_channels=config['in_channels'], 
                                           embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'],
                                           num_heads=config['num_heads'], num_layers=config['num_layers'],
                                           num_classes=config['num_classes']),
        "ViT": ViT(img_size=config['img_size'], in_channels=config['in_channels'], patch_size=config['patch_size'],
                   embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim'],
                   num_heads=config['num_heads'], num_layers=config['num_layers'],
                   num_classes=config['num_classes']),
        "CNN": CNN(in_channels=config['in_channels'], num_classes=config['num_classes'])
    }
    
    # --- Run Experiments ---
    results = {}
    for name, model in models.items():
        print(f"\n{'='*30}\nTraining {name}\n{'='*30}")
        torch.manual_seed(config['seed'])
        
        # A simple way to re-initialize model weights
        def weight_reset(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.reset_parameters()
        model.apply(weight_reset)
        
        history = train_and_evaluate(model, name, train_loader, test_loader, config['epochs'], config['lr'], config['device'])
        results[name] = history
        print(f"Final Test Accuracy for {name}: {history['test_acc'][-1]:.4f}")

    # --- Plot Final Results ---
    plot_results(results)