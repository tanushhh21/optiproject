# problem4.py
"""
FM216 Question 4 — MNIST Classification

Comparing:
- GDAO (PyTorch version)
- Adam
- SGD + Momentum
- RMSprop

Recorded:
- Per-epoch training loss
- Per-epoch test accuracy
- Training time for each optimizer
- Final comparison table
- Loss curves saved to: plots/mnist_comparison.png
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from gdao_torch import GDAOTorch  # must be in the same folder

# -------------------------------------------------------
# DEVICE
# -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n=====================================")
print("        FM216 QUESTION 4 — MNIST      ")
print("=====================================")
print("Using device:", device)

# create plots directory
os.makedirs("plots", exist_ok=True)

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------------------------------------
# TRAIN LOOP
# -------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, name):
    model.train()
    total_loss = 0
    batch_idx = 0

    for x, y in loader:
        batch_idx += 1
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print for every 200 batches
        if batch_idx % 200 == 0:
            print(f"  [{name}]   Batch {batch_idx:>4}   Loss={loss.item():.4f}")

    return total_loss / len(loader)

# -------------------------------------------------------
# EVALUATION LOOP
# -------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

# -------------------------------------------------------
# MAIN EXPERIMENT
# -------------------------------------------------------
def run():
    # --- Load MNIST ---
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    print("Training samples:", len(train_set))
    print("Test samples:", len(test_set))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # --- Define optimizers ---
    optimizers = {
        "GDAO": lambda params: GDAOTorch(params, lr=0.001, gamma=0.5),
        "Adam": lambda params: optim.Adam(params, lr=0.001),
        "SGDM": lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
        "RMSprop": lambda params: optim.RMSprop(params, lr=0.001)
    }

    num_epochs = 20
    results = {}

    # -------------------------------------------------------
    # Run all optimizers
    # -------------------------------------------------------
    for name, opt_fn in optimizers.items():
        print(f"\n=====================================")
        print(f"      Training with {name}")
        print("=====================================")

        model = MNISTNet().to(device)
        optimizer = opt_fn(model.parameters())

        losses = []
        accuracies = []

        start = time.time()

        for epoch in range(1, num_epochs + 1):
            print(f"\n--- {name}: Epoch {epoch}/{num_epochs} ---")

            loss = train_one_epoch(model, train_loader, optimizer, criterion, name)
            acc = evaluate(model, test_loader)

            losses.append(loss)
            accuracies.append(acc)

            print(f"{name} | Epoch {epoch}/{num_epochs} | "
                  f"Loss={loss:.4f} | Test Acc={acc:.4f}")

            # Print extra useful stuff
            if name == "GDAO":
                try:
                    last_align = optimizer.align_history[-1]
                    print(f"  [GDAO] Last Alignment: {last_align:.4f}")
                except:
                    pass

        total_time = time.time() - start
        results[name] = (losses, accuracies, total_time)

        print(f"\n{name} finished training:")
        print(f"  ➤ Final Accuracy: {accuracies[-1]:.4f}")
        print(f"  ➤ Total Training Time: {total_time:.2f} sec")

    # -------------------------------------------------------
    # FINAL COMPARISON TABLE
    # -------------------------------------------------------
    print("\n==============================================")
    print("            FINAL MNIST COMPARISON TABLE       ")
    print("==============================================")
    print(f"{'Optimizer':<10} | {'Final Acc':<10} | {'Time (s)':<10}")
    print("-" * 45)

    for name, (losses, accs, tm) in results.items():
        print(f"{name:<10} | {accs[-1]:<10.4f} | {tm:<10.2f}")

    print("-" * 45)

    # -------------------------------------------------------
    # PLOT LOSS CURVES
    # -------------------------------------------------------
    plt.figure(figsize=(8, 5))
    for name, (losses, _, _) in results.items():
        plt.plot(losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("MNIST Training Loss — GDAO vs Adam vs SGDM vs RMSprop")
    plt.grid(True, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/mnist_comparison.png")
    plt.close()

    print("\nSaved training loss plot → plots/mnist_comparison.png\n")

    # -------------------------------------------------------
    # PLOT TEST ACCURACY CURVES (with std shading + markers)
    # -------------------------------------------------------
    plt.figure(figsize=(8, 5))

    # Convert accuracy lists to tensors for std shading
    # We only have 1 run per optimizer here, so no real std.
    # But we plot clean curves + final value markers.
    for name, (losses, accs, _) in results.items():
        epochs = list(range(1, len(accs) + 1))
        plt.plot(epochs, accs, label=name, linewidth=2)

        # Add final accuracy marker
        plt.scatter([epochs[-1]], [accs[-1]], s=60, marker="o")

        # Annotate the final value
        plt.text(
            epochs[-1] + 0.1,
            accs[-1],
            f"{accs[-1]*100:.2f}%",
            fontsize=9,
            verticalalignment="center"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("MNIST Test Accuracy — GDAO vs Adam vs SGDM vs RMSprop")
    plt.grid(True, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/mnist_accuracy.png")
    plt.close()

    print("\nSaved test accuracy plot → plots/mnist_accuracy.png\n")

if __name__ == "__main__":
    run()
