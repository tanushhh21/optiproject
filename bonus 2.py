import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from gdao_torch import GDAOTorch   # ← your optimizer


# ===============================================================
# Hyperparameters
# ===============================================================
LR_GDAO = 1e-2
LR_ADAM = 1e-2
LR_RMS  = 1e-2
LR_SGDM = 1e-2

MOMENTUM = 0.9
ALPHA_RMS = 0.99
BATCH_SIZE = 128
EPOCHS = 20
L2_WEIGHT_DECAY = 1e-4      # small L2 to stabilize


# ===============================================================
# Load Fashion-MNIST
# ===============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 → 784
])

train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_data  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=512, shuffle=False)


# ===============================================================
# Softmax Regression Model (Linear Only)
# ===============================================================
class SoftmaxRegression(nn.Module):
    def __init__(self, in_dim=784, out_dim=10):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)   # CrossEntropyLoss applies softmax internally


# ===============================================================
# Accuracy helper
# ===============================================================
def compute_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    model.train()
    return correct / total


# ===============================================================
# Training function
# ===============================================================
def train_model(opt_name):
    model = SoftmaxRegression()
    loss_fn = nn.CrossEntropyLoss()

    # Select optimizer
    if opt_name == "GDAO":
        optimizer = GDAOTorch(model.parameters(), lr=LR_GDAO)
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
    elif opt_name == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LR_RMS, alpha=ALPHA_RMS)
    elif opt_name == "SGDM":
        optimizer = torch.optim.SGD(model.parameters(), lr=LR_SGDM, momentum=MOMENTUM)
    else:
        raise ValueError("Unknown optimizer")

    loss_curve = []

    for epoch in range(EPOCHS):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)

            # Add small L2 stabilizer (weight decay)
            l2 = sum((p**2).sum() for p in model.parameters())
            loss = loss + L2_WEIGHT_DECAY * l2

            loss.backward()
            optimizer.step()

        loss_curve.append(loss.item())

        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            print(f"[{opt_name}] Epoch {epoch:02d} | Loss = {loss.item():.4f}")

    train_acc = compute_accuracy(model, train_loader)
    test_acc  = compute_accuracy(model, test_loader)

    return {
        "optimizer": opt_name,
        "loss_curve": loss_curve,
        "train_acc": train_acc,
        "test_acc": test_acc
    }


# ===============================================================
# Run all optimizers
# ===============================================================
optimizers = ["GDAO", "Adam", "RMSProp", "SGDM"]
results = {}

for opt in optimizers:
    print(f"\n=== Training with {opt} ===")
    results[opt] = train_model(opt)


# ===============================================================
# Summary
# ===============================================================
print("\n======================= Summary =======================")
for opt in optimizers:
    stats = results[opt]
    print(f"{opt:7s} -> Train Acc: {stats['train_acc']:.4f} | "
          f"Test Acc: {stats['test_acc']:.4f}")


# ===============================================================
# Plotting loss curves
# ===============================================================
plt.figure(figsize=(10,6))
for opt in optimizers:
    plt.plot(results[opt]["loss_curve"], label=opt, linewidth=2)

plt.title("Softmax Regression on Fashion-MNIST — Loss Curves", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
