"""
Multi-class Classification using PyTorch and Synthetic Data

This script builds a neural network classifier to distinguish between 
multiple classes generated using Scikit-Learn's make_blobs().
"""

# Import Modules
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
from torchmetrics import Accuracy

# Hyperparameters
NUM_CLASSES = 4
NUM_FEATURES = 2
HIDDEN_UNITS = 8
EPOCHS = 100
LR = 0.1
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data Generation
X, y = make_blobs(n_samples=1000,
                  n_features=NUM_FEATURES,
                  centers=NUM_CLASSES,
                  cluster_std=1.5,
                  random_state=RANDOM_SEED)

# Tensor Conversion
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# Data Visualization
def plot_data(X, y, title="Data"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=y.cpu(), cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

# Uncomment to view data
# plot_data(X, y, "Synthetic Multi-Class Data")

# Model Definition
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )

    def forward(self, x):
        return self.network(x)

# Instantiate Model
model = BlobModel(input_features=NUM_FEATURES,
                  output_features=NUM_CLASSES,
                  hidden_units=HIDDEN_UNITS).to(device)

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# Accuracy Function
def accuracy_fn(y_true, y_pred):
    correct = (y_pred == y_true).sum().item()
    return 100 * correct / len(y_true)

# Training and Evaluation Loop
for epoch in range(EPOCHS):
    model.train()
    logits = model(X_train)
    y_pred = torch.softmax(logits, dim=1).argmax(dim=1)

    loss = loss_fn(logits, y_train)
    acc = accuracy_fn(y_train, y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Evaluation
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_preds)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03}: | Train Loss: {loss:.4f} | Train Acc: {acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

# Final Evaluation
print("\nFinal Test Evaluation:")
print(f"Accuracy: {accuracy_fn(y_test, test_preds):.2f}%")

# Metrics with TorchMetrics
metric = Accuracy(task='multiclass', num_classes=NUM_CLASSES).to(device)
torchmetrics_acc = metric(test_preds, y_test)
print(f"TorchMetrics Accuracy: {torchmetrics_acc.item()*100:.2f}%")
