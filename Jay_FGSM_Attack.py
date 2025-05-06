import os
import pickle
import tarfile
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def extract_cifar10(file_path, extract_path="./cifar-10-batches-py"):
    if not os.path.exists(extract_path):
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall()
    return extract_path

def load_batch(batch_file):
    with open(batch_file, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        data = entry['data']
        labels = entry['labels']
        data = data.reshape(-1, 3, 32, 32).astype(np.uint8)
        data = np.transpose(data, (0, 2, 3, 1))  # (N, H, W, C)
        return data, labels


class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # keep as numpy array!
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)  # Transform will convert it to tensor
        return img, label


def prepare_dataloaders(data_path, batch_size=128):
    # Load training data
    train_data, train_labels = [], []
    for i in range(1, 6):
        data, labels = load_batch(os.path.join(data_path, f"data_batch_{i}"))
        train_data.append(data)
        train_labels += labels
    train_data = np.concatenate(train_data)

    # Load test data
    test_data, test_labels = load_batch(os.path.join(data_path, "test_batch"))

    # Transform (PyTorch tensors, normalization optional)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Already 0-1, so ToTensor suffices
    ])

    train_set = CIFAR10Dataset(train_data, train_labels, transform)
    test_set = CIFAR10Dataset(test_data, test_labels, transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class SimpleVGG(nn.Module):
    def __init__(self):
        super(SimpleVGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_model(model, train_loader, test_loader, device, epochs=50):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        test_model(model, test_loader, device)

def test_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_dir = extract_cifar10("cifar-10-python.tar.gz")
train_loader, test_loader = prepare_dataloaders(data_dir)

# Model Training
model = SimpleVGG()
train_model(model, train_loader, test_loader, device)

####################################################
#NEW CODE
######################################################




def fgsm_attack(image, epsilon, data_grad):
    # FGSM step: add sign of gradient multiplied by epsilon
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad

    # Clamp to [0,1] to keep pixel values valid
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def run_fgsm_experiments(model, device, test_loader, epsilons=[0, 0.05, 0.1, 0.15, 0.2, 0.25]):
    model.eval()
    accuracies = []
    examples = []

    for eps in epsilons:
        correct = 0
        adv_examples = []

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data.requires_grad = True

            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]

            if init_pred.item() != target.item():
                continue

            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            perturbed_data = fgsm_attack(data, eps, data_grad)
            output = model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]

            if final_pred.item() == target.item():
                correct += 1

            if eps == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        final_acc = correct / float(len(test_loader))
        accuracies.append(final_acc)
        print(f"Epsilon: {eps:.3f} \tTest Accuracy = {final_acc * 100:.2f}%")

    return accuracies


# Make sure 'model', 'device', and 'test_loader' are defined in earlier cells (from Kevin's code)
fgsm_accuracies = run_fgsm_experiments(model, device, test_loader)
