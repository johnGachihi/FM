import argparse
import yaml
from pathlib import Path
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.galileo import Encoder
from src.data.utils import construct_galileo_input
import numpy as np
from tqdm import tqdm


class H5CropDataset(Dataset):
    def __init__(self, h5_file, bands, label_key="label"):
        self.h5_file = h5_file
        self.bands = bands
        self.label_key = label_key
        with h5py.File(h5_file, "r") as f:
            self.length = len(f[label_key])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, "r") as f:
            inputs = {}
            for band in self.bands:
                inputs[band] = f[band][idx]  # Expected shape: [T, H, W, C]
            label = f[self.label_key][idx]

        input_tensor = construct_galileo_input(
            **inputs, normalize=True  # Ensure Galileo pretraining normalization
        )
        return input_tensor, label


class GalileoClassifier(nn.Module):
    def __init__(self, encoder_ckpt_path, num_classes):
        super().__init__()
        self.encoder = Encoder.load_from_folder(Path(encoder_ckpt_path))
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # Modify if different feature dim
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, masked_output):
        feat = self.encoder.forward_masked_output(masked_output)
        return self.classifier(feat)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for x, y in tqdm(loader, desc="Training"):
        for key in x:
            x[key] = x[key].to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation"):
            for key in x:
                x[key] = x[key].to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = H5CropDataset(cfg["train_data"], cfg["bands"], cfg["label_key"])
    val_ds = H5CropDataset(cfg["val_data"], cfg["bands"], cfg["label_key"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = GalileoClassifier(cfg["encoder_ckpt"], cfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Optionally save model
        torch.save(model.state_dict(), f"galileo_classifier_epoch{epoch + 1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
