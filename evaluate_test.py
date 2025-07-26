import argparse
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import CustomPatchDataset
from src.galileo import Encoder
from src.data.utils import construct_galileo_input
from train_classifier import PatchClassifier  # Assumes same definition used in training

def evaluate(model, dataloader, device):
    model.eval()
    total_correct, total = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += y.size(0)

    acc = total_correct / total
    print(f"[Test] Accuracy: {acc:.4f}")
    return acc

def main(args):
    # Use same label2idx as training
    train_dataset = CustomPatchDataset(root_dir=args.data_dir, split="train")
    label2idx = train_dataset.label2idx
    test_dataset = CustomPatchDataset(root_dir=args.data_dir, split="test", label2idx=label2idx)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load encoder and model
    encoder = Encoder.load_from_folder(Path(args.encoder_ckpt))
    model = PatchClassifier(encoder, num_classes=len(label2idx), freeze_encoder=False).to(args.device)

    # Load trained weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    print(f"Loaded model from {args.checkpoint}")

    evaluate(model, test_loader, args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with train/val/test folders")
    parser.add_argument("--encoder_ckpt", type=str, required=True, help="Path to encoder folder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args)
