import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import PixelwisePatchDataset
from src.galileo import Encoder
from src.data.utils import construct_galileo_input


class PixelwisePatchClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.classifier = nn.Conv2d(
            in_channels=self.encoder.embedding_size,
            out_channels=num_classes,
            kernel_size=1
        )

    def encode_features(self, x):
        B, C, H, W = x.shape
        x = x.view(B, 6, 12, H, W).permute(0, 1, 3, 4, 2).contiguous()

        inputs = []
        for b in range(B):
            s1 = x[b, ..., :2].permute(1, 2, 0, 3).contiguous().float()
            s2 = x[b, ..., 2:].permute(1, 2, 0, 3).contiguous().float()
            masked = construct_galileo_input(s1=s1, s2=s2, normalize=True)
            inputs.append(masked)

        batched_input = {
            k: torch.stack([getattr(i, k).float() if k != "months" else getattr(i, k).long() for i in inputs])
            for k in inputs[0]._fields
        }

        feats, *_ = self.encoder(
            batched_input["space_time_x"],
            batched_input["space_x"],
            batched_input["time_x"],
            batched_input["static_x"],
            batched_input["space_time_mask"],
            batched_input["space_mask"],
            batched_input["time_mask"],
            batched_input["static_mask"],
            batched_input["months"],
            patch_size=H,
        )
        return feats

    def forward(self, x):
        feats = self.encode_features(x)
        while feats.dim() > 5:
            feats = feats.squeeze(1)
        feats = feats[:, -1, :, :, :]  # [B, H, W, C]
        feats = feats.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        logits = self.classifier(feats)  # [B, num_classes, H, W]
        return logits


@torch.no_grad()
def test(args):
    print(f"[INFO] Loading test dataset from: {args.data_dir}")
    test_dataset = PixelwisePatchDataset(root_dir=args.data_dir, split="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_classes = test_dataset.num_classes
    print(f"[INFO] Number of classes: {num_classes}")

    encoder = Encoder.load_from_folder(Path(args.encoder_ckpt))
    model = PixelwisePatchClassifier(encoder, num_classes=num_classes, freeze_encoder=args.freeze_encoder)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.to(args.device)
    model.eval()

    correct, total = 0, 0
    for x, mask in tqdm(test_loader, desc="Evaluating"):
        x, mask = x.to(args.device), mask.to(args.device)
        logits = model(x)
        logits = F.interpolate(logits, size=mask.shape[1:], mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1)
        correct += ((preds == mask) & (mask != 255)).sum().item()
        total += (mask != 255).sum().item()

    test_acc = correct / total
    print(f"[Test Accuracy] = {test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with test split")
    parser.add_argument("--encoder_ckpt", type=str, required=True, help="Path to encoder checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pt file")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze_encoder", action="store_true")
    args = parser.parse_args()

    test(args)
