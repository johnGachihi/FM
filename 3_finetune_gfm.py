'''
To run in terminal:
python 3_finetune_gfm.py \
--data_dir /cluster/archiving/GIZ/data/patches_25/ \
--encoder_ckpt models/nano/ \
--save_dir /cluster/archiving/GIZ/data/checkpoints/ \
--batch_size 8 \
--epochs 50 \
--lr 0.0001


To run the model in server backgroung
nohup python 3_finetune_gfm.py \
--data_dir /cluster/archiving/GIZ/data/patches_25/ \
--encoder_ckpt models/nano/ \
--save_dir /cluster/archiving/GIZ/data/checkpoints/ \
--batch_size 8 \
--epochs 50 \
--lr 0.0001 \
> finetune_allseasons_with_2025.log  2>&1 &
'''

import timeit
start_time = timeit.default_timer()
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassJaccardIndex

from src.data.dataset import PixelwisePatchDataset
from src.galileo import Encoder
from src.data.utils import construct_galileo_input

valid_labels = {0, 1, 2, 3} #XXXXXIndicate class labels in training data
nsteps = 5         # 5 timesteps
nbands = 12        # 2 (S1) + 10 (S2)
nstatic = 11       # 1 (elevation) + 1 (slope) + 9 DW bands
modelWeightsName = 'gfm_model_with_2025.pt'

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

        # Split temporal and static bands
        temporal = x[:, :nsteps * nbands, :, :].view(B, nsteps, nbands, H, W)  # [B, 5, 12, H, W]
        temporal = temporal.permute(0, 1, 3, 4, 2).contiguous()                # [B, 5, H, W, 12]

        inputs = []

        for b in range(B):
            # Sentinel-1: VV, VH (first 2 channels)
            s1 = temporal[b, ..., :2].permute(1, 2, 0, 3).float()  # [H, W, 5, 2]

            # Sentinel-2: next 10 channels
            s2 = temporal[b, ..., 2:].permute(1, 2, 0, 3).float()  # [H, W, 5, 10]

            # Static bands: elevation (1), slope (1), DW (9)
            static = x[b, -nstatic:, :, :]  # [11, H, W]
            srtm = static[:2, :, :].permute(1, 2, 0).float()  # [H, W, 2]
            dw = static[2:, :, :].permute(1, 2, 0).float()    # [H, W, 9]

            # Call construct_galileo_input
            masked = construct_galileo_input(s1=s1, s2=s2, srtm=srtm, dw=dw, normalize=True)
            inputs.append(masked)

        # Stack per-batch tensors into batched input
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
        while feats.dim() > nsteps:
            feats = feats.squeeze(1)
        feats = feats[:, -1, :, :, :]  # [B, H, W, C]
        feats = feats.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return self.classifier(feats)


def compute_class_weights(dataset, num_classes):
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    class_counts = torch.zeros(num_classes)

    for _, mask in loader:
        for cls in range(num_classes):
            class_counts[cls] += (mask == cls).sum()

    weights = 1.0 / (class_counts + 1e-6)
    weights /= weights.sum()
    return weights


def train(args):
    print(f"[INFO] Loading datasets from: {args.data_dir}")
    train_dataset = PixelwisePatchDataset(root_dir=args.data_dir, split="train", valid_labels = valid_labels)
    val_dataset = PixelwisePatchDataset(root_dir=args.data_dir, split="val", valid_labels = valid_labels)

    num_classes = train_dataset.num_classes
    print(f"[INFO] Number of classes: {num_classes}", flush =True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    encoder = Encoder.load_from_folder(Path(args.encoder_ckpt))
    model = PixelwisePatchClassifier(encoder, num_classes=num_classes, freeze_encoder=args.freeze_encoder).to(args.device)

    weights = compute_class_weights(train_dataset, num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weights)
    # Optimizer and Scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_miou = 0.0

    val_miou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(args.device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, mask in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
            x, mask = x.to(args.device), mask.to(args.device)

            optimizer.zero_grad()
            logits = model(x)
            logits = F.interpolate(logits, size=mask.shape[1:], mode="bilinear", align_corners=False)

            loss = criterion(logits, mask)
            if torch.isnan(loss) or torch.isinf(loss):
                print("[WARN] Skipping batch due to invalid loss.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += ((preds == mask) & (mask != 255)).sum().item()
            total += (mask != 255).sum().item()

        train_acc = correct / total
        '''
        if total == 0:
            print(f"[WARNING] No valid training pixels found for epoch {epoch}. Setting train accuracy to 0.")
        train_acc = correct / total if total != 0 else 0.0
        '''
        model.eval()
        val_correct, val_total = 0, 0
        val_miou_metric.reset()

        with torch.no_grad():
            for x, mask in val_loader:
                x, mask = x.to(args.device), mask.to(args.device)
                logits = model(x)
                logits = F.interpolate(logits, size=mask.shape[1:], mode="bilinear", align_corners=False)
                preds = logits.argmax(dim=1)
                val_correct += ((preds == mask) & (mask != 255)).sum().item()
                val_total += (mask != 255).sum().item()
                val_miou_metric.update(preds, mask)

        val_acc = val_correct / val_total
        '''
        if val_total == 0:
            print(f"[WARNING] No valid validation pixels found for epoch {epoch}. Setting val accuracy to 0.")
        val_acc = val_correct / val_total if val_total != 0 else 0.0
        '''
        mean_miou = val_miou_metric.compute().item()
        scheduler.step(mean_miou)

        print(f"[Epoch {epoch}] LR: {optimizer.param_groups[0]['lr']:.6f}, Train Loss = {total_loss:.4f}, "
              f"Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Val mIoU = {mean_miou:.4f}")

        if mean_miou > best_val_miou:
            best_val_miou = mean_miou
            torch.save(model.state_dict(), os.path.join(args.save_dir, modelWeightsName))
            print(f"[INFO] Best model saved based on mIoU = {mean_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="pixelwise_checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze_encoder", action="store_true")
    args = parser.parse_args()

    train(args)
print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)


