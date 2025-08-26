'''
To run in terminal:
python 3_finetune_gfm.py \
--data_dir /cluster/archiving/GIZ/data/patches_25/ \
--encoder_ckpt models/nano/ \
--decoder_ckpt models/nano/ \
--save_dir /cluster/archiving/GIZ/data/checkpoints/ \
--batch_size 8 \
--epochs 50 \
--lr 0.0001


To run the model in server backgroung
nohup python /home/bkenduiywo/FM/3b_finetune_gfm.py \
--data_dir /cluster/archiving/GIZ/data/WC_SA/ \
--encoder_ckpt models/nano/ \
--save_dir /cluster/archiving/GIZ/data/checkpoints/ \
--batch_size 8 \
--epochs 50 \
--lr 0.0001 \
--recon_weight 0.5 \
--freeze_encoder \
> finetune_WC_SA.log 2>&1 &
'''
# ------------------------------------------------------------
# Author: Joseph Chemut and Benson Kenduiywo
# Accuracy assessment using pre-rasterized reference labels
# ------------------------------------------------------------
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

valid_labels = {0, 1, 2, 3, 4, 5, 6}  # Crop classes: banana, bean, etc.
nsteps = 5  # 5 timesteps
nbands = 12  # 2 (S1) + 10 (S2)
nstatic = 11  # 1 (elevation) + 1 (slope) + 9 (DW)
modelWeightsName = 'gfm_model_WC_SA.pt'

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
        self.decoder = nn.Conv2d(
            in_channels=self.encoder.embedding_size,
            out_channels=nsteps * nbands + nstatic,  # Reconstruct 71 bands
            kernel_size=1
        )

    def encode_features(self, x):
        B, C, H, W = x.shape
        assert C == nsteps * nbands + nstatic, f"Expected {nsteps * nbands + nstatic} channels, got {C}"

        temporal = x[:, :nsteps * nbands, :, :].view(B, nsteps, nbands, H, W)  # [B, 5, 12, H, W]
        temporal = temporal.permute(0, 1, 3, 4, 2).contiguous()  # [B, 5, H, W, 12]

        inputs = []
        for b in range(B):
            s1 = temporal[b, ..., :2].permute(1, 2, 0, 3).float()  # [H, W, 5, 2]
            s2 = temporal[b, ..., 2:].permute(1, 2, 0, 3).float()  # [H, W, 5, 10]
            static = x[b, -nstatic:, :, :]  # [11, H, W]
            srtm = static[:2, :, :].permute(1, 2, 0).float()  # [H, W, 2]
            dw = static[2:, :, :].permute(1, 2, 0).float()    # [H, W, 9]

            masked = construct_galileo_input(s1=s1, s2=s2, srtm=srtm, dw=dw, normalize=True)
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
        print("Encoder output shape:", feats.shape)  # Debug
        return feats

    def forward(self, x):
        feats = self.encode_features(x)
        if feats.dim() == 6:
            feats = feats[:, -1, 0, :, :, :]  # [B, H, W, C]
        elif feats.dim() == 5:
            feats = feats[:, -1, :, :, :]  # [B, H, W, C]
        else:
            raise RuntimeError(f"Unexpected feats shape: {feats.shape}")
        feats = feats.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        logits = self.classifier(feats)  # [B, num_classes, H, W]
        recon = self.decoder(feats)      # [B, 71, H, W]
        return logits, recon

def compute_class_weights(dataset, num_classes):
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    class_counts = torch.zeros(num_classes)
    total_pixels = 0

    for _, mask, _ in loader:
        for cls in range(num_classes):
            class_counts[cls] += (mask == cls).sum()
        total_pixels += (mask != 255).sum()

    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum()
    print("Class counts:", class_counts.tolist())
    print("Class weights:", weights.tolist())
    print("Valid pixels (not 255):", total_pixels.item())
    return weights

def reconstruction_loss(pred, target, missing_mask):
    mse = F.mse_loss(pred, target, reduction='none')  # [B, 71, H, W]
    missing_mask = missing_mask.unsqueeze(1)  # [B, 1, H, W]
    mse = mse * missing_mask
    if missing_mask.sum() > 0:
        loss = mse.sum() / missing_mask.sum()
        if torch.isnan(loss) or torch.isinf(loss):
            print("[WARN] Invalid reconstruction loss. Pred max:", pred.max().item(), "Target max:", target.max().item())
            return torch.tensor(0.0, device=pred.device)
        return loss
    print("[WARN] No missing pixels in batch; reconstruction loss set to 0.")
    return torch.tensor(0.0, device=pred.device)

def train(args):
    print(f"[INFO] Loading datasets from: {args.data_dir}")
    train_dataset = PixelwisePatchDataset(root_dir=args.data_dir, split="train", valid_labels=valid_labels)
    val_dataset = PixelwisePatchDataset(root_dir=args.data_dir, split="val", valid_labels=valid_labels)

    num_classes = train_dataset.num_classes
    print(f"[INFO] Number of classes: {num_classes}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    encoder = Encoder.load_from_folder(Path(args.encoder_ckpt))
    model = PixelwisePatchClassifier(encoder, num_classes=num_classes, freeze_encoder=args.freeze_encoder).to(args.device)

    weights = compute_class_weights(train_dataset, num_classes).to(args.device)
    cls_criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_miou = 0.0
    val_miou_metric = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(args.device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_cls_loss, total_rec_loss, correct, total = 0.0, 0.0, 0.0, 0, 0
        batch_count = 0

        for x, mask, missing_mask in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
            x, mask, missing_mask = x.to(args.device), mask.to(args.device), missing_mask.to(args.device)

            optimizer.zero_grad()
            logits, recon = model(x)
            logits = F.interpolate(logits, size=mask.shape[1:], mode="bilinear", align_corners=False)
            recon = F.interpolate(recon, size=x.shape[2:], mode="bilinear", align_corners=False)

            cls_loss = cls_criterion(logits, mask)
            rec_loss = reconstruction_loss(recon, x, missing_mask)
            loss = cls_loss + args.recon_weight * rec_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print("[WARN] Skipping batch due to invalid loss. Logits max:", logits.max().item(), 
                      "Recon max:", recon.max().item(), "Input max:", x.max().item())
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_rec_loss += rec_loss.item()
            preds = logits.argmax(dim=1)
            correct += ((preds == mask) & (mask != 255)).sum().item()
            total += (mask != 255).sum().item()
            batch_count += 1

            if batch_count % 100 == 0:
                print(f"Batch {batch_count}: Preds unique:", torch.unique(preds).tolist(), 
                      "Mask unique:", torch.unique(mask).tolist(), 
                      "Missing pixels:", missing_mask.sum().item())

        train_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        avg_cls_loss = total_cls_loss / batch_count if batch_count > 0 else 0.0
        avg_rec_loss = total_rec_loss / batch_count if batch_count > 0 else 0.0

        model.eval()
        val_correct, val_total, val_total_rec_loss = 0, 0, 0.0
        val_batch_count = 0
        val_miou_metric.reset()

        with torch.no_grad():
            for x, mask, missing_mask in val_loader:
                x, mask, missing_mask = x.to(args.device), mask.to(args.device), missing_mask.to(args.device)
                logits, recon = model(x)
                logits = F.interpolate(logits, size=mask.shape[1:], mode="bilinear", align_corners=False)
                recon = F.interpolate(recon, size=x.shape[2:], mode="bilinear", align_corners=False)
                preds = logits.argmax(dim=1)
                val_correct += ((preds == mask) & (mask != 255)).sum().item()
                val_total += (mask != 255).sum().item()
                rec_loss = reconstruction_loss(recon, x, missing_mask)
                if not torch.isnan(rec_loss) and not torch.isinf(rec_loss):
                    val_total_rec_loss += rec_loss.item()
                    val_batch_count += 1
                val_miou_metric.update(preds, mask)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        avg_val_rec_loss = val_total_rec_loss / val_batch_count if val_batch_count > 0 else 0.0
        mean_miou = val_miou_metric.compute().item()
        scheduler.step(mean_miou)

        print(f"[Epoch {epoch}] LR: {optimizer.param_groups[0]['lr']:.6f}, "
              f"Train Loss = {avg_loss:.4f}, Train Cls Loss = {avg_cls_loss:.4f}, "
              f"Train Rec Loss = {avg_rec_loss:.4f}, Train Acc = {train_acc:.4f}, "
              f"Val Acc = {val_acc:.4f}, Val mIoU = {mean_miou:.4f}, Val Rec Loss = {avg_val_rec_loss:.4f}")

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
    parser.add_argument("--recon_weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze_encoder", action="store_true")
    args = parser.parse_args()

    train(args)
print("Done! Elapsed time (hours):", (timeit.default_timer() - start_time) / 3600.0)