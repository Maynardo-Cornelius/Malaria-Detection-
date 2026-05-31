""" 
Install dependencies: pip install torch torchvision matplotlib scikit-learn pillow tqdm
Jalankan: python train.py --data_dir ./cell_images
"""
 
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
 
# ARGUMEN
parser = argparse.ArgumentParser(description="Training CNN Deteksi Malaria")
parser.add_argument("--data_dir",   type=str, default="./cell_images", help="Path ke folder dataset")
parser.add_argument("--epochs",     type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--img_size",   type=int, default=128)
parser.add_argument("--val_split",  type=float, default=0.15, help="Proporsi validasi (0.0-1.0)")
parser.add_argument("--test_split", type=float, default=0.15, help="Proporsi test (0.0-1.0)")
parser.add_argument("--output_dir", type=str, default="./output", help="Folder simpan model & grafik")
args = parser.parse_args()
 
os.makedirs(args.output_dir, exist_ok=True)
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device   : {DEVICE}")
print(f"Dataset  : {args.data_dir}")
print(f"Epochs   : {args.epochs}")
print(f"Img size : {args.img_size}x{args.img_size}\n")
 
# DATA AUGMENTATION & LOADING
IMG_MEAN = [0.485, 0.456, 0.406]   
IMG_STD  = [0.229, 0.224, 0.225]
 
transform_train = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])
 
transform_eval = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])
 
# Load semua data dengan transform train dulu untuk split
full_dataset = ImageFolder(root=args.data_dir, transform=transform_train)
CLASSES      = full_dataset.classes          # ['Parasitized', 'Uninfected']
NUM_CLASSES  = len(CLASSES)
N_TOTAL      = len(full_dataset)
 
print(f"Kelas ditemukan : {CLASSES}")
print(f"Total gambar    : {N_TOTAL:,}")
for cls, idx in full_dataset.class_to_idx.items():
    count = sum(1 for _, l in full_dataset.samples if l == idx)
    print(f"  {cls}: {count:,} gambar")
 
# Split train / val / test
n_test  = int(N_TOTAL * args.test_split)
n_val   = int(N_TOTAL * args.val_split)
n_train = N_TOTAL - n_val - n_test
 
train_set, val_set, test_set = random_split(
    full_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)
 
# Terapkan transform eval untuk val & test
val_set.dataset  = ImageFolder(root=args.data_dir, transform=transform_eval)
test_set.dataset = ImageFolder(root=args.data_dir, transform=transform_eval)
 
print(f"\nSplit data  : Train={n_train:,} | Val={n_val:,} | Test={n_test:,}")
 
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
 
# MODEL — Transfer Learning (ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
 
# Freeze semua layer, kecuali layer terakhir
for param in model.parameters():
    param.requires_grad = False
 
# Ganti classifier terakhir
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, NUM_CLASSES)
)
 
model = model.to(DEVICE)
 
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel      : ResNet18 (Transfer Learning)")
print(f"Total param    : {total_params:,}")
print(f"Trainable param: {trainable_params:,} (hanya classifier head)\n")
 
# LOSS, OPTIMIZER, SCHEDULER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
 
# TRAINING LOOP
def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
 
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
 
    with ctx:
        for images, labels in tqdm(loader, desc="Train" if is_train else "Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
 
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
            total_loss += loss.item() * images.size(0)
            correct    += outputs.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)
 
    return total_loss / total, 100.0 * correct / total
 
 
history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0
best_model_path = os.path.join(args.output_dir, "best_model.pth")
 
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
print("-" * 65)
 
for epoch in range(1, args.epochs + 1):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
    vl_loss, vl_acc = run_epoch(model, val_loader,   criterion)
    scheduler.step()
 
    current_lr = optimizer.param_groups[0]["lr"]
 
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)
 
    # Simpan model terbaik
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "img_size": args.img_size,
            "val_acc": vl_acc,
        }, best_model_path)
        marker = " ✓ BEST"
    else:
        marker = ""
 
    print(f"{epoch:>5} | {tr_loss:>10.4f} | {tr_acc:>8.2f}% | {vl_loss:>8.4f} | {vl_acc:>6.2f}%"
          f" | {current_lr:.2e}{marker}")
 
print(f"\nTraining selesai! Best Val Acc: {best_val_acc:.2f}%")
print(f"Model terbaik disimpan di: {best_model_path}")

# EVALUASI FINAL DI TEST SET
print("\n── Evaluasi di Test Set ──")
 
# Load model terbaik
checkpoint = torch.load(best_model_path, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
 
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        preds  = model(images).argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
 
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))
 
# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix - Test Set")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.show()
print(f"Confusion matrix disimpan di: {cm_path}")
 
# GRAFIK TRAINING
epochs_range = range(1, args.epochs + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
 
ax1.plot(epochs_range, history["train_loss"], label="Train")
ax1.plot(epochs_range, history["val_loss"],   label="Val")
ax1.set_title("Loss per Epoch"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(True)
 
ax2.plot(epochs_range, history["train_acc"], label="Train")
ax2.plot(epochs_range, history["val_acc"],   label="Val")
ax2.set_title("Akurasi per Epoch"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Akurasi (%)")
ax2.legend(); ax2.grid(True)
 
plt.tight_layout()
hist_path = os.path.join(args.output_dir, "training_history.png")
plt.savefig(hist_path, dpi=150)
plt.show()
print(f"Grafik training disimpan di: {hist_path}")
 