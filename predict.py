"""
Cara pakai:
    python predict.py --image ./contoh_sel.png
    python predict.py --image ./contoh_sel.png --model ./output/best_model.pth
"""

import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

# ARGUMEN
parser = argparse.ArgumentParser(description="Prediksi Malaria dari Gambar Sel Darah Merah")
parser.add_argument("--image", type=str, required=True, help="Path ke gambar sel darah merah")
parser.add_argument("--model", type=str, default="./output/best_model.pth", help="Path ke file model (.pth)")
args = parser.parse_args()

# Validasi file
if not os.path.exists(args.image):
    print(f"[ERROR] Gambar tidak ditemukan: {args.image}")
    sys.exit(1)
if not os.path.exists(args.model):
    print(f"[ERROR] Model tidak ditemukan: {args.model}")
    print("Pastikan kamu sudah menjalankan train.py terlebih dahulu.")
    sys.exit(1)

# LOAD MODEL
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(args.model, map_location=DEVICE)
CLASSES    = checkpoint["classes"]     # ['Parasitized', 'Uninfected']
IMG_SIZE   = checkpoint.get("img_size", 128)
VAL_ACC    = checkpoint.get("val_acc", None)

# Bangun ulang arsitektur yang sama dengan train.py
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(CLASSES))
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print(f"Model loaded  : {args.model}")
if VAL_ACC:
    print(f"Val Accuracy  : {VAL_ACC:.2f}%")
print(f"Kelas         : {CLASSES}")
print(f"Device        : {DEVICE}\n")

# PREPROCESSING GAMBAR
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

img_original = Image.open(args.image).convert("RGB")
img_tensor   = transform(img_original).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

# PREDIKSI
with torch.no_grad():
    logits       = model(img_tensor)                           # (1, 2)
    probs        = torch.softmax(logits, dim=1)[0]            # (2,)
    pred_idx     = probs.argmax().item()
    pred_label   = CLASSES[pred_idx]
    confidence   = probs[pred_idx].item() * 100

# Prob per kelas
prob_parasitized = probs[CLASSES.index("Parasitized")].item() * 100
prob_uninfected  = probs[CLASSES.index("Uninfected")].item()  * 100

is_malaria = pred_label == "Parasitized"

# TAMPILKAN HASIL DI TERMINAL
print("=" * 45)
print("        HASIL DETEKSI MALARIA")
print("=" * 45)
print(f"  File gambar  : {os.path.basename(args.image)}")
print(f"  Prediksi     : {'🔴 POSITIF MALARIA' if is_malaria else '🟢 NEGATIF (Normal)'}")
print(f"  Confidence   : {confidence:.1f}%")
print()
print(f"  Prob Parasitized : {prob_parasitized:.1f}%")
print(f"  Prob Uninfected  : {prob_uninfected:.1f}%")
print("=" * 45)

if is_malaria:
    print("\nPERHATIAN: Sel darah terdeteksi TERINFEKSI parasit malaria.")
else:
    print("\nSel darah tampak NORMAL (tidak terdeteksi parasit).")

# 6. VISUALISASI HASIL
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# ── Panel kiri: gambar asli ─────────────
axes[0].imshow(img_original)
axes[0].set_title("Input: Gambar Sel Darah Merah", fontsize=11)
axes[0].axis("off")

border_color = "#e53935" if is_malaria else "#43a047"
for spine in axes[0].spines.values():
    spine.set_edgecolor(border_color)
    spine.set_linewidth(4)

# ── Panel kanan: bar chart probabilitas ─
bar_colors = []
for cls in CLASSES:
    if cls == pred_label:
        bar_colors.append("#e53935" if is_malaria else "#43a047")
    else:
        bar_colors.append("#bdbdbd")

probs_list = [prob_parasitized, prob_uninfected]
bars = axes[1].barh(CLASSES, probs_list, color=bar_colors, edgecolor="white", height=0.5)

# Tambahkan label persentase di ujung bar
for bar, prob in zip(bars, probs_list):
    axes[1].text(
        min(prob + 1, 97), bar.get_y() + bar.get_height() / 2,
        f"{prob:.1f}%", va="center", ha="left", fontsize=11, fontweight="bold"
    )

axes[1].set_xlim(0, 105)
axes[1].set_xlabel("Probabilitas (%)", fontsize=10)
axes[1].set_title("Distribusi Probabilitas Prediksi", fontsize=11)
axes[1].axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
axes[1].grid(axis="x", alpha=0.3)

# Judul keseluruhan
status_text = "🔴 POSITIF MALARIA" if is_malaria else "🟢 NEGATIF (Normal)"
fig.suptitle(
    f"Hasil Deteksi: {status_text}  |  Confidence: {confidence:.1f}%",
    fontsize=13, fontweight="bold",
    color="#e53935" if is_malaria else "#2e7d32"
)

plt.tight_layout()
out_path = os.path.splitext(args.image)[0] + "_result.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nHasil visual disimpan di: {out_path}")