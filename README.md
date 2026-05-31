# Malaria Detection Using CNN with Transfer Learning

Automated malaria parasite detection in red blood cell images using ResNet18-based transfer learning — achieving 93% accuracy on the NIH Malaria Cell Image Dataset.

---

## Overview

This project presents a deep learning system for binary classification of blood cell images into two categories:
- **Parasitized** — cell infected with Plasmodium parasite
- **Uninfected** — healthy red blood cell

The model is built on a pretrained **ResNet18** backbone fine-tuned using transfer learning, and supports **single-image inference** with confidence score output.

---

## Results

| Metric    | Parasitized | Uninfected | Macro Avg |
|-----------|-------------|------------|-----------|
| Precision | 0.92        | 0.93       | 0.93      |
| Recall    | 0.93        | 0.92       | 0.93      |
| F1-Score  | 0.92        | 0.93       | 0.93      |
| Accuracy  |             |            | 93%       |

---

## Project Structure

```
malaria-detection-cnn/
├── cell_images/              <- dataset (not included, see below)
│   ├── Parasitized/
│   └── Uninfected/
├── output/                   <- generated after training
│   ├── best_model.pth
│   ├── training_history.png
│   └── confusion_matrix.png
├── train.py                  <- training script
├── predict.py                <- single-image inference script
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses the **NIH Malaria Cell Image Dataset**, a publicly available benchmark comprising 27,560 segmented red blood cell images (13,780 Parasitized + 13,780 Uninfected).

Download the dataset from:
- NIH Official Repository: https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html
- Kaggle Mirror: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

After downloading, extract and place inside the `cell_images/` folder with the following structure:
```
cell_images/
├── Parasitized/
└── Uninfected/
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/USERNAME/malaria-detection-cnn.git
cd malaria-detection-cnn
```

**2. Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

### Training
```bash
python train.py --data_dir ./cell_images
```

Optional arguments:
```bash
python train.py \
  --data_dir ./cell_images \
  --epochs 20 \
  --batch_size 32 \
  --lr 0.0001
```

Training outputs saved to `./output/`:
- `best_model.pth` — best model checkpoint
- `training_history.png` — loss and accuracy curves
- `confusion_matrix.png` — test set evaluation

### Inference (Single Image)
```bash
python predict.py --image ./path/to/cell_image.png
```

---

## Model Architecture

| Component         | Detail                                                      |
|-------------------|-------------------------------------------------------------|
| Backbone          | ResNet18 (pretrained on ImageNet)                           |
| Strategy          | Feature extraction (backbone frozen)                        |
| Classifier        | Linear(512->256) -> ReLU -> Dropout(0.4) -> Linear(256->2) |
| Trainable params  | ~131,000                                                    |
| Total params      | ~11,000,000                                                 |

---

## Training Configuration

| Parameter     | Value                        |
|---------------|------------------------------|
| Optimizer     | Adam                         |
| Learning rate | 1e-4                         |
| Weight decay  | 1e-4                         |
| Scheduler     | CosineAnnealingLR (T_max=20) |
| Loss function | CrossEntropyLoss             |
| Epochs        | 20                           |
| Batch size    | 32                           |
| Image size    | 128x128                      |

---

## Requirements

```
torch
torchvision
matplotlib
scikit-learn
seaborn
pillow
tqdm
```

Install all at once:
```bash
pip install torch torchvision matplotlib scikit-learn seaborn pillow tqdm
```

---

## Disclaimer

This system is intended as a research and screening assistance tool only. It is not a substitute for professional medical diagnosis. Always consult a qualified medical professional for clinical decisions.

---

## License

This project is for academic and educational purposes.

---
