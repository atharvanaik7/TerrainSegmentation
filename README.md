# Off-Road Terrain Segmentation

Pixel-level semantic segmentation of off-road terrain using a **ResNet34 + U-Net** architecture, trained on the [Yamaha-CMU Off-Road Dataset](http://theairlab.org/yamaha-offroad-dataset/). The model classifies each pixel in a driving image into terrain categories (e.g., sky, vegetation, dirt, obstacle), enabling perception for autonomous off-road vehicles.

---

## Overview

| | |
|---|---|
| **Architecture** | U-Net with ResNet34 encoder (ImageNet pretrained) |
| **Framework** | TensorFlow / Keras + `segmentation_models` |
| **Dataset** | Yamaha-CMU Off-Road (yamaha_v0) |
| **Input Size** | 128 × 128 × 3 (RGB) |
| **Loss** | Categorical Cross-Entropy |
| **Optimizer** | Adam |
| **Metrics** | Accuracy, Mean IoU |
| **Epochs** | 100 |
| **Batch Size** | 16 |
| **Train/Test Split** | 80% / 20% |

---

## How It Works

### 1. Data Loading
RGB training images and their corresponding RGB segmentation masks are loaded from disk and resized to 128×128. A `class_dict.csv` file maps each RGB color in the mask to a class index (e.g., sky, dirt, vegetation, obstacle, etc.).

### 2. Label Encoding
Each RGB mask is converted to a single-channel label map using the class dictionary — every pixel becomes an integer class index. Labels are then one-hot encoded for multi-class classification.

### 3. Model
A U-Net is instantiated with a **ResNet34 encoder** pretrained on ImageNet via the `segmentation_models` library. The encoder captures rich spatial features at multiple scales; the U-Net decoder upsamples them back to the original resolution with skip connections to preserve spatial detail.

```python
import segmentation_models as sm

BACKBONE = 'resnet34'
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 4. Preprocessing
Inputs are preprocessed using the ResNet34-specific normalization from `segmentation_models` to match the pretrained encoder's expected input distribution.

### 5. Training
The model is trained for 100 epochs with a batch size of 16. Training and validation accuracy curves are plotted post-training. The best model is saved as `semantic_yamaha_augmented.h5`.

### 6. Inference
The trained model runs inference on:
- **Static test images** — side-by-side visualization of input, ground truth, and predicted label map
- **Individual validation images** from the Yamaha dataset
- **Video feed** — frame-by-frame real-time prediction overlaid on driving footage from an Unreal Engine simulation environment

Predicted class index maps are converted back to RGB using the colormap from `class_dict.csv` for visualization.

---

## Project Structure

```
.
├── resnet34_unet.ipynb         # Main training and inference notebook
├── class_dict.csv              # RGB color → class label mapping
├── semantic_yamaha_augmented.h5  # Saved trained model
└── README.md
```

> **Note:** The Yamaha dataset and raw image/label directories are not included in this repo due to size. See dataset setup below.

---

## Dataset Setup

1. Download the [Yamaha-CMU Off-Road Dataset](http://theairlab.org/yamaha-offroad-dataset/)
2. Extract and organize as follows:
```
E:/Offroad database/
├── Rest of the files/
│   └── Training Images and Labels/
│       ├── images/
│       └── labels/
└── yamaha_seg/
    ├── class_dict.csv
    └── yamaha_v0/
        └── valid/
            └── iid000856/
                ├── rgb.jpg
                └── labels.png
```
3. Update the directory paths in the notebook cells to match your local setup.

---

## Requirements

```
tensorflow
keras
segmentation-models
opencv-python
numpy
pandas
matplotlib
seaborn
scikit-learn
patchify
Pillow
tqdm
```

Install with:
```bash
pip install -r requirements.txt
```

> **Note:** `segmentation_models` requires TensorFlow as its backend. Set it explicitly before use:
> ```python
> import segmentation_models as sm
> sm.set_framework('tf.keras')
> ```

---

## Results

The model was evaluated on a held-out 20% test split. Predictions capture the general terrain structure across classes. Performance is visualized via:
- Training vs. validation accuracy curves
- Side-by-side image / ground truth / prediction plots
- Live frame-by-frame prediction on a simulated driving video

---

## Key Design Decisions

- **ResNet34 backbone** selected over lighter backbones (e.g., ResNet18) based on feature embedding quality — t-SNE cluster separability analysis on encoder outputs showed better class separation with the deeper backbone.
- **ImageNet pretrained weights** used to leverage low-level feature detectors (edges, textures) relevant to outdoor terrain.
- **128×128 input resolution** chosen to balance spatial detail with memory and compute constraints.
- **Categorical cross-entropy** used over Dice loss given the multi-class nature of the task (up to ~24 terrain classes in the Yamaha dataset).
