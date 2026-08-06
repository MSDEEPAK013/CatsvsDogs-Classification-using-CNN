# Cats vs Dogs Image Classifier 

A binary image classifier that distinguishes cats from dogs, built and iterated on as a first deep learning project — starting from a custom CNN and evolving into a transfer-learning model using MobileNetV2.

Final validation accuracy: ~95%

Overview

This project walks through the full lifecycle of building an image classifier:
1. A custom CNN built from scratch (Conv2D + BatchNorm + MaxPooling blocks)
2. Diagnosing training instability from the logs (fluctuating val_loss, a loss spike)
3. Fixing the architecture and training process
4. Switching to transfer learning (MobileNetV2) for a stronger final result

Dataset

- Source: [Dogs vs. Cats — Kaggle](https://www.kaggle.com/c/dogs-vs-cats)
- Classes: 2 (Cat, Dog)
- Input size: 256×256×3

## Architecture

### v1 — Custom CNN (baseline)

RandomFlip → RandomRotation → RandomZoom
Conv2D(32) → BatchNorm → MaxPool
Conv2D(64) → BatchNorm → MaxPool
Conv2D(128) → BatchNorm → MaxPool
Flatten
Dense(64) → BatchNorm → Dropout(0.5)
Dense(32) → BatchNorm → Dropout(0.5)
Dense(1, sigmoid)

**Issue found:** `Flatten()` fed a very large number of parameters into the dense head, and the learning rate was too aggressive — together these caused unstable validation loss across epochs (e.g. a spike to 2.70 at epoch 8 despite steady training accuracy).

### v2 — Improved custom CNN
Key changes:
- Added `ReduceLROnPlateau` and `EarlyStopping` callbacks to stabilize training and prevent wasted epochs
- Tuned learning rate to reduce the val_loss instability seen in v1
- Kept `Flatten()` for the transition into the dense head

### v3 — Transfer Learning (final model)

MobileNetV2 (frozen, imagenet weights) → Flatten
Dense(128, relu) → Dropout(0.3)
Dense(1, sigmoid)

Later fine-tuned by unfreezing the top ~30 layers of MobileNetV2 with a very low learning rate (1e-5) for a further accuracy boost.

## Key Decisions & Why

| Decision | Reasoning |
|---|---|
| `padding='same'` over `'valid'` | Keeps spatial dimensions predictable across blocks and avoids losing edge information, since MaxPooling already handles downsampling |
| `Flatten()` before the dense head | Preserves full spatial feature detail going into the classifier; worked well here paired with the pretrained MobileNetV2 backbone and dropout to control overfitting |
| `BatchNormalization` after every Conv/Dense layer it stabilizes the distribution of activations between layers, allows higher learning rates, smooths the loss landscape |
| `Dropout` after Dense layers | Prevents co-adapted neurons and overfitting; rates graduated (0.4) rather than flat 0.5 across all layers |
| `ReduceLROnPlateau` + `EarlyStopping` | Detects and corrects learning rate instability mid-training, and stops once validation loss stops improving, restoring best weights |
| Transfer learning (MobileNetV2) | Faster convergence and higher accuracy ceiling than training from scratch on a moderate-sized dataset; also relevant for edge/embedded deployment given lightweight architecture |

## Results

| Model Version | Val Accuracy |

| v1 — Custom CNN (baseline) | ~80% (unstable across epochs) |
| v2 — Improved Custom CNN | ~88–90% |
| v3 — Transfer Learning (MobileNetV2) | **~95%** |



