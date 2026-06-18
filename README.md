# 🏛️ AI-Powered Egyptian Artifact Recognition — From Museum Walls to Your Pocket

### A Full-Stack Graduation Project: Computer Vision · NLP · Mobile Application · Real-World Deployment

---

## 💡 The Idea

> *"Point your phone at an ancient Egyptian artifact — and instantly learn its story."*

We built an **end-to-end AI chatbot** that allows users to photograph any Egyptian artifact and receive rich, contextual information about it in natural language. The system works in two tightly integrated layers:

1. **A Computer Vision model** — identifies the artifact from the image with 99.65% accuracy across 74 artifact classes
2. **An NLP model** — takes the classification output and generates a detailed, human-readable description of the artifact, its history, and its significance

The result is a **seamless conversational experience**: the user asks with a photo, and the AI answers with knowledge.

---

## 📱 Deployed as a Mobile Application

This is not just a research project — **the full system is live**.

The computer vision pipeline and NLP model were integrated into a **mobile application published on both Google Play and the App Store**, making Egyptian heritage accessible to tourists, students, and history enthusiasts worldwide — directly from their smartphones.

---

## 🗺️ Tested in the Real World — The Grand Egyptian Museum

The model was not only validated in the lab. **It was tested in real conditions at the Grand Egyptian Museum (GEM)**, one of the largest archaeological museums in the world. Visitors' photographs taken on-site — through glass cases, under museum lighting, from varying angles — were fed into the system, and the model performed with exceptional reliability.

This real-world validation confirmed that the museum-aware augmentation strategy and the OOD rejection filter were the right design choices.

---

## 🧠 Computer Vision Model — EfficientNet-B4

### Architecture
- **Backbone:** EfficientNet-B4 pretrained on ImageNet-1K at native resolution **380×380 px**
- **Custom Classification Head:**
  ```
  1792 → Dropout(0.50) → Linear(1024) → ReLU → BatchNorm →
         Dropout(0.40) → Linear(512)  → ReLU → BatchNorm →
         Dropout(0.30) → Linear(74)
  ```
- **Dataset:** ~31,700 images across 74 Egyptian artifact classes, sourced from museum archives and the internet

### Training Strategy — Progressive Unfreezing (3 Phases)

```
Phase 1 │ Classifier head only       │ LR = 5e-4  │ Backbone fully frozen
Phase 2 │ Last 15 backbone layers    │ LR = 1e-4  │ Partial unfreezing
Phase 3 │ Entire network             │ LR = 1e-5  │ Full end-to-end fine-tuning
```

- **Optimizer:** AdamW (weight decay = 1e-4)
- **Scheduler:** CosineAnnealingLR — smooth decay, avoids abrupt LR drops
- **Loss:** CrossEntropyLoss with Label Smoothing = 0.10 (prevents overconfidence across 74 classes)
- **Mixed Precision Training (AMP):** ~40–50% VRAM reduction
- **Converged at epoch 28 out of 50** — early stopping triggered

---

## 🔍 OOD Similarity Filter — Knowing What the Model Doesn't Know

One of the most important components of this system is the **Out-of-Distribution (OOD) Rejection Filter** — a safety mechanism that protects the pipeline from making incorrect predictions on images it was never trained to handle.

### The Core Idea
The model has only ever seen **74 specific artifact classes**. If a user photographs something unrelated — a random object, a person, or an artifact outside the training domain — a naive classifier would still force a prediction. The OOD filter prevents this entirely.

### How It Works
The filter computes **cosine similarity** between the incoming image and the full training set:

1. The backbone encoder extracts a **1792-dimensional embedding** from the input image
2. This embedding is compared against **~25,000 stored training embeddings** using cosine similarity
3. The **maximum similarity score** is extracted — measuring how close the new image is to the nearest known training example
4. If this score falls **below the calibrated threshold**, the image is flagged as out-of-domain and **rejected before classification**

### Threshold Calibration
- The threshold is set at the **5th percentile** of validation similarity scores
- This ensures **95% of known, legitimate artifact images** are accepted
- Tested against synthetic OOD inputs (random noise, solid fills, blank images) — **all reliably rejected**

```
Input Image ──► Cosine Similarity vs Training Set ──► Score < Threshold? ──► REJECT
                                                                │
                                                            Score ≥ Threshold
                                                                │
                                                           ──► PROCEED TO CLASSIFICATION
```

---

## 🤝 4-Model Ensemble — Triple External Verification

After passing the OOD filter, the image goes through **four models** in a coordinated ensemble to maximize prediction reliability.

### The Four Models

| # | Model | Type | Resolution |
|---|-------|------|------------|
| 1 | **EfficientNet-B4** (fine-tuned) | Supervised CNN | 380×380 |
| 2 | **CLIP ViT-L/14@336** | Zero-shot Vision-Language | 336×336 |
| 3 | **CLIP ViT-B/32** | Zero-shot Vision-Language | 224×224 |
| 4 | **CLIP ViT-B/16** | Zero-shot Vision-Language | 224×224 |

The three CLIP models serve as **independent external verifiers** — they were not trained on this dataset and bring a completely different, zero-shot perspective to each prediction. This design means the final answer is never the opinion of a single model alone.

### Voting Logic

```
If EfficientNet-B4 confidence ≥ 45%  →  Direct return (CLIP skipped — fast path)
If EfficientNet-B4 confidence < 45%  →  All 3 CLIP models cast votes
                                         └─► ≥ 2 models agree + avg conf ≥ 42%  →  Consensus winner
                                         └─► All disagree  →  Fall back to EfficientNet-B4
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Total Dataset Size | 31,692 images |
| Training Set | 25,353 images |
| Validation Set | 6,339 images |
| Number of Classes | 74 artifact types |
| Best Validation Accuracy | **99.65%** |
| Test-Time Augmentation (TTA) Accuracy | **99.62%** |
| Final Train / Val Gap | **−0.08%** ✅ |
| Epochs to Convergence | 28 / 50 |

The near-zero generalisation gap (−0.08%) confirms **zero overfitting** — the model generalises to unseen images just as reliably as it performs on training data.

---

## 🖼️ Museum-Aware Data Augmentation

The training pipeline was specifically engineered to bridge the domain gap between clean internet photographs and real museum conditions:

| Augmentation | Why It Matters |
|---|---|
| `RandomResizedCrop (scale 0.6–1.0)` | Variable framing and zoom from different photographers |
| `ColorJitter (brightness/contrast 0.50)` | UV spotlights and fluorescent display cabinets |
| `GaussianBlur (σ 0.1–2.0)` | Museum glass reflections and handheld camera blur |
| `RandomAdjustSharpness` | Soft focus through display case glass surfaces |
| `RandomGrayscale (p=0.10)` | Historical monochrome archive photographs |
| `RandomPosterize` | JPEG compression artifacts from web-sourced images |
| `RandomEqualize` | Over-exposed and under-exposed museum captures |
| `RandomErasing (p=0.45)` | Forces the network to use global, distributed features |

---

## ⚡ Full Inference Pipeline

```
📷 Input Image
       │
       ▼
┌──────────────────────────────────┐
│   OOD Cosine Similarity Filter   │ ──► ✗ REJECT  (image too dissimilar to training set)
└──────────────────────────────────┘
       │ ✓ IN-DOMAIN
       ▼
┌──────────────────────────────────┐
│   EfficientNet-B4 Inference      │ ──► ✓ DIRECT RETURN  (if confidence ≥ 45%)
└──────────────────────────────────┘
       │ confidence < 45%
       ▼
┌──────────────────────────────────────────────────────┐
│   CLIP Ensemble — 3 External Zero-Shot Verifiers     │
│   · CLIP ViT-L/14@336                               │
│   · CLIP ViT-B/32                                   │
│   · CLIP ViT-B/16                                   │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│   Weighted Majority Vote         │ ──► Final Class + Confidence + Top-K
└──────────────────────────────────┘
       │
       ▼
🔤 NLP Model generates rich artifact description
       │
       ▼
💬 User receives conversational answer in the mobile app
```

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `EfficientNet-B4` · `OpenAI CLIP` · `Hugging Face Transformers`
`torchvision` · `Mixed Precision AMP` · `scikit-learn` · `Google Play` · `App Store`

---

## 🏺 74 Artifact Classes

Spanning pharaohs, queens, priests, temples, pyramids, statues, and royal artifacts — including:

*Akhenaten · Cleopatra · Nefertiti · Tutankhamun · Mask of Tutankhamun · Ramesses II · Sphinx · Khafre Pyramid · Khufu Pyramid · Hatshepsut Temple in Luxor · Karnak Temple · The Great Temple of Ramses II (Abu Simbel) · Goddess Isis · Goddess of War Sekhmet · Queen Hatshepsut · Ra-Horakhty · and 58 more ...*

---

*Graduation Project in Artificial Intelligence — integrating computer vision, natural language processing, ensemble learning, and mobile deployment to make Egypt's ancient heritage universally accessible.*

Download the Model : https://drive.google.com/drive/folders/1PAoJQX48Nl2tR7470-ysvwArvMQ4wN9i?usp=sharing
